from __future__ import annotations

from .pose import (
    LEFT_ELBOW,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ELBOW,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    Pose,
    angle,
)


class Pose67Gate:
    """Robust 6/7 rep counter.

    The 6/7 gesture is both forearms held up and alternately rocked up/down like a
    balance scale. A naive "which wrist is higher, did it swap?" counter false-fires
    on idle keypoint jitter and on any casual motion. This gate only counts a swap as
    a real rep when:

    * posture is valid (arm keypoints confident, elbows bent),
    * arms are actually raised (the higher wrist is at/above elbow level),
    * the wrists reach a real vertical separation (``min_wrist_amplitude``) on each
      side, using ``min_wrist_gap`` as a hysteresis deadband so tiny crossings near
      level do not count, and
    * the motion is sustained — if no qualifying swap happens for ``idle_reset_frames``
      the rep count decays back to zero (you have to keep going).

    The count also fully resets if the person leaves frame for more than
    ``max_missing_frames`` frames.
    """

    def __init__(
        self,
        required_movements: int = 6,
        window_frames: int = 90,
        min_conf: float = 0.25,
        max_elbow_angle: float = 140,
        min_wrist_gap: float = 20,
        max_missing_frames: int = 12,
        min_wrist_amplitude: float = 55,
        require_arms_up: bool = True,
        idle_reset_frames: int = 45,
    ) -> None:
        self.required_movements = required_movements
        self.min_conf = min_conf
        self.max_elbow_angle = max_elbow_angle
        self.min_wrist_gap = min_wrist_gap
        self.max_missing_frames = max_missing_frames
        # amplitude must clear the deadband, otherwise the deadband is meaningless
        self.min_wrist_amplitude = max(min_wrist_amplitude, min_wrist_gap)
        self.require_arms_up = require_arms_up
        self.idle_reset_frames = idle_reset_frames
        self.confirmed_side: str | None = None
        self.movements = 0
        self.missing_frames = 0
        self.frames_since_flip = 0

    def update(self, poses: list[Pose]) -> bool:
        pose = self._best(poses)
        if pose is None or not self._valid(pose):
            self._missing()
            return self._done

        self.missing_frames = 0
        ly, ry = pose.xy[LEFT_WRIST][1], pose.xy[RIGHT_WRIST][1]
        if self.require_arms_up and not self._arms_up(pose):
            self._idle()
            return self._done

        # diff > 0  => left wrist higher (smaller y); diff < 0 => right wrist higher
        diff = float(ry - ly)
        if abs(diff) < self.min_wrist_amplitude:
            self._idle()
            return self._done

        side = "left_high" if diff > 0 else "right_high"
        if self.confirmed_side is None:
            self.confirmed_side = side
            self.frames_since_flip = 0
        elif side != self.confirmed_side:
            self.movements += 1
            self.confirmed_side = side
            self.frames_since_flip = 0
        else:
            self._idle()
        return self._done

    @property
    def progress(self) -> str:
        return f"{self.movement_count}/{self.required_movements}"

    @property
    def movement_count(self) -> int:
        return min(self.movements, self.required_movements)

    @property
    def _done(self) -> bool:
        return self.movements >= self.required_movements

    def _idle(self) -> None:
        """A valid-but-not-swapping frame: age the rep streak and decay if stale."""
        self.frames_since_flip += 1
        if self.frames_since_flip > self.idle_reset_frames:
            self.movements = 0
            self.confirmed_side = None
            self.frames_since_flip = 0

    def _missing(self) -> None:
        self.missing_frames += 1
        if self.missing_frames > self.max_missing_frames:
            self.movements = 0
            self.confirmed_side = None
            self.frames_since_flip = 0

    def _best(self, poses: list[Pose]) -> Pose | None:
        if not poses:
            return None
        return max(poses, key=_area)

    def _valid(self, p: Pose) -> bool:
        needed = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST]
        if any(p.conf[i] < self.min_conf for i in needed):
            return False
        left_angle = angle(p.xy[LEFT_SHOULDER], p.xy[LEFT_ELBOW], p.xy[LEFT_WRIST])
        right_angle = angle(p.xy[RIGHT_SHOULDER], p.xy[RIGHT_ELBOW], p.xy[RIGHT_WRIST])
        return left_angle < self.max_elbow_angle and right_angle < self.max_elbow_angle

    def _arms_up(self, p: Pose) -> bool:
        """The higher of the two wrists must be at/above the elbows (arms raised)."""
        higher_wrist_y = min(p.xy[LEFT_WRIST][1], p.xy[RIGHT_WRIST][1])
        elbow_y = (p.xy[LEFT_ELBOW][1] + p.xy[RIGHT_ELBOW][1]) / 2.0
        return higher_wrist_y <= elbow_y


def _area(p: Pose) -> float:
    if p.box is None:
        return 0.0
    x1, y1, x2, y2 = p.box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)
