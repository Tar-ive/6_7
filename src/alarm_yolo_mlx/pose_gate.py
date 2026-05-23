from __future__ import annotations

from collections import deque

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
    def __init__(
        self,
        required_movements: int = 6,
        window_frames: int = 90,
        min_conf: float = 0.25,
        max_elbow_angle: float = 140,
        min_wrist_gap: float = 20,
        max_missing_frames: int = 12,
    ) -> None:
        self.required_movements = required_movements
        self.min_conf = min_conf
        self.max_elbow_angle = max_elbow_angle
        self.min_wrist_gap = min_wrist_gap
        self.max_missing_frames = max_missing_frames
        self.orders: deque[str] = deque(maxlen=window_frames)
        self.last_order: str | None = None
        self.movements = 0
        self.missing_frames = 0

    def update(self, poses: list[Pose]) -> bool:
        pose = self._best(poses)
        if pose is None or not self._valid(pose):
            self._missing()
            return False
        ly, ry = pose.xy[LEFT_WRIST][1], pose.xy[RIGHT_WRIST][1]
        if abs(ly - ry) < self.min_wrist_gap:
            self._missing("level")
            return False
        self.missing_frames = 0
        order = "left_high" if ly < ry else "right_high"
        self.orders.append(order)
        if self.last_order and order != self.last_order:
            self.movements += 1
        self.last_order = order
        return self.movements >= self.required_movements

    @property
    def progress(self) -> str:
        return f"{min(self.movements, self.required_movements)}/{self.required_movements}"

    @property
    def movement_count(self) -> int:
        return min(self.movements, self.required_movements)

    def _missing(self, label: str = "missing") -> None:
        self.orders.append(label)
        self.missing_frames += 1
        if self.missing_frames > self.max_missing_frames:
            self.last_order = None

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


def _area(p: Pose) -> float:
    if p.box is None:
        return 0.0
    x1, y1, x2, y2 = p.box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)
