from __future__ import annotations

import random
import subprocess
import time
from collections import deque

import cv2

from .alarm import Alarm, play_once
from .camera import Camera
from .config import PoseConfig
from .detector import YoloMlxDetector
from .pose import (
    LEFT_ELBOW,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ELBOW,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    make_pose_detector,
)
from .pose_gate import Pose67Gate


class PoseAlarmApp:
    def __init__(self, cfg: PoseConfig, alarm: Alarm | None = None, detector=None) -> None:
        self.cfg = cfg
        self.alarm = alarm or Alarm(cfg.alarm_sound)
        self.camera = Camera(cfg.source or cfg.camera_index)
        self.detector = detector or make_pose_detector(
            cfg.backend, cfg.weights, cfg.conf, cfg.imgsz
        )
        print(f"pose backend={self.detector.backend_name} imgsz={cfg.imgsz} conf={cfg.conf}")
        proof_classes = [*(cfg.mug_classes or []), *(cfg.phone_classes or [])]
        self.object_detector = YoloMlxDetector(
            cfg.object_weights,
            cfg.object_conf,
            cfg.imgsz,
            proof_classes,
            cfg.object_class_names,
        )
        print(f"object backend=mlx:{cfg.object_weights} conf={cfg.object_conf}")
        self.movement_sets = random.randint(cfg.movement_sets_min, cfg.movement_sets_max)
        self.movements_per_set = cfg.required_movements
        self.completed_sets = 0
        self.gate = Pose67Gate(
            self.movements_per_set * self.movement_sets,
            cfg.window_frames,
            cfg.min_keypoint_conf,
            cfg.max_elbow_angle,
            cfg.min_wrist_gap,
            cfg.max_missing_frames,
            min_wrist_amplitude=cfg.min_wrist_amplitude,
            require_arms_up=cfg.require_arms_up,
            idle_reset_frames=cfg.idle_reset_frames,
        )
        self.pose_done = False
        self.mug_history: deque[bool] = deque(maxlen=cfg.detection_window)
        self.phone_history: deque[bool] = deque(maxlen=cfg.detection_window)
        self.mug_frames = 0
        self.spoof_seen = False
        self.full_phone_warning_played = False
        self.mug_encouragement_spoken = False
        self.alarm_resume_at = 0.0
        self.object_detections = []
        self.voice_proc: subprocess.Popen | None = None
        self.voice_tag: str | None = None
        self.voice_pauses_detection = False
        self.voice_queue: list[tuple[str, str | None, bool, str | None]] = []

    def run(self) -> bool:
        self.alarm.set_volume(self.cfg.alarm_full_volume)
        self.alarm.start()
        self._queue_speech(
            "Do the 6 7.",
            self.cfg.start_prompt_sound,
        )
        stopped = False
        try:
            for frame in self.camera.frames():
                self._update_speech()
                detection_paused = self.voice_pauses_detection
                if not self.pose_done:
                    poses = [] if detection_paused else self.detector.detect(frame)
                    if not detection_paused:
                        self.pose_done = self.gate.update(poses) or self.pose_done
                        self._speak_set_progress()
                        if self.pose_done:
                            self._queue_speech(
                                "Get the mug.",
                                self.cfg.mug_prompt_sound,
                            )
                else:
                    poses = []
                    if not detection_paused:
                        self.object_detections = self.object_detector.detect(frame)
                        stopped = self._update_mug_gate()
                self._draw(frame, poses, stopped)
                if stopped:
                    self.alarm.stop()
                elif time.monotonic() >= self.alarm_resume_at:
                    self.alarm.tick()
                if self.cfg.show and cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                if stopped:
                    break
        finally:
            self.alarm.stop()
            self._stop_speech()
            self.camera.close()
            cv2.destroyAllWindows()
        if stopped:
            play_once(
                self.cfg.stopped_sound,
                wait=True,
                text="Good morning. Rise and shine!",
            )
        return stopped

    def _update_mug_gate(self) -> bool:
        # Temporal smoothing: require N-of-M recent frames so a single mis-detection
        # neither resets the mug hold nor false-triggers the "na na buddy" spoof voice.
        self.mug_history.append(any(self._is_mug(d) for d in self.object_detections))
        self.phone_history.append(any(self._is_phone(d) for d in self.object_detections))
        has_mug = sum(self.mug_history) >= self.cfg.mug_min_hits
        has_phone = sum(self.phone_history) >= self.cfg.phone_min_hits
        if has_phone:
            self.mug_frames = 0
            if not self._phone_speech_active:
                tag = "phone_full" if not self.full_phone_warning_played else "phone"
                self._queue_speech(
                    "Na na buddy! That won't work, not with me.",
                    self.cfg.spoof_sound,
                    tag=tag,
                )
                self.full_phone_warning_played = True
            return False
        self._stop_phone_speech()
        self.mug_frames = self.mug_frames + 1 if has_mug else 0
        if (
            self.mug_frames >= self.cfg.mug_encouragement_frames
            and not self.mug_encouragement_spoken
        ):
            self._queue_speech(
                "That's it. Hold steady.",
                tag="mug_encouragement",
            )
            self.mug_encouragement_spoken = True
        return self.mug_frames >= self.cfg.required_mug_frames

    def _speak_set_progress(self) -> None:
        done = self.gate.movement_count // self.movements_per_set
        if done <= self.completed_sets:
            return
        self.completed_sets = done
        remaining = self.movement_sets - done
        prefix = "One" if done == 1 else str(done)
        if remaining == 1:
            self._queue_speech(f"{prefix} down. One more set.")
        elif remaining > 1:
            self._queue_speech(f"{prefix} down. {remaining} more.")

    def _queue_speech(
        self,
        text: str,
        sound: str | None = None,
        *,
        pause_detection: bool = False,
        tag: str | None = None,
    ) -> None:
        self.voice_queue.append((text, sound, pause_detection, tag))
        self._update_speech()

    def _update_speech(self) -> None:
        if self.voice_proc and self.voice_proc.poll() is None:
            return
        if self.voice_proc:
            self.voice_proc = None
            self.voice_tag = None
            self.voice_pauses_detection = False
            self.alarm.set_volume(self.cfg.alarm_full_volume)
        if not self.voice_queue:
            return
        text, sound, pause_detection, tag = self.voice_queue.pop(0)
        self.alarm.set_volume(self.cfg.alarm_duck_volume)
        self.voice_proc = play_once(sound, text=text)
        self.voice_tag = tag
        self.voice_pauses_detection = pause_detection
        if not self.voice_proc:
            self.voice_tag = None
            self.voice_pauses_detection = False
            self.alarm.set_volume(self.cfg.alarm_full_volume)

    def _stop_speech(self) -> None:
        self.voice_queue.clear()
        if self.voice_proc and self.voice_proc.poll() is None:
            self.voice_proc.terminate()
        self.voice_proc = None
        self.voice_tag = None
        self.voice_pauses_detection = False

    @property
    def _phone_speech_active(self) -> bool:
        if self.voice_tag == "phone" and self.voice_proc and self.voice_proc.poll() is None:
            return True
        return any(tag == "phone" for _, _, _, tag in self.voice_queue)

    def _stop_phone_speech(self) -> None:
        self.voice_queue = [item for item in self.voice_queue if item[3] != "phone"]
        if self.voice_tag != "phone":
            return
        if self.voice_proc and self.voice_proc.poll() is None:
            self.voice_proc.terminate()
        self.voice_proc = None
        self.voice_tag = None
        self.voice_pauses_detection = False
        self.alarm.set_volume(self.cfg.alarm_full_volume)

    def _is_mug(self, detection) -> bool:
        return detection.conf >= self.cfg.mug_conf and _matches(detection, self.cfg.mug_classes or [])

    def _is_phone(self, detection) -> bool:
        return (
            detection.conf >= self.cfg.phone_conf
            and _matches(detection, self.cfg.phone_classes or [])
        )

    def _draw(self, frame, poses, stopped: bool) -> None:
        if not self.cfg.show:
            return
        h, w = frame.shape[:2]
        for pose in poses:
            if pose.box:
                x1, y1, x2, y2 = map(int, pose.box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 180, 255), 2)
                if pose.track_id:
                    cv2.putText(
                        frame,
                        f"person #{pose.track_id}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (80, 180, 255),
                        2,
                    )
            for a, b in [
                (LEFT_SHOULDER, LEFT_ELBOW),
                (LEFT_ELBOW, LEFT_WRIST),
                (RIGHT_SHOULDER, RIGHT_ELBOW),
                (RIGHT_ELBOW, RIGHT_WRIST),
            ]:
                if (
                    pose.conf[a] > self.cfg.min_keypoint_conf
                    and pose.conf[b] > self.cfg.min_keypoint_conf
                ):
                    pa, pb = tuple(map(int, pose.xy[a])), tuple(map(int, pose.xy[b]))
                    cv2.line(frame, pa, pb, (30, 220, 70), 3)
                    cv2.circle(frame, pa, 4, (30, 220, 70), -1)
                    cv2.circle(frame, pb, 4, (30, 220, 70), -1)
        label = "ALARM STOPPED" if stopped else "ALARM ACTIVE"
        color = (30, 220, 70) if stopped else (20, 20, 230)
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(
            frame,
            self._status_line(),
            (20, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "YOLO26 MLX | Apple Silicon",
            (20, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (160, 160, 160),
            1,
        )
        if not stopped:
            self._draw_stage_progress(frame, w)
        self._draw_objects(frame)
        cv2.imshow("YOLO Pose Alarm", frame)

    def _draw_stage_progress(self, frame, frame_width: int) -> None:
        bar_x = 20
        bar_y = 130
        bar_w = frame_width - 40
        bar_h = 18
        if not self.pose_done:
            total = self.gate.required_movements
            done = min(self.gate.movements, total)
            self._draw_progress_bar(
                frame, bar_x, bar_y, bar_w, bar_h,
                done / total if total > 0 else 0,
                (30, 200, 80),
                f"{done}/{total} movements",
            )
        else:
            required = self.cfg.required_mug_frames
            self._draw_progress_bar(
                frame, bar_x, bar_y, bar_w, bar_h,
                min(self.mug_frames / required, 1.0) if required > 0 else 0,
                (40, 180, 255),
                f"{min(self.mug_frames, required)}/{required} mug frames",
            )

    @staticmethod
    def _draw_progress_bar(
        frame,
        x: int,
        y: int,
        width: int,
        height: int,
        progress: float,
        color_fill,
        label: str = "",
    ) -> None:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 40, 40), -1)
        fill_w = int(width * min(max(progress, 0.0), 1.0))
        if fill_w > 0:
            cv2.rectangle(frame, (x, y), (x + fill_w, y + height), color_fill, -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (100, 100, 100), 1)
        if label:
            cv2.putText(
                frame,
                label,
                (x + 6, y + height - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (230, 230, 230),
                1,
            )

    def _status_line(self) -> str:
        if not self.pose_done:
            return f"6_7 set {self.completed_sets}/{self.movement_sets} | movements {self.gate.progress}"
        return f"GET MUG {self.mug_frames}/{self.cfg.required_mug_frames} | phone blocks unlock"

    def _draw_objects(self, frame) -> None:
        if not self.pose_done:
            return
        for det in self.object_detections:
            if self._is_mug(det):
                color, label = (40, 220, 80), "mug"
            elif self._is_phone(det):
                color, label = (40, 40, 240), "phone - na na buddy"
            else:
                continue
            x1, y1, x2, y2 = map(int, det.xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} {det.conf:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )


def _matches(detection, classes: list) -> bool:
    wanted = {str(c) for c in classes}
    return str(detection.cls) in wanted or detection.name in wanted
