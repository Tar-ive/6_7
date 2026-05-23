from __future__ import annotations

import time

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
        self.gate = Pose67Gate(
            cfg.required_movements,
            cfg.window_frames,
            cfg.min_keypoint_conf,
            cfg.max_elbow_angle,
            cfg.min_wrist_gap,
            cfg.max_missing_frames,
        )
        self.pose_done = False
        self.mug_frames = 0
        self.spoof_seen = False
        self.alarm_resume_at = 0.0
        self.object_detections = []

    def run(self) -> bool:
        self.alarm.start()
        stopped = False
        try:
            for frame in self.camera.frames():
                if not self.pose_done:
                    poses = self.detector.detect(frame)
                    self.pose_done = self.gate.update(poses) or self.pose_done
                else:
                    poses = []
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
            self.camera.close()
            cv2.destroyAllWindows()
        if stopped:
            play_once(self.cfg.stopped_sound)
        return stopped

    def _update_mug_gate(self) -> bool:
        has_mug = any(self._is_mug(d) for d in self.object_detections)
        has_phone = any(self._is_phone(d) for d in self.object_detections)
        if has_phone:
            self.mug_frames = 0
            if not self.spoof_seen:
                self.alarm.stop()
                play_once(self.cfg.spoof_sound)
                self.alarm_resume_at = time.monotonic() + 1.5
                self.spoof_seen = True
            return False
        self.spoof_seen = False
        self.mug_frames = self.mug_frames + 1 if has_mug else 0
        return self.mug_frames >= self.cfg.required_mug_frames

    def _is_mug(self, detection) -> bool:
        return _matches(detection, self.cfg.mug_classes or [])

    def _is_phone(self, detection) -> bool:
        return _matches(detection, self.cfg.phone_classes or [])

    def _draw(self, frame, poses, stopped: bool) -> None:
        if not self.cfg.show:
            return
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
            f"backend {self.detector.backend_name}",
            (20, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        self._draw_objects(frame)
        cv2.imshow("YOLO Pose Alarm", frame)

    def _status_line(self) -> str:
        if not self.pose_done:
            return f"6_7 movements {self.gate.progress}"
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
