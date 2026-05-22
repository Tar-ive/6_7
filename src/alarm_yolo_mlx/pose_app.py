from __future__ import annotations

import cv2

from .alarm import Alarm
from .camera import Camera
from .config import PoseConfig
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
    def __init__(self, cfg: PoseConfig) -> None:
        self.cfg = cfg
        self.camera = Camera(cfg.source or cfg.camera_index)
        self.detector = make_pose_detector(cfg.backend, cfg.weights, cfg.conf, cfg.imgsz)
        print(f"pose backend={self.detector.backend_name} imgsz={cfg.imgsz} conf={cfg.conf}")
        self.gate = Pose67Gate(
            cfg.required_movements,
            cfg.window_frames,
            cfg.min_keypoint_conf,
            cfg.max_elbow_angle,
            cfg.min_wrist_gap,
            cfg.max_missing_frames,
        )
        self.alarm = Alarm(cfg.alarm_sound)

    def run(self) -> bool:
        self.alarm.start()
        stopped = False
        try:
            for frame in self.camera.frames():
                poses = self.detector.detect(frame)
                stopped = self.gate.update(poses)
                self._draw(frame, poses, stopped)
                self.alarm.stop() if stopped else self.alarm.tick()
                if self.cfg.show and cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                if stopped:
                    break
        finally:
            self.alarm.stop()
            self.camera.close()
            cv2.destroyAllWindows()
        return stopped

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
            f"6_7 movements {self.gate.progress}",
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
        cv2.imshow("YOLO Pose Alarm", frame)
