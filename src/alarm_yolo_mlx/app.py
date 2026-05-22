from __future__ import annotations

import cv2

from .alarm import Alarm
from .camera import Camera
from .config import AppConfig
from .detector import Detection, YoloMlxDetector
from .gate import StopGate
from .motion_gate import AlternatingHandsGate
from .tracker import IouTracker


class AlarmStopApp:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.camera = Camera(cfg.source or cfg.camera_index)
        self.detector = YoloMlxDetector(
            cfg.weights,
            cfg.conf,
            cfg.imgsz,
            cfg.target_classes,
            cfg.class_names,
        )
        self.tracker = IouTracker(cfg.track_iou)
        self.gate = StopGate(cfg.required_hits, cfg.window_frames)
        self.motion_gate = AlternatingHandsGate(cfg.motion_class, cfg.min_switches)
        self.alarm = Alarm(cfg.alarm_sound)

    def run(self) -> bool:
        self.alarm.start()
        stopped = False
        try:
            for frame in self.camera.frames():
                detections = self.detector.detect(frame)
                detections = self.tracker.update(detections)
                stopped = self._stopped(detections)
                self._draw(frame, detections, stopped)
                if stopped:
                    self.alarm.stop()
                else:
                    self.alarm.tick()
                if self.cfg.show and cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                if stopped:
                    break
        finally:
            self.alarm.stop()
            self.camera.close()
            cv2.destroyAllWindows()
        return stopped

    def _stopped(self, detections: list[Detection]) -> bool:
        if self.cfg.stop_mode == "static":
            return self.gate.update(bool(detections))
        return self.motion_gate.update(detections)

    def _draw(self, frame, detections: list[Detection], stopped: bool) -> None:
        if not self.cfg.show:
            return
        for d in detections:
            x1, y1, x2, y2 = map(int, d.xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 220, 70), 2)
            text = f"{d.name} {d.conf:.2f}" + (f" #{d.track_id}" if d.track_id else "")
            cv2.putText(
                frame,
                text,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (30, 220, 70),
                2,
            )
        label = "ALARM STOPPED" if stopped else "ALARM ACTIVE"
        color = (30, 220, 70) if stopped else (20, 20, 230)
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.imshow("MLX YOLO Alarm", frame)
