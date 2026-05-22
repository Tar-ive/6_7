#!/usr/bin/env python3
from __future__ import annotations

import argparse

from alarm_yolo_mlx.camera import Camera
from alarm_yolo_mlx.config import AppConfig
from alarm_yolo_mlx.detector import YoloMlxDetector
from alarm_yolo_mlx.motion_gate import AlternatingHandsGate
from alarm_yolo_mlx.tracker import IouTracker


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/alarm.yaml")
    p.add_argument("--source", required=True)
    p.add_argument("--weights")
    p.add_argument("--conf", type=float)
    p.add_argument("--max-frames", type=int, default=300)
    args = p.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    if args.weights:
        cfg.weights = args.weights
    if args.conf is not None:
        cfg.conf = args.conf

    camera = Camera(args.source)
    detector = YoloMlxDetector(cfg.weights, cfg.conf, cfg.imgsz, cfg.target_classes, cfg.class_names)
    tracker = IouTracker(cfg.track_iou)
    gate = AlternatingHandsGate(cfg.motion_class, cfg.min_switches)

    try:
        for frame_no, frame in enumerate(camera.frames(), start=1):
            detections = tracker.update(detector.detect(frame))
            if gate.update(detections):
                print(f"stop=true frame={frame_no} detections={len(detections)}")
                return
            if frame_no >= args.max_frames:
                break
    finally:
        camera.close()
    print("stop=false")


if __name__ == "__main__":
    main()
