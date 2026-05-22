#!/usr/bin/env python3
from __future__ import annotations

import argparse

from alarm_yolo_mlx.camera import Camera
from alarm_yolo_mlx.config import PoseConfig
from alarm_yolo_mlx.pose import YoloPoseDetector
from alarm_yolo_mlx.pose_gate import Pose67Gate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pose_alarm.yaml")
    p.add_argument("--source", required=True)
    p.add_argument("--weights")
    p.add_argument("--conf", type=float)
    p.add_argument("--max-frames", type=int, default=300)
    args = p.parse_args()

    cfg = PoseConfig.from_yaml(args.config)
    if args.weights:
        cfg.weights = args.weights
    if args.conf is not None:
        cfg.conf = args.conf

    camera = Camera(args.source)
    detector = YoloPoseDetector(cfg.weights, cfg.conf, cfg.imgsz)
    gate = Pose67Gate(
        cfg.required_movements,
        cfg.window_frames,
        cfg.min_keypoint_conf,
        cfg.max_elbow_angle,
        cfg.min_wrist_gap,
        cfg.max_missing_frames,
    )
    try:
        for frame_no, frame in enumerate(camera.frames(), 1):
            poses = detector.detect(frame)
            if gate.update(poses):
                print(f"stop=true frame={frame_no} poses={len(poses)} movements={gate.progress}")
                return
            if frame_no >= args.max_frames:
                break
    finally:
        camera.close()
    print(f"stop=false movements={gate.progress}")


if __name__ == "__main__":
    main()
