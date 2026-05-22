#!/usr/bin/env python3
from __future__ import annotations

import argparse

from alarm_yolo_mlx.config import PoseConfig
from alarm_yolo_mlx.pose_app import PoseAlarmApp


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pose_alarm.yaml")
    p.add_argument("--weights")
    p.add_argument("--backend", choices=["mlx", "ultralytics"])
    p.add_argument("--source")
    p.add_argument("--camera-index", type=int)
    p.add_argument("--conf", type=float)
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    cfg = PoseConfig.from_yaml(args.config)
    if args.weights:
        cfg.weights = args.weights
    if args.backend:
        cfg.backend = args.backend
    if args.source:
        cfg.source = args.source
    if args.camera_index is not None:
        cfg.camera_index = args.camera_index
    if args.conf is not None:
        cfg.conf = args.conf
    if args.no_show:
        cfg.show = False

    print("alarm stopped" if PoseAlarmApp(cfg).run() else "alarm exited")


if __name__ == "__main__":
    main()
