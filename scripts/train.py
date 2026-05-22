#!/usr/bin/env python3
from __future__ import annotations

import argparse

from yolo26mlx import YOLO


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="configs/hand_alarm.yaml")
    p.add_argument("--weights", default="models/yolo26n.npz")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--name", default="hand_alarm_n")
    p.add_argument("--project", default="runs/train")
    p.add_argument("--exist-ok", action="store_true")
    args = p.parse_args()

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
    )


if __name__ == "__main__":
    main()
