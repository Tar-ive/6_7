from __future__ import annotations

import argparse

from .app import AlarmStopApp
from .config import AppConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/alarm.yaml")
    p.add_argument("--weights")
    p.add_argument("--camera-index", type=int)
    p.add_argument("--conf", type=float)
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    if args.weights:
        cfg.weights = args.weights
    if args.camera_index is not None:
        cfg.camera_index = args.camera_index
    if args.conf is not None:
        cfg.conf = args.conf
    if args.no_show:
        cfg.show = False

    stopped = AlarmStopApp(cfg).run()
    print("alarm stopped" if stopped else "alarm exited")


if __name__ == "__main__":
    main()
