#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pt", default="models/yolo26n.pt")
    p.add_argument("--out", default="models/yolo26n.npz")
    p.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    yolo = Path(sys.executable).with_name("yolo26")
    cmd = [str(yolo), "converters", "convert", args.pt, "-o", args.out]
    if args.verify:
        cmd.append("--verify")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
