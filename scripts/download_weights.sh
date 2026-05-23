#!/usr/bin/env bash
set -euo pipefail

mkdir -p models
curl -L -o models/yolo26n.pt \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
curl -L -o models/yolo26n-pose.pt \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt
