# MLX YOLO Alarm Stopper

Train a YOLO model with MLX, run webcam inference, and stop an alarm only when a trained hand/stop gesture is detected.

## Layout

```text
configs/              Dataset and runtime config
models/               Put converted .npz weights here
scripts/              Setup, conversion, training, webcam commands
src/alarm_yolo_mlx/   App code
tests/                Small logic tests
```

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,convert]"
```

This project depends on `yolo-mlx` from GitHub because the MLX YOLO API lives in `yolo26mlx`.

## Dataset

Expected YOLO detection layout:

```text
data/hand_alarm/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt
```

Your current labels are LabelMe-style JSON boxes for `forearm` and `underarm`. Convert them first:

```bash
python scripts/prepare_dataset.py \
  --sources data/frames data/people-balancing-pose \
  --out data/yolo_hand_alarm
```

That creates:

```text
data/yolo_hand_alarm/
  images/train
  images/val
  labels/train
  labels/val
```

## Weights

```bash
scripts/download_weights.sh
python scripts/convert_weights.py --pt models/yolo26n.pt --out models/yolo26n.npz
```

## Train

```bash
python scripts/train.py \
  --data configs/hand_alarm.yaml \
  --weights models/yolo26n.npz \
  --epochs 3 \
  --batch 2 \
  --name hand_alarm_smoke \
  --exist-ok
```

The best model should appear under `runs/train/hand_alarm_smoke/`.

Current smoke run on the tiny local dataset reached `mAP50=0.9594`, but precision is still low. Treat it as a wiring proof, not a production model.

## Run Webcam Alarm

Pose-based path, recommended:

```bash
python scripts/run_pose_alarm.py --config configs/pose_alarm.yaml
```

Box-detector path:

```bash
python scripts/run_webcam_alarm.py \
  --weights runs/train/hand_alarm_smoke/best.safetensors \
  --config configs/alarm.yaml
```

Press `q` to quit. On macOS, allow Terminal/Codex/Python camera access when prompted.

To test against a saved video instead of the webcam:

```bash
python scripts/probe_pose_video.py \
  --source /path/to/video.webm \
  --weights models/yolo26n-pose.pt \
  --conf 0.25
```

## Detection Strategy

Recommended path: use YOLO pose to detect shoulders, elbows, and wrists, then use ByteTrack plus a motion rule to detect the 6_7 alternating hand movement.

Fallback path: YOLO can learn the arm parts you labelled: `forearm` and `underarm`.
The alarm app then tracks detections frame-to-frame and stops only when the `forearm` detections alternate vertical order, matching the 6_7 balancing motion. This is better than a single static “gesture” class because the alarm should respond to the movement, not just one pose.
