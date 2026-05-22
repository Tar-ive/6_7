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
python3 -m venv .venv
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

Edit [configs/hand_alarm.yaml](configs/hand_alarm.yaml) so `path` points to your labelled dataset and `names` matches your label IDs.

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
  --epochs 100 \
  --batch 8 \
  --name hand_alarm_n
```

The best model should appear under `runs/train/hand_alarm_n/`.

## Run Webcam Alarm

```bash
python scripts/run_webcam_alarm.py \
  --weights runs/train/hand_alarm_n/best.safetensors \
  --config configs/alarm.yaml
```

Press `q` to quit. On macOS, allow Terminal/Codex/Python camera access when prompted.
