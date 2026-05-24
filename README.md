# MLX YOLO Alarm Stopper

Train a YOLO model with MLX, run webcam inference, and stop an alarm only when a trained hand/stop gesture is detected.

## What 6_7 Means

`6_7` refers to the 2025 "six seven" Internet meme and gesture: both hands are held forward with palms up, then moved in an alternating up/down motion like a balancing scale. In this project, that gesture is the alarm-off signal. The camera keeps the alarm active until the pose pipeline sees enough repeated 6_7 hand movements.

Reference: [6-7 on Wikipedia](https://en.wikipedia.org/wiki/6-7).

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

If `python3.12` is unavailable, use your installed Python 3:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,convert]"
```

This project depends on `yolo-mlx` from GitHub because the MLX YOLO API lives in `yolo26mlx`.

On macOS, give camera permission to the terminal app you launch from:

```text
System Settings > Privacy & Security > Camera
```

## Dataset

Expected YOLO detection layout:

```text
data/hand_alarm/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt
```

Dataset

```text
data/yolo_hand_alarm/
  images/train
  images/val
  labels/train
  labels/val
```

## Weights

The default pose config and hackathon demo config both use the MLX backend with `models/yolo26n-pose.npz`. If you see Ultralytics downloading `yolo26n-pose.pt`, you are running with `backend: ultralytics` or overriding the demo with `--backend ultralytics`.

Prepare the MLX pose model:

```bash
scripts/download_weights.sh
python scripts/convert_weights.py \
  --pt models/yolo26n-pose.pt \
  --out models/yolo26n-pose.npz
```

Check that the converted file exists and is an NPZ archive:

```bash
ls -lh models/yolo26n-pose.npz
python -c "import mlx.core as mx; print(len(mx.load('models/yolo26n-pose.npz')))"
```

## Alarm Audio

The pose alarm uses your recorded alarm loop through the whole wake-up game:

```text
assets/voice/alarm_loop.m4a
```

The spoken game prompts use these cached ElevenLabs clips when available:

```text
assets/voice/alarm_time_do_67.mp3
assets/voice/get_coffee_cup_right_now.mp3
assets/voice/good_morning_rise_shine.mp3
assets/voice/nah_buddy_not_with_me.mp3
```

Regenerate the voice clips with the ElevenLabs voice ID `eXpIbVcVbLo8ZJQDlDnl`:

```bash
export ELEVENLABS_API_KEY="..."
python scripts/generate_voice_cache.py
unset ELEVENLABS_API_KEY
```

At runtime, `alarm_loop.m4a` keeps playing until the randomized 6_7 sets and mug proof are reached. Each run randomly asks for 1-3 full sets, where one set is the configured `required_movements` count. Spoken prompts duck the alarm to 20%, then restore full volume. After each non-final set, the app tells you how many sets remain. After all sets complete, it says "Get the coffee cup right now!" If the detector sees a phone during the mug proof step, it says "Na na buddy! That won't work, not with me." while the alarm keeps looping. Once the mug is held long enough, the alarm stops and the app says "Good morning. Rise and shine!"

Optional box-detector model conversion:

```bash
python scripts/convert_weights.py \
  --pt models/yolo26n.pt \
  --out models/yolo26n.npz
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

## Run Alarm Demo

Browser GUI path, recommended:

```bash
python scripts/run_alarm_gui.py --config configs/pose_alarm_demo.yaml
```

The command prints a local setup URL such as:

```text
alarm setup ui: http://127.0.0.1:8767
```

Open that URL if the browser does not open automatically.

User flow:

```text
1. Set the alarm time, or click 10 sec demo / 1 min demo / 5 min demo.
2. Keep the terminal process running.
3. The browser page shows the countdown and live alarm status.
4. When the alarm rings, the camera window opens.
5. Do the requested 6_7 sets, then show the coffee mug to stop the alarm.
```

The browser status should move through:

```text
Alarm Armed -> Preparing model -> Alarm Ringing -> Alarm Stopped
```

Press `q` in the camera window to quit manually.

## CLI Alternatives

Pose-based CLI path:

```bash
python scripts/run_pose_alarm.py --config configs/pose_alarm.yaml
```

Ring at a clock time:

```bash
python scripts/run_pose_alarm.py --config configs/pose_alarm.yaml --alarm-at 07:30
```

Ring after a delay, useful for testing:

```bash
python scripts/run_pose_alarm.py --config configs/pose_alarm.yaml --alarm-in 10s
python scripts/run_pose_alarm.py --config configs/pose_alarm.yaml --alarm-in 5m
```

Box-detector path:

```bash
python scripts/run_webcam_alarm.py \
  --weights runs/train/hand_alarm_smoke/best.safetensors \
  --config configs/alarm.yaml
```

To test against a saved video instead of the webcam:

```bash
python scripts/probe_pose_video.py \
  --source /path/to/video.webm \
  --weights models/yolo26n-pose.npz \
  --conf 0.25
```

## Troubleshooting

If the alarm rings but no camera window appears, check the terminal logs. On macOS, camera permission is usually the issue. Enable camera access for Terminal, iTerm, VS Code, Cursor, or whichever app launched Python, then fully quit and reopen that app.

If the first run downloads `yolo26n-pose.pt`, the app is not using MLX:

```text
Downloading ... yolo26n-pose.pt
```

Check that your config contains `backend: mlx` and `weights: models/yolo26n-pose.npz`.

If you cannot hear the alarm, test the sound file directly:

```bash
afplay assets/voice/alarm_loop.m4a
afplay assets/voice/good_morning_alarm_off.mp3
afplay assets/voice/nah_buddy_photo.mp3
```

## Detection Strategy

Recommended path: use YOLO pose to detect shoulders, elbows, and wrists, then use ByteTrack plus a motion rule to detect the 6_7 alternating hand movement.

Fallback path: YOLO can learn the arm parts you labelled: `forearm` and `underarm`.
The alarm app then tracks detections frame-to-frame and stops only when the `forearm` detections alternate vertical order, matching the 6_7 balancing motion. This is better than a single static “gesture” class because the alarm should respond to the movement, not just one pose.
