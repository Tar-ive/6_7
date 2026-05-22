# YOLO26 Pose Support In MLX

This document explains the minimum patch needed to make `yolo26n-pose.pt` run as MLX inference inside `thewebAI/yolo-mlx`.

## Goal

We want this to work:

```python
from yolo26mlx import YOLO

model = YOLO("models/yolo26n-pose.npz", task="pose")
results = model.predict(frame)
keypoints = results[0].keypoints
```

The target model is only the smallest nano checkpoint for now:

```text
models/yolo26n-pose.pt
models/yolo26n-pose.npz
```

The goal is **inference only**. Pose training is out of scope.

## Repos Needed

- `ultralytics/ultralytics`
  - Source of truth for YOLO26 pose architecture, weight layout, keypoint decode, and pose result construction.
  - Relevant local installed files:
    - `.venv/lib/python3.12/site-packages/ultralytics/nn/modules/head.py`
    - `.venv/lib/python3.12/site-packages/ultralytics/models/yolo/pose/predict.py`
    - `.venv/lib/python3.12/site-packages/ultralytics/utils/ops.py`

- `thewebAI/yolo-mlx`
  - MLX runtime we want to patch.
  - Relevant local installed files:
    - `.venv/lib/python3.12/site-packages/yolo26mlx/nn/modules/head.py`
    - `.venv/lib/python3.12/site-packages/yolo26mlx/nn/tasks.py`
    - `.venv/lib/python3.12/site-packages/yolo26mlx/engine/model.py`
    - `.venv/lib/python3.12/site-packages/yolo26mlx/engine/predictor.py`
    - `.venv/lib/python3.12/site-packages/yolo26mlx/cfg/models/26/yolo26.yaml`

## Current State In This Repo

The alarm system now has two paths:

- Box path: MLX YOLO detection trained on `forearm` and `underarm`.
- Pose path: selectable `mlx` or `ultralytics`.

The default pose config now uses MLX:

```bash
python scripts/run_pose_alarm.py --config configs/pose_alarm.yaml
```

```text
backend: mlx
models/yolo26n-pose.npz
```

The app prints and overlays the active backend:

```text
pose backend=mlx:models/yolo26n-pose.npz imgsz=640 conf=0.25
```

The fork patch lives in:

```text
/Users/tarive/yolo-mlx
branch: codex/pose26-mlx
remote: https://github.com/Tar-ive/yolo-mlx.git
```

## What Ultralytics Actually Uses

`models/yolo26n-pose.pt` loads as:

```text
type: Pose26
nc: 1
kpt_shape: [17, 3]
nk: 51
nl: 3
reg_max: 1
end2end: True
```

The important point: this is **Pose26**, not the generic older `Pose` head.

## 1. Model Head Shape

### Ultralytics `Pose26`

For the nano model, the three detection scales use input channels:

```text
P3: 64
P4: 128
P5: 256
```

Detection heads:

```text
cv2: box head
  layer0: 64 -> 16 -> 16 -> 4
  layer1: 128 -> 16 -> 16 -> 4
  layer2: 256 -> 16 -> 16 -> 4

cv3: class head
  layer0: 64 -> 64 -> 64 -> 1
  layer1: 128 -> 64 -> 64 -> 1
  layer2: 256 -> 64 -> 64 -> 1
```

Pose heads:

```text
cv4:
  layer0: 64 -> 85 -> 85
  layer1: 128 -> 85 -> 85
  layer2: 256 -> 85 -> 85

cv4_kpts:
  layer0: 85 -> 51
  layer1: 85 -> 51
  layer2: 85 -> 51

cv4_sigma:
  layer0: 85 -> 34
  layer1: 85 -> 34
  layer2: 85 -> 34
```

End-to-end one-to-one copies:

```text
one2one_cv2
one2one_cv3
one2one_cv4
one2one_cv4_kpts
one2one_cv4_sigma
```

`51 = 17 keypoints * 3 values`.

`34 = 17 keypoints * 2 sigma values`.

### Current `yolo26mlx` Pose

`yolo26mlx` currently has a generic `Pose`:

```text
cv4:
  Conv(x, c4, 3)
  Conv(c4, c4, 3)
  Conv2d(c4, nk, 1)
```

That does not match `Pose26`, because YOLO26 pose separates:

```text
shared pose features: cv4
keypoint output: cv4_kpts
sigma output: cv4_sigma
```

### Minimum MLX Head Patch

Add a `Pose26` class to `yolo26mlx/nn/modules/head.py`.

It should mirror Ultralytics:

```python
class Pose26(Pose):
    def __init__(self, nc=1, kpt_shape=(17, 3), reg_max=1, end2end=True, ch=()):
        super().__init__(nc, kpt_shape, reg_max, end2end, ch)
        c4 = max(ch[0] // 4, kpt_shape[0] * (kpt_shape[1] + 2))
        self.cv4 = {
            f"layer{i}": Sequential(Conv(x, c4, 3), Conv(c4, c4, 3))
            for i, x in enumerate(ch)
        }
        self.cv4_kpts = {
            f"layer{i}": nn.Conv2d(c4, self.nk, 1)
            for i, _ in enumerate(ch)
        }
        self.cv4_sigma = {
            f"layer{i}": nn.Conv2d(c4, self.kpt_shape[0] * 2, 1)
            for i, _ in enumerate(ch)
        }
```

For inference, sigma can be loaded but ignored at first. The needed output is decoded `kpts`.

## 2. Exact Weight Name Mapping

PyTorch keys use `model.23...`; MLX internal keys use `layers.23...`.

Existing `yolo26mlx` already maps most detection and generic `cv4` names. It needs mappings for the YOLO26-specific pose heads.

### Required New Mappings

For shared pose features:

```text
layers.23.cv4.0.0.* -> layers.23.cv4.layer0.layers.0.*
layers.23.cv4.0.1.* -> layers.23.cv4.layer0.layers.1.*
layers.23.cv4.1.0.* -> layers.23.cv4.layer1.layers.0.*
layers.23.cv4.1.1.* -> layers.23.cv4.layer1.layers.1.*
layers.23.cv4.2.0.* -> layers.23.cv4.layer2.layers.0.*
layers.23.cv4.2.1.* -> layers.23.cv4.layer2.layers.1.*
```

For keypoint output:

```text
layers.23.cv4_kpts.0.* -> layers.23.cv4_kpts.layer0.*
layers.23.cv4_kpts.1.* -> layers.23.cv4_kpts.layer1.*
layers.23.cv4_kpts.2.* -> layers.23.cv4_kpts.layer2.*
```

For sigma output:

```text
layers.23.cv4_sigma.0.* -> layers.23.cv4_sigma.layer0.*
layers.23.cv4_sigma.1.* -> layers.23.cv4_sigma.layer1.*
layers.23.cv4_sigma.2.* -> layers.23.cv4_sigma.layer2.*
```

For one-to-one heads:

```text
layers.23.one2one_cv4.0.0.* -> layers.23.one2one_cv4.layer0.layers.0.*
layers.23.one2one_cv4.0.1.* -> layers.23.one2one_cv4.layer0.layers.1.*
layers.23.one2one_cv4.1.0.* -> layers.23.one2one_cv4.layer1.layers.0.*
layers.23.one2one_cv4.1.1.* -> layers.23.one2one_cv4.layer1.layers.1.*
layers.23.one2one_cv4.2.0.* -> layers.23.one2one_cv4.layer2.layers.0.*
layers.23.one2one_cv4.2.1.* -> layers.23.one2one_cv4.layer2.layers.1.*

layers.23.one2one_cv4_kpts.0.* -> layers.23.one2one_cv4_kpts.layer0.*
layers.23.one2one_cv4_kpts.1.* -> layers.23.one2one_cv4_kpts.layer1.*
layers.23.one2one_cv4_kpts.2.* -> layers.23.one2one_cv4_kpts.layer2.*

layers.23.one2one_cv4_sigma.0.* -> layers.23.one2one_cv4_sigma.layer0.*
layers.23.one2one_cv4_sigma.1.* -> layers.23.one2one_cv4_sigma.layer1.*
layers.23.one2one_cv4_sigma.2.* -> layers.23.one2one_cv4_sigma.layer2.*
```

### Weight Shapes To Match

From the nano checkpoint:

```text
model.23.cv4_kpts.0.weight: (51, 85, 1, 1)
model.23.cv4_kpts.1.weight: (51, 85, 1, 1)
model.23.cv4_kpts.2.weight: (51, 85, 1, 1)

model.23.cv4_sigma.0.weight: (34, 85, 1, 1)
model.23.cv4_sigma.1.weight: (34, 85, 1, 1)
model.23.cv4_sigma.2.weight: (34, 85, 1, 1)
```

In MLX NHWC conv format, converted weights should become:

```text
(1, 1, 85, 51)
(1, 1, 85, 34)
```

The minimum validation is:

```text
matching_weights should include cv4, cv4_kpts, cv4_sigma, and one2one equivalents.
no pose-head weights should be skipped except num_batches_tracked.
```

## 3. Keypoint Decode

Ultralytics `Pose26.kpts_decode` does:

```python
y[:, 2::ndim] = sigmoid(y[:, 2::ndim])
y[:, 0::ndim] = (y[:, 0::ndim] + anchors[0]) * strides
y[:, 1::ndim] = (y[:, 1::ndim] + anchors[1]) * strides
```

For normal generic `Pose`, Ultralytics uses:

```python
x = (x * 2.0 + (anchor_x - 0.5)) * stride
y = (y * 2.0 + (anchor_y - 0.5)) * stride
```

But for **Pose26**, do not use the `* 2.0 - 0.5` formula. Use:

```text
decoded_x = (raw_x + anchor_x) * stride
decoded_y = (raw_y + anchor_y) * stride
decoded_conf = sigmoid(raw_conf)
```

After model-space decode, the predictor must undo letterboxing:

```text
x = (x - pad_x) / gain
y = (y - pad_y) / gain
clip to original image width/height
```

This mirrors Ultralytics `ops.scale_coords`.

## 4. YOLO26 End-To-End Pose Format

YOLO26 uses `end2end=True`.

For detection, Ultralytics `Detect.forward`:

```text
1. compute one2many predictions
2. compute one2one predictions from detached features
3. during inference, decode one2one predictions
4. postprocess top-k scores
5. output [x1, y1, x2, y2, conf, class]
```

For pose, the output must keep keypoints aligned with the selected top-k detections:

```text
raw shape before postprocess:
  [batch, anchors, 4 + nc + nk]

postprocess output:
  [batch, max_det, 6 + nk]

last dimension:
  x1, y1, x2, y2, conf, class, kpt0_x, kpt0_y, kpt0_conf, ...
```

The MLX implementation already has end-to-end detection infrastructure. What is missing is:

- `Pose26` producing `kpts` alongside `boxes` and `scores`.
- Pose postprocess preserving keypoints when selecting top-k boxes.
- `Keypoints` result object populated instead of `Keypoints(None, orig_shape)`.

## 5. Side-By-Side Mapping For `yolo26n-pose`

### Architecture Config

Add:

```text
yolo26mlx/cfg/models/26/yolo26-pose.yaml
```

It should be the same as `yolo26.yaml`, except:

```yaml
nc: 1
kpt_shape: [17, 3]
end2end: True
reg_max: 1

head:
  ...
  - [[16, 19, 22], 1, Pose26, [nc, kpt_shape]]
```

Also update:

```text
yolo26mlx/nn/modules/__init__.py
yolo26mlx/nn/tasks.py
yolo26mlx/engine/model.py
```

So `Pose26` can be resolved and `task="pose"` selects `yolo26-pose.yaml`.

### Runtime Comparison

Ultralytics:

```python
from ultralytics import YOLO

model = YOLO("models/yolo26n-pose.pt")
result = model(frame)[0]
xy = result.keypoints.xy
conf = result.keypoints.conf
```

Target MLX:

```python
from yolo26mlx import YOLO

model = YOLO("models/yolo26n-pose.npz", task="pose")
result = model.predict(frame)[0]
xy = result.keypoints.xy
conf = result.keypoints.conf
```

### Acceptance Test

Run both backends on the same frame:

```text
data/frames/frame_016.jpg
```

Compare these six keypoints:

```text
left_shoulder: 5
right_shoulder: 6
left_elbow: 7
right_elbow: 8
left_wrist: 9
right_wrist: 10
```

Expected acceptance:

```text
person box roughly overlaps
keypoints are on the same body parts
shoulder/elbow/wrist coordinates are close enough for the 6_7 gate
no crash on webcam or video input
```

## Minimum Implementation Order

1. Add `Pose26` MLX head.
2. Add `yolo26-pose.yaml`.
3. Register `Pose26` in parser/imports.
4. Make `task="pose"` select the pose YAML.
5. Add key mappings for `cv4_kpts`, `cv4_sigma`, and one-to-one copies.
6. Verify converted weights load without missing pose-head keys.
7. Implement `Pose26.kpts_decode`.
8. Implement predictor `_postprocess_pose`.
9. Compare one image against Ultralytics.
10. Run `scripts/probe_pose_video.py` with MLX weights.

## What We Understand

The original blocker was not the checkpoint. The checkpoint downloads and converts.

The blocker was that `yolo26mlx` implemented enough detection infrastructure for YOLO26, but its pose path was incomplete:

- missing packaged pose YAML
- missing `Pose26` head
- incomplete pose weight mapping
- TODO pose postprocess

The current fork patch adds those four pieces for inference. The alarm app now swaps from Ultralytics pose to MLX pose without changing the higher-level gesture logic.
