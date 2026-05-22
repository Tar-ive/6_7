# Dataset Notes

Observed local data:

- `data/frames`: 150 reference-video frames, 13 labelled JSON files.
- `data/people-balancing-pose`: 20 generated/staged positive pose images, 20 labelled JSON files.
- Label schema: LabelMe rectangles with `forearm` and `underarm`.

Training target:

- Train YOLO to detect `forearm` and `underarm`.
- Do not train only one static `stop_hand` class yet. The alarm condition is a motion: the left/right forearms alternate vertical order like a balancing scale.

Runtime decision:

- Run YOLO on webcam frames.
- Track detections frame-to-frame.
- Stop the alarm when the tracked `forearm` detections switch vertical order enough times.

More data needed:

- Add negatives: normal sitting, waving, hands down, palms down, one hand only, phone use.
- Add more real webcam clips from different people and lighting.
- Label more frames from the actual 6_7 video, especially across the full alternating motion.
