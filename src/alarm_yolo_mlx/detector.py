from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from yolo26mlx import YOLO


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    conf: float
    cls: int
    name: str


class YoloMlxDetector:
    def __init__(
        self,
        weights: str,
        conf: float = 0.45,
        imgsz: int = 640,
        target_classes: list[str] | None = None,
    ) -> None:
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.target_classes = {str(c) for c in target_classes or []}
        self.names = getattr(self.model, "names", {}) or {}

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz)[0]
        detections = list(self._parse_boxes(getattr(result, "boxes", None)))
        if not self.target_classes:
            return detections
        return [d for d in detections if d.name in self.target_classes or str(d.cls) in self.target_classes]

    def _parse_boxes(self, boxes: Any):
        if boxes is None:
            return
        data = getattr(boxes, "data", None)
        if data is None and all(hasattr(boxes, attr) for attr in ("xyxy", "conf", "cls")):
            xyxy = np.asarray(boxes.xyxy)
            conf = np.asarray(boxes.conf).reshape(-1)
            cls = np.asarray(boxes.cls).reshape(-1)
            data = np.column_stack([xyxy, conf, cls])
        if data is None:
            data = boxes
        data = np.asarray(data)
        if data.size == 0:
            return
        for row in data.reshape(-1, data.shape[-1]):
            if len(row) < 6:
                continue
            cls = int(row[5])
            name = self.names.get(cls, str(cls)) if isinstance(self.names, dict) else str(cls)
            yield Detection(tuple(map(float, row[:4])), float(row[4]), cls, name)
