from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10


@dataclass(frozen=True)
class Pose:
    xy: np.ndarray
    conf: np.ndarray
    box: tuple[float, float, float, float] | None = None
    track_id: int | None = None


class UltralyticsPoseDetector:
    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 640) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.backend_name = f"ultralytics:{weights}"

    def detect(self, frame: np.ndarray) -> list[Pose]:
        result = self.model.track(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )[0]
        if result.keypoints is None:
            return []
        xys = result.keypoints.xy.cpu().numpy()
        confs = result.keypoints.conf.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else [None] * len(xys)
        ids = (
            result.boxes.id.cpu().numpy().astype(int)
            if result.boxes is not None and result.boxes.id is not None
            else [None] * len(xys)
        )
        return [
            Pose(xy, cf, tuple(box) if box is not None else None, track_id)
            for xy, cf, box, track_id in zip(xys, confs, boxes, ids)
        ]


class MlxPoseDetector:
    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 640) -> None:
        from yolo26mlx import YOLO

        self.model = YOLO(weights, task="pose", verbose=False)
        self.conf = conf
        self.imgsz = imgsz
        self.backend_name = f"mlx:{weights}"

    def detect(self, frame: np.ndarray) -> list[Pose]:
        result = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz)[0]
        if result.keypoints is None:
            return []
        xys = result.keypoints.xy
        confs = result.keypoints.conf
        boxes = result.boxes.xyxy if result.boxes is not None else [None] * len(xys)
        return [
            Pose(xy, cf, tuple(box) if box is not None else None)
            for xy, cf, box in zip(xys, confs, boxes)
        ]


def make_pose_detector(backend: str, weights: str, conf: float, imgsz: int):
    if backend == "mlx":
        return MlxPoseDetector(weights, conf, imgsz)
    if backend == "ultralytics":
        return UltralyticsPoseDetector(weights, conf, imgsz)
    raise ValueError(f"unknown pose backend: {backend}")


YoloPoseDetector = UltralyticsPoseDetector


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba, bc = a - b, c - b
    denom = max(np.linalg.norm(ba) * np.linalg.norm(bc), 1e-6)
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1, 1))))
