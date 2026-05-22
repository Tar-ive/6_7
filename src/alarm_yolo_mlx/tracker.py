from __future__ import annotations

from dataclasses import replace

from .detector import Detection


class IouTracker:
    def __init__(self, iou_threshold: float = 0.2, max_age: int = 8) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id = 1
        self.tracks: dict[int, tuple[Detection, int]] = {}

    def update(self, detections: list[Detection]) -> list[Detection]:
        matched: set[int] = set()
        out = []
        for det in detections:
            track_id, score = self._best(det, matched)
            if track_id is None or score < self.iou_threshold:
                track_id = self.next_id
                self.next_id += 1
            matched.add(track_id)
            tracked = replace(det, track_id=track_id)
            self.tracks[track_id] = (tracked, 0)
            out.append(tracked)
        self._age(matched)
        return out

    def _best(self, det: Detection, used: set[int]) -> tuple[int | None, float]:
        best_id, best_iou = None, 0.0
        for track_id, (prev, _) in self.tracks.items():
            if track_id in used or prev.cls != det.cls:
                continue
            score = iou(prev.xyxy, det.xyxy)
            if score > best_iou:
                best_id, best_iou = track_id, score
        return best_id, best_iou

    def _age(self, matched: set[int]) -> None:
        for track_id, (det, age) in list(self.tracks.items()):
            if track_id in matched:
                continue
            if age + 1 > self.max_age:
                del self.tracks[track_id]
            else:
                self.tracks[track_id] = (det, age + 1)


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-9)
