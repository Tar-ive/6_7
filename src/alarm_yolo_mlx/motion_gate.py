from __future__ import annotations

from collections import deque

from .detector import Detection


class AlternatingHandsGate:
    def __init__(self, motion_class: str = "forearm", min_switches: int = 2, window: int = 18) -> None:
        self.motion_class = motion_class
        self.min_switches = min_switches
        self.orders: deque[str] = deque(maxlen=window)

    def update(self, detections: list[Detection]) -> bool:
        arms = [d for d in detections if d.name == self.motion_class or str(d.cls) == self.motion_class]
        if len(arms) < 2:
            self.orders.append("missing")
            return False
        left, right = sorted(arms, key=lambda d: _center(d)[0])[:2]
        order = "left_high" if _center(left)[1] < _center(right)[1] else "right_high"
        self.orders.append(order)
        clean = [o for o in self.orders if o != "missing"]
        switches = sum(a != b for a, b in zip(clean, clean[1:]))
        return switches >= self.min_switches


def _center(d: Detection) -> tuple[float, float]:
    x1, y1, x2, y2 = d.xyxy
    return (x1 + x2) / 2, (y1 + y2) / 2
