from __future__ import annotations

from collections import deque


class StopGate:
    def __init__(self, required_hits: int = 5, window_frames: int = 8) -> None:
        if required_hits < 1 or window_frames < required_hits:
            raise ValueError("window_frames must be >= required_hits >= 1")
        self.required_hits = required_hits
        self.window = deque(maxlen=window_frames)

    def update(self, detected: bool) -> bool:
        self.window.append(bool(detected))
        return sum(self.window) >= self.required_hits
