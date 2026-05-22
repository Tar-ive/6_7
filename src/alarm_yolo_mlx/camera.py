from __future__ import annotations

import cv2


class Camera:
    def __init__(self, source: int | str = 0) -> None:
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source {source}")

    def frames(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame

    def close(self) -> None:
        self.cap.release()
