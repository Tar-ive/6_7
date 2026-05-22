from __future__ import annotations

import cv2


class Camera:
    def __init__(self, index: int = 0) -> None:
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam index {index}")

    def frames(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                raise RuntimeError("Webcam frame read failed")
            yield frame

    def close(self) -> None:
        self.cap.release()
