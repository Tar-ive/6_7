from __future__ import annotations

import threading
import time

import cv2


class Camera:
    """Threaded camera reader — always yields the latest frame, never stale buffers."""

    def __init__(self, source: int | str = 0) -> None:
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source {source}")
        self._frame = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        t = threading.Thread(target=self._read_loop, daemon=True)
        t.start()

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if ok:
                with self._lock:
                    self._frame = frame
                    self._seq += 1

    def frames(self):
        last_seq = -1
        while not self._stop.is_set():
            with self._lock:
                if self._seq == last_seq:
                    frame = None
                else:
                    frame = self._frame
                    last_seq = self._seq
            if frame is None:
                time.sleep(0.005)
                continue
            yield frame

    def close(self) -> None:
        self._stop.set()
        time.sleep(0.1)
        self.cap.release()
