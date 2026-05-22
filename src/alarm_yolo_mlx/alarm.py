from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


class Alarm:
    def __init__(self, sound: str | None = None, interval: float = 0.8) -> None:
        self.sound = Path(sound) if sound else None
        self.interval = interval
        self._proc: subprocess.Popen[bytes] | None = None
        self._last_beep = 0.0

    def start(self) -> None:
        if self.running:
            return
        if self.sound and self.sound.exists() and shutil.which("afplay"):
            self._proc = subprocess.Popen(["afplay", str(self.sound)])
            return
        self._beep()

    def tick(self) -> None:
        if self.running:
            return
        if self.sound and self.sound.exists() and shutil.which("afplay"):
            self.start()
            return
        if time.monotonic() - self._last_beep >= self.interval:
            self._beep()

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _beep(self) -> None:
        self._last_beep = time.monotonic()
        os.write(sys.stdout.fileno(), b"\a")
