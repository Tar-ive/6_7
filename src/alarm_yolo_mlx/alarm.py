from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


class Alarm:
    def __init__(
        self, sound: str | None = None, interval: float = 0.8, volume: float = 0.8
    ) -> None:
        self.sound = Path(sound) if sound else None
        self.interval = interval
        self.volume = volume
        self._proc: subprocess.Popen | None = None
        self._last_beep = 0.0

    def start(self) -> None:
        if self.running:
            return
        self._start_sound()

    def set_volume(self, volume: float) -> None:
        self.volume = volume
        if not self.running or not self._proc or not self._proc.stdin:
            return
        try:
            self._proc.stdin.write(f"volume {volume}\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self._proc = None

    def _start_sound(self) -> None:
        if not self.sound or not self.sound.exists():
            self._beep()
            return
        helper = Path(__file__).with_name("loop_alarm_player.swift")
        if shutil.which("swift") and helper.exists():
            self._proc = subprocess.Popen(
                ["swift", str(helper), str(self.sound), str(self.volume)],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            return
        if shutil.which("afplay"):
            self._proc = subprocess.Popen(["afplay", "-v", str(self.volume), str(self.sound)])
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
            if self._proc.stdin:
                try:
                    self._proc.stdin.write("stop\n")
                    self._proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
            try:
                self._proc.wait(timeout=0.3)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
        self._proc = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _beep(self) -> None:
        self._last_beep = time.monotonic()
        os.write(sys.stdout.fileno(), b"\a")


def play_once(
    sound: str | None,
    *,
    wait: bool = False,
    text: str | None = None,
) -> subprocess.Popen | None:
    path = Path(sound) if sound else None
    if path and path.exists() and shutil.which("afplay"):
        proc = subprocess.Popen(["afplay", str(path)])
        if wait:
            proc.wait()
        return proc
    if text and shutil.which("say"):
        proc = subprocess.Popen(["say", text])
        if wait:
            proc.wait()
        return proc
    return None
