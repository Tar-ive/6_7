#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


VOICE_ID = "eXpIbVcVbLo8ZJQDlDnl"
MODEL_ID = "eleven_multilingual_v2"
OUTPUT_FORMAT = "mp3_44100_128"
CACHE_DIR = Path("assets/voice")
MIN_AUDIO_BYTES = 4_000
MIN_MAX_VOLUME_DB = -30.0

CLIPS = {
    "alarm_time_do_67.mp3": "Alarm time. You need to do the 6 7 movement.",
    "get_coffee_cup_right_now.mp3": "Get the coffee cup right now!",
    "good_morning_rise_shine.mp3": "Good morning. Rise and shine!",
    "nah_buddy_not_with_me.mp3": "Na na buddy! That won't work, not with me.",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Regenerate existing cached clips.")
    parser.add_argument("--no-validate", action="store_true", help="Skip ffmpeg loudness checks.")
    args = parser.parse_args()

    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise SystemExit("Set ELEVENLABS_API_KEY before generating the voice cache.")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for filename, text in CLIPS.items():
        out = CACHE_DIR / filename
        if out.exists() and out.stat().st_size > 0 and not args.force:
            _validate_audio(out, required=not args.no_validate)
            print(f"cached {out}")
            continue
        _generate(key, text, out, validate=not args.no_validate)
        print(f"generated {out}")


def _generate(api_key: str, text: str, out: Path, validate: bool) -> None:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}?output_format={OUTPUT_FORMAT}"
    payload = {
        "text": text,
        "model_id": MODEL_ID,
        "voice_settings": {
            "stability": 0.35,
            "similarity_boost": 0.75,
            "use_speaker_boost": True,
        },
    }
    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "xi-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=60) as resp:
            content_type = resp.headers.get("Content-Type", "")
            data = resp.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"ElevenLabs request failed: {exc.code} {detail}") from exc

    if "audio" not in content_type:
        preview = data[:300].decode("utf-8", errors="replace")
        raise SystemExit(f"ElevenLabs returned non-audio content: {content_type} {preview}")
    if len(data) < MIN_AUDIO_BYTES:
        raise SystemExit(f"ElevenLabs returned too little audio data: {len(data)} bytes")
    if data[:1] == b"{":
        raise SystemExit(data.decode("utf-8", errors="replace"))
    tmp = out.with_suffix(".tmp.mp3")
    tmp.write_bytes(data)
    try:
        _validate_audio(tmp, required=validate)
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)


def _validate_audio(path: Path, required: bool = True) -> None:
    if not required:
        return
    if not shutil.which("ffmpeg"):
        print(f"warning: ffmpeg missing, cannot validate {path}")
        return
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg could not decode {path}:\n{result.stderr}")
    max_volume = _parse_max_volume(result.stderr)
    if max_volume is None:
        raise SystemExit(f"could not read max_volume for {path}")
    if max_volume < MIN_MAX_VOLUME_DB:
        raise SystemExit(
            f"{path} looks too quiet or silent: max_volume={max_volume:.1f} dB. "
            "Regeneration did not replace the cache."
        )
    print(f"validated {path} max_volume={max_volume:.1f} dB")


def _parse_max_volume(stderr: str) -> float | None:
    for line in stderr.splitlines():
        if "max_volume:" not in line:
            continue
        value = line.rsplit("max_volume:", 1)[1].strip().split(" ", 1)[0]
        return float(value)
    return None


if __name__ == "__main__":
    main()
