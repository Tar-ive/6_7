#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


VOICE_ID = "eXpIbVcVbLo8ZJQDlDnl"
MODEL_ID = "eleven_multilingual_v2"
OUTPUT_FORMAT = "mp3_44100_128"
CACHE_DIR = Path("assets/voice")

CLIPS = {
    "get_up_67_alarm.mp3": "Get up! Get up! Do the six-seven movement to turn the alarm off!",
    "lets_go_67.mp3": "Let's go!",
}


def main() -> None:
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise SystemExit("Set ELEVENLABS_API_KEY before generating the voice cache.")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for filename, text in CLIPS.items():
        out = CACHE_DIR / filename
        if out.exists() and out.stat().st_size > 0:
            print(f"cached {out}")
            continue
        _generate(key, text, out)
        print(f"generated {out}")


def _generate(api_key: str, text: str, out: Path) -> None:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}?output_format={OUTPUT_FORMAT}"
    payload = {
        "text": text,
        "model_id": MODEL_ID,
        "voice_settings": {
            "stability": 0.42,
            "similarity_boost": 0.85,
            "style": 0.55,
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
            out.write_bytes(resp.read())
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"ElevenLabs request failed: {exc.code} {detail}") from exc


if __name__ == "__main__":
    main()
