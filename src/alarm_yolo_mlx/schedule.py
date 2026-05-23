from __future__ import annotations

import argparse
import re
import time
from datetime import datetime, timedelta


_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smh]?)\s*$", re.IGNORECASE)


def add_alarm_schedule_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--alarm-at",
        metavar="HH:MM",
        type=validate_alarm_at,
        help="wait until this local 24-hour time before ringing",
    )
    group.add_argument(
        "--alarm-in",
        metavar="DURATION",
        type=parse_duration,
        help="wait before ringing, for example 10s, 5m, 1h, or bare seconds",
    )


def scheduled_alarm_time(args: argparse.Namespace, now: datetime | None = None) -> datetime | None:
    now = now or datetime.now()
    if args.alarm_at:
        return next_alarm_at(args.alarm_at, now)
    if args.alarm_in:
        return now + args.alarm_in
    return None


def validate_alarm_at(value: str) -> str:
    try:
        hour_s, minute_s = value.split(":", 1)
        hour, minute = int(hour_s), int(minute_s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--alarm-at must use HH:MM, like 07:30") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise argparse.ArgumentTypeError("--alarm-at must use a valid 24-hour time")
    return value


def next_alarm_at(value: str, now: datetime | None = None) -> datetime:
    now = now or datetime.now()
    value = validate_alarm_at(value)
    hour_s, minute_s = value.split(":", 1)
    hour, minute = int(hour_s), int(minute_s)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def parse_duration(value: str) -> timedelta:
    match = _DURATION_RE.match(value)
    if not match:
        raise argparse.ArgumentTypeError("--alarm-in must look like 10s, 5m, 1h, or 30")
    amount = float(match.group(1))
    unit = match.group(2).lower() or "s"
    seconds = amount
    if unit == "m":
        seconds *= 60
    elif unit == "h":
        seconds *= 3600
    if seconds <= 0:
        raise argparse.ArgumentTypeError("--alarm-in must be greater than zero")
    return timedelta(seconds=seconds)


def wait_until_alarm(target: datetime | None) -> None:
    if target is None:
        return

    print(f"alarm armed for {target:%Y-%m-%d %H:%M:%S}")
    while True:
        remaining = (target - datetime.now()).total_seconds()
        if remaining <= 0:
            print("alarm ringing")
            return
        if remaining >= 60:
            mins = int(remaining // 60)
            secs = int(remaining % 60)
            print(f"ringing in {mins}m {secs}s", end="\r", flush=True)
        else:
            print(f"ringing in {int(remaining)}s   ", end="\r", flush=True)
        time.sleep(min(1.0, remaining))
