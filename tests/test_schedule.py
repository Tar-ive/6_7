from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from alarm_yolo_mlx.schedule import next_alarm_at, parse_duration


def test_next_alarm_at_uses_today_for_future_time():
    now = datetime(2026, 5, 22, 7, 15)

    assert next_alarm_at("07:30", now) == datetime(2026, 5, 22, 7, 30)


def test_next_alarm_at_rolls_past_time_to_tomorrow():
    now = datetime(2026, 5, 22, 7, 45)

    assert next_alarm_at("07:30", now) == datetime(2026, 5, 23, 7, 30)


def test_parse_duration_units():
    assert parse_duration("10") == timedelta(seconds=10)
    assert parse_duration("10s") == timedelta(seconds=10)
    assert parse_duration("5m") == timedelta(minutes=5)
    assert parse_duration("1h") == timedelta(hours=1)


def test_parse_duration_rejects_zero():
    with pytest.raises(Exception):
        parse_duration("0")
