import pytest

from alarm_yolo_mlx.gate import StopGate


def test_gate_requires_stable_hits():
    gate = StopGate(required_hits=3, window_frames=5)
    assert not gate.update(True)
    assert not gate.update(False)
    assert not gate.update(True)
    assert gate.update(True)


def test_gate_validates_window():
    with pytest.raises(ValueError):
        StopGate(required_hits=4, window_frames=3)
