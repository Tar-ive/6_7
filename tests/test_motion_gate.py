from alarm_yolo_mlx.detector import Detection
from alarm_yolo_mlx.motion_gate import AlternatingHandsGate


def arm(x1, y1, x2, y2):
    return Detection((x1, y1, x2, y2), 0.9, 0, "forearm")


def test_alternating_gate_requires_switches():
    gate = AlternatingHandsGate(min_switches=2)
    assert not gate.update([arm(0, 10, 10, 20), arm(20, 30, 30, 40)])
    assert not gate.update([arm(0, 10, 10, 20), arm(20, 30, 30, 40)])
    assert not gate.update([arm(0, 30, 10, 40), arm(20, 10, 30, 20)])
    assert gate.update([arm(0, 10, 10, 20), arm(20, 30, 30, 40)])
