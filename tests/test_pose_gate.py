import numpy as np

from alarm_yolo_mlx.pose import Pose
from alarm_yolo_mlx.pose_gate import Pose67Gate


def pose(left_wrist_y, right_wrist_y):
    xy = np.zeros((17, 2), dtype=float)
    conf = np.ones(17, dtype=float)
    xy[5], xy[7], xy[9] = (100, 100), (100, 170), (160, left_wrist_y)
    xy[6], xy[8], xy[10] = (300, 100), (300, 170), (240, right_wrist_y)
    return Pose(xy, conf, (50, 50, 350, 400))


def test_pose_gate_detects_alternating_wrists():
    gate = Pose67Gate(required_movements=2, min_wrist_gap=10)
    assert not gate.update([pose(150, 210)])
    assert not gate.update([pose(210, 150)])
    assert gate.update([pose(150, 210)])
    assert gate.progress == "2/2"
