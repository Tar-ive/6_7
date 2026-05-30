import numpy as np

from alarm_yolo_mlx.pose import Pose
from alarm_yolo_mlx.pose_gate import Pose67Gate


def pose(left_wrist_y, right_wrist_y, elbow_y=170):
    xy = np.zeros((17, 2), dtype=float)
    conf = np.ones(17, dtype=float)
    xy[5], xy[7], xy[9] = (100, 100), (100, elbow_y), (160, left_wrist_y)
    xy[6], xy[8], xy[10] = (300, 100), (300, elbow_y), (240, right_wrist_y)
    return Pose(xy, conf, (50, 50, 350, 400))


def test_pose_gate_detects_alternating_wrists():
    gate = Pose67Gate(required_movements=2, min_wrist_gap=10, min_wrist_amplitude=40)
    assert not gate.update([pose(150, 210)])
    assert not gate.update([pose(210, 150)])
    assert gate.movement_count == 1
    assert gate.update([pose(150, 210)])
    assert gate.progress == "2/2"
    assert gate.movement_count == 2


def test_jitter_below_amplitude_does_not_count():
    gate = Pose67Gate(required_movements=2, min_wrist_amplitude=60)
    # wrists alternate which is higher, but only by ~10px of jitter
    for ly, ry in [(150, 160), (160, 150), (150, 160), (160, 150)]:
        gate.update([pose(ly, ry)])
    assert gate.movement_count == 0


def test_arms_down_does_not_count():
    gate = Pose67Gate(required_movements=2, min_wrist_amplitude=40)
    # full-amplitude alternation but both wrists below the elbows (hands in lap)
    for ly, ry in [(250, 320), (320, 250), (250, 320), (320, 250)]:
        gate.update([pose(ly, ry, elbow_y=170)])
    assert gate.movement_count == 0


def test_rep_streak_decays_when_motion_stops():
    gate = Pose67Gate(required_movements=4, min_wrist_amplitude=40, idle_reset_frames=5)
    assert not gate.update([pose(150, 210)])
    assert gate.update([pose(210, 150)]) is False
    assert gate.movement_count == 1
    # hold the same side past idle_reset_frames -> streak decays to zero
    for _ in range(7):
        gate.update([pose(210, 150)])
    assert gate.movement_count == 0


def test_missing_person_resets():
    gate = Pose67Gate(required_movements=4, min_wrist_amplitude=40, max_missing_frames=3)
    gate.update([pose(150, 210)])
    gate.update([pose(210, 150)])
    assert gate.movement_count == 1
    for _ in range(5):
        gate.update([])
    assert gate.movement_count == 0
