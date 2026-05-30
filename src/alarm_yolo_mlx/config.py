from __future__ import annotations

from dataclasses import dataclass, fields

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


@dataclass
class AppConfig:
    weights: str
    source: str | None = None
    camera_index: int = 0
    imgsz: int = 640
    conf: float = 0.45
    target_classes: list[str] | None = None
    class_names: dict[int, str] | None = None
    required_hits: int = 5
    window_frames: int = 8
    stop_mode: str = "motion"
    motion_class: str = "forearm"
    min_switches: int = 2
    track_iou: float = 0.2
    alarm_sound: str | None = None
    show: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "AppConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) if yaml else _simple_yaml(f.read())
        allowed = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in (raw or {}).items() if k in allowed})


@dataclass
class PoseConfig:
    backend: str = "mlx"
    weights: str = "models/yolo26n-pose.npz"
    source: str | None = None
    camera_index: int = 0
    imgsz: int = 640
    conf: float = 0.25
    required_movements: int = 4
    window_frames: int = 90
    min_keypoint_conf: float = 0.25
    max_elbow_angle: float = 140
    min_wrist_gap: float = 20
    max_missing_frames: int = 12
    min_wrist_amplitude: float = 55
    require_arms_up: bool = True
    idle_reset_frames: int = 45
    detection_window: int = 8
    mug_min_hits: int = 5
    phone_min_hits: int = 4
    alarm_sound: str | None = None
    alarm_full_volume: float = 0.8
    alarm_duck_volume: float = 0.2
    movement_sets_min: int = 1
    movement_sets_max: int = 3
    start_prompt_sound: str | None = None
    mug_prompt_sound: str | None = None
    stopped_sound: str | None = None
    spoof_sound: str | None = None
    object_weights: str = "models/yolo26n.npz"
    object_conf: float = 0.35
    mug_conf: float = 0.30
    phone_conf: float = 0.35
    mug_classes: list[str] | None = None
    phone_classes: list[str] | None = None
    object_class_names: dict[int, str] | None = None
    mug_encouragement_frames: int = 5
    required_mug_frames: int = 5
    show: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "PoseConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) if yaml else _simple_yaml(f.read())
        allowed = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in (raw or {}).items() if k in allowed})


def _simple_yaml(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, val = [part.strip() for part in line.split(":", 1)]
        out[key] = _value(val)
    return out


def _value(val: str):
    if val in {"true", "false"}:
        return val == "true"
    if val.startswith("[") and val.endswith("]"):
        return [_value(v.strip().strip("\"'")) for v in val[1:-1].split(",") if v.strip()]
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val.strip("\"'")
