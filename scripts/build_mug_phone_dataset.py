#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import yaml
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mug", default="data/roboflow/coffee_mug")
    parser.add_argument("--phone", default="data/roboflow/phone")
    parser.add_argument("--out", default="data/yolo_mug_phone")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for split in ["train", "val"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    mug_items = _mug_items(Path(args.mug))
    phone_items = _phone_items(Path(args.phone))
    _write_items(out, "mug", mug_items, cls=0, val_ratio=args.val_ratio)
    _write_items(out, "phone", phone_items, cls=1, val_ratio=args.val_ratio)
    _write_yaml(out)
    print(f"wrote {out}: mug={len(mug_items)} phone={len(phone_items)}")


def _mug_items(root: Path) -> list[tuple[Path, list[str]]]:
    items = []
    for image in sorted((root / "train" / "images").iterdir()):
        if image.suffix not in IMAGE_EXTS:
            continue
        label = root / "train" / "labels" / f"{image.stem}.txt"
        if label.exists() and label.read_text().strip():
            items.append((image, label.read_text().splitlines()))
    return items


def _phone_items(root: Path) -> list[tuple[Path, list[str]]]:
    ann_path = root / "train" / "_annotations.coco.json"
    data = json.loads(ann_path.read_text())
    images = {img["id"]: img for img in data["images"]}
    grouped: dict[int, list[dict]] = {}
    for ann in data["annotations"]:
        grouped.setdefault(ann["image_id"], []).append(ann)

    items = []
    for image_id, anns in grouped.items():
        meta = images[image_id]
        lines = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            xc = (x + w / 2) / meta["width"]
            yc = (y + h / 2) / meta["height"]
            lines.append(f"0 {xc:.8f} {yc:.8f} {w / meta['width']:.8f} {h / meta['height']:.8f}")
        if lines:
            items.append((root / "train" / meta["file_name"], lines))
    return sorted(items, key=lambda item: item[0].name)


def _write_items(
    out: Path, prefix: str, items: list[tuple[Path, list[str]]], cls: int, val_ratio: float
) -> None:
    val_every = max(int(1 / val_ratio), 2)
    for i, (image, lines) in enumerate(items):
        split = "val" if i % val_every == 0 else "train"
        name = f"{prefix}_{image.name}"
        image_dst = out / "images" / split / name
        label_dst = out / "labels" / split / f"{Path(name).stem}.txt"
        _link_or_copy(image, image_dst)
        label_dst.write_text("\n".join(_with_class(lines, cls)) + "\n")


def _with_class(lines: list[str], cls: int) -> list[str]:
    out = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 5:
            out.append(" ".join([str(cls), *parts[1:5]]))
    return out


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _write_yaml(out: Path) -> None:
    data = {
        "path": str(out),
        "train": "images/train",
        "val": "images/val",
        "nc": 2,
        "names": {0: "mug", 1: "phone"},
    }
    Path("configs/mug_phone.yaml").write_text(yaml.safe_dump(data, sort_keys=False))


if __name__ == "__main__":
    main()
