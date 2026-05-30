#!/usr/bin/env python3
"""Merge mug + phone source datasets into one YOLO dataset.

Each source contributes exactly one class regardless of its own internal category
ids: the mug source -> class 0 (mug), the phone source -> class 1 (phone). Each
source may be either a Roboflow COCO export (``train/_annotations.coco.json``) or a
YOLO-txt export (``train/images`` + ``train/labels``); the layout is autodetected.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mug", default="data/roboflow/new2", help="coffee mug source dir")
    parser.add_argument("--phone", default="data/roboflow/new1", help="phone source dir")
    parser.add_argument("--out", default="data/yolo_mug_phone")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for split in ["train", "val"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    mug_items = _load_items(Path(args.mug))
    phone_items = _load_items(Path(args.phone))
    _write_items(out, "mug", mug_items, cls=0, val_ratio=args.val_ratio)
    _write_items(out, "phone", phone_items, cls=1, val_ratio=args.val_ratio)
    _write_yaml(out)
    print(f"wrote {out}: mug={len(mug_items)} phone={len(phone_items)}")


def _load_items(root: Path) -> list[tuple[Path, list[tuple[float, float, float, float]]]]:
    """Return [(image_path, [(xc, yc, w, h) normalized, ...]), ...] for a source dir."""
    coco = root / "train" / "_annotations.coco.json"
    if coco.exists():
        return _coco_items(root, coco)
    if (root / "train" / "labels").exists():
        return _yolo_items(root)
    raise SystemExit(f"no recognizable dataset layout under {root}/train")


def _coco_items(root: Path, ann_path: Path) -> list[tuple[Path, list]]:
    data = json.loads(ann_path.read_text())
    images = {img["id"]: img for img in data["images"]}
    grouped: dict[int, list[dict]] = {}
    for ann in data["annotations"]:
        grouped.setdefault(ann["image_id"], []).append(ann)

    items = []
    for image_id, anns in grouped.items():
        meta = images[image_id]
        boxes = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append(
                (
                    (x + w / 2) / meta["width"],
                    (y + h / 2) / meta["height"],
                    w / meta["width"],
                    h / meta["height"],
                )
            )
        if boxes:
            items.append((root / "train" / meta["file_name"], boxes))
    return sorted(items, key=lambda item: item[0].name)


def _yolo_items(root: Path) -> list[tuple[Path, list]]:
    items = []
    for image in sorted((root / "train" / "images").iterdir()):
        if image.suffix not in IMAGE_EXTS:
            continue
        label = root / "train" / "labels" / f"{image.stem}.txt"
        if not (label.exists() and label.read_text().strip()):
            continue
        boxes = []
        for line in label.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                boxes.append(tuple(float(p) for p in parts[1:5]))
        if boxes:
            items.append((image, boxes))
    return items


def _write_items(out: Path, prefix: str, items: list[tuple[Path, list]], cls: int, val_ratio: float) -> None:
    val_every = max(int(1 / val_ratio), 2)
    for i, (image, boxes) in enumerate(items):
        split = "val" if i % val_every == 0 else "train"
        name = f"{prefix}_{image.name}"
        image_dst = out / "images" / split / name
        label_dst = out / "labels" / split / f"{Path(name).stem}.txt"
        _link_or_copy(image, image_dst)
        lines = [f"{cls} {xc:.8f} {yc:.8f} {w:.8f} {h:.8f}" for xc, yc, w, h in boxes]
        label_dst.write_text("\n".join(lines) + "\n")


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _write_yaml(out: Path) -> None:
    lines = [
        f"path: {out}",
        "train: images/train",
        "val: images/val",
        "nc: 2",
        "names:",
        "  0: mug",
        "  1: phone",
        "",
    ]
    Path("configs/mug_phone.yaml").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
