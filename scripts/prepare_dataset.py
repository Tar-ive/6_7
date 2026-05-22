#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="+", default=["data/frames", "data/people-balancing-pose"])
    p.add_argument("--out", default="data/yolo_hand_alarm")
    p.add_argument("--classes", nargs="+", default=["forearm", "underarm"])
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    classes = {name: i for i, name in enumerate(args.classes)}
    samples = collect(args.sources)
    out = Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    for i, (img, label) in enumerate(samples):
        stride = max(2, round(1 / args.val_ratio))
        part = "val" if i % stride == 0 else "train"
        write_sample(img, label, out, part, classes)
    print(f"wrote {len(samples)} labelled images to {args.out}")


def collect(sources: list[str]) -> list[tuple[Path, Path]]:
    samples = []
    for source in sources:
        for label in sorted(Path(source).glob("*.json")):
            meta = json.loads(label.read_text())
            image = label.with_name(meta.get("imagePath", label.with_suffix(".jpg").name))
            if not image.exists():
                image = next((label.with_suffix(ext) for ext in IMG_EXTS if label.with_suffix(ext).exists()), None)
            if image and image.exists():
                samples.append((image, label))
    if not samples:
        raise SystemExit("no labelled image/json pairs found")
    return samples


def write_sample(img: Path, label: Path, out: Path, part: str, classes: dict[str, int]) -> None:
    meta = json.loads(label.read_text())
    w, h = meta["imageWidth"], meta["imageHeight"]
    stem = f"{img.parent.name}_{img.stem}"
    img_out = out / "images" / part / f"{stem}{img.suffix.lower()}"
    label_out = out / "labels" / part / f"{stem}.txt"
    img_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, img_out)
    rows = []
    for shape in meta.get("shapes", []):
        cls = classes.get(shape.get("label"))
        if cls is None or shape.get("shape_type") != "rectangle":
            continue
        (x1, y1), (x2, y2) = shape["points"]
        x1, x2 = sorted((max(0, x1), min(w, x2)))
        y1, y2 = sorted((max(0, y1), min(h, y2)))
        xc, yc = ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        rows.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    label_out.write_text("\n".join(rows) + ("\n" if rows else ""))


if __name__ == "__main__":
    main()
