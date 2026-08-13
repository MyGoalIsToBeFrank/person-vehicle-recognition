#!/usr/bin/env python3
"""只从人工属性金标准导出人物检测 COCO 数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "independent"))
sys.path.insert(0, str(HERE))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from dataset_schema import load_gold, resolved_path  # noqa: E402


def split_name(full_image: str) -> str:
    bucket = int(hashlib.sha1(full_image.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "test" if bucket == 0 else "val" if bucket == 1 else "train"


def image_size(path: Path) -> tuple[int, int]:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取人物检测原图: {path}")
    height, width = image.shape[:2]
    return width, height


def export(gold_path: Path, output_dir: Path) -> dict[str, tuple[int, int]]:
    gold = load_gold(gold_path)
    images_dir = output_dir / "images"
    annotations_dir = output_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    grouped = {name: [] for name in ("train", "val", "test")}
    for frame in gold["属性"]:
        grouped[split_name(frame["全图"])].append(frame)

    summary = {}
    for split, frames in grouped.items():
        images = []
        annotations = []
        annotation_id = 1
        for image_id, frame in enumerate(frames, 1):
            source = resolved_path(frame["全图"], config.PROJECT_ROOT)
            suffix = source.suffix.lower() if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"} else ".jpg"
            name = f"{hashlib.sha1(frame['全图'].encode('utf-8')).hexdigest()[:24]}{suffix}"
            target = images_dir / name
            if not target.is_file():
                shutil.copy2(source, target)
            width, height = image_size(source)
            images.append({"id": image_id, "file_name": name, "width": width, "height": height})
            for item in frame["框"]:
                left, top, right, bottom = map(float, item["框"])
                box_width, box_height = right - left, bottom - top
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [left, top, box_width, box_height],
                        "area": box_width * box_height,
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1
        data = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
        }
        path = annotations_dir / f"instances_{split}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary[split] = (len(images), len(annotations))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="导出人工人物检测 COCO 金标准")
    parser.add_argument("--gold", type=Path, default=config.GOLD_LABELS_PATH)
    parser.add_argument("--output-dir", type=Path, default=config.PERSON_DETECTION_COCO_DIR)
    args = parser.parse_args()
    summary = export(args.gold, args.output_dir)
    print("人物检测 COCO:", ", ".join(f"{name}={images}图/{boxes}框" for name, (images, boxes) in summary.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
