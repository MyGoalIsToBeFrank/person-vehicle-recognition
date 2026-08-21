#!/usr/bin/env python3
"""把 1_detection/confirmed.json（及 4_augmented/detection 训练划分）导出为训练用 COCO。

增强整图只进入其源图所属划分为 train 时的训练集；验证/测试集只用人工确认原图，
避免同一原图的增强副本跨集泄漏。输出到 5_export/person_detection_coco。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import config  # noqa: E402
from dataset_schema import load_dataset, resolved_path, split_name  # noqa: E402


def collect(confirmed: dict, augmented: dict | None) -> dict[str, list[dict]]:
    grouped = {name: [] for name in ("train", "val", "test")}
    for image in confirmed["images"]:
        grouped[split_name(image["id"])].append(
            (image, [a for a in confirmed["annotations"] if a["image_id"] == image["id"]])
        )
    if augmented is not None:
        for image in augmented["images"]:
            if split_name(image["source_image_id"]) != "train":
                continue
            grouped["train"].append(
                (image, [a for a in augmented["annotations"] if a["image_id"] == image["id"]])
            )
    return grouped


def export(output_dir: Path) -> dict[str, tuple[int, int]]:
    confirmed = load_dataset(config.DETECTION_CONFIRMED_PATH, "detection", "human", gold=True)
    augmented_path = config.AUGMENTED_DATA_DIR / "detection/annotations.json"
    augmented = (
        load_dataset(augmented_path, "detection", "augmented", gold=True)
        if augmented_path.is_file()
        else None
    )
    grouped = collect(confirmed, augmented)
    images_dir = output_dir / "images"
    annotations_dir = output_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split, rows in grouped.items():
        images = []
        annotations = []
        annotation_id = 1
        for image_id, (image, items) in enumerate(rows, 1):
            source = resolved_path(image["file_name"], config.PROJECT_ROOT)
            suffix = source.suffix.lower() if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"} else ".jpg"
            name = f"{image['id']}{suffix}"
            target = images_dir / name
            if not target.is_file():
                shutil.copy2(source, target)
            images.append(
                {"id": image_id, "file_name": name, "width": image["width"], "height": image["height"]}
            )
            for item in items:
                x, y, width, height = map(float, item["bbox"])
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [x, y, width, height],
                        "area": width * height,
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
    parser = argparse.ArgumentParser(description="导出人物检测训练 COCO（含增强训练集）")
    parser.add_argument("--output-dir", type=Path, default=config.PERSON_DETECTION_COCO_DIR)
    args = parser.parse_args()
    summary = export(args.output_dir)
    print("人物检测 COCO:", ", ".join(f"{name}={images}图/{boxes}框" for name, (images, boxes) in summary.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
