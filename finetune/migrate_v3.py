#!/usr/bin/env python3
"""把版本 2 的 candidates.json 迁移到版本 3 的阶段化 COCO 布局。

- 属性栏（模型人物框候选）→ 1_detection/candidates.json
- 口罩栏（AIZOO 人脸框）→ 3_mask/candidates.json
- 旧 labels.json、images/、gold_labels.json、gold_images/、candidates.json → _legacy/

只读图片文件头获取宽高，不做完整解码。迁移前会校验，失败时不写任何新文件。
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "inference"))
sys.path.insert(0, str(HERE))

import config  # noqa: E402
from dataset_schema import (  # noqa: E402
    annotation_id,
    empty_dataset,
    image_id,
    mask_category_id,
    save_dataset,
    stored_path,
    xyxy_to_xywh,
)


def image_size_fast(path: Path) -> tuple[int, int]:
    """只读文件头解析 JPEG/PNG/BMP 宽高。"""
    header = path.open("rb")
    with header as stream:
        magic = stream.read(24)
        if magic[:2] == b"\xff\xd8":  # JPEG
            stream.seek(2)
            while True:
                marker_prefix = stream.read(1)
                if not marker_prefix:
                    break
                if marker_prefix != b"\xff":
                    continue
                marker = stream.read(1)
                while marker == b"\xff":
                    marker = stream.read(1)
                if marker in (b"\xd8", b"\xd9") or b"\xd0" <= marker <= b"\xd7":
                    continue
                length = struct.unpack(">H", stream.read(2))[0]
                if marker in (b"\xc0", b"\xc1", b"\xc2", b"\xc3", b"\xc5", b"\xc6", b"\xc7",
                              b"\xc9", b"\xca", b"\xcb", b"\xcd", b"\xce", b"\xcf"):
                    stream.read(1)
                    height, width = struct.unpack(">HH", stream.read(4))
                    return width, height
                stream.seek(length - 2, 1)
        elif magic[:8] == b"\x89PNG\r\n\x1a\n":
            width, height = struct.unpack(">II", magic[16:24])
            return width, height
        elif magic[:2] == b"BM":
            stream.seek(18)
            width, height = struct.unpack("<ii", stream.read(8))
            return width, abs(height)
    raise ValueError(f"无法解析图片宽高: {path}")


def load_legacy_candidates(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("版本") != 2:
        raise ValueError(f"候选文件版本不是 2: {path}")
    return data


def migrate_detection(frames: list[dict], target: Path, force: bool) -> tuple[int, int]:
    if target.exists() and not force:
        raise FileExistsError(f"目标已存在，使用 --force 覆盖: {target}")
    dataset = empty_dataset("detection", "model")
    box_count = 0
    for frame in frames:
        full_image = frame["全图"]
        source = (PROJECT_ROOT / full_image).resolve()
        width, height = image_size_fast(source)
        new_image_id = image_id("detection", full_image)
        dataset["images"].append(
            {"id": new_image_id, "file_name": full_image, "width": width, "height": height}
        )
        for number, item in enumerate(frame["框"]):
            bbox = [float(v) for v in xyxy_to_xywh([float(v) for v in item["框"]])]
            dataset["annotations"].append(
                {
                    "id": annotation_id(new_image_id, number),
                    "image_id": new_image_id,
                    "category_id": 1,
                    "bbox": bbox,
                }
            )
            box_count += 1
    save_dataset(target, dataset, "detection", "model", gold=False)
    return len(dataset["images"]), box_count


def migrate_mask(frames: list[dict], target: Path, force: bool) -> tuple[int, int]:
    if target.exists() and not force:
        raise FileExistsError(f"目标已存在，使用 --force 覆盖: {target}")
    dataset = empty_dataset("mask", "aizoo")
    box_count = 0
    for frame in frames:
        full_image = frame["全图"]
        source = (PROJECT_ROOT / full_image).resolve()
        width, height = image_size_fast(source)
        new_image_id = image_id("mask", full_image)
        dataset["images"].append(
            {"id": new_image_id, "file_name": full_image, "width": width, "height": height}
        )
        for number, item in enumerate(frame["框"]):
            bbox = [float(v) for v in xyxy_to_xywh([float(v) for v in item["框"]])]
            dataset["annotations"].append(
                {
                    "id": annotation_id(new_image_id, number),
                    "image_id": new_image_id,
                    "category_id": mask_category_id(item["标签"]),
                    "bbox": bbox,
                }
            )
            box_count += 1
    save_dataset(target, dataset, "mask", "aizoo", gold=False)
    return len(dataset["images"]), box_count


def archive_legacy(processed: Path) -> list[str]:
    legacy = processed / "_legacy"
    moved = []
    for name in ("labels.json", "images", "gold_labels.json", "gold_images", "candidates.json"):
        source = processed / name
        if source.exists():
            target = legacy / name
            if target.exists():
                raise FileExistsError(f"_legacy 中已存在: {target}")
            legacy.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            moved.append(name)
    return moved


def main() -> int:
    parser = argparse.ArgumentParser(description="迁移版本 2 候选数据到阶段化 COCO 布局")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的新布局文件")
    args = parser.parse_args()
    legacy_candidates = config.PROCESSED_DATA_DIR / "candidates.json"
    if not legacy_candidates.is_file():
        raise FileNotFoundError(f"旧候选文件不存在: {legacy_candidates}")
    data = load_legacy_candidates(legacy_candidates)
    det_images, det_boxes = migrate_detection(
        data["属性"], config.DETECTION_CANDIDATES_PATH, args.force
    )
    mask_images, mask_boxes = migrate_mask(
        data["口罩"], config.MASK_CANDIDATES_PATH, args.force
    )
    moved = archive_legacy(config.PROCESSED_DATA_DIR)
    print(f"1_detection/candidates.json: {det_images} 图 / {det_boxes} 框")
    print(f"3_mask/candidates.json: {mask_images} 图 / {mask_boxes} 框")
    print(f"归档到 _legacy/: {', '.join(moved) or '无'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
