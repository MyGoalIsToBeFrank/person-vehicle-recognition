#!/usr/bin/env python3
"""第三阶段入口：把 AIZOO 整图、人脸框与口罩类别导入口罩候选（COCO）。"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import config  # noqa: E402
from dataset_schema import (  # noqa: E402
    annotation_id,
    image_id,
    load_dataset,
    mask_category_id,
    save_dataset,
    stored_path,
    xyxy_to_xywh,
)


def image_for_xml(xml_path: Path, filename: str) -> Path:
    direct = xml_path.parent / Path(filename).name
    if direct.is_file() and direct.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        return direct
    alternatives = [xml_path.with_suffix(suffix) for suffix in (".jpg", ".jpeg", ".png", ".bmp")]
    path = next((candidate for candidate in alternatives if candidate.is_file()), direct)
    if not path.is_file():
        raise FileNotFoundError(f"XML 对应图片不存在: {xml_path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="导入 AIZOO 口罩整图候选（COCO）")
    parser.add_argument("--input-dir", type=Path, default=config.RAW_DATA_DIR / "aizoo")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    args = parser.parse_args()
    xml_files = sorted(args.input_dir.resolve().rglob("*.xml"), key=lambda path: path.as_posix().casefold())
    if args.limit is not None:
        xml_files = xml_files[: args.limit]

    candidates = load_dataset(config.MASK_CANDIDATES_PATH, "mask", "aizoo", gold=False)
    gold = load_dataset(config.MASK_GOLD_PATH, "mask", "human", gold=True)
    known = {image["id"] for image in candidates["images"]} | {
        image["source_image_id"] for image in gold["images"]
    }
    added_frames = 0
    added_boxes = 0
    for xml_number, xml_path in enumerate(xml_files, 1):
        root = ET.parse(xml_path).getroot()
        filename = root.findtext("filename")
        if not filename:
            raise ValueError(f"XML 缺少 filename: {xml_path}")
        image_path = image_for_xml(xml_path, filename)
        full_path = stored_path(image_path, config.PROJECT_ROOT)
        candidate_id = image_id("mask", full_path)
        if candidate_id in known:
            continue
        size = root.find("size")
        width = int(size.findtext("width", "0")) if size is not None else 0
        height = int(size.findtext("height", "0")) if size is not None else 0
        annotations = []
        for box_number, item in enumerate(root.findall("object")):
            source_name = item.findtext("name")
            if source_name not in {"face", "face_mask", "face_nask"}:
                raise ValueError(f"未知 AIZOO 类别: {source_name} ({xml_path})")
            bounds = item.find("bndbox")
            if bounds is None:
                continue
            left = max(0, int(float(bounds.findtext("xmin", "0"))))
            top = max(0, int(float(bounds.findtext("ymin", "0"))))
            right = min(width, int(float(bounds.findtext("xmax", "0")))) if width else int(float(bounds.findtext("xmax", "0")))
            bottom = min(height, int(float(bounds.findtext("ymax", "0")))) if height else int(float(bounds.findtext("ymax", "0")))
            if right <= left or bottom <= top:
                continue
            annotations.append(
                {
                    "id": annotation_id(candidate_id, box_number),
                    "image_id": candidate_id,
                    "category_id": mask_category_id(
                        "w/ mask" if source_name in {"face_mask", "face_nask"} else "w/o mask"
                    ),
                    "bbox": [float(v) for v in xyxy_to_xywh([left, top, right, bottom])],
                }
            )
        candidates["images"].append(
            {"id": candidate_id, "file_name": full_path, "width": width, "height": height}
        )
        candidates["annotations"].extend(annotations)
        known.add(candidate_id)
        added_frames += 1
        added_boxes += len(annotations)
        if xml_number % args.checkpoint_every == 0:
            save_dataset(config.MASK_CANDIDATES_PATH, candidates, "mask", "aizoo", gold=False)
            print(f"[{xml_number}/{len(xml_files)}] 新增整图 {added_frames}，人脸框 {added_boxes}", flush=True)
    save_dataset(config.MASK_CANDIDATES_PATH, candidates, "mask", "aizoo", gold=False)
    print(f"完成: 新增口罩整图 {added_frames}，人脸框 {added_boxes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
