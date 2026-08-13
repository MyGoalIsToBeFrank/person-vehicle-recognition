#!/usr/bin/env python3
"""把 AIZOO 整图、人脸框与口罩标签导入候选队列；不生成训练裁剪。"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "independent"))
sys.path.insert(0, str(HERE))

import config  # noqa: E402
from dataset_schema import (  # noqa: E402
    frame_id,
    load_candidates,
    load_gold,
    save_candidates,
    stored_path,
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
    parser = argparse.ArgumentParser(description="导入 AIZOO 口罩整图候选")
    parser.add_argument("--input-dir", type=Path, default=config.RAW_DATA_DIR / "aizoo")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    args = parser.parse_args()
    xml_files = sorted(args.input_dir.resolve().rglob("*.xml"), key=lambda path: path.as_posix().casefold())
    if args.limit is not None:
        xml_files = xml_files[: args.limit]

    candidates = load_candidates(config.CANDIDATES_PATH)
    gold = load_gold(config.GOLD_LABELS_PATH)
    known = {frame["id"] for frame in candidates["口罩"]} | {frame["id"] for frame in gold["口罩"]}
    added_frames = 0
    added_boxes = 0
    for xml_number, xml_path in enumerate(xml_files, 1):
        root = ET.parse(xml_path).getroot()
        filename = root.findtext("filename")
        if not filename:
            raise ValueError(f"XML 缺少 filename: {xml_path}")
        image_path = image_for_xml(xml_path, filename)
        full_path = stored_path(image_path, config.PROJECT_ROOT)
        candidate_id = frame_id("口罩", full_path)
        if candidate_id in known:
            continue
        size = root.find("size")
        width = int(size.findtext("width", "0")) if size is not None else 0
        height = int(size.findtext("height", "0")) if size is not None else 0
        boxes = []
        for box_number, item in enumerate(root.findall("object")):
            source_name = item.findtext("name")
            if source_name not in {"face", "face_mask", "face_nask"}:
                raise ValueError(f"未知 AIZOO 类别: {source_name} ({xml_path})")
            bounds = item.find("bndbox")
            if bounds is None:
                continue
            box = [
                max(0, int(float(bounds.findtext("xmin", "0")))),
                max(0, int(float(bounds.findtext("ymin", "0")))),
                min(width, int(float(bounds.findtext("xmax", "0")))) if width else int(float(bounds.findtext("xmax", "0"))),
                min(height, int(float(bounds.findtext("ymax", "0")))) if height else int(float(bounds.findtext("ymax", "0"))),
            ]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            boxes.append(
                {
                    "id": f"{candidate_id}_{box_number:03d}",
                    "框": box,
                    "标签": "w/ mask" if source_name in {"face_mask", "face_nask"} else "w/o mask",
                }
            )
        candidates["口罩"].append({"id": candidate_id, "全图": full_path, "框": boxes})
        known.add(candidate_id)
        added_frames += 1
        added_boxes += len(boxes)
        if xml_number % args.checkpoint_every == 0:
            save_candidates(config.CANDIDATES_PATH, candidates)
            print(f"[{xml_number}/{len(xml_files)}] 新增整图 {added_frames}，人脸框 {added_boxes}", flush=True)
    save_candidates(config.CANDIDATES_PATH, candidates)
    print(f"完成: 新增口罩整图 {added_frames}，人脸框 {added_boxes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
