#!/usr/bin/env python3
"""清空全部人工成果（确认框、属性/口罩金标准、增强集），让所有整图回到待核对候选。

已确认的检测整图连同当前框会重新并入 1_detection/candidates.json 作为候选，
候选顺序由 WebUI 启动时随机打乱，与文件内顺序无关。raw 原图不受影响。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "inference"))
sys.path.insert(0, str(HERE))

import config  # noqa: E402
from dataset_schema import load_dataset, save_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="清空人工标记进度，回到全候选状态")
    parser.add_argument("--yes", action="store_true", help="确认执行（不可逆，需显式指定）")
    args = parser.parse_args()
    if not args.yes:
        print("该操作会删除所有已确认/已保存标记。确认执行请加 --yes")
        return 1

    restored = 0
    if config.DETECTION_CONFIRMED_PATH.is_file():
        confirmed = load_dataset(config.DETECTION_CONFIRMED_PATH, "detection", "human", gold=True)
        candidates = load_dataset(config.DETECTION_CANDIDATES_PATH, "detection", "model", gold=False)
        existing = {image["id"] for image in candidates["images"]}
        for image in confirmed["images"]:
            if image["id"] in existing:
                continue
            candidates["images"].append(image)
            candidates["annotations"].extend(
                item for item in confirmed["annotations"] if item["image_id"] == image["id"]
            )
            existing.add(image["id"])
            restored += 1
        save_dataset(config.DETECTION_CANDIDATES_PATH, candidates, "detection", "model", gold=False)
        config.DETECTION_CONFIRMED_PATH.unlink()

    removed = []
    for path in (
        config.ATTRIBUTE_CANDIDATES_PATH,
        config.ATTRIBUTE_GOLD_PATH,
        config.MASK_GOLD_PATH,
    ):
        if path.is_file():
            path.unlink()
            removed.append(path.name)
    for directory in (
        config.ATTRIBUTE_IMAGES_DIR,
        config.MASK_IMAGES_DIR,
        config.AUGMENTED_DATA_DIR,
    ):
        if directory.is_dir():
            shutil.rmtree(directory)
            removed.append(f"{directory.name}/")
    print(f"已把 {restored} 张确认整图并回检测候选；删除: {', '.join(removed) or '无'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
