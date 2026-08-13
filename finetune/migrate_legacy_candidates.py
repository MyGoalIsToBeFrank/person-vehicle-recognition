#!/usr/bin/env python3
"""一次性把旧扁平预标注转成整图候选；不把旧预测冒充人工金标准。"""

from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "independent"))
sys.path.insert(0, str(HERE))

import config  # noqa: E402
from dataset_schema import empty_dataset, frame_id, save_candidates, save_gold  # noqa: E402


def group(rows: list[dict], section: str) -> list[dict]:
    frames: OrderedDict[str, dict] = OrderedDict()
    for row in rows:
        full_image = row["全图"]
        identifier = frame_id(section, full_image)
        frame = frames.setdefault(identifier, {"id": identifier, "全图": full_image, "框": []})
        frame["框"].append({"id": row["id"], "框": row["框"], "标签": row["标签"]})
    return list(frames.values())


def main() -> int:
    if config.CANDIDATES_PATH.exists() or config.GOLD_LABELS_PATH.exists():
        raise FileExistsError("新候选或金标准已经存在，拒绝覆盖")
    legacy = json.loads(config.LEGACY_LABELS_PATH.read_text(encoding="utf-8"))
    candidates = empty_dataset()
    candidates["属性"] = group(legacy["全人"], "属性")

    archive = config.CANDIDATE_ARCHIVE_DIR / "labels_before_review_cap.json"
    face_source = json.loads(archive.read_text(encoding="utf-8")) if archive.is_file() else legacy
    candidates["口罩"] = group(face_source["脸部"], "口罩")
    save_candidates(config.CANDIDATES_PATH, candidates)
    save_gold(config.GOLD_LABELS_PATH, empty_dataset())
    print(
        f"迁移完成: 属性整图={len(candidates['属性'])}, 口罩整图={len(candidates['口罩'])}; "
        "gold_labels.json 为空，只有新 WebUI 保存才进入金标准"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
