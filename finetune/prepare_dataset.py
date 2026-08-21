#!/usr/bin/env python3
"""第一阶段：只对原图运行人物检测模型，生成整图级框候选（COCO，无标签）。

属性预标注推迟到 WebUI 确认整图框之后逐裁剪进行，见 review_server.py。
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
import config  # noqa: E402


def enter_project_environment() -> None:
    python_name = "python.exe" if os.name == "nt" else "python"
    python_dir = "Scripts" if os.name == "nt" else "bin"
    expected = config.INFERENCE_VENV_DIR / python_dir / python_name
    if expected.is_file() and Path(sys.executable).resolve() != expected.resolve():
        completed = subprocess.run([str(expected), str(__file__), *sys.argv[1:]], cwd=PROJECT_ROOT)
        raise SystemExit(completed.returncode)


enter_project_environment()
sys.path.insert(0, str(HERE))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

_DLL_HANDLES = config.configure_runtime_dlls(config.INFERENCE_VENV_DIR)

from dataset_schema import (  # noqa: E402
    annotation_id,
    empty_dataset,
    image_id,
    load_dataset,
    save_dataset,
    stored_path,
    xyxy_to_xywh,
)
from prelabel_models import PaddleDetector  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成待人工核对的整图人物框候选（COCO）")
    parser.add_argument("--input-dir", type=Path, default=config.RAW_DATA_DIR)
    parser.add_argument("--device", choices=("CPU", "GPU"), default=config.DEVICE)
    parser.add_argument("--limit", type=int, help="本次最多读取多少张原图；不限制候选总量")
    parser.add_argument("--shuffle-seed", type=int, help="处理前确定性打乱原图")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="丢弃旧检测候选并从原图重新推理；已确认框不受影响",
    )
    return parser.parse_args()


def decode_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法解码图片: {path}")
    return image


def clipped_box(box: tuple[float, float, float, float], width: int, height: int) -> list[int] | None:
    left = max(0, min(width, round(box[0])))
    top = max(0, min(height, round(box[1])))
    right = max(0, min(width, round(box[2])))
    bottom = max(0, min(height, round(box[3])))
    return [left, top, right, bottom] if right > left and bottom > top else None


def main() -> int:
    args = arguments()
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every 必须大于零")
    images = sorted(
        (path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.relative_to(input_dir).as_posix().casefold(),
    )
    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(images)
    if args.limit is not None:
        images = images[: args.limit]

    candidates = (
        empty_dataset("detection", "model")
        if args.replace
        else load_dataset(config.DETECTION_CANDIDATES_PATH, "detection", "model", gold=False)
    )
    confirmed = load_dataset(config.DETECTION_CONFIRMED_PATH, "detection", "human", gold=True)
    known = {image["id"] for image in candidates["images"]} | {
        image["id"] for image in confirmed["images"]
    }
    detector = PaddleDetector(config.PERSON_DETECTOR_DIR, args.device)

    added_frames = 0
    added_boxes = 0
    for image_number, path in enumerate(images, 1):
        full_path = stored_path(path, config.PROJECT_ROOT)
        candidate_id = image_id("detection", full_path)
        if candidate_id in known:
            continue
        image = decode_image(path)
        height, width = image.shape[:2]
        candidates["images"].append(
            {"id": candidate_id, "file_name": full_path, "width": width, "height": height}
        )
        for box_number, detection in enumerate(detector.predict(image)):
            box = clipped_box(detection.box, width, height)
            if box is None:
                continue
            candidates["annotations"].append(
                {
                    "id": annotation_id(candidate_id, box_number),
                    "image_id": candidate_id,
                    "category_id": 1,
                    "bbox": [float(v) for v in xyxy_to_xywh(box)],
                }
            )
            added_boxes += 1
        known.add(candidate_id)
        added_frames += 1
        if image_number % args.checkpoint_every == 0:
            save_dataset(
                config.DETECTION_CANDIDATES_PATH, candidates, "detection", "model", gold=False
            )
            print(
                f"[{image_number}/{len(images)}] 新增整图 {added_frames}，人物框 {added_boxes}",
                flush=True,
            )
    save_dataset(config.DETECTION_CANDIDATES_PATH, candidates, "detection", "model", gold=False)
    print(f"完成: 新增检测整图 {added_frames}，人物框 {added_boxes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
