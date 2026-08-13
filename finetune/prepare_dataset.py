#!/usr/bin/env python3
"""用现有人物检测与属性模型生成整图级属性候选；不生成训练裁剪。"""

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
sys.path.insert(0, str(PROJECT_ROOT / "independent"))

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
    BODY_ATTRIBUTES,
    frame_id,
    load_candidates,
    load_gold,
    save_candidates,
    stored_path,
)
from model_adapters import (  # noqa: E402
    PaddleAttributeModel,
    PaddleDetector,
    paddle_model_files,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成待人工核对的整图人物框与属性候选")
    parser.add_argument("--input-dir", type=Path, default=config.RAW_DATA_DIR)
    parser.add_argument("--device", choices=("CPU", "GPU"), default=config.DEVICE)
    parser.add_argument("--limit", type=int, help="本次最多读取多少张原图；不限制候选总量")
    parser.add_argument("--shuffle-seed", type=int, help="处理前确定性打乱原图")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="丢弃旧属性候选并从原图重新推理；已保存金标准不受影响",
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


def initial_body_labels(scores: np.ndarray) -> dict[str, bool]:
    labels = {name: False for name in BODY_ATTRIBUTES}
    labels["Hat"] = bool(scores[0] > 0.5)
    labels["Glasses"] = bool(scores[1] > 0.3)
    labels["LongSleeve" if scores[3] > scores[2] else "ShortSleeve"] = True
    upper = int(np.argmax(scores[4:8]))
    if scores[4 + upper] > 0.5:
        labels[BODY_ATTRIBUTES[4 + upper]] = True
    lower = [index for index in range(8, 14) if scores[index] > 0.5]
    for index in lower or [8 + int(np.argmax(scores[8:14]))]:
        labels[BODY_ATTRIBUTES[index]] = True
    labels["Boots"] = bool(scores[14] > 0.5)
    bag = int(np.argmax(scores[15:18]))
    if scores[15 + bag] > 0.5:
        labels[BODY_ATTRIBUTES[15 + bag]] = True
    labels["HoldObjectsInFront"] = bool(scores[18] > 0.6)
    labels[BODY_ATTRIBUTES[19 + int(np.argmax(scores[19:22]))]] = True
    labels["Female"] = bool(scores[22] > 0.5)
    labels[BODY_ATTRIBUTES[23 + int(np.argmax(scores[23:26]))]] = True
    return labels


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

    candidates = load_candidates(config.CANDIDATES_PATH)
    gold = load_gold(config.GOLD_LABELS_PATH)
    if args.replace:
        candidates["属性"] = []
        save_candidates(config.CANDIDATES_PATH, candidates)
    known = {frame["id"] for frame in candidates["属性"]} | {
        frame["id"] for frame in gold["属性"]
    }
    detector = PaddleDetector(config.PERSON_DETECTOR_DIR, args.device)
    model_file, params_file = paddle_model_files(config.PERSON_ATTRIBUTE_DIR, "inference")
    attributes = PaddleAttributeModel(model_file, params_file, (192, 256), args.device)

    added_frames = 0
    added_boxes = 0
    for image_number, path in enumerate(images, 1):
        full_path = stored_path(path, config.PROJECT_ROOT)
        candidate_id = frame_id("属性", full_path)
        if candidate_id in known:
            continue
        image = decode_image(path)
        height, width = image.shape[:2]
        boxes = []
        for box_number, detection in enumerate(detector.predict(image)):
            box = clipped_box(detection.box, width, height)
            if box is None:
                continue
            left, top, right, bottom = box
            labels = initial_body_labels(attributes.predict(image[top:bottom, left:right]))
            boxes.append({"id": f"{candidate_id}_{box_number:03d}", "框": box, "标签": labels})
        candidates["属性"].append({"id": candidate_id, "全图": full_path, "框": boxes})
        known.add(candidate_id)
        added_frames += 1
        added_boxes += len(boxes)
        if image_number % args.checkpoint_every == 0:
            save_candidates(config.CANDIDATES_PATH, candidates)
            print(
                f"[{image_number}/{len(images)}] 新增整图 {added_frames}，人物框 {added_boxes}",
                flush=True,
            )
    save_candidates(config.CANDIDATES_PATH, candidates)
    print(f"完成: 新增属性整图 {added_frames}，人物框 {added_boxes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
