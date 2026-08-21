"""第四阶段：尘土化/昏黄化离线扩充。

对已有金标准（检测 confirmed、属性 gold、口罩 gold）施加更重的贴近业务现场的
光学退化，**几何不变、标签不变**，产物写入 ``4_augmented/<stage>/``。
这是唯一的增强手段；训练侧不做任何在线增强。

每次运行重建对应阶段的整个增强集（确定性种子，同样的输入得到同样的产物）。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import config  # noqa: E402


def enter_project_environment() -> None:
    python_name = "python.exe" if os.name == "nt" else "python"
    python_dir = "Scripts" if os.name == "nt" else "bin"
    expected = config.INFERENCE_VENV_DIR / python_dir / python_name
    if expected.is_file() and Path(sys.executable).resolve() != expected.resolve():
        completed = subprocess.run([str(expected), str(__file__), *sys.argv[1:]], cwd=PROJECT_ROOT)
        raise SystemExit(completed.returncode)


enter_project_environment()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from dataset_schema import (  # noqa: E402
    STAGES,
    annotation_id,
    empty_dataset,
    load_dataset,
    save_dataset,
    split_name,
    stored_path,
)


@dataclass
class AugmentParams:
    variants: int = 2  # 每个源样本生成多少个增强副本
    intensity: float = 1.0  # 强度倍率，0.5–2.0
    dust: bool = True  # 尘土：污渍斑块 + 颗粒 + 拖痕
    yellow: bool = True  # 昏黄：暖色偏移 + 大气幕罩 + 降亮度对比度
    low_light: bool = True
    blur: bool = True
    noise: bool = True
    jpeg: bool = True
    seed: int = 20260817

    def summary(self) -> dict[str, Any]:
        return asdict(self)


EFFECTS = ("dust", "yellow", "low_light", "blur", "noise", "jpeg")


def _rng_for(params: AugmentParams, source_id: str, variant: int) -> random.Random:
    digest = hashlib.sha1(f"{params.seed}\0{source_id}\0{variant}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:12], 16))


def apply_dust(value: np.ndarray, rng: random.Random, intensity: float) -> np.ndarray:
    """多层暗色/灰色斑块、尘土颗粒与拖痕。"""
    height, width = value.shape[:2]
    overlay = value.copy()
    blobs = max(2, int(rng.uniform(3, 10) * intensity))
    palette = ((40, 52, 66), (58, 66, 74), (30, 38, 48), (70, 74, 70))
    for _ in range(blobs):
        center = (rng.randrange(width), rng.randrange(height))
        axes = (
            max(2, int(min(width, height) * rng.uniform(0.02, 0.10) * intensity)),
            max(2, int(min(width, height) * rng.uniform(0.02, 0.10) * intensity)),
        )
        cv2.ellipse(overlay, center, axes, rng.uniform(0, 180), 0, 360, rng.choice(palette), -1)
    value = cv2.addWeighted(overlay, rng.uniform(0.20, 0.45), value, 0.75, 0)
    # 尘土颗粒：随机散布的暗/亮点。
    density = min(0.004, 0.0012 * intensity)
    grain = np.random.default_rng(rng.randrange(2**32))
    points = grain.random((height, width)) < density
    dark = grain.integers(20, 90, size=(int(points.sum()), 3), dtype=np.uint8)
    value[points] = dark
    # 一到两条斜向拖痕。
    for _ in range(rng.randint(1, 2)):
        p1 = (rng.randrange(width), rng.randrange(height))
        p2 = (min(width - 1, p1[0] + rng.randint(-width // 3, width // 3)),
              min(height - 1, p1[1] + rng.randint(-height // 3, height // 3)))
        streak = value.copy()
        cv2.line(streak, p1, p2, rng.choice(palette), max(1, int(2 * intensity)))
        value = cv2.addWeighted(streak, rng.uniform(0.10, 0.25), value, 0.85, 0)
    return value


def apply_yellow(value: np.ndarray, rng: random.Random, intensity: float) -> np.ndarray:
    """昏黄化：暖色通道偏移 + 黄色大气幕罩 + 亮度对比度下降。"""
    result = value.astype(np.float32)
    result[:, :, 2] *= rng.uniform(1.15, 1.40) * (0.5 + 0.5 * intensity)
    result[:, :, 1] *= rng.uniform(1.00, 1.12)
    result[:, :, 0] *= max(0.4, rng.uniform(0.55, 0.85) / (0.5 + 0.5 * intensity))
    result = np.clip(result, 0, 255).astype(np.uint8)
    # 大气幕罩：与昏黄色均匀层（可选垂直渐变）混合。
    veil_color = np.array([105, 140, 165], dtype=np.float32)  # BGR 昏黄
    height, width = result.shape[:2]
    if rng.random() < 0.5:
        gradient = np.linspace(0.6, 1.2, height, dtype=np.float32)[:, None, None]
        veil = np.clip(veil_color * gradient, 0, 255).repeat(width, axis=1)
    else:
        veil = np.full_like(result, veil_color, dtype=np.float32)
    alpha = min(0.45, rng.uniform(0.10, 0.28) * intensity)
    result = cv2.addWeighted(veil.astype(np.uint8), alpha, result, 1 - alpha, 0)
    # 降亮度与对比度。
    beta = rng.uniform(-40, -10) * intensity
    alpha_c = max(0.55, rng.uniform(0.75, 0.95) - 0.1 * (intensity - 1))
    return cv2.convertScaleAbs(result, alpha=alpha_c, beta=beta)


def apply_degradation(
    image: np.ndarray, params: AugmentParams, rng: random.Random
) -> np.ndarray:
    """按开关与强度随机组合各退化效果；顺序固定以保证可复现。"""
    intensity = max(0.5, min(2.0, params.intensity))
    value = image
    if params.low_light and rng.random() < 0.7:
        value = cv2.convertScaleAbs(value, alpha=rng.uniform(0.45, 0.85), beta=0)
    if params.yellow and rng.random() < 0.7:
        value = apply_yellow(value, rng, intensity)
    if params.dust and rng.random() < 0.7:
        value = apply_dust(value, rng, intensity)
    if params.blur and rng.random() < 0.5:
        kernel = rng.choice((3, 5))
        value = cv2.GaussianBlur(value, (kernel, kernel), rng.uniform(0.5, 2.0) * intensity)
    if params.noise and rng.random() < 0.5:
        noise = np.random.normal(0, rng.uniform(4, 16) * intensity, value.shape)
        value = np.clip(value.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if params.jpeg and rng.random() < 0.5:
        quality = rng.randint(max(15, int(45 - 15 * (intensity - 1))), 65)
        ok, encoded = cv2.imencode(".jpg", value, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            value = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return value


def _decode(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片: {path}")
    return image


def _save_jpeg(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError(f"无法编码图片: {path}")
    temporary = path.with_suffix(".jpg.tmp")
    encoded.tofile(temporary)
    temporary.replace(path)


def stage_source(stage: str) -> tuple[Path, str]:
    """返回该阶段增强的金标准来源（路径, source 标记）。"""
    if stage == "detection":
        return config.DETECTION_CONFIRMED_PATH, "human"
    if stage == "attribute":
        return config.ATTRIBUTE_GOLD_PATH, "human"
    return config.MASK_GOLD_PATH, "human"


def augment_stage(
    stage: str,
    params: AugmentParams,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """重建一个阶段的增强集，返回统计摘要。"""
    if stage not in STAGES:
        raise ValueError(f"stage 只能是 {STAGES}: {stage}")
    if params.variants < 1 or params.variants > 5:
        raise ValueError("variants 必须在 1–5 之间")
    source_path, source_name = stage_source(stage)
    gold = load_dataset(source_path, stage, source_name, gold=True)

    output_dir = config.AUGMENTED_DATA_DIR / stage
    images_dir = output_dir / "images"
    augmented = empty_dataset(stage, "augmented")
    total = len(gold["images"]) * params.variants
    done = 0
    for image in gold["images"]:
        source_image = _decode(config.PROJECT_ROOT / image["file_name"])
        height, width = source_image.shape[:2]
        for variant in range(params.variants):
            rng = _rng_for(params, image["id"], variant)
            degraded = apply_degradation(source_image, params, rng)
            digest = hashlib.sha1(f"aug\0{image['id']}\0{variant}".encode("utf-8")).hexdigest()[:20]
            new_image_id = f"aug_{digest}"
            target = images_dir / f"{new_image_id}.jpg"
            _save_jpeg(target, degraded)
            record = {
                "id": new_image_id,
                "file_name": stored_path(target, config.PROJECT_ROOT),
                "width": width,
                "height": height,
                "source_image_id": image.get("source_image_id", image["id"]),
                "augmentation": {**params.summary(), "variant": variant},
            }
            if "source_bbox" in image:
                record["source_bbox"] = image["source_bbox"]
            augmented["images"].append(record)
            for number, annotation in enumerate(
                item for item in gold["annotations"] if item["image_id"] == image["id"]
            ):
                copied = {
                    "id": annotation_id(new_image_id, number),
                    "image_id": new_image_id,
                    "category_id": annotation["category_id"],
                    "bbox": list(annotation["bbox"]),
                    "source_id": annotation["id"],
                }
                if "attributes" in annotation:
                    copied["attributes"] = dict(annotation["attributes"])
                augmented["annotations"].append(copied)
            done += 1
            if progress is not None:
                progress(done, total)
    save_dataset(output_dir / "annotations.json", augmented, stage, "augmented", gold=True)
    split_keys = {image["id"]: image["source_image_id"] for image in augmented["images"]}
    train = sum(
        1
        for annotation in augmented["annotations"]
        if split_name(split_keys[annotation["image_id"]]) == "train"
    )
    return {
        "stage": stage,
        "源图片": len(gold["images"]),
        "增强图片": len(augmented["images"]),
        "增强标注": len(augmented["annotations"]),
        "其中训练集标注": train,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="尘土化/昏黄化离线扩充（标签不变）")
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--intensity", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260817)
    for name in EFFECTS:
        parser.add_argument(f"--no-{name.replace('_', '-')}", dest=name,
                            action="store_false", help=f"关闭 {name} 效果")
    args = parser.parse_args()
    params = AugmentParams(
        variants=args.variants,
        intensity=args.intensity,
        seed=args.seed,
        **{name: getattr(args, name) for name in EFFECTS},
    )
    summary = augment_stage(stage=args.stage, params=params)
    print("增强完成:", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
