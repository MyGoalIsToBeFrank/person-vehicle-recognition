#!/usr/bin/env python3
"""Paddle → ONNX 数值回归：同一输入下两个引擎的最终业务输出必须一致。

用法（在 .venv 里，ONNX 文件放在 models/onnx/<名称>/）：

    .venv/Scripts/python.exe -X utf8 inference/onnx_regression.py

检测器比对的是生产代码路径（PaddleDetector vs OnnxDetector 的 predict），
即阈值过滤后的检出框；属性模型比对 [1,N] 原始输出。
通过标准：属性模型逐元素最大绝对误差 <= 1e-4；检测器检出框数量完全一致，
框坐标/分数误差 <= 5e-4。不达标时应回退 Paddle GPU 方案，不要带病上线。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config as project_config  # noqa: E402,F401  (设置 FLAGS_enable_pir_api 供 Paddle 基准)
from model_adapters import (  # noqa: E402
    OnnxAttributeModel,
    OnnxDetector,
    PaddleAttributeModel,
    PaddleDetector,
    paddle_model_files,
)

TOLERANCE = 1e-4
# 检测器比对跨引擎（Paddle GPU vs onnxruntime CPU），卷积浮点舍入差异会放大到
# 1e-4 量级，故框坐标/分数容差放宽到 5e-4（约千分之一个像素）；检出框数量必须严格一致。
DETECTOR_TOLERANCE = 5e-4


def compare(name: str, paddle_out: np.ndarray, onnx_out: np.ndarray) -> bool:
    if paddle_out.shape != onnx_out.shape:
        print(f"[FAIL] {name}: 形状不一致 {paddle_out.shape} vs {onnx_out.shape}")
        return False
    diff = float(np.max(np.abs(paddle_out.astype(np.float64) - onnx_out.astype(np.float64))))
    ok = diff <= TOLERANCE
    print(f"[{'OK' if ok else 'FAIL'}] {name}: shape={paddle_out.shape} max|diff|={diff:.3e}")
    return ok


def detector_case(
    name: str,
    paddle_model_dir: Path,
    onnx_model_dir: Path,
    image: np.ndarray,
) -> bool:
    # Paddle 3.3 的 PIR 检测图在 CPU/oneDNN 下跑不了 nearest_interp（ArrayAttribute<Double>
    # 未实现），GPU 路径正常，所以基准侧用 GPU；ONNX 侧用 CPU 反而更严格。
    paddle_dets = PaddleDetector(paddle_model_dir, "GPU").predict(image)
    onnx_dets = OnnxDetector(onnx_model_dir, "CPU").predict(image)
    if len(paddle_dets) != len(onnx_dets):
        print(f"[FAIL] {name}: 检出数量不一致 {len(paddle_dets)} vs {len(onnx_dets)}")
        return False
    paddle_sorted = sorted((d.score, *d.box) for d in paddle_dets)
    onnx_sorted = sorted((d.score, *d.box) for d in onnx_dets)
    if not paddle_sorted:
        print(f"[OK] {name}: 两侧均无检出（空场景一致）")
        return True
    diff = float(np.max(np.abs(np.asarray(paddle_sorted) - np.asarray(onnx_sorted))))
    ok = diff <= DETECTOR_TOLERANCE
    print(f"[{'OK' if ok else 'FAIL'}] {name}: {len(paddle_dets)} 个检出框, max|diff|={diff:.3e}")
    return ok


def attribute_case(
    name: str,
    paddle_model_dir: Path,
    stem: str,
    onnx_model_dir: Path,
    size: tuple[int, int],
    image: np.ndarray,
) -> bool:
    model, params = paddle_model_files(paddle_model_dir, stem)
    paddle_out = PaddleAttributeModel(model, params, size, "CPU").predict(image)
    onnx_out = OnnxAttributeModel(onnx_model_dir, size, "CPU").predict(image)
    return compare(name, np.asarray(paddle_out).reshape(1, -1), np.asarray(onnx_out).reshape(1, -1))


def main() -> int:
    parser = argparse.ArgumentParser(description="Paddle→ONNX 数值回归")
    parser.add_argument("--onnx-dir", type=Path, default=HERE.parent / "models/onnx")
    parser.add_argument("--sample", type=Path, default=HERE.parent / "easy_test/images (5).jpg")
    args = parser.parse_args()

    encoded = np.fromfile(args.sample, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"样例图片读取失败: {args.sample}")

    model_dir = HERE.parent / "models"
    results = [
        detector_case(
            "person_detector",
            model_dir / "finetuned/person_detector",
            args.onnx_dir / "person_detector",
            image,
        ),
        detector_case(
            "vehicle_detector",
            model_dir / "vehicle/mot_ppyoloe_s_36e_ppvehicle",
            args.onnx_dir / "vehicle_detector",
            image,
        ),
        attribute_case(
            "person_attribute",
            model_dir / "finetuned/person_attribute",
            "inference",
            args.onnx_dir / "person_attribute",
            (192, 256),
            image,
        ),
        attribute_case(
            "vehicle_attribute",
            model_dir / "vehicle/vehicle_attribute_model",
            "model",
            args.onnx_dir / "vehicle_attribute",
            (256, 192),
            image,
        ),
    ]
    print("=" * 50)
    if all(results):
        print("全部通过：ONNX 与 Paddle 输出一致，可以切换引擎。")
        return 0
    print("存在不一致：请勿切换，回退 Paddle GPU 方案。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
