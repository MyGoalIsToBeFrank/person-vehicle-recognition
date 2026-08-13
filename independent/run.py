#!/usr/bin/env python3
"""独立识别入口：读取图片，运行模型，原子写出 result.json。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
_DLL_HANDLES: list[Any] = []

import config as project_config  # noqa: E402


def enter_project_environment() -> None:
    """从系统 Python 启动时，切换到根目录已有的项目环境。"""
    python_name = "python.exe" if os.name == "nt" else "python"
    python_dir = "Scripts" if os.name == "nt" else "bin"
    expected_python = project_config.INFERENCE_VENV_DIR / python_dir / python_name
    if expected_python.is_file():
        try:
            if Path(sys.executable).resolve() == expected_python.resolve():
                return
        except OSError:
            pass
        completed = subprocess.run(
            [str(expected_python), str(__file__), *sys.argv[1:]],
            cwd=PROJECT_ROOT,
        )
        raise SystemExit(completed.returncode)

    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("没有找到根目录 .venv，也没有找到 uv。")
    environment = os.environ.copy()
    environment["UV_CACHE_DIR"] = str(PROJECT_ROOT / ".uv-cache")
    environment["UV_PYTHON_INSTALL_DIR"] = str(PROJECT_ROOT / ".uv-python")
    completed = subprocess.run(
        [
            uv,
            "run",
            "--project",
            str(PROJECT_ROOT),
            "python",
            str(__file__),
            *sys.argv[1:],
        ],
        cwd=PROJECT_ROOT,
        env=environment,
    )
    raise SystemExit(completed.returncode)


def configure_runtime_dlls() -> None:
    """让 Windows 找到项目环境内的 CUDA 和 cuDNN 动态库。"""
    _DLL_HANDLES.extend(project_config.configure_runtime_dlls(project_config.INFERENCE_VENV_DIR))


enter_project_environment()
configure_runtime_dlls()

from pipeline import RecognitionPipeline  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="独立的行人、车辆结构化识别")
    parser.add_argument("--data-dir", help="临时覆盖图片目录")
    parser.add_argument("--person-detector-dir", help="临时覆盖行人检测模型目录")
    parser.add_argument("--person-attribute-dir", help="临时覆盖行人属性模型目录")
    parser.add_argument("--vehicle-detector-dir", help="临时覆盖车辆检测模型目录")
    parser.add_argument("--vehicle-attribute-dir", help="临时覆盖车辆属性模型目录")
    parser.add_argument("--face-mask-dir", help="临时覆盖口罩模型目录")
    parser.add_argument("--plate-model-dir", help="临时覆盖车牌模型目录")
    parser.add_argument("--result-json", help="临时覆盖 JSON 输出路径")
    parser.add_argument("--device", choices=("CPU", "GPU"), help="临时覆盖推理设备")
    parser.add_argument("--limit", type=int, help="只处理排序后的前 N 张图片")
    return parser.parse_args()


def resolve_override(value: str | None, default: Path) -> Path:
    if value is None:
        return default.resolve()
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def find_images(data_dir: Path, limit: int | None) -> list[Path]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    images = sorted(
        (
            path.resolve()
            for path in data_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.relative_to(data_dir).as_posix().casefold(),
    )
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit 必须大于零")
        images = images[:limit]
    if not images:
        raise ValueError(f"数据目录中没有支持的图片: {data_dir}")
    return images


def write_json_atomically(rows: list[dict[str, Any]], result_path: Path) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix="result.",
            suffix=".tmp",
            dir=result_path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(rows, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, result_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main() -> int:
    args = parse_arguments()
    data_dir = resolve_override(args.data_dir, project_config.DATA_DIR)
    result_path = resolve_override(args.result_json, project_config.RESULT_JSON)
    device = (args.device or project_config.DEVICE).upper()
    if device not in {"CPU", "GPU"}:
        raise ValueError("device 只能是 CPU 或 GPU")

    images = find_images(data_dir, args.limit)
    model_paths = {
        "person_detector_dir": resolve_override(
            args.person_detector_dir, project_config.PERSON_DETECTOR_DIR
        ),
        "person_attribute_dir": resolve_override(
            args.person_attribute_dir, project_config.PERSON_ATTRIBUTE_DIR
        ),
        "vehicle_detector_dir": resolve_override(
            args.vehicle_detector_dir, project_config.VEHICLE_DETECTOR_DIR
        ),
        "vehicle_attribute_dir": resolve_override(
            args.vehicle_attribute_dir, project_config.VEHICLE_ATTRIBUTE_DIR
        ),
        "face_mask_dir": resolve_override(args.face_mask_dir, project_config.FACE_MASK_DIR),
        "plate_model_dir": resolve_override(
            args.plate_model_dir, project_config.PLATE_MODEL_DIR
        ),
    }
    print(f"加载模型: device={device}", flush=True)
    pipeline = RecognitionPipeline(
        device=device,
        face_mask_sha256=project_config.FACE_MASK_SHA256,
        person_attribute_crop_scale=project_config.PERSON_ATTRIBUTE_CROP_SCALE,
        **model_paths,
    )

    rows: list[dict[str, Any]] = []
    for index, image_path in enumerate(images, 1):
        started = time.perf_counter()
        content = pipeline.recognize(image_path)
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        rows.append(
            {
                "图片位置": str(image_path),
                "处理耗时（毫秒）": elapsed_ms,
                "识别内容": content,
            }
        )
        print(
            f"[{index}/{len(images)}] {image_path.name}: "
            f"行人={len(content['行人'])}, 车辆={len(content['车辆'])}, "
            f"{elapsed_ms:.3f} ms",
            flush=True,
        )

    write_json_atomically(rows, result_path)
    print(f"完成: {result_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
