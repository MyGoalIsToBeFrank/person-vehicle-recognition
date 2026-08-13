"""项目的唯一日常配置入口。

所有相对目录都在这里先解析成绝对路径。切换微调前后的模型时，只需要修改
``PERSON_DETECTOR_DIR``、``PERSON_ATTRIBUTE_DIR`` 和 ``FACE_MASK_DIR``。
"""

from __future__ import annotations

import json
import os
from pathlib import Path


# 现有官方 Paddle `.pdmodel` 是旧静态图格式，必须在导入 Paddle 前指定。
os.environ.setdefault("FLAGS_enable_pir_api", "0")


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

# 推理输入、设备和输出。
DATA_DIR = PROJECT_ROOT / "easy_test"
DEVICE = "GPU"
RESULT_JSON = HERE / "result.json"
RESULT_XLSX = HERE / "result.xlsx"
INFERENCE_VENV_DIR = PROJECT_ROOT / ".venv"

# 每个模型单独配置，避免为了替换一个微调模型而复制整套 models 目录。
MODEL_DIR = PROJECT_ROOT / "models"
PERSON_DETECTOR_DIR = MODEL_DIR / "human/mot_ppyoloe_s_36e_pipeline"
PERSON_ATTRIBUTE_DIR = MODEL_DIR / "human/PPHGNet_small_person_attribute_954_infer"
VEHICLE_DETECTOR_DIR = MODEL_DIR / "vehicle/mot_ppyoloe_s_36e_ppvehicle"
VEHICLE_ATTRIBUTE_DIR = MODEL_DIR / "vehicle/vehicle_attribute_model"
FACE_MASK_DIR = MODEL_DIR / "face_mask_yolov5"
PLATE_MODEL_DIR = MODEL_DIR / "vehicle"

# 官网属性模型沿用旧工作流的 1.3 倍人物裁剪；微调模型按 WebUI 最终红框训练，切换时设 1.0。
PERSON_ATTRIBUTE_CROP_SCALE = 1.3

# 原口罩权重必须校验；切换到微调模型时填写新 ONNX 的 SHA-256。
FACE_MASK_SHA256: str | None = (
    "ebe6674b727ba090fd94f394b81ca487d61f0f54fe1f48f2fa2443d9bd5fc280"
)

# 数据准备、人工复核和训练共用这些路径。
DATASET_ROOT = PROJECT_ROOT / "dataset"
RAW_DATA_DIR = DATASET_ROOT / "raw"
PROCESSED_DATA_DIR = DATASET_ROOT / "processed"
LEGACY_LABELS_PATH = PROCESSED_DATA_DIR / "labels.json"
CANDIDATES_PATH = PROCESSED_DATA_DIR / "candidates.json"
GOLD_LABELS_PATH = PROCESSED_DATA_DIR / "gold_labels.json"
GOLD_IMAGES_DIR = PROCESSED_DATA_DIR / "gold_images"
CANDIDATE_ARCHIVE_DIR = DATASET_ROOT / "candidate_archives"
PERSON_DETECTION_COCO_DIR = PROCESSED_DATA_DIR / "person_detection_coco"
TRAINING_OUTPUT_DIR = PROJECT_ROOT / "models/finetuned"
TRAINING_VENV_DIR = PROJECT_ROOT / ".venv-train"
TORCH_PACKAGE_DIR = PROJECT_ROOT / ".torch-cu130"
PADDLECLAS_DIR = PROJECT_ROOT / "vendor/PaddleClas"
YOLOV5_DIR = PROJECT_ROOT / "vendor/yolov5"
PADDLEDETECTION_DIR = PROJECT_ROOT / "vendor/PaddleDetection"
PERSON_DETECTOR_TRAINING_WEIGHTS = (
    MODEL_DIR / "original/person_detector/mot_ppyoloe_s_36e_pipeline.pdparams"
)


def configure_runtime_dlls(environment_dir: Path) -> list[object]:
    """在 Windows 上注册指定 Python 环境里的 CUDA/cuDNN DLL。"""
    if os.name != "nt":
        return []
    site_packages = environment_dir / "Lib/site-packages"
    candidates = (
        site_packages / "nvidia/cu13/bin/x86_64",
        site_packages / "nvidia/cudnn/bin",
        site_packages / "nvidia/cublas/bin",
    )
    dll_dirs = [path for path in candidates if path.is_dir()]
    if not dll_dirs:
        return []
    os.environ["PATH"] = os.pathsep.join(map(str, dll_dirs)) + os.pathsep + os.environ["PATH"]
    if not hasattr(os, "add_dll_directory"):
        return []
    return [os.add_dll_directory(str(path)) for path in dll_dirs]


def exported_paths() -> dict[str, str]:
    """返回供 Python 入口和 Excel 导出共同读取的扁平配置。"""
    names = (
        "PROJECT_ROOT",
        "DATA_DIR",
        "DEVICE",
        "RESULT_JSON",
        "RESULT_XLSX",
        "INFERENCE_VENV_DIR",
        "MODEL_DIR",
        "PERSON_DETECTOR_DIR",
        "PERSON_ATTRIBUTE_DIR",
        "VEHICLE_DETECTOR_DIR",
        "VEHICLE_ATTRIBUTE_DIR",
        "FACE_MASK_DIR",
        "PLATE_MODEL_DIR",
        "PERSON_ATTRIBUTE_CROP_SCALE",
        "FACE_MASK_SHA256",
        "DATASET_ROOT",
        "RAW_DATA_DIR",
        "PROCESSED_DATA_DIR",
        "LEGACY_LABELS_PATH",
        "CANDIDATES_PATH",
        "GOLD_LABELS_PATH",
        "GOLD_IMAGES_DIR",
        "CANDIDATE_ARCHIVE_DIR",
        "PERSON_DETECTION_COCO_DIR",
        "TRAINING_OUTPUT_DIR",
        "TRAINING_VENV_DIR",
        "TORCH_PACKAGE_DIR",
        "PADDLECLAS_DIR",
        "YOLOV5_DIR",
        "PADDLEDETECTION_DIR",
        "PERSON_DETECTOR_TRAINING_WEIGHTS",
    )
    values = globals()
    return {
        name: str(values[name]) if isinstance(values[name], Path) else values[name]
        for name in names
    }


if __name__ == "__main__":
    print(json.dumps(exported_paths(), ensure_ascii=False))
