"""inference 推理应用端的唯一日常配置入口，训练侧脚本也共用其中的路径。

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
PERSON_DETECTOR_DIR = MODEL_DIR / "finetuned/person_detector"
PERSON_ATTRIBUTE_DIR = MODEL_DIR / "finetuned/person_attribute"
VEHICLE_DETECTOR_DIR = MODEL_DIR / "vehicle/mot_ppyoloe_s_36e_ppvehicle"
VEHICLE_ATTRIBUTE_DIR = MODEL_DIR / "vehicle/vehicle_attribute_model"
FACE_MASK_DIR = MODEL_DIR / "finetuned/face_mask"
PLATE_MODEL_DIR = MODEL_DIR / "vehicle"

# 官网属性模型沿用旧工作流的 1.3 倍人物裁剪；微调模型按 WebUI 最终红框训练，切换时设 1.0。
PERSON_ATTRIBUTE_CROP_SCALE = 1.0

# 数据准备、人工复核和训练共用这些路径。processed 下按管线阶段编号：
# 1_detection 目标识别 → 2_attribute 属性标注 → 3_mask 口罩标注 → 4_augmented 尘土化
# → 5_export 训练导出。所有 JSON 均为 COCO 风格（images/annotations/categories）。
DATASET_ROOT = PROJECT_ROOT / "dataset"
RAW_DATA_DIR = DATASET_ROOT / "raw"
PROCESSED_DATA_DIR = DATASET_ROOT / "processed"

DETECTION_DATA_DIR = PROCESSED_DATA_DIR / "1_detection"
DETECTION_CANDIDATES_PATH = DETECTION_DATA_DIR / "candidates.json"
DETECTION_CONFIRMED_PATH = DETECTION_DATA_DIR / "confirmed.json"

ATTRIBUTE_DATA_DIR = PROCESSED_DATA_DIR / "2_attribute"
ATTRIBUTE_IMAGES_DIR = ATTRIBUTE_DATA_DIR / "images"
ATTRIBUTE_CANDIDATES_PATH = ATTRIBUTE_DATA_DIR / "candidates.json"
ATTRIBUTE_GOLD_PATH = ATTRIBUTE_DATA_DIR / "gold.json"

MASK_DATA_DIR = PROCESSED_DATA_DIR / "3_mask"
MASK_IMAGES_DIR = MASK_DATA_DIR / "images"
MASK_CANDIDATES_PATH = MASK_DATA_DIR / "candidates.json"
MASK_GOLD_PATH = MASK_DATA_DIR / "gold.json"

AUGMENTED_DATA_DIR = PROCESSED_DATA_DIR / "4_augmented"
PERSON_DETECTION_COCO_DIR = PROCESSED_DATA_DIR / "5_export/person_detection_coco"
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
    """导出 Node Excel 工具实际消费的路径（DATA_DIR 用于解析相对图片位置）。"""
    return {
        "DATA_DIR": str(DATA_DIR),
        "RESULT_JSON": str(RESULT_JSON),
        "RESULT_XLSX": str(RESULT_XLSX),
    }


if __name__ == "__main__":
    print(json.dumps(exported_paths(), ensure_ascii=False))
