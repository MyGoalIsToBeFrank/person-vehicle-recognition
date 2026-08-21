"""Paths shared only by the offline relabeling and fine-tuning tools.

The production service deliberately does not import this module. Keeping the
offline configuration here prevents the removed v1 Python inference pipeline
from becoming an accidental runtime dependency again.
"""

from __future__ import annotations

import os
from pathlib import Path


os.environ.setdefault("FLAGS_enable_pir_api", "0")

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

DEVICE = "GPU"
INFERENCE_VENV_DIR = PROJECT_ROOT / ".venv"
TRAINING_VENV_DIR = PROJECT_ROOT / ".venv-train"
TORCH_PACKAGE_DIR = PROJECT_ROOT / ".torch-cu130"

MODEL_DIR = PROJECT_ROOT / "models"
PERSON_DETECTOR_DIR = MODEL_DIR / "finetuned/person_detector"
PERSON_ATTRIBUTE_DIR = MODEL_DIR / "finetuned/person_attribute"

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
TRAINING_OUTPUT_DIR = MODEL_DIR / "finetuned"

PADDLECLAS_DIR = PROJECT_ROOT / "vendor/PaddleClas"
YOLOV5_DIR = PROJECT_ROOT / "vendor/yolov5"
PADDLEDETECTION_DIR = PROJECT_ROOT / "vendor/PaddleDetection"
PERSON_DETECTOR_TRAINING_WEIGHTS = (
    MODEL_DIR / "original/person_detector/mot_ppyoloe_s_36e_pipeline.pdparams"
)


def configure_runtime_dlls(environment_dir: Path) -> list[object]:
    """Register CUDA/cuDNN DLL directories for the selected Windows venv."""
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
