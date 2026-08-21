"""Small Paddle adapters used only to create human-review candidates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from paddle.inference import Config, create_predictor


@dataclass(frozen=True)
class Detection:
    score: float
    box: tuple[float, float, float, float]


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少离线模型文件，程序不会联网补齐:\n  " + "\n  ".join(missing))


def paddle_model_files(model_dir: Path, stem: str) -> tuple[Path, Path]:
    pdmodel = model_dir / f"{stem}.pdmodel"
    model = pdmodel if pdmodel.is_file() else model_dir / f"{stem}.json"
    return model, model_dir / f"{stem}.pdiparams"


def _predictor(model: Path, params: Path, device: str):
    _require_files([model, params])
    settings = Config(str(model), str(params))
    if device == "GPU":
        settings.enable_use_gpu(500, 0)
    else:
        settings.disable_gpu()
        settings.set_cpu_math_library_num_threads(1)
    settings.switch_ir_optim(True)
    if model.suffix == ".pdmodel":
        settings.enable_memory_optim()
    settings.disable_glog_info()
    return create_predictor(settings)


def _run(predictor, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    names = predictor.get_input_names()
    unexpected = set(names) - set(inputs)
    if unexpected:
        raise RuntimeError(f"模型出现未知输入: {sorted(unexpected)}")
    for name in names:
        predictor.get_input_handle(name).copy_from_cpu(np.ascontiguousarray(inputs[name]))
    predictor.run()
    return {
        name: predictor.get_output_handle(name).copy_to_cpu()
        for name in predictor.get_output_names()
    }


class PaddleDetector:
    def __init__(self, model_dir: Path, device: str, threshold: float = 0.5):
        model, params = paddle_model_files(model_dir, "model")
        self.predictor = _predictor(model, params, device)
        self.threshold = threshold

    def predict(self, image: np.ndarray) -> list[Detection]:
        height, width = image.shape[:2]
        resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_CUBIC)
        tensor = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32)
        scale = np.asarray([640.0 / height, 640.0 / width], dtype=np.float32)
        outputs = _run(
            self.predictor,
            {"image": tensor[None, ...], "scale_factor": scale[None, ...]},
        )
        rows = [value for value in outputs.values() if value.ndim == 2 and value.shape[-1] == 6]
        if len(rows) != 1:
            shapes = {name: value.shape for name, value in outputs.items()}
            raise RuntimeError(f"人物检测候选输出结构异常: {shapes}")
        detections = []
        for class_id, score, left, top, right, bottom in rows[0]:
            if int(class_id) != 0 or float(score) < self.threshold or right <= left or bottom <= top:
                continue
            detections.append(
                Detection(float(score), (float(left), float(top), float(right), float(bottom)))
            )
        return detections


class PaddleAttributeModel:
    def __init__(self, model: Path, params: Path, size: tuple[int, int], device: str):
        self.predictor = _predictor(model, params, device)
        self.width, self.height = size
        self.mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        self.std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

    def predict(self, crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            raise ValueError("属性预标注收到空裁片")
        resized = cv2.resize(crop, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        tensor = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32)
        tensor = (tensor / 255.0 - self.mean) / self.std
        names = self.predictor.get_input_names()
        if len(names) != 1:
            raise RuntimeError(f"属性模型输入结构异常: {names}")
        outputs = _run(self.predictor, {names[0]: tensor[None, ...]})
        if len(outputs) != 1:
            shapes = {name: value.shape for name, value in outputs.items()}
            raise RuntimeError(f"属性模型输出结构异常: {shapes}")
        return next(iter(outputs.values()))[0]
