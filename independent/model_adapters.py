"""五类模型的薄适配层；这里只处理模型文件、张量和运行时。"""

from __future__ import annotations

import hashlib
import importlib
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from paddle.inference import Config, create_predictor


MASK_SHA256 = "ebe6674b727ba090fd94f394b81ca487d61f0f54fe1f48f2fa2443d9bd5fc280"


@dataclass(frozen=True)
class Detection:
    score: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class MaskDetection:
    score: float
    class_id: int
    box: tuple[float, float, float, float]


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        formatted = "\n  ".join(missing)
        raise FileNotFoundError(f"缺少模型文件，程序不会联网补齐:\n  {formatted}")


def paddle_model_files(model_dir: Path, stem: str) -> tuple[Path, Path]:
    """同时接受官网旧静态图 `.pdmodel` 和微调后 Paddle 3 的 `.json`。"""
    pdmodel = model_dir / f"{stem}.pdmodel"
    model = pdmodel if pdmodel.is_file() else model_dir / f"{stem}.json"
    return model, model_dir / f"{stem}.pdiparams"


def _paddle_predictor(model: Path, params: Path, device: str):
    _require_files([model, params])
    config = Config(str(model), str(params))
    if device == "GPU":
        config.enable_use_gpu(500, 0)
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(1)
    config.switch_ir_optim(True)
    config.enable_memory_optim()
    config.disable_glog_info()
    return create_predictor(config)


def _run_predictor(predictor, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    input_names = predictor.get_input_names()
    unexpected = set(input_names) - set(inputs)
    if unexpected:
        raise RuntimeError(f"模型出现未知输入: {sorted(unexpected)}")
    for name in input_names:
        predictor.get_input_handle(name).copy_from_cpu(np.ascontiguousarray(inputs[name]))
    predictor.run()
    return {
        name: predictor.get_output_handle(name).copy_to_cpu()
        for name in predictor.get_output_names()
    }


class PaddleDetector:
    def __init__(self, model_dir: Path, device: str, threshold: float = 0.5):
        model_file, params_file = paddle_model_files(model_dir, "model")
        self.predictor = _paddle_predictor(
            model_file,
            params_file,
            device,
        )
        self.threshold = threshold

    def predict(self, image: np.ndarray) -> list[Detection]:
        height, width = image.shape[:2]
        resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1)[None].astype(np.float32)
        scale = np.asarray([[640.0 / height, 640.0 / width]], dtype=np.float32)
        outputs = _run_predictor(
            self.predictor,
            {"image": tensor, "scale_factor": scale},
        )
        candidates = [
            value
            for value in outputs.values()
            if value.ndim == 2 and value.shape[-1] == 6
        ]
        if len(candidates) != 1:
            shapes = {name: value.shape for name, value in outputs.items()}
            raise RuntimeError(f"检测模型输出结构异常: {shapes}")

        detections: list[Detection] = []
        for class_id, score, left, top, right, bottom in candidates[0]:
            if int(class_id) != 0 or float(score) < self.threshold:
                continue
            if right <= left or bottom <= top:
                continue
            detections.append(
                Detection(
                    score=float(score),
                    box=(float(left), float(top), float(right), float(bottom)),
                )
            )
        return detections


class PaddleAttributeModel:
    def __init__(
        self,
        model: Path,
        params: Path,
        size: tuple[int, int],
        device: str,
    ):
        self.predictor = _paddle_predictor(model, params, device)
        self.width, self.height = size
        self.mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        self.std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

    def predict(self, crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            raise ValueError("属性模型收到空裁片")
        resized = cv2.resize(crop, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        tensor = ((tensor - self.mean) / self.std)[None]
        input_names = self.predictor.get_input_names()
        if len(input_names) != 1:
            raise RuntimeError(f"属性模型输入结构异常: {input_names}")
        outputs = _run_predictor(self.predictor, {input_names[0]: tensor})
        if len(outputs) != 1:
            shapes = {name: value.shape for name, value in outputs.items()}
            raise RuntimeError(f"属性模型输出结构异常: {shapes}")
        return next(iter(outputs.values()))[0]


class FaceMaskModel:
    def __init__(
        self,
        model_dir: Path,
        confidence: float = 0.5,
        iou: float = 0.45,
        expected_sha256: str | None = MASK_SHA256,
    ):
        model_path = model_dir / "face_mask_detection.onnx"
        classes_path = model_dir / "synset.txt"
        _require_files([model_path, classes_path])
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(f"口罩模型 SHA-256 不匹配: {digest}")
        classes = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines()]
        if classes != ["w/o mask", "w/ mask"]:
            raise ValueError(f"口罩类别文件内容异常: {classes}")
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.confidence = confidence
        self.iou = iou

    def predict(self, crop: np.ndarray) -> str:
        detections = self.detect(crop)
        if not detections:
            return "未识别"
        best = max(detections, key=lambda detection: detection.score)
        return "佩戴口罩" if best.class_id == 1 else "未佩戴口罩"

    def detect(self, crop: np.ndarray) -> list[MaskDetection]:
        """返回人脸框和类别；现有业务 ``predict`` 仍只取最高分标签。"""
        if crop.size == 0:
            return []
        tensor, scale, pad_left, pad_top = self._letterbox(crop)
        prediction = self.session.run(None, {self.input_name: tensor})[0][0]
        class_ids = np.argmax(prediction[:, 5:], axis=1)
        scores = prediction[:, 4] * prediction[
            np.arange(prediction.shape[0]), class_ids + 5
        ]
        selected = scores >= self.confidence
        if not np.any(selected):
            return []

        boxes = prediction[selected, :4].astype(np.float32)
        selected_scores = scores[selected].astype(np.float32)
        selected_classes = class_ids[selected].astype(np.int64)
        boxes[:, 0] -= boxes[:, 2] / 2.0
        boxes[:, 1] -= boxes[:, 3] / 2.0
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]

        kept: list[int] = []
        for class_id in np.unique(selected_classes):
            indices = np.flatnonzero(selected_classes == class_id)
            kept.extend(self._nms(boxes, selected_scores, indices))
        height, width = crop.shape[:2]
        detections: list[MaskDetection] = []
        for index in kept:
            left, top, right, bottom = boxes[index]
            mapped = (
                max(0.0, min(float(width), (float(left) - pad_left) / scale)),
                max(0.0, min(float(height), (float(top) - pad_top) / scale)),
                max(0.0, min(float(width), (float(right) - pad_left) / scale)),
                max(0.0, min(float(height), (float(bottom) - pad_top) / scale)),
            )
            if mapped[2] <= mapped[0] or mapped[3] <= mapped[1]:
                continue
            detections.append(
                MaskDetection(
                    score=float(selected_scores[index]),
                    class_id=int(selected_classes[index]),
                    box=mapped,
                )
            )
        return detections

    @staticmethod
    def _letterbox(image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        height, width = image.shape[:2]
        scale = min(640.0 / width, 640.0 / height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        left = (640 - new_width) // 2
        top = (640 - new_height) // 2
        canvas[top : top + new_height, left : left + new_width] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return tensor, scale, left, top

    def _nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        indices: np.ndarray,
    ) -> list[int]:
        order = indices[np.argsort(scores[indices])[::-1]]
        kept: list[int] = []
        while order.size:
            current = int(order[0])
            kept.append(current)
            if order.size == 1:
                break
            rest = order[1:]
            left = np.maximum(boxes[current, 0], boxes[rest, 0])
            top = np.maximum(boxes[current, 1], boxes[rest, 1])
            right = np.minimum(boxes[current, 2], boxes[rest, 2])
            bottom = np.minimum(boxes[current, 3], boxes[rest, 3])
            intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
            current_area = max(0.0, boxes[current, 2] - boxes[current, 0]) * max(
                0.0, boxes[current, 3] - boxes[current, 1]
            )
            rest_area = np.maximum(0.0, boxes[rest, 2] - boxes[rest, 0]) * np.maximum(
                0.0, boxes[rest, 3] - boxes[rest, 1]
            )
            union = current_area + rest_area - intersection
            overlap = np.divide(
                intersection,
                union,
                out=np.zeros_like(intersection),
                where=union > 0,
            )
            order = rest[overlap <= self.iou]
        return kept


class PlateRecognizer:
    REQUIRED_NAMES = (
        "y5fu_320x_sim.onnx",
        "y5fu_640x_sim.onnx",
        "rpv3_mdict_160_r3.onnx",
        "litemodel_cls_96x_r1.onnx",
    )
    PROVINCES = set("浙粤京津冀晋蒙辽黑沪吉苏皖赣鲁豫鄂湘桂琼渝川贵云藏陕甘青宁闽")

    def __init__(self, vehicle_model_dir: Path):
        cache_root = vehicle_model_dir / ".hyperlpr3"
        onnx_dir = cache_root / "20230229" / "onnx"
        _require_files([onnx_dir / name for name in self.REQUIRED_NAMES])

        # 0.1.3 在导入时检查 HOMEPATH。先指向已验证的外部模型目录，
        # 可阻止包在用户目录中隐式下载；导入后立即恢复进程环境。
        previous_homepath = os.environ.get("HOMEPATH")
        os.environ["HOMEPATH"] = str(vehicle_model_dir)
        try:
            hyperlpr3 = importlib.import_module("hyperlpr3")
        finally:
            if previous_homepath is None:
                os.environ.pop("HOMEPATH", None)
            else:
                os.environ["HOMEPATH"] = previous_homepath
        self.engine = hyperlpr3.LicensePlateCatcher(folder=str(cache_root))

    def predict(self, crop: np.ndarray) -> str:
        if crop.size == 0:
            return "未识别"
        # 现有识别模型的部署链路使用 RGB 裁片；颜色空间在适配层显式转换。
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = self.engine(rgb)
        if not results:
            return "未识别"
        best = max(results, key=lambda item: float(item[1]))
        text = "".join(
            char.upper()
            for char in str(best[0]).strip()
            if char in self.PROVINCES or (char.isascii() and char.isalnum())
        )
        return text or "未识别"
