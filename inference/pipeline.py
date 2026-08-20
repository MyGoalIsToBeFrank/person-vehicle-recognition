"""业务管线：裁切检测对象，并把模型数值转换为精简中文语义。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from model_adapters import (
    Detection,
    FaceMaskModel,
    PaddleAttributeModel,
    PaddleDetector,
    PlateRecognizer,
    paddle_model_files,
)


PERSON_VALUE = {
    "Female": "女",
    "Male": "男",
    "AgeLess18": "未满18岁",
    "Age18-60": "18至60岁",
    "AgeOver60": "60岁以上",
    "Front": "正面",
    "Side": "侧面",
    "Back": "背面",
    "HandBag": "手提包",
    "ShoulderBag": "单肩包",
    "Backpack": "双肩包",
    "No bag": "无",
    "LongSleeve": "长袖",
    "ShortSleeve": "短袖",
    "UpperStride": "条纹",
    "UpperLogo": "标志",
    "UpperPlaid": "格纹",
    "UpperSplice": "拼接",
    "LowerStripe": "条纹",
    "LowerPattern": "图案",
    "LongCoat": "长外套",
    "Trousers": "长裤",
    "Shorts": "短裤",
    "Skirt&Dress": "裙装",
    "Boots": "靴子",
    "No boots": "非靴子",
}

VEHICLE_COLORS = [
    "黄色",
    "橙色",
    "绿色",
    "灰色",
    "红色",
    "蓝色",
    "白色",
    "金色",
    "棕色",
    "黑色",
]

VEHICLE_TYPES = [
    "轿车",
    "SUV",
    "厢式货车",
    "掀背车",
    "MPV",
    "皮卡",
    "公交车",
    "卡车",
    "旅行车",
]


@dataclass(frozen=True)
class VehicleResult:
    detection: Detection
    content: dict[str, str]


class RecognitionPipeline:
    def __init__(
        self,
        *,
        device: str,
        person_detector_dir: Path,
        person_attribute_dir: Path,
        vehicle_detector_dir: Path,
        vehicle_attribute_dir: Path,
        face_mask_dir: Path,
        plate_model_dir: Path,
        person_attribute_crop_scale: float,
    ):
        """从唯一配置入口传入全部模型路径，不推断隐含目录。"""
        if person_attribute_crop_scale < 1.0:
            raise ValueError("person_attribute_crop_scale 不能小于 1.0")

        person_detector_dir = Path(person_detector_dir)
        person_attribute_dir = Path(person_attribute_dir)
        vehicle_detector_dir = Path(vehicle_detector_dir)
        vehicle_attribute_dir = Path(vehicle_attribute_dir)
        face_mask_dir = Path(face_mask_dir)
        plate_model_dir = Path(plate_model_dir)
        self.person_detector = PaddleDetector(
            person_detector_dir,
            device,
        )
        person_model, person_params = paddle_model_files(person_attribute_dir, "inference")
        self.person_attributes = PaddleAttributeModel(person_model, person_params, (192, 256), device)
        self.vehicle_detector = PaddleDetector(
            vehicle_detector_dir,
            device,
        )
        vehicle_model, vehicle_params = paddle_model_files(vehicle_attribute_dir, "model")
        self.vehicle_attributes = PaddleAttributeModel(
            vehicle_model, vehicle_params, (256, 192), device
        )
        self.face_mask = FaceMaskModel(face_mask_dir)
        self.plate = PlateRecognizer(plate_model_dir)
        self.person_attribute_crop_scale = person_attribute_crop_scale

    def recognize(self, image_path: Path) -> dict[str, list[dict[str, Any]]]:
        return self.recognize_array(self._decode_image(image_path))

    def recognize_array(self, image: np.ndarray) -> dict[str, list[dict[str, Any]]]:
        """对已解码的 BGR 图像直接识别（FastAPI 服务走这个入口，不落盘）。"""
        return self.recognize_batch([image])[0]

    def recognize_batch(
        self, images: list[np.ndarray]
    ) -> list[dict[str, list[dict[str, Any]]]]:
        """批量识别：整图预处理一次共享给行人/车辆两个检测器，两个检测各自
        一次批量前向；随后跨图收集全部人物/车辆裁片，属性模型各跑一次大批。
        吞吐场景（FastAPI worker）必须走这里而不是逐张 recognize_array。"""

        tensors, scales = zip(
            *(self.person_detector.preprocess(image) for image in images)
        )
        person_dets = self.person_detector.predict_prepped(list(tensors), list(scales))
        vehicle_dets = self.vehicle_detector.predict_prepped(list(tensors), list(scales))

        # 跨图收集人物裁片：(图序号, 头部裁片, 身体裁片)
        person_jobs: list[tuple[int, np.ndarray | None, np.ndarray]] = []
        for index, (image, detections) in enumerate(zip(images, person_dets)):
            for detection in detections:
                body = self._expanded_crop(
                    image, detection.box, self.person_attribute_crop_scale
                )
                if body is None:
                    continue
                head = self._upper_crop(image, detection.box, 0.40)
                person_jobs.append((index, head, body))

        person_results: dict[int, list[dict[str, Any]]] = {
            i: [] for i in range(len(images))
        }
        if person_jobs:
            scores_batch = self.person_attributes.predict_batch(
                [body for _, _, body in person_jobs]
            )
            for (index, head, _), scores in zip(person_jobs, scores_batch):
                attributes = decode_person_attributes(scores)
                attributes["口罩"] = (
                    self.face_mask.predict(head) if head is not None else "未佩戴口罩"
                )
                person_results[index].append(attributes)

        # 跨图收集车辆裁片
        vehicle_jobs: list[tuple[int, Detection, np.ndarray]] = []
        for index, (image, detections) in enumerate(zip(images, vehicle_dets)):
            for detection in detections:
                vehicle = self._expanded_crop(image, detection.box)
                if vehicle is None:
                    continue
                vehicle_jobs.append((index, detection, vehicle))

        vehicle_results: dict[int, list[VehicleResult]] = {
            i: [] for i in range(len(images))
        }
        if vehicle_jobs:
            scores_batch = self.vehicle_attributes.predict_batch(
                [crop for _, _, crop in vehicle_jobs]
            )
            for (index, detection, vehicle), scores in zip(vehicle_jobs, scores_batch):
                content = decode_vehicle_attributes(scores)
                content["车牌"] = self.plate.predict(vehicle)
                vehicle_results[index].append(
                    VehicleResult(detection=detection, content=content)
                )

        return [
            {
                "行人": person_results[index],
                "车辆": [
                    item.content for item in suppress_same_plate(vehicle_results[index])
                ],
            }
            for index in range(len(images))
        ]

    @staticmethod
    def _decode_image(path: Path) -> np.ndarray:
        try:
            encoded = np.fromfile(path, dtype=np.uint8)
        except OSError as exc:
            raise ValueError(f"无法读取图片: {path}") from exc
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"图片损坏或格式不受支持: {path}")
        return image

    @staticmethod
    def _expanded_crop(
        image: np.ndarray,
        box: tuple[float, float, float, float],
        scale: float = 1.3,
    ) -> np.ndarray | None:
        height, width = image.shape[:2]
        left, top, right, bottom = map(int, box)
        half_height = (bottom - top) * scale / 2.0
        half_width = (right - left) * scale / 2.0
        if half_height > half_width * 4.0 / 3.0:
            half_width = half_height * 0.75
        center_y = (top + bottom) / 2.0
        center_x = (left + right) / 2.0
        top = max(0, int(center_y - half_height))
        bottom = min(height - 1, int(center_y + half_height))
        left = max(0, int(center_x - half_width))
        right = min(width - 1, int(center_x + half_width))
        if right <= left or bottom <= top:
            return None
        return image[top:bottom, left:right]

    @staticmethod
    def _upper_crop(
        image: np.ndarray,
        box: tuple[float, float, float, float],
        ratio: float,
    ) -> np.ndarray | None:
        height, width = image.shape[:2]
        left = max(0, min(width, int(box[0])))
        top = max(0, min(height, int(box[1])))
        right = max(0, min(width, int(box[2])))
        original_bottom = max(0, min(height, int(box[3])))
        bottom = min(original_bottom, top + max(1, int((original_bottom - top) * ratio)))
        if right <= left or bottom <= top:
            return None
        return image[top:bottom, left:right]


def decode_person_attributes(scores: np.ndarray) -> dict[str, Any]:
    if scores.size < 26:
        raise RuntimeError(f"人体属性输出不足 26 项: {scores.size}")
    age_names = ["AgeLess18", "Age18-60", "AgeOver60"]
    direction_names = ["Front", "Side", "Back"]
    bag_names = ["HandBag", "ShoulderBag", "Backpack"]
    upper_names = ["UpperStride", "UpperLogo", "UpperPlaid", "UpperSplice"]
    lower_names = [
        "LowerStripe",
        "LowerPattern",
        "LongCoat",
        "Trousers",
        "Shorts",
        "Skirt&Dress",
    ]

    bag_index = int(np.argmax(scores[15:18]))
    bag = bag_names[bag_index] if scores[15 + bag_index] > 0.5 else "No bag"
    upper_scores = scores[4:8]
    upper_styles = []
    if float(np.max(upper_scores)) > 0.5:
        upper_styles.append(PERSON_VALUE[upper_names[int(np.argmax(upper_scores))]])
    lower_scores = scores[8:14]
    lower = [
        PERSON_VALUE[name]
        for name, score in zip(lower_names, lower_scores)
        if float(score) > 0.5
    ]
    if not lower:
        lower.append(PERSON_VALUE[lower_names[int(np.argmax(lower_scores))]])

    return {
        "性别": "女" if scores[22] > 0.5 else "男",
        "年龄": PERSON_VALUE[age_names[int(np.argmax(scores[19:22]))]],
        "朝向": PERSON_VALUE[direction_names[int(np.argmax(scores[23:26]))]],
        "佩戴眼镜": "是" if scores[1] > 0.3 else "否",
        "佩戴帽子": "是" if scores[0] > 0.5 else "否",
        "手持物品": "是" if scores[18] > 0.6 else "否",
        "包": PERSON_VALUE[bag],
        "上装": {
            "袖长": "长袖" if scores[3] > scores[2] else "短袖",
            "款式": upper_styles,
        },
        "下装": lower,
        "鞋靴": "靴子" if scores[14] > 0.5 else "非靴子",
    }


def decode_vehicle_attributes(scores: np.ndarray) -> dict[str, str]:
    if scores.size < 19:
        raise RuntimeError(f"车辆属性输出不足 19 项: {scores.size}")
    color_index = int(np.argmax(scores[:10]))
    type_index = int(np.argmax(scores[10:19]))
    return {
        "颜色": VEHICLE_COLORS[color_index] if scores[color_index] >= 0.5 else "未知",
        "车型": VEHICLE_TYPES[type_index] if scores[type_index + 10] >= 0.5 else "未知",
    }


def suppress_same_plate(items: list[VehicleResult]) -> list[VehicleResult]:
    kept: list[VehicleResult] = []
    for item in sorted(items, key=lambda value: value.detection.score, reverse=True):
        duplicate = any(
            item.content["车牌"] != "未识别"
            and item.content["车牌"] == existing.content["车牌"]
            and contained_overlap(item.detection.box, existing.detection.box) >= 0.50
            for existing in kept
        )
        if not duplicate:
            kept.append(item)
    return kept


def contained_overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    smaller = min(first_area, second_area)
    return intersection / smaller if smaller > 0 else 0.0
