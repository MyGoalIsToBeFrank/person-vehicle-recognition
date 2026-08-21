#!/usr/bin/env python3
"""四阶段人工核对 WebUI：目标识别 → 属性标注 → 口罩标注 → 尘土化。

- 目标识别：整图核对人物框（无标签编辑），确认后生成人物裁剪并用属性模型预标注。
- 属性标注：只看裁剪小图，检查并修改模型预标注的 26 项属性。
- 口罩标注：只看带上下文的人脸裁剪，检查并修改 AIZOO 类别。
- 尘土化：对任一阶段金标准做离线退化扩充，标签不变。

启动时会重入 ``.venv`` 推理环境，以便在确认检测框时同步完成属性预标注；
``--no-prelabel`` 可完全跳过模型加载。
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import mimetypes
import os
import random
import subprocess
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
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
    annotation_id,
    crop_image_id,
    empty_dataset,
    image_index,
    load_dataset,
    resolved_path,
    save_dataset,
    stored_path,
    validate_bbox,
    xywh_to_xyxy,
    xyxy_to_xywh,
)
from dust_augment import AugmentParams, apply_degradation, _rng_for, augment_stage  # noqa: E402


def initial_body_labels(scores: np.ndarray) -> dict[str, bool]:
    """把属性模型的 26 维分数转成初始布尔标签（与训练标签同序）。"""
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


class ReviewStore:
    def __init__(self, prelabel: bool, device: str, shuffle_seed: int | None = None):
        self.lock = threading.Lock()
        self.prelabel_enabled = prelabel
        self.device = device
        self.prelabel_warning: str | None = None
        self._attribute_model = None
        self._full_image_cache: tuple[str, np.ndarray] | None = None
        self.detection_candidates = load_dataset(
            config.DETECTION_CANDIDATES_PATH, "detection", "model", gold=False
        )
        self.detection_confirmed = load_dataset(
            config.DETECTION_CONFIRMED_PATH, "detection", "human", gold=True
        )
        self.attribute_candidates = load_dataset(
            config.ATTRIBUTE_CANDIDATES_PATH, "attribute", "model", gold=False
        )
        self.attribute_gold = load_dataset(
            config.ATTRIBUTE_GOLD_PATH, "attribute", "human", gold=True
        )
        self.mask_candidates = load_dataset(
            config.MASK_CANDIDATES_PATH, "mask", "aizoo", gold=False
        )
        self.mask_gold = load_dataset(config.MASK_GOLD_PATH, "mask", "human", gold=True)
        # 候选按随机顺序出图：相邻图片高度相关（同摄像头连拍），顺序核对会让
        # 早期样本集中在少数摄像头。顺序在启动时打乱，本次会话内保持稳定。
        rng = random.Random(shuffle_seed if shuffle_seed is not None else int(time.time()))
        self._orders = {
            "detection": [image["id"] for image in self.detection_candidates["images"]],
            "attribute": [image["id"] for image in self.attribute_candidates["images"]],
            "mask": [item["id"] for item in self.mask_candidates["annotations"]],
        }
        for keys in self._orders.values():
            rng.shuffle(keys)
        self.augment_status: dict = {"running": False, "stage": None, "done": 0, "total": 0,
                                     "error": None, "summary": None}
        self._augment_thread: threading.Thread | None = None

    # ---- 汇总 ----

    def _augmented_counts(self) -> dict:
        counts = {}
        for stage in ("detection", "attribute", "mask"):
            path = config.AUGMENTED_DATA_DIR / stage / "annotations.json"
            if path.is_file():
                data = load_dataset(path, stage, "augmented", gold=True)
                counts[stage] = {"图片": len(data["images"]), "标注": len(data["annotations"])}
        return counts

    def summary(self) -> dict:
        return {
            "detection": {
                "待核对": len(self._detection_pending()),
                "已确认": len(self.detection_confirmed["images"]),
                "确认框": len(self.detection_confirmed["annotations"]),
            },
            "attribute": {
                "待标注": len(self.attribute_candidates["annotations"]),
                "已保存": len(self.attribute_gold["annotations"]),
            },
            "mask": {
                "待核对": len(self.mask_candidates["annotations"]),
                "已保存": len(self.mask_gold["annotations"]),
            },
            "augmented": self._augmented_counts(),
            "预标注": "开启" if self.prelabel_enabled else "关闭",
            "预标注警告": self.prelabel_warning,
        }

    # ---- 阶段一：目标识别 ----

    def _detection_pending(self) -> list[dict]:
        images = image_index(self.detection_candidates)
        return [images[key] for key in self._orders["detection"] if key in images]

    def detection_record(self, index: int, scope: str = "pending") -> dict:
        if scope == "saved":
            rows = list(reversed(self.detection_confirmed["images"]))
            source = self.detection_confirmed
            empty = "还没有已确认的整图"
        else:
            rows = self._detection_pending()
            source = self.detection_candidates
            empty = "目标识别没有待核对原图"
        if not rows:
            raise IndexError(empty)
        index = max(0, min(index, len(rows) - 1))
        image = copy.deepcopy(rows[index])
        annotations = [
            copy.deepcopy(item)
            for item in source["annotations"]
            if item["image_id"] == image["id"]
        ]
        return {"index": index, "count": len(rows), "scope": scope,
                "image": image, "annotations": annotations}

    def detection_image_path(self, image_id_value: str) -> Path:
        images = image_index(self.detection_candidates) | image_index(self.detection_confirmed)
        if image_id_value not in images:
            raise ValueError("整图不存在")
        return resolved_path(images[image_id_value]["file_name"], config.PROJECT_ROOT)

    def detection_confirm(self, image_id_value: str, boxes: object) -> dict:
        if not isinstance(boxes, list):
            raise ValueError("annotations 必须是数组")
        with self.lock:
            candidates_index = image_index(self.detection_candidates)
            confirmed_index = image_index(self.detection_confirmed)
            if image_id_value in candidates_index:
                image_record = candidates_index[image_id_value]
                first_save = True
            elif image_id_value in confirmed_index:
                # 回改已确认整图：按新框更新确认集，并 reconcile 派生的属性裁剪。
                image_record = confirmed_index[image_id_value]
                first_save = False
            else:
                raise ValueError("该原图不在候选或已确认队列")
            full = self._decode(resolved_path(image_record["file_name"], config.PROJECT_ROOT))
            height, width = full.shape[:2]
            used: set[str] = set()
            normalized = []
            for number, item in enumerate(boxes):
                if not isinstance(item, dict) or not set(item) <= {"id", "bbox"}:
                    raise ValueError("每个框只能包含 id、bbox")
                validate_bbox(item["bbox"], str(item.get("id", number)), "bbox")
                x, y, w, h = (round(float(v)) for v in item["bbox"])
                if x < 0 or y < 0 or x + w > width or y + h > height:
                    raise ValueError(f"框超出原图: {[x, y, w, h]}, image={width}x{height}")
                if w < 4 or h < 4:
                    raise ValueError("框过小（宽高至少 4 像素）")
                identifier = str(item.get("id") or annotation_id(image_id_value, number))
                if identifier in used:
                    raise ValueError(f"框 id 重复: {identifier}")
                used.add(identifier)
                normalized.append((identifier, [x, y, w, h]))

            if first_save:
                self.detection_confirmed["images"].append(copy.deepcopy(image_record))
                self.detection_candidates["images"] = [
                    image for image in self.detection_candidates["images"]
                    if image["id"] != image_id_value
                ]
                self.detection_candidates["annotations"] = [
                    item for item in self.detection_candidates["annotations"]
                    if item["image_id"] != image_id_value
                ]
            else:
                self.detection_confirmed["annotations"] = [
                    item for item in self.detection_confirmed["annotations"]
                    if item["image_id"] != image_id_value
                ]
            for identifier, bbox in normalized:
                self.detection_confirmed["annotations"].append(
                    {"id": identifier, "image_id": image_id_value, "category_id": 1, "bbox": bbox}
                )
            save_dataset(
                config.DETECTION_CONFIRMED_PATH, self.detection_confirmed, "detection", "human",
                gold=True,
            )
            save_dataset(
                config.DETECTION_CANDIDATES_PATH, self.detection_candidates, "detection", "model",
                gold=False,
            )
            prelabeled = self._reconcile_attribute_crops(image_id_value, normalized, full)
            return {"saved": image_id_value, "updated": not first_save,
                    "预标注框": prelabeled, "summary": self.summary()}

    def detection_exclude(self, image_id_value: str) -> dict:
        with self.lock:
            before = len(self.detection_candidates["images"])
            self.detection_candidates["images"] = [
                image for image in self.detection_candidates["images"] if image["id"] != image_id_value
            ]
            if len(self.detection_candidates["images"]) == before:
                raise ValueError("候选整图不存在")
            self.detection_candidates["annotations"] = [
                item for item in self.detection_candidates["annotations"]
                if item["image_id"] != image_id_value
            ]
            save_dataset(
                config.DETECTION_CANDIDATES_PATH, self.detection_candidates, "detection", "model",
                gold=False,
            )
            return {"excluded": image_id_value, "summary": self.summary()}

    def _reconcile_attribute_crops(
        self, source_id: str, boxes: list[tuple[str, list[int]]], full: np.ndarray
    ) -> int:
        """把属性裁剪与整图最终框按框 id 对齐。

        新框生成裁剪并预标注；框位置未变的裁剪与标签原样保留；框变了的重裁剪但
        保留已有标签；整图里删掉的框连同其裁剪（含已保存金标准）一并移除。
        """
        wanted = {
            crop_image_id("attribute", source_id, identifier): bbox
            for identifier, bbox in boxes
        }
        for dataset in (self.attribute_candidates, self.attribute_gold):
            removed = {
                image["id"]
                for image in dataset["images"]
                if image["source_image_id"] == source_id and image["id"] not in wanted
            }
            if removed:
                dataset["images"] = [image for image in dataset["images"] if image["id"] not in removed]
                dataset["annotations"] = [
                    item for item in dataset["annotations"] if item["image_id"] not in removed
                ]
                for crop_id in removed:
                    (config.ATTRIBUTE_IMAGES_DIR / f"{crop_id}.jpg").unlink(missing_ok=True)

        prelabeled = 0
        candidates_images = image_index(self.attribute_candidates)
        gold_images = image_index(self.attribute_gold)
        for identifier, bbox in boxes:
            crop_id = crop_image_id("attribute", source_id, identifier)
            existing = candidates_images.get(crop_id) or gold_images.get(crop_id)
            if existing is not None and [int(v) for v in existing.get("source_bbox", [])] == bbox:
                continue  # 框未动，裁剪与标签都保留
            x, y, w, h = bbox
            crop = full[y : y + h, x : x + w]
            crop_height, crop_width = crop.shape[:2]
            target = config.ATTRIBUTE_IMAGES_DIR / f"{crop_id}.jpg"
            self._save_crop(target, crop)
            if existing is not None:
                # 框变了：重裁剪、更新元数据，标签保留
                existing["width"] = crop_width
                existing["height"] = crop_height
                existing["source_bbox"] = bbox
                for dataset in (self.attribute_candidates, self.attribute_gold):
                    for item in dataset["annotations"]:
                        if item["image_id"] == crop_id:
                            item["bbox"] = [0, 0, crop_width, crop_height]
                continue
            attributes = self._prelabel(crop)
            if attributes is not None:
                prelabeled += 1
            self.attribute_candidates["images"].append(
                {
                    "id": crop_id,
                    "file_name": stored_path(target, config.PROJECT_ROOT),
                    "width": crop_width,
                    "height": crop_height,
                    "source_image_id": source_id,
                    "source_bbox": bbox,
                }
            )
            self.attribute_candidates["annotations"].append(
                {
                    "id": annotation_id(crop_id, 0),
                    "image_id": crop_id,
                    "category_id": 1,
                    "bbox": [0, 0, crop_width, crop_height],
                    "attributes": attributes,
                }
            )
            self._orders["attribute"].append(crop_id)
        save_dataset(
            config.ATTRIBUTE_CANDIDATES_PATH, self.attribute_candidates, "attribute", "model",
            gold=False,
        )
        save_dataset(config.ATTRIBUTE_GOLD_PATH, self.attribute_gold, "attribute", "human", gold=True)
        return prelabeled

    def _prelabel(self, crop: np.ndarray) -> dict[str, bool] | None:
        if not self.prelabel_enabled:
            return None
        try:
            if self._attribute_model is None:
                from prelabel_models import PaddleAttributeModel, paddle_model_files

                model_file, params_file = paddle_model_files(
                    config.PERSON_ATTRIBUTE_DIR, "inference"
                )
                self._attribute_model = PaddleAttributeModel(
                    model_file, params_file, (192, 256), self.device
                )
            return initial_body_labels(self._attribute_model.predict(crop))
        except Exception as exc:  # 预标注失败不阻塞保存
            self.prelabel_warning = f"属性预标注不可用: {exc}"
            print(self.prelabel_warning, flush=True)
            return None

    # ---- 阶段二：属性标注 ----

    def attribute_record(self, index: int, scope: str = "pending") -> dict:
        if scope == "saved":
            dataset = self.attribute_gold
            rows = list(reversed(dataset["annotations"]))  # 最近保存的在前，便于回改
            empty = "还没有已保存的属性标注"
        else:
            dataset = self.attribute_candidates
            images_all = image_index(dataset)
            by_image: dict[str, list[dict]] = {}
            for item in dataset["annotations"]:
                by_image.setdefault(item["image_id"], []).append(item)
            rows = [
                item
                for key in self._orders["attribute"]
                if key in images_all
                for item in by_image.get(key, [])
            ]
            empty = "属性标注没有待核对裁剪"
        if not rows:
            raise IndexError(empty)
        index = max(0, min(index, len(rows) - 1))
        annotation = copy.deepcopy(rows[index])
        images = image_index(dataset)
        return {
            "index": index,
            "count": len(rows),
            "scope": scope,
            "image": copy.deepcopy(images[annotation["image_id"]]),
            "annotation": annotation,
        }

    def attribute_image_path(self, image_id_value: str) -> Path:
        images = image_index(self.attribute_candidates) | image_index(self.attribute_gold)
        if image_id_value not in images:
            raise ValueError("裁剪不存在")
        return resolved_path(images[image_id_value]["file_name"], config.PROJECT_ROOT)

    def attribute_context_path(self, image_id_value: str) -> Path:
        images = image_index(self.attribute_candidates) | image_index(self.attribute_gold)
        if image_id_value not in images:
            raise ValueError("裁剪不存在")
        source_id = images[image_id_value]["source_image_id"]
        return self.detection_image_path(source_id)

    def attribute_save(self, image_id_value: str, attributes: object) -> dict:
        # 先校验再落库，避免非法 payload 污染内存中的金标准。
        if (
            not isinstance(attributes, dict)
            or tuple(attributes) != BODY_ATTRIBUTES
            or not all(type(value) is bool for value in attributes.values())
        ):
            raise ValueError("属性标签必须是固定顺序的 26 项布尔值")
        with self.lock:
            images = image_index(self.attribute_candidates)
            if image_id_value in images:
                annotation = next(
                    (item for item in self.attribute_candidates["annotations"]
                     if item["image_id"] == image_id_value),
                    None,
                )
                if annotation is None:
                    raise ValueError("候选裁剪缺少标注")
                saved = copy.deepcopy(annotation)
                saved["attributes"] = attributes
                self.attribute_gold["images"].append(copy.deepcopy(images[image_id_value]))
                self.attribute_gold["annotations"].append(saved)
                self._remove_candidate(self.attribute_candidates, image_id_value)
                save_dataset(
                    config.ATTRIBUTE_GOLD_PATH, self.attribute_gold, "attribute", "human", gold=True
                )
                save_dataset(
                    config.ATTRIBUTE_CANDIDATES_PATH, self.attribute_candidates, "attribute",
                    "model", gold=False,
                )
                return {"saved": image_id_value, "updated": False, "summary": self.summary()}
            # 回改已保存记录：原地更新属性，裁剪与划分不变。
            gold_images = image_index(self.attribute_gold)
            if image_id_value not in gold_images:
                raise ValueError("该裁剪不在候选或金标准中")
            annotation = next(
                item for item in self.attribute_gold["annotations"]
                if item["image_id"] == image_id_value
            )
            annotation["attributes"] = attributes
            save_dataset(
                config.ATTRIBUTE_GOLD_PATH, self.attribute_gold, "attribute", "human", gold=True
            )
            return {"saved": image_id_value, "updated": True, "summary": self.summary()}

    def attribute_exclude(self, image_id_value: str) -> dict:
        with self.lock:
            images = image_index(self.attribute_candidates)
            if image_id_value not in images:
                raise ValueError("候选裁剪不存在")
            crop_path = resolved_path(images[image_id_value]["file_name"], config.PROJECT_ROOT)
            self._remove_candidate(self.attribute_candidates, image_id_value)
            save_dataset(
                config.ATTRIBUTE_CANDIDATES_PATH, self.attribute_candidates, "attribute", "model",
                gold=False,
            )
            crop_path.unlink(missing_ok=True)
            return {"excluded": image_id_value, "summary": self.summary()}

    # ---- 阶段三：口罩标注 ----

    def mask_record(self, index: int, scope: str = "pending") -> dict:
        if scope == "saved":
            dataset = self.mask_gold
            rows = list(reversed(dataset["annotations"]))  # 最近保存的在前，便于回改
            empty = "还没有已保存的口罩标注"
        else:
            dataset = self.mask_candidates
            by_id = {item["id"]: item for item in dataset["annotations"]}
            rows = [by_id[key] for key in self._orders["mask"] if key in by_id]
            empty = "口罩标注没有待核对裁剪"
        if not rows:
            raise IndexError(empty)
        index = max(0, min(index, len(rows) - 1))
        annotation = copy.deepcopy(rows[index])
        images = image_index(dataset)
        return {
            "index": index,
            "count": len(rows),
            "scope": scope,
            "image": copy.deepcopy(images[annotation["image_id"]]),
            "annotation": annotation,
        }

    def mask_crop(self, annotation_id_value: str) -> bytes:
        candidate = next(
            (item for item in self.mask_candidates["annotations"]
             if item["id"] == annotation_id_value),
            None,
        )
        if candidate is None:
            # 已保存的金标准直接读落盘的裁剪文件。
            gold_annotation = next(
                (item for item in self.mask_gold["annotations"] if item["id"] == annotation_id_value),
                None,
            )
            if gold_annotation is None:
                raise ValueError("人脸框不存在")
            gold_image = image_index(self.mask_gold)[gold_annotation["image_id"]]
            return resolved_path(gold_image["file_name"], config.PROJECT_ROOT).read_bytes()
        image = image_index(self.mask_candidates)[candidate["image_id"]]
        full = self._decode_cached(resolved_path(image["file_name"], config.PROJECT_ROOT))
        crop, _ = self._mask_context_crop(full, candidate["bbox"], image)
        ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise ValueError("无法编码口罩裁剪")
        return encoded.tobytes()

    def mask_save(self, annotation_id_value: str, category_id: object) -> dict:
        if category_id not in (1, 2):
            raise ValueError("口罩类别只能是 1（未佩戴）或 2（佩戴）")
        with self.lock:
            if not any(
                item["id"] == annotation_id_value for item in self.mask_candidates["annotations"]
            ):
                # 回改已保存记录：原地更新类别，裁剪不变。
                gold_annotation = next(
                    (item for item in self.mask_gold["annotations"]
                     if item["id"] == annotation_id_value),
                    None,
                )
                if gold_annotation is None:
                    raise ValueError("该人脸框不在候选或金标准中")
                gold_annotation["category_id"] = int(category_id)
                save_dataset(config.MASK_GOLD_PATH, self.mask_gold, "mask", "human", gold=True)
                return {"saved": annotation_id_value, "updated": True, "summary": self.summary()}
            annotation, image = self._mask_candidate(annotation_id_value)
            full = self._decode_cached(resolved_path(image["file_name"], config.PROJECT_ROOT))
            crop, inner_box = self._mask_context_crop(full, annotation["bbox"], image)
            crop_id = crop_image_id("mask", image["id"], annotation_id_value)
            target = config.MASK_IMAGES_DIR / f"{crop_id}.jpg"
            self._save_crop(target, crop)
            crop_height, crop_width = crop.shape[:2]
            self.mask_gold["images"].append(
                {
                    "id": crop_id,
                    "file_name": stored_path(target, config.PROJECT_ROOT),
                    "width": crop_width,
                    "height": crop_height,
                    "source_image_id": image["id"],
                }
            )
            self.mask_gold["annotations"].append(
                {
                    "id": annotation_id(crop_id, 0),
                    "image_id": crop_id,
                    "category_id": int(category_id),
                    "bbox": [float(v) for v in xyxy_to_xywh(inner_box)],
                }
            )
            self._remove_mask_candidate(annotation, image)
            save_dataset(config.MASK_GOLD_PATH, self.mask_gold, "mask", "human", gold=True)
            save_dataset(
                config.MASK_CANDIDATES_PATH, self.mask_candidates, "mask", "aizoo", gold=False
            )
            return {"saved": crop_id, "updated": False, "summary": self.summary()}

    def mask_exclude(self, annotation_id_value: str) -> dict:
        with self.lock:
            annotation, image = self._mask_candidate(annotation_id_value)
            self._remove_mask_candidate(annotation, image)
            save_dataset(
                config.MASK_CANDIDATES_PATH, self.mask_candidates, "mask", "aizoo", gold=False
            )
            return {"excluded": annotation_id_value, "summary": self.summary()}

    def _mask_candidate(self, annotation_id_value: str) -> tuple[dict, dict]:
        annotation = next(
            (item for item in self.mask_candidates["annotations"]
             if item["id"] == annotation_id_value),
            None,
        )
        if annotation is None:
            raise ValueError("该人脸框已核对或不在候选队列")
        images = image_index(self.mask_candidates)
        return annotation, images[annotation["image_id"]]

    def _remove_mask_candidate(self, annotation: dict, image: dict) -> None:
        self.mask_candidates["annotations"] = [
            item for item in self.mask_candidates["annotations"] if item["id"] != annotation["id"]
        ]
        remaining = any(
            item["image_id"] == image["id"] for item in self.mask_candidates["annotations"]
        )
        if not remaining:
            self.mask_candidates["images"] = [
                item for item in self.mask_candidates["images"] if item["id"] != image["id"]
            ]

    @staticmethod
    def _mask_context_crop(
        full: np.ndarray, bbox: list[float], image: dict
    ) -> tuple[np.ndarray, list[int]]:
        """以人脸框为中心取 1.6 倍边长的带上下文裁剪，返回裁剪与人脸在裁剪内的框。"""
        height, width = full.shape[:2]
        left, top, right, bottom = xywh_to_xyxy(bbox)
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        side = max(right - left, bottom - top) * 1.6
        crop_box = [
            max(0, int(center_x - side / 2.0)),
            max(0, int(center_y - side / 2.0)),
            min(width, int(center_x + side / 2.0)),
            min(height, int(center_y + side / 2.0)),
        ]
        x1, y1, x2, y2 = crop_box
        inner = [
            int(left) - x1, int(top) - y1,
            min(int(right), x2) - x1, min(int(bottom), y2) - y1,
        ]
        return full[y1:y2, x1:x2], inner

    # ---- 阶段四：尘土化 ----

    def augment_preview(self, stage: str, params: AugmentParams) -> dict:
        from dust_augment import stage_source

        source_path, source_name = stage_source(stage)
        gold = load_dataset(source_path, stage, source_name, gold=True)
        if not gold["images"]:
            raise ValueError("该阶段还没有金标准数据")
        step = max(1, len(gold["images"]) // 4)
        samples = []
        for image in gold["images"][::step][:4]:
            source = self._decode(resolved_path(image["file_name"], config.PROJECT_ROOT))
            degraded = apply_degradation(source, params, _rng_for(params, image["id"], 0))
            samples.append(
                {"before": self._data_url(source), "after": self._data_url(degraded)}
            )
        return {"samples": samples}

    def augment_run(self, stage: str, params: AugmentParams) -> dict:
        with self.lock:
            if self.augment_status["running"]:
                raise ValueError("已有增强任务在运行")
            self.augment_status = {"running": True, "stage": stage, "done": 0, "total": 0,
                                   "error": None, "summary": None}

            def progress(done: int, total: int) -> None:
                self.augment_status["done"] = done
                self.augment_status["total"] = total

            def work() -> None:
                try:
                    summary = augment_stage(stage, params, progress)
                    self.augment_status["summary"] = summary
                except Exception as exc:  # 后台线程错误回传到状态
                    self.augment_status["error"] = str(exc)
                finally:
                    self.augment_status["running"] = False

            self._augment_thread = threading.Thread(target=work, daemon=True)
            self._augment_thread.start()
            return {"started": stage}

    # ---- 共用工具 ----

    @staticmethod
    def _remove_candidate(dataset: dict, image_id_value: str) -> None:
        dataset["images"] = [image for image in dataset["images"] if image["id"] != image_id_value]
        dataset["annotations"] = [
            item for item in dataset["annotations"] if item["image_id"] != image_id_value
        ]

    @staticmethod
    def _decode(path: Path) -> np.ndarray:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取图片: {path}")
        return image

    def _decode_cached(self, path: Path) -> np.ndarray:
        if self._full_image_cache is not None and self._full_image_cache[0] == str(path):
            return self._full_image_cache[1]
        image = self._decode(path)
        self._full_image_cache = (str(path), image)
        return image

    @staticmethod
    def _save_crop(path: Path, image: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise ValueError(f"无法编码裁剪: {path}")
        temporary = path.with_suffix(".jpg.tmp")
        encoded.tofile(temporary)
        temporary.replace(path)

    @staticmethod
    def _data_url(image: np.ndarray) -> str:
        height, width = image.shape[:2]
        scale = min(1.0, 320.0 / max(height, width))
        if scale < 1.0:
            image = cv2.resize(image, (int(width * scale), int(height * scale)))
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise ValueError("无法编码预览图")
        return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


class ReviewHandler(BaseHTTPRequestHandler):
    store: ReviewStore

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                self._send_file(HERE / "review_webui.html", "text/html; charset=utf-8")
            elif parsed.path == "/api/summary":
                self._json(self.store.summary())
            elif parsed.path == "/api/detection/record":
                self._json(
                    self.store.detection_record(self._index(query), self._scope(query))
                )
            elif parsed.path == "/api/detection/image":
                self._send_file(
                    self.store.detection_image_path(self._query_id(query)), self._mime()
                )
            elif parsed.path == "/api/attribute/record":
                self._json(
                    self.store.attribute_record(self._index(query), self._scope(query))
                )
            elif parsed.path == "/api/attribute/image":
                self._send_file(
                    self.store.attribute_image_path(self._query_id(query)), "image/jpeg"
                )
            elif parsed.path == "/api/attribute/context":
                self._send_file(
                    self.store.attribute_context_path(self._query_id(query)), self._mime()
                )
            elif parsed.path == "/api/mask/record":
                self._json(self.store.mask_record(self._index(query), self._scope(query)))
            elif parsed.path == "/api/mask/crop":
                content = self.store.mask_crop(self._query_id(query))
                self._send_bytes(content, "image/jpeg")
            elif parsed.path == "/api/augment/preview":
                self._json(
                    self.store.augment_preview(
                        self._query_stage(query), self._query_params(query)
                    )
                )
            elif parsed.path == "/api/augment/status":
                self._json(self.store.augment_status)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except (ValueError, IndexError, OSError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            payload = self._body()
            if parsed.path == "/api/detection/confirm":
                self._json(
                    self.store.detection_confirm(str(payload["image_id"]), payload["annotations"])
                )
            elif parsed.path == "/api/detection/exclude":
                self._json(self.store.detection_exclude(str(payload["image_id"])))
            elif parsed.path == "/api/attribute/save":
                self._json(
                    self.store.attribute_save(str(payload["image_id"]), payload["attributes"])
                )
            elif parsed.path == "/api/attribute/exclude":
                self._json(self.store.attribute_exclude(str(payload["image_id"])))
            elif parsed.path == "/api/mask/save":
                self._json(
                    self.store.mask_save(str(payload["annotation_id"]), payload["category_id"])
                )
            elif parsed.path == "/api/mask/exclude":
                self._json(self.store.mask_exclude(str(payload["annotation_id"])))
            elif parsed.path == "/api/augment/run":
                params = payload.get("params", {})
                if not isinstance(params, dict):
                    raise ValueError("params 必须是对象")
                self._json(
                    self.store.augment_run(
                        str(payload["stage"]), AugmentParams(**params)
                    )
                )
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except (KeyError, TypeError, ValueError, IndexError, OSError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")

    @staticmethod
    def _index(query: dict) -> int:
        return int(query.get("index", ["0"])[0])

    @staticmethod
    def _scope(query: dict) -> str:
        return "saved" if query.get("scope", [""])[0] == "saved" else "pending"

    @staticmethod
    def _query_id(query: dict) -> str:
        values = query.get("id")
        if not values or not values[0]:
            raise ValueError("缺少 id 参数")
        return values[0]

    @staticmethod
    def _query_stage(query: dict) -> str:
        stage = query.get("stage", ["attribute"])[0]
        if stage not in ("detection", "attribute", "mask"):
            raise ValueError("stage 只能是 detection、attribute 或 mask")
        return stage

    @staticmethod
    def _query_params(query: dict) -> AugmentParams:
        def number(name: str, default: float) -> float:
            return float(query.get(name, [str(default)])[0])

        def flag(name: str) -> bool:
            return query.get(name, ["1"])[0] != "0"

        return AugmentParams(
            variants=int(number("variants", 2)),
            intensity=number("intensity", 1.0),
            seed=int(number("seed", 20260817)),
            dust=flag("dust"), yellow=flag("yellow"), low_light=flag("low_light"),
            blur=flag("blur"), noise=flag("noise"), jpeg=flag("jpeg"),
        )

    @staticmethod
    def _mime() -> str:
        return "application/octet-stream"

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 5_000_000:
            raise ValueError("请求正文大小无效")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("请求正文必须是对象")
        return value

    def _json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        content = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self._send_bytes(content, "application/json; charset=utf-8", status)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json({"error": message}, status)

    def _send_file(self, path: Path, mime: str) -> None:
        guessed = mimetypes.guess_type(path.name)[0]
        self._send_bytes(path.read_bytes(), guessed or mime)

    def _send_bytes(
        self, content: bytes, mime: str, status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)


def main() -> int:
    parser = argparse.ArgumentParser(description="四阶段人工核对与尘土化扩充 WebUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", choices=("CPU", "GPU"), default=config.DEVICE)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-prelabel", action="store_true", help="不加载属性模型，预标注留空")
    parser.add_argument("--seed", type=int, help="候选随机出图顺序的种子；默认每次启动随机")
    args = parser.parse_args()
    for path in (config.DETECTION_CANDIDATES_PATH, config.MASK_CANDIDATES_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"请先生成候选文件: {path}")
    ReviewHandler.store = ReviewStore(
        prelabel=not args.no_prelabel, device=args.device, shuffle_seed=args.seed
    )
    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"核对界面: {url}")
    print(f"检测候选: {config.DETECTION_CANDIDATES_PATH}")
    print(f"属性裁剪: {config.ATTRIBUTE_IMAGES_DIR}")
    print(f"口罩候选: {config.MASK_CANDIDATES_PATH}")
    print(f"尘土化产物: {config.AUGMENTED_DATA_DIR}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
