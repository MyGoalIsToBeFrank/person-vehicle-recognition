"""管线中间数据的 COCO 风格（版本 3）结构、校验和原子持久化。

所有阶段共用 COCO 顶层结构 ``info / images / annotations / categories``，
框一律为 ``bbox = [x, y, w, h]``。各阶段的差异：

- ``detection``：整图 + person 框，无标签。候选是模型初检，确认后是检测训练金标准。
- ``attribute``：人物裁剪小图 + 26 项布尔属性。image 记录带 ``source_image_id``
  与 ``source_bbox``（框在原图中的位置），annotation 带 ``attributes`` 字典。
- ``mask``：候选是整图 + 人脸框（类别即标签）；金标准是带上下文的人脸裁剪小图，
  annotation 的 ``bbox`` 为人脸在裁剪内的位置。

裁剪图片划分 train/val/test 时一律按 ``source_image_id`` 哈希，保证同一原图的
所有目标与增强副本落在同一划分。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 3
STAGES = ("detection", "attribute", "mask")

BODY_ATTRIBUTES = (
    "Hat", "Glasses", "ShortSleeve", "LongSleeve", "UpperStride", "UpperLogo",
    "UpperPlaid", "UpperSplice", "LowerStripe", "LowerPattern", "LongCoat",
    "Trousers", "Shorts", "Skirt&Dress", "Boots", "HandBag", "ShoulderBag",
    "Backpack", "HoldObjectsInFront", "AgeLess18", "Age18-60", "AgeOver60",
    "Female", "Front", "Side", "Back",
)
MASK_LABELS = ("w/o mask", "w/ mask")

_IMAGE_PREFIX = {"detection": "det", "attribute": "attr", "mask": "mask"}


def stage_categories(stage: str) -> list[dict[str, Any]]:
    if stage == "detection":
        return [{"id": 1, "name": "person", "supercategory": "person"}]
    if stage == "attribute":
        return [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person",
                "attributes": list(BODY_ATTRIBUTES),
            }
        ]
    if stage == "mask":
        return [
            {"id": index, "name": name, "supercategory": "face"}
            for index, name in enumerate(MASK_LABELS, 1)
        ]
    raise ValueError(f"未知阶段: {stage}")


def mask_category_id(label: str) -> int:
    if label not in MASK_LABELS:
        raise ValueError(f"口罩标签无效: {label}")
    return MASK_LABELS.index(label) + 1


def mask_label(category_id: int) -> str:
    if category_id not in (1, 2):
        raise ValueError(f"口罩类别 id 无效: {category_id}")
    return MASK_LABELS[category_id - 1]


def empty_dataset(stage: str, source: str) -> dict[str, Any]:
    _check_stage(stage)
    return {
        "info": {"version": SCHEMA_VERSION, "stage": stage, "source": source},
        "images": [],
        "annotations": [],
        "categories": stage_categories(stage),
    }


def load_dataset(path: Path, stage: str, source: str, gold: bool) -> dict[str, Any]:
    if not path.exists():
        return empty_dataset(stage, source)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取数据文件: {path}") from exc
    validate_dataset(data, stage, source, gold)
    return data


def save_dataset(path: Path, data: dict[str, Any], stage: str, source: str, gold: bool) -> None:
    validate_dataset(data, stage, source, gold)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", prefix=f"{path.stem}.",
            suffix=".tmp", dir=path.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def validate_dataset(data: Any, stage: str, source: str, gold: bool) -> None:
    _check_stage(stage)
    if not isinstance(data, dict) or set(data) != {"info", "images", "annotations", "categories"}:
        raise ValueError("数据顶层只能包含 info、images、annotations、categories")
    info = data["info"]
    if not isinstance(info, dict) or info.get("version") != SCHEMA_VERSION:
        raise ValueError(f"不支持的数据版本: {info!r}")
    if info.get("stage") != stage or info.get("source") != source:
        raise ValueError(f"数据阶段或来源不符: {info!r}")
    if data["categories"] != stage_categories(stage):
        raise ValueError(f"{stage} 类别表错误")
    images = data["images"]
    annotations = data["annotations"]
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("images 和 annotations 必须是数组")
    image_ids: set[str] = set()
    for image in images:
        validate_image(stage, image, gold)
        if image["id"] in image_ids:
            raise ValueError(f"image id 重复: {image['id']}")
        image_ids.add(image["id"])
    annotation_ids: set[str] = set()
    for annotation in annotations:
        validate_annotation(stage, annotation, gold)
        if annotation["id"] in annotation_ids:
            raise ValueError(f"annotation id 重复: {annotation['id']}")
        if annotation["image_id"] not in image_ids:
            raise ValueError(f"annotation 引用了不存在的 image: {annotation['id']}")
        annotation_ids.add(annotation["id"])


def validate_image(stage: str, image: Any, gold: bool) -> None:
    if not isinstance(image, dict):
        raise ValueError("image 必须是对象")
    # attribute 的候选与金标准都是裁剪小图；mask 只有金标准是裁剪，候选仍是整图。
    crop = stage == "attribute" or (stage == "mask" and gold)
    required = {"id", "file_name", "width", "height"} | ({"source_image_id"} if crop else set())
    allowed = required | {"source_image_id", "source_bbox", "augmentation"}
    if not required <= set(image) or not set(image) <= allowed:
        raise ValueError(f"image 字段错误: {sorted(image)}")
    if not isinstance(image["id"], str) or not image["id"]:
        raise ValueError("image id 必须是非空字符串")
    if not isinstance(image["file_name"], str) or not image["file_name"]:
        raise ValueError(f"{image['id']} 的 file_name 必须是路径字符串")
    for name in ("width", "height"):
        if not isinstance(image[name], int) or image[name] <= 0:
            raise ValueError(f"{image['id']} 的 {name} 必须是正整数")
    if "source_image_id" in image:
        if not isinstance(image["source_image_id"], str) or not image["source_image_id"]:
            raise ValueError(f"{image['id']} 的 source_image_id 必须是非空字符串")
    elif crop:
        raise ValueError(f"{image['id']} 缺少 source_image_id")
    if "source_bbox" in image:
        validate_bbox(image["source_bbox"], image["id"], "source_bbox")
    if "augmentation" in image and not isinstance(image["augmentation"], dict):
        raise ValueError(f"{image['id']} 的 augmentation 必须是对象")


def validate_annotation(stage: str, annotation: Any, gold: bool) -> None:
    if not isinstance(annotation, dict):
        raise ValueError("annotation 必须是对象")
    required = {"id", "image_id", "category_id", "bbox"}
    allowed = required | {"attributes", "source_id"}
    if not required <= set(annotation) or not set(annotation) <= allowed:
        raise ValueError(f"annotation 字段错误: {sorted(annotation)}")
    if not isinstance(annotation["id"], str) or not annotation["id"]:
        raise ValueError("annotation id 必须是非空字符串")
    if not isinstance(annotation["image_id"], str) or not annotation["image_id"]:
        raise ValueError(f"{annotation['id']} 的 image_id 必须是非空字符串")
    validate_bbox(annotation["bbox"], annotation["id"], "bbox")
    category_ids = {category["id"] for category in stage_categories(stage)}
    if annotation["category_id"] not in category_ids:
        raise ValueError(f"{annotation['id']} 的 category_id 无效: {annotation['category_id']}")
    if stage == "attribute":
        attributes = annotation.get("attributes")
        if attributes is None:
            if gold:
                raise ValueError(f"{annotation['id']} 缺少属性标签")
        elif (
            not isinstance(attributes, dict)
            or tuple(attributes) != BODY_ATTRIBUTES
            or not all(type(value) is bool for value in attributes.values())
        ):
            raise ValueError(f"{annotation['id']} 的属性标签必须是固定顺序的 26 项布尔值")
    elif "attributes" in annotation:
        raise ValueError(f"{stage} 阶段的 annotation 不能带 attributes")
    if "source_id" in annotation and not isinstance(annotation["source_id"], str):
        raise ValueError(f"{annotation['id']} 的 source_id 必须是字符串")


def validate_bbox(value: Any, record_id: str, name: str) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or not all(isinstance(number, (int, float)) and not isinstance(number, bool) for number in value)
        or value[2] <= 0
        or value[3] <= 0
    ):
        raise ValueError(f"{record_id} 的 {name} 无效: {value}")


def image_id(stage: str, full_image: str) -> str:
    """由阶段与项目相对路径得到确定性的整图 id。"""
    _check_stage(stage)
    digest = hashlib.sha1(f"{stage}\0{full_image}".encode("utf-8")).hexdigest()[:20]
    return f"{_IMAGE_PREFIX[stage]}_{digest}"


def crop_image_id(stage: str, source_id: str, annotation_id_value: str) -> str:
    digest = hashlib.sha1(f"{stage}\0{source_id}\0{annotation_id_value}".encode("utf-8")).hexdigest()[:20]
    return f"{_IMAGE_PREFIX[stage]}_{digest}"


def annotation_id(image_id_value: str, number: int) -> str:
    return f"{image_id_value}_{number:03d}"


def xyxy_to_xywh(box: list[float]) -> list[float]:
    left, top, right, bottom = box
    return [left, top, right - left, bottom - top]


def xywh_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, width, height = bbox
    return [x, y, x + width, y + height]


def split_name(source_key: str) -> str:
    """按源整图 id（或路径）哈希划分，同图所有目标与增强副本同划分。"""
    bucket = int(hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "test" if bucket == 0 else "val" if bucket == 1 else "train"


def image_index(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {image["id"]: image for image in data["images"]}


def flatten_annotations(data: dict[str, Any]) -> list[dict[str, Any]]:
    """把 annotation 与其 image 记录合并成行，供训练与导出使用。"""
    images = image_index(data)
    return [{**annotation, "image": images[annotation["image_id"]]} for annotation in data["annotations"]]


def stored_path(path: Path, project_root: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def resolved_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _check_stage(stage: str) -> None:
    if stage not in STAGES:
        raise ValueError(f"stage 只能是 {STAGES}: {stage}")
