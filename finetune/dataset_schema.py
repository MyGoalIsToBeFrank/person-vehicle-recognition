"""模型候选与人工金标准的数据结构、校验和原子持久化。"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


BODY_ATTRIBUTES = (
    "Hat", "Glasses", "ShortSleeve", "LongSleeve", "UpperStride", "UpperLogo",
    "UpperPlaid", "UpperSplice", "LowerStripe", "LowerPattern", "LongCoat",
    "Trousers", "Shorts", "Skirt&Dress", "Boots", "HandBag", "ShoulderBag",
    "Backpack", "HoldObjectsInFront", "AgeLess18", "Age18-60", "AgeOver60",
    "Female", "Front", "Side", "Back",
)
MASK_LABELS = ("w/o mask", "w/ mask")
SECTIONS = ("属性", "口罩")


def empty_dataset() -> dict[str, Any]:
    return {"版本": 2, "属性": [], "口罩": []}


def load_candidates(path: Path) -> dict[str, Any]:
    data = _load(path)
    validate_dataset(data, gold=False)
    return data


def load_gold(path: Path) -> dict[str, Any]:
    data = _load(path)
    validate_dataset(data, gold=True)
    return data


def save_candidates(path: Path, data: dict[str, Any]) -> None:
    validate_dataset(data, gold=False)
    _save(path, data)


def save_gold(path: Path, data: dict[str, Any]) -> None:
    validate_dataset(data, gold=True)
    _save(path, data)


def validate_dataset(data: Any, gold: bool) -> None:
    if not isinstance(data, dict) or set(data) != {"版本", *SECTIONS}:
        raise ValueError("数据顶层只能包含版本、属性、口罩")
    if data["版本"] != 2:
        raise ValueError(f"不支持的数据版本: {data['版本']}")
    for section in SECTIONS:
        frames = data[section]
        if not isinstance(frames, list):
            raise ValueError(f"{section}必须是数组")
        frame_ids: set[str] = set()
        for frame in frames:
            validate_frame(section, frame, gold)
            if frame["id"] in frame_ids:
                raise ValueError(f"{section}整图 id 重复: {frame['id']}")
            frame_ids.add(frame["id"])


def validate_frame(section: str, frame: Any, gold: bool) -> None:
    if not isinstance(frame, dict) or set(frame) != {"id", "全图", "框"}:
        raise ValueError(f"{section}整图字段错误")
    if not isinstance(frame["id"], str) or not frame["id"]:
        raise ValueError("整图 id 必须是非空字符串")
    if not isinstance(frame["全图"], str) or not frame["全图"]:
        raise ValueError(f"{frame['id']} 的全图必须是路径字符串")
    if not isinstance(frame["框"], list):
        raise ValueError(f"{frame['id']} 的框必须是数组")
    box_ids: set[str] = set()
    for item in frame["框"]:
        validate_box_record(section, item, gold)
        if item["id"] in box_ids:
            raise ValueError(f"{frame['id']} 的框 id 重复: {item['id']}")
        box_ids.add(item["id"])


def validate_box_record(section: str, item: Any, gold: bool) -> None:
    common = {"id", "框", "标签"}
    expected = common | ({"图片"} if gold else set())
    if section == "口罩" and gold:
        expected.add("训练框")
    if not isinstance(item, dict) or set(item) != expected:
        raise ValueError(f"{section}框字段错误")
    if not isinstance(item["id"], str) or not item["id"]:
        raise ValueError("框 id 必须是非空字符串")
    validate_box(item["框"], item["id"], "框")
    if gold and (not isinstance(item["图片"], str) or not item["图片"]):
        raise ValueError(f"{item['id']} 的图片必须是路径字符串")
    if section == "口罩" and gold:
        validate_box(item["训练框"], item["id"], "训练框")
    validate_label(section, item["标签"], allow_empty=not gold)


def validate_label(section: str, label: Any, allow_empty: bool) -> None:
    if allow_empty and label is None:
        return
    if section == "属性":
        if not isinstance(label, dict) or tuple(label) != BODY_ATTRIBUTES:
            raise ValueError("属性标签必须包含固定顺序的 26 项")
        if not all(type(value) is bool for value in label.values()):
            raise ValueError("属性标签值必须是布尔值")
    elif section == "口罩":
        if label not in MASK_LABELS:
            raise ValueError(f"口罩标签无效: {label}")
    else:
        raise ValueError(f"未知栏目: {section}")


def validate_box(value: Any, record_id: str, name: str) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or not all(isinstance(number, (int, float)) and not isinstance(number, bool) for number in value)
        or value[2] <= value[0]
        or value[3] <= value[1]
    ):
        raise ValueError(f"{record_id} 的{name}无效: {value}")


def frame_id(section: str, full_image: str) -> str:
    digest = hashlib.sha1(f"{section}\0{full_image}".encode("utf-8")).hexdigest()[:20]
    return f"{'body' if section == '属性' else 'mask'}_{digest}"


def flatten_gold(data: dict[str, Any], section: str) -> list[dict[str, Any]]:
    rows = []
    for frame in data[section]:
        for item in frame["框"]:
            rows.append({"全图": frame["全图"], **item})
    return rows


def stored_path(path: Path, project_root: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def resolved_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_dataset()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取数据文件: {path}") from exc


def _save(path: Path, data: dict[str, Any]) -> None:
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
