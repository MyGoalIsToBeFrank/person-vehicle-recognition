#!/usr/bin/env python3
"""整图级人工核对：可增删改框，保存后才生成金标准裁剪与标签。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import mimetypes
import os
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "independent"))
sys.path.insert(0, str(HERE))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from dataset_schema import (  # noqa: E402
    SECTIONS,
    load_candidates,
    load_gold,
    resolved_path,
    save_candidates,
    save_gold,
    stored_path,
    validate_box,
    validate_label,
)


class ReviewStore:
    def __init__(self, candidates_path: Path, gold_path: Path, crop_dir: Path):
        self.candidates_path = candidates_path
        self.gold_path = gold_path
        self.crop_dir = crop_dir
        self.candidates = load_candidates(candidates_path)
        self.gold = load_gold(gold_path)
        self.lock = threading.Lock()

    def pending(self, section: str) -> list[dict]:
        self._section(section)
        reviewed = {frame["id"] for frame in self.gold[section]}
        return [frame for frame in self.candidates[section] if frame["id"] not in reviewed]

    def summary(self) -> dict:
        return {
            section: {
                "待核对": len(self.pending(section)),
                "已保存": len(self.gold[section]),
                "金标准框": sum(len(frame["框"]) for frame in self.gold[section]),
            }
            for section in SECTIONS
        }

    def record(self, section: str, index: int) -> tuple[dict, int, int]:
        rows = self.pending(section)
        if not rows:
            raise IndexError(f"{section}没有待核对原图")
        index = max(0, min(index, len(rows) - 1))
        return copy.deepcopy(rows[index]), len(rows), index

    def save_review(self, section: str, frame_id: str, boxes: object) -> dict:
        self._section(section)
        if not isinstance(boxes, list):
            raise ValueError("框必须是数组")
        with self.lock:
            frame = next((item for item in self.pending(section) if item["id"] == frame_id), None)
            if frame is None:
                raise ValueError("该原图已核对或不在候选队列")
            image = self._decode(resolved_path(frame["全图"], config.PROJECT_ROOT))
            height, width = image.shape[:2]
            normalized = []
            used_ids: set[str] = set()
            for number, item in enumerate(boxes):
                if not isinstance(item, dict) or set(item) != {"id", "框", "标签"}:
                    raise ValueError("每个框只能包含 id、框、标签")
                identifier = str(item["id"] or f"{frame_id}_new_{number:03d}")
                if identifier in used_ids:
                    raise ValueError(f"框 id 重复: {identifier}")
                used_ids.add(identifier)
                box = self._integer_box(item["框"], identifier, width, height)
                validate_label(section, item["标签"], allow_empty=False)
                normalized.append((identifier, box, copy.deepcopy(item["标签"])))
            gold_boxes = [
                self._gold_box(section, frame_id, identifier, box, label, image)
                for identifier, box, label in normalized
            ]
            gold_frame = {"id": frame_id, "全图": frame["全图"], "框": gold_boxes}
            self.gold[section].append(gold_frame)
            save_gold(self.gold_path, self.gold)
            return {"saved": frame_id, "summary": self.summary()}

    def exclude(self, section: str, frame_id: str) -> dict:
        """排除不可用整图：从候选移除，不写入金标准，也不删除原图。"""
        self._section(section)
        with self.lock:
            before = len(self.candidates[section])
            self.candidates[section] = [
                frame for frame in self.candidates[section] if frame["id"] != frame_id
            ]
            if len(self.candidates[section]) == before:
                raise ValueError("候选整图不存在")
            save_candidates(self.candidates_path, self.candidates)
            return {"excluded": frame_id, "summary": self.summary()}

    def _gold_box(
        self,
        section: str,
        frame_id: str,
        identifier: str,
        box: list[int],
        label: object,
        image: np.ndarray,
    ) -> dict:
        height, width = image.shape[:2]
        if section == "属性":
            crop_box = box
            training_box = None
        else:
            crop_box = self._context_box(box, width, height)
            training_box = [
                box[0] - crop_box[0], box[1] - crop_box[1],
                box[2] - crop_box[0], box[3] - crop_box[1],
            ]
        left, top, right, bottom = crop_box
        digest = hashlib.sha1(f"{frame_id}\0{identifier}".encode("utf-8")).hexdigest()[:24]
        output = self.crop_dir / ("body" if section == "属性" else "mask") / f"{digest}.jpg"
        self._save_crop(output, image[top:bottom, left:right])
        result = {
            "id": identifier,
            "框": box,
            "图片": stored_path(output, config.PROJECT_ROOT),
            "标签": copy.deepcopy(label),
        }
        if training_box is not None:
            result["训练框"] = training_box
        return result

    @staticmethod
    def _context_box(box: list[int], width: int, height: int) -> list[int]:
        left, top, right, bottom = box
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        side = max(right - left, bottom - top) * 1.6
        return [
            max(0, int(center_x - side / 2.0)),
            max(0, int(center_y - side / 2.0)),
            min(width, int(center_x + side / 2.0)),
            min(height, int(center_y + side / 2.0)),
        ]

    @staticmethod
    def _integer_box(value: object, identifier: str, width: int, height: int) -> list[int]:
        validate_box(value, identifier, "框")
        box = [round(float(number)) for number in value]  # type: ignore[arg-type]
        if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
            raise ValueError(f"{identifier} 的框超出原图: {box}, image={width}x{height}")
        if box[2] - box[0] < 4 or box[3] - box[1] < 4:
            raise ValueError(f"{identifier} 的框过小")
        return box

    @staticmethod
    def _decode(path: Path) -> np.ndarray:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取原图: {path}")
        return image

    @staticmethod
    def _save_crop(path: Path, image: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise ValueError(f"无法编码裁剪: {path}")
        temporary = path.with_suffix(".jpg.tmp")
        encoded.tofile(temporary)
        os.replace(temporary, path)

    @staticmethod
    def _section(section: str) -> None:
        if section not in SECTIONS:
            raise ValueError("section 只能是属性或口罩")


class ReviewHandler(BaseHTTPRequestHandler):
    store: ReviewStore

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send_file(HERE / "review_webui.html", "text/html; charset=utf-8")
            elif parsed.path == "/api/summary":
                self._json(self.store.summary())
            elif parsed.path == "/api/record":
                section, index = self._selection(parsed.query)
                record, count, index = self.store.record(section, index)
                self._json({"section": section, "index": index, "count": count, "record": record})
            elif parsed.path == "/api/image":
                section, index = self._selection(parsed.query)
                record, _, _ = self.store.record(section, index)
                image_path = resolved_path(record["全图"], config.PROJECT_ROOT)
                mime = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
                self._send_file(image_path, mime)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except (ValueError, IndexError, OSError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/review":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self._body()
            self._json(
                self.store.save_review(str(payload["section"]), str(payload["id"]), payload["框"])
            )
        except (KeyError, TypeError, ValueError, IndexError, OSError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))

    def do_DELETE(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/candidate":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self._body()
            self._json(self.store.exclude(str(payload["section"]), str(payload["id"])))
        except (KeyError, TypeError, ValueError, OSError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")

    @staticmethod
    def _selection(query: str) -> tuple[str, int]:
        values = parse_qs(query)
        return values.get("section", ["属性"])[0], int(values.get("index", ["0"])[0])

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
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json({"error": message}, status)

    def _send_file(self, path: Path, mime: str) -> None:
        content = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)


def main() -> int:
    parser = argparse.ArgumentParser(description="打开整图人物框、属性与口罩金标准核对界面")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not config.CANDIDATES_PATH.is_file():
        raise FileNotFoundError(f"请先生成候选文件: {config.CANDIDATES_PATH}")
    ReviewHandler.store = ReviewStore(
        config.CANDIDATES_PATH, config.GOLD_LABELS_PATH, config.GOLD_IMAGES_DIR
    )
    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"核对界面: {url}")
    print(f"候选输入: {config.CANDIDATES_PATH}")
    print(f"人工金标准: {config.GOLD_LABELS_PATH}")
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
