#!/usr/bin/env python3
"""FastAPI 异步识别服务：提交图片拿 sessionId，凭 sessionId 轮询结果。

面向视频抽帧/抓拍流：上游逐帧 POST 到 /v1/tasks，立即拿到 sessionId；
随后用 GET /v1/tasks/{session_id} 轮询，识别完成后取走结构化中文结果。
提交与识别完全异步。识别在单个后台 worker 里串行执行（GPU 模型非线程安全）。

防积压崩溃（上游可能一秒几千张，远超 GPU 吞吐）：

- 识别队列有硬上限（环境变量 ``BACKLOG``，默认 512）。队列满时 POST 立即
  返回 **429**，上游应丢弃该帧或稍后重试——服务宁可拒收也绝不无限堆积。
- 结果只在内存保留最近 ``MAX_KEPT`` 条（默认 20000，超出按提交顺序淘汰），
  调用方应及时取走。
- 单帧解码或识别失败只标记该帧 ``error``，不影响后续帧。

接口：

- `POST /v1/tasks`：multipart 表单字段 `file` 上传一张图片。
  202 返回 `{"session_id": "...", "status": "pending"}`；队列满返回 429。
- `GET /v1/tasks/{session_id}`：返回 `{"session_id", "status", ...}`。
  `status` 为 `pending` / `done`（带 `result`、`elapsed_ms`）/ `error`（带 `error`）。
  `result` 结构与批量版 result.json 的 `识别内容` 字段完全一致
  （`行人` / `车辆` 两个数组，字段定义见 inference/README.md）。
- `GET /v1/health`：服务自检，返回设备、队列水位与结果保有量。

本地启动：`python service/app.py`（自动切到根目录 .venv）；
容器内由 deploy/docker_entrypoint.sh 直接拉起 uvicorn。
"""

from __future__ import annotations

import os
import sys
import threading
import queue
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "inference"))

import config as project_config  # noqa: E402


def enter_project_environment() -> None:
    """与 run.py 同款的环境切换：从系统 Python 启动时切到根目录 .venv。"""
    if os.environ.get("INFERENCE_RUN_IN_PLACE") == "1":
        return
    import shutil
    import subprocess

    python_name = "python.exe" if os.name == "nt" else "python"
    python_dir = "Scripts" if os.name == "nt" else "bin"
    expected_python = project_config.INFERENCE_VENV_DIR / python_dir / python_name
    if expected_python.is_file():
        try:
            if Path(sys.executable).resolve() == expected_python.resolve():
                return
        except OSError:
            pass
        completed = subprocess.run(
            [str(expected_python), str(__file__), *sys.argv[1:]],
            cwd=PROJECT_ROOT,
        )
        raise SystemExit(completed.returncode)
    raise RuntimeError("没有找到根目录 .venv；请先 uv sync 或在容器内运行。")


_DLL_HANDLES: list[Any] = []
enter_project_environment()
_DLL_HANDLES.extend(project_config.configure_runtime_dlls(project_config.INFERENCE_VENV_DIR))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402

from pipeline import RecognitionPipeline  # noqa: E402

BACKLOG = int(os.environ.get("BACKLOG", "512"))
MAX_KEPT = int(os.environ.get("MAX_KEPT", "20000"))

app = FastAPI(title="person-vehicle-recognition", version="1.0")

_pipeline: RecognitionPipeline | None = None
_tasks: "queue.Queue[tuple[str, np.ndarray]]" = queue.Queue(maxsize=BACKLOG)
_results: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_results_lock = threading.Lock()


def _store(session_id: str, record: dict[str, Any]) -> None:
    with _results_lock:
        _results[session_id] = record
        _results.move_to_end(session_id)
        while len(_results) > MAX_KEPT:
            _results.popitem(last=False)


def _worker() -> None:
    while True:
        session_id, image = _tasks.get()
        started = time.perf_counter()
        try:
            content = _pipeline.recognize_array(image)
        except Exception as exc:  # 单帧失败不能拖垮整个流
            _store(session_id, {"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            continue
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        _store(
            session_id,
            {"status": "done", "result": content, "elapsed_ms": elapsed_ms},
        )


@app.on_event("startup")
def _startup() -> None:
    global _pipeline
    device = os.environ.get("DEVICE", project_config.DEVICE).upper()
    _pipeline = RecognitionPipeline(
        device=device,
        person_detector_dir=project_config.PERSON_DETECTOR_DIR,
        person_attribute_dir=project_config.PERSON_ATTRIBUTE_DIR,
        vehicle_detector_dir=project_config.VEHICLE_DETECTOR_DIR,
        vehicle_attribute_dir=project_config.VEHICLE_ATTRIBUTE_DIR,
        face_mask_dir=project_config.FACE_MASK_DIR,
        plate_model_dir=project_config.PLATE_MODEL_DIR,
        person_attribute_crop_scale=project_config.PERSON_ATTRIBUTE_CROP_SCALE,
    )
    threading.Thread(target=_worker, daemon=True).start()


@app.post("/v1/tasks", status_code=202)
async def submit(file: UploadFile = File(...)) -> dict[str, str]:
    payload = await file.read()
    array = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="图片损坏或格式不受支持")
    session_id = uuid.uuid4().hex
    try:
        _tasks.put_nowait((session_id, image))
    except queue.Full:
        raise HTTPException(
            status_code=429,
            detail=f"识别队列已满（{BACKLOG}），请丢弃该帧或稍后重试",
        )
    _store(session_id, {"status": "pending"})
    return {"session_id": session_id, "status": "pending"}


@app.get("/v1/tasks/{session_id}")
def query(session_id: str) -> dict[str, Any]:
    with _results_lock:
        record = _results.get(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="session_id 不存在或已被淘汰")
    return {"session_id": session_id, **record}


@app.get("/v1/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "device": os.environ.get("DEVICE", project_config.DEVICE).upper(),
        "backlog": BACKLOG,
        "queue": _tasks.qsize(),
        "kept_results": len(_results),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        log_level="info",
    )
