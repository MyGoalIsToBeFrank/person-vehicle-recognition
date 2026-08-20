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

吞吐：``WORKERS``（默认 4）个**独立进程**（绕开 Python GIL 限制，CPU 预处理
才能真正并行），每个进程持有一整套模型实例（启动时串行创建，避免并发初始化
冲突）；主进程的派发线程把队列里的帧攒成最多 ``BATCH_SIZE``（默认 8）张一批
发给空闲进程批量推理——整图预处理在行人/车辆两个检测器间共享，全部裁片的
属性识别跨图合并成大批。GPU 显存约占用 WORKERS × 2GB。

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
import multiprocessing
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
WORKERS = int(os.environ.get("WORKERS", "4"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))

app = FastAPI(title="person-vehicle-recognition", version="1.0")

_tasks: "queue.Queue[tuple[str, np.ndarray]]" = queue.Queue(maxsize=BACKLOG)
_results: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_results_lock = threading.Lock()

# 跨进程队列在 startup 里创建（fork/spawn 语义不同，避免模块级创建）。
_mp_task_q = None
_mp_result_q = None


def _store(session_id: str, record: dict[str, Any]) -> None:
    with _results_lock:
        _results[session_id] = record
        _results.move_to_end(session_id)
        while len(_results) > MAX_KEPT:
            _results.popitem(last=False)


def _inference_process(task_q, result_q, device: str) -> None:
    """识别子进程入口：独占一套模型实例，收一批、推理、回传结果。"""
    pipeline = RecognitionPipeline(
        device=device,
        person_detector_dir=project_config.PERSON_DETECTOR_DIR,
        person_attribute_dir=project_config.PERSON_ATTRIBUTE_DIR,
        vehicle_detector_dir=project_config.VEHICLE_DETECTOR_DIR,
        vehicle_attribute_dir=project_config.VEHICLE_ATTRIBUTE_DIR,
        face_mask_dir=project_config.FACE_MASK_DIR,
        plate_model_dir=project_config.PLATE_MODEL_DIR,
        person_attribute_crop_scale=project_config.PERSON_ATTRIBUTE_CROP_SCALE,
    )
    while True:
        batch = task_q.get()
        started = time.perf_counter()
        try:
            contents = pipeline.recognize_batch([image for _, image in batch])
        except Exception:  # 批失败时逐张降级，隔离坏帧不影响整批
            contents = []
            for session_id, image in batch:
                try:
                    contents.append(pipeline.recognize_array(image))
                except Exception as exc:
                    result_q.put(
                        (session_id, {"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                    )
                    contents.append(None)
        per_image_ms = round((time.perf_counter() - started) * 1000.0 / len(batch), 3)
        for (session_id, _), content in zip(batch, contents):
            if content is None:
                continue  # 降级路径里已标记 error
            result_q.put(
                (session_id, {"status": "done", "result": content, "elapsed_ms": per_image_ms})
            )


def _dispatcher() -> None:
    """主进程派发：从有界内存队列攒批，送进跨进程队列（满了自然阻塞，
    背压一路传导回 POST 的 429）。"""
    while True:
        first = _tasks.get()
        batch = [first]
        while len(batch) < BATCH_SIZE:
            try:
                batch.append(_tasks.get_nowait())
            except queue.Empty:
                break
        _mp_task_q.put(batch)


def _collector() -> None:
    while True:
        session_id, record = _mp_result_q.get()
        _store(session_id, record)


@app.on_event("startup")
def _startup() -> None:
    global _mp_task_q, _mp_result_q
    device = os.environ.get("DEVICE", project_config.DEVICE).upper()
    # 必须 spawn：fork 会继承父进程已初始化的 CUDA 状态，
    # 子进程 create_predictor 直接报 cudaErrorInitializationError。
    ctx = multiprocessing.get_context("spawn")
    # 跨进程任务队列按批计，最多积压 WORKERS×2 批，防止大图数组在内存里堆积。
    _mp_task_q = ctx.Queue(maxsize=WORKERS * 2)
    _mp_result_q = ctx.Queue()
    for _ in range(WORKERS):  # 串行拉起，避免多 predictor 并发初始化冲突
        ctx.Process(
            target=_inference_process,
            args=(_mp_task_q, _mp_result_q, device),
            daemon=True,
        ).start()
    threading.Thread(target=_dispatcher, daemon=True).start()
    threading.Thread(target=_collector, daemon=True).start()


@app.post("/v1/tasks", status_code=202)
def submit(file: UploadFile = File(...)) -> dict[str, str]:
    # 同步端点：FastAPI 放进线程池执行，解码不阻塞事件循环，提交路径可并行。
    payload = file.file.read()
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
        "workers": WORKERS,
        "batch_size": BATCH_SIZE,
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
