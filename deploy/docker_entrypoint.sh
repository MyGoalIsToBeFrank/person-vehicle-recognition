#!/bin/bash
# 容器入口：启动 FastAPI 识别服务。
# PORT / DEVICE / BACKLOG / MAX_KEPT 均可用 -e 覆盖。
set -e
cd /app/service
exec python3 -m uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}"
