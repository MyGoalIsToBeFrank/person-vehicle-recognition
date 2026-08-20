# service/ — FastAPI 异步识别服务

把 `inference/` 的识别管线包装成 HTTP 服务，面向视频抽帧 / 抓拍流场景。
**输入与输出分离（异步）**：提交图片立即拿到 `session_id`，结果另行轮询获取，
配合队列上限防止高并发抓拍时服务积压崩溃。

## 启动

本地：

```bash
cd service
../.venv/Scripts/python.exe app.py        # Windows
# ../.venv/bin/python app.py              # Linux
```

环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `PORT` | `8000` | 监听端口 |
| `DEVICE` | `GPU` | 推理设备，`GPU` / `CPU` |
| `BACKLOG` | `512` | 队列硬上限，满则拒收（429） |
| `MAX_KEPT` | `20000` | 内存中最多保留的结果数，超出按先进先出淘汰 |
| `INFERENCE_RUN_IN_PLACE` | — | 置 `1` 时跳过 venv 切换（容器内置位） |

Docker 部署见根目录 `deploy/`。

## 接口

### `POST /v1/tasks` — 提交图片

- 请求：`multipart/form-data`，字段名 `file`，内容为图片（jpg/png）。
- 响应 `202`：

```json
{"session_id": "afd8d588d6c242abbb33c18b5932221c", "status": "pending"}
```

- 响应 `429`：队列已满，**调用方应丢弃本帧或稍后重试**，不要无脑重发。

### `GET /v1/tasks/{session_id}` — 查询结果

处理中：

```json
{"session_id": "...", "status": "pending"}
```

完成（`result` 格式与 `inference/result.json` 完全一致，详见 `inference/README.md`）：

```json
{
  "session_id": "...",
  "status": "done",
  "elapsed_ms": 620.5,
  "result": {
    "行人": [{"性别": "女", "年龄": "18至60岁", "…": "…"}],
    "车辆": [{"颜色": "白色", "车型": "轿车", "车牌": "鲁A66666"}]
  }
}
```

失败：`{"session_id": "...", "status": "error", "error": "..."}`

### `GET /v1/health` — 健康检查

```json
{"status": "ok", "device": "GPU", "backlog": 512, "queue": 3, "kept_results": 128}
```

## 视频抽帧客户端 `video_client.py`

从视频文件或 RTSP 流按参数化帧率抽帧，异步提交给服务，最后汇总写出 JSONL。

```bash
python video_client.py --source video.mp4 --fps 2 --output results.jsonl
python video_client.py --source rtsp://192.168.1.10/stream --fps 1 \
    --server http://server:8000 --output results.jsonl
```

参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--source` | 必填 | 视频文件路径、RTSP 地址或摄像头编号 |
| `--fps` | `1.0` | 每秒抽帧数 |
| `--server` | `http://127.0.0.1:8000` | 服务地址 |
| `--output` | `results.jsonl` | 输出文件（每行一条 JSON） |
| `--timeout` | `600` | 结果轮询总超时（秒） |

收到 429 时客户端直接丢弃该帧并计数，保证服务端永不积压。

输出行格式：

```json
{"frame_index": 50, "timestamp_ms": 2000, "session_id": "...", "status": "done",
 "elapsed_ms": 310.2, "result": {"行人": [...], "车辆": [...]}}
```
