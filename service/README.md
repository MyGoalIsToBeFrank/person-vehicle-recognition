# 异步 HTTP 接口

提交与取结果是两个独立步骤。成功接收图片只表示它已进入有界队列；服务立即返回 `session_id`，不会占住连接等待 GPU。调用方随后单独查询结果。队列满时立即 429，因此图片突增不会形成无界积压。

## 单图

```http
POST /v1/tasks
Content-Type: multipart/form-data
```

字段名为 `file`，成功响应：

```http
HTTP/1.1 202 Accepted

{"session_id":"...","status":"pending"}
```

```http
GET /v1/tasks/{session_id}
```

状态为 `pending`、`running`、`done` 或 `error`。完成响应包含 `timing_ms.queue/inference/total` 和中文 `行人/车辆` 业务结果，车牌位于每条车辆记录内。未知 ID 为 404，已过期结果为 410。

## 高吞吐批提交

```http
POST /v1/task-batches
Content-Type: application/vnd.pvr.tasks-v1
```

所有整数均为 little-endian：

```text
magic[4] = "PVRB"
version:u16 = 1
flags:u16 = 0
image_count:u32
repeat image_count:
  media_type:u8   # 1 JPEG, 2 PNG, 3 BMP, 4 WebP
  reserved:u8 = 0
  reserved:u16 = 0
  payload_size:u32
  payload[payload_size]
```

声明格式必须与文件签名一致。成功返回与输入顺序一致的 `session_ids`。同一镜像中的 benchmark 和视频客户端已经使用该协议，不需要自行拼包。

## 批量取结果

```http
POST /v1/results:batch
Content-Type: application/json

{"session_ids":["...","..."]}
```

一次最多查询 512 个 ID，响应顺序与请求一致，避免为数千张图片发起数千个独立轮询请求。

## 限制与状态码

| 限制 | 默认值 |
|---|---:|
| 单图 | 8MiB、20MP |
| 批请求 | 64 图、64MiB |
| 原生队列 | 8192 图且不超过 1GiB |
| 聚批等待 | 2ms |
| 结果 TTL | 60 秒且总计不超过 1GiB |
| 结果记录 | 262144 |
| 同时摄入请求 | 2（防止最坏 64MiB 请求并发耗尽内存） |

- 413：请求体、图片数、图片字节或像素超限。
- 415：Content-Type 或图片格式不支持。
- 429：有界队列已满，带 `Retry-After`；调用方应退避或丢弃低价值帧。
- 503：engine 仍在构建或服务尚未 ready。
- 410：结果已过期。

已接收批次中的坏图只把自身置为 `error`，不使其他图片失败。JPEG 纳入吞吐 SLA；PNG/BMP/WebP 兼容但不计入千张吞吐指标。

## 运维接口

- `GET /v1/health`：readiness、初始化错误、GPU、engine 精度/缓存键、队列、结果缓存和显存。
- `GET /metrics`：Prometheus 文本格式的接受/完成吞吐、延迟、429、错误、batch、队列和显存指标。

## 同镜像工具

```bash
docker run --rm --network host person-vehicle-recognition:v2.0.0 \
  video-client --camera-id gate-1 --source rtsp://... --sample-fps 2

docker run --rm --network host person-vehicle-recognition:v2.0.0 \
  benchmark --input-dir /images --server http://127.0.0.1:8000 \
  --batch-size 64 --duration 600
```

推理容器默认只运行服务。每个摄像头采集客户端可以使用同一镜像独立横向扩展。
