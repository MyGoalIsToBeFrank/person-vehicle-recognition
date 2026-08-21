# PVR v2.0.0 镜像交付与部署手册

本文面向镜像接收方，说明 `person-vehicle-recognition:v2.0.0` 的功能、接口、部署、验证、调试和后续维护。分享包已经包含镜像、校验文件、测试客户端、样例图片和参考输出；接收方不需要源码或 Python 第三方包即可完成基本验收。

## 1. 功能与实现边界

服务对独立抓拍图片执行完整识别：

- 行人检测、人体属性和口罩状态；
- 车辆检测、车辆颜色和车型属性；
- 在车辆区域内使用 PaddleDetection PP-OCRv3 det + rec 检测并识别车牌；
- JPEG 使用 GPU 解码/预处理热路径；PNG、BMP、WebP 可用，但不计入高吞吐 SLA；
- FastAPI 只负责异步 HTTP 边界，内部是单实例 C++/CUDA/TensorRT worker、有界队列和有界结果缓存。

最终运行镜像不包含 Paddle 推理服务、ONNX Runtime、DALI 或多 Python 进程模型副本。当前镜像使用已有微调权重，构建过程不会运行或改写 `finetune/`。运行栈固定为 CUDA 12.6.3、cuDNN 9.5、TensorRT 10.5；Python 依赖由 `uv 0.11.2` 锁定，构建过程不调用 `pip`。

当前镜像身份：

```text
镜像标签：person-vehicle-recognition:v2.0.0
镜像 ID：sha256:9aebf7312a1607e07d1c184ec8bebc288488a254dd382d5c485f770b3adc7439
离线镜像 SHA-256：e8d4823e3967ae40e29e26b33628c201c150854b3b68558a13ad9550a7cb1f25
```

外部分发前必须确认图片和所有模型权重的再分发权利，尤其口罩模型。样例中可能有清晰车牌，只能向有权处理这些数据的接收方受控分享，不应建立无期限公开链接。

## 2. 分享包内容和完整性校验

完整分享包解压后应为：

```text
pvr-v2.0.0-share/
├── DEPLOY.md
├── MANIFEST.sha256
├── README.md
├── person-vehicle-recognition-v2.0.0.tar.zst
├── person-vehicle-recognition-v2.0.0.tar.zst.sha256
├── reference_results_rtx3080ti_fp16.json
├── test_service.py
└── samples/
    ├── sample_front_vehicle.jpg
    ├── sample_multi_vehicle.jpg
    └── sample_rear_vehicle.jpg
```

发送方会另外提供外层文件 `pvr-v2.0.0-share-bundle.tar.zst.sha256`。Linux 接收方先校验、解包，再校验包内每个文件：

```bash
sha256sum -c pvr-v2.0.0-share-bundle.tar.zst.sha256
zstd -dc pvr-v2.0.0-share-bundle.tar.zst | tar -xf -
cd pvr-v2.0.0-share
sha256sum -c MANIFEST.sha256
sha256sum -c person-vehicle-recognition-v2.0.0.tar.zst.sha256
```

Windows 可用 7-Zip ZS 或 WSL 解压；推荐在 WSL 中执行同样的 `sha256sum`、`zstd` 和 `tar` 命令。校验不通过时不要加载镜像，应重新传输损坏文件。

## 3. 主机准备

最低部署条件：

- Linux x86_64；Docker 24+；NVIDIA Container Toolkit；
- NVIDIA 驱动能够运行 CUDA 12.6 容器；
- 目标生产 GPU 为 A30 24GB，首次生成 TensorRT engine 期间需要充足磁盘和显存；
- 端口 8000 可供调用方访问。

先验证 Docker 和 GPU 透传：

```bash
docker version
nvidia-smi
docker run --rm --gpus all \
  m.daocloud.io/docker.io/nvidia/cuda:12.6.3-base-ubuntu22.04 \
  nvidia-smi
```

若最后一条失败，先修复 NVIDIA 驱动或 Container Toolkit，不要在应用容器内绕过 GPU 透传问题。

## 4. 加载和启动

在解包目录加载离线镜像：

```bash
zstd -dc person-vehicle-recognition-v2.0.0.tar.zst | docker load
docker image inspect person-vehicle-recognition:v2.0.0 \
  --format '{{.Id}}'
```

返回的 ID 应与第 1 节一致。创建持久化 engine cache 并启动：

```bash
docker volume create pvr-engine-cache

docker run -d --name pvr-v2 \
  --gpus '"device=0"' \
  --restart unless-stopped \
  -p 8000:8000 \
  -v pvr-engine-cache:/var/cache/pvr \
  person-vehicle-recognition:v2.0.0
```

首次启动会针对当前 GPU、SM、TensorRT 版本、精度、模型哈希和输入 profile 构建 7 个专用 engine，可能耗时几十分钟。此时进程存活，但 `/v1/health` 返回 HTTP 503；用以下命令观察：

```bash
docker logs -f pvr-v2
curl -i http://127.0.0.1:8000/v1/health
```

只有 HTTP 200 且 JSON 中 `ready: true` 才能接收图片。之后重启会命中命名卷中的缓存并显著加快 ready。不要把其他 GPU 生成的 engine 复制进该卷。

## 5. 异步输入与输出

提交和获取结果是两个完全分开的步骤。提交成功只代表图片进入有界队列，服务立即返回 session ID，不会让 HTTP 连接等待 GPU：

```bash
curl -i -F 'file=@samples/sample_front_vehicle.jpg;type=image/jpeg' \
  http://127.0.0.1:8000/v1/tasks
```

成功响应为 HTTP 202：

```json
{"session_id":"b8...","status":"pending"}
```

随后独立查询：

```bash
curl http://127.0.0.1:8000/v1/tasks/b8...
```

状态依次可能为 `pending`、`running`、`done` 或 `error`。完成记录示例：

```json
{
  "session_id": "b8...",
  "status": "done",
  "result": {
    "行人": [],
    "车辆": [
      {"颜色": "白色", "车型": "轿车", "车牌": "皖D5H594"}
    ]
  },
  "timing_ms": {"queue": 0.143, "inference": 238.4, "total": 238.7}
}
```

行人记录可能包含以下字段：

```json
{
  "性别": "女",
  "年龄": "18至60岁",
  "朝向": "正面",
  "佩戴眼镜": "否",
  "佩戴帽子": "否",
  "手持物品": "否",
  "包": "无",
  "上装": {"袖长": "短袖", "款式": []},
  "下装": ["长外套", "长裤", "裙装"],
  "鞋靴": "非靴子",
  "口罩": "未佩戴口罩"
}
```

车牌是每条车辆记录内的字段；无法识别时为 `未识别`。服务目前不返回检测框坐标。已接受的坏图只会把自身置为 `error`，不会使同批其他图片失败。

### 高吞吐批协议

高吞吐调用使用 `POST /v1/task-batches`，而不是并发发送大量 multipart 请求。`Content-Type` 固定为 `application/vnd.pvr.tasks-v1`，最多 64 图。所有整数均为 little-endian：

```text
magic[4] = "PVRB"
version:u16 = 1
flags:u16 = 0
image_count:u32
repeat image_count:
  media_type:u8       # 1 JPEG, 2 PNG, 3 BMP, 4 WebP
  reserved:u8 = 0
  reserved:u16 = 0
  payload_size:u32
  payload[payload_size]
```

成功返回与输入顺序一致的 `session_ids`。随后通过一个请求查询最多 512 个 ID：

```http
POST /v1/results:batch
Content-Type: application/json

{"session_ids":["id-1","id-2"]}
```

响应 `results` 顺序与请求一致。调用方应保留 `文件/业务主键 ↔ session_id` 映射，直到拿到终态。结果只在单容器内存中保存；超时、重启或关机后，未确认完成的图片必须由调用方重新提交。

### 默认边界与状态码

| 项目 | 默认值 |
|---|---:|
| 单图 | 8 MiB、20 MP |
| 单批请求 | 64 图、64 MiB |
| 原生队列 | 8192 图且不超过 1 GiB |
| 聚批等待 | 2 ms |
| 结果缓存 | 60 秒、1 GiB、262144 条 |
| 并发摄入大请求 | 2 |

- `413`：图片、像素、图片数或请求字节超限；
- `415`：Content-Type、图片格式或二进制协议不支持；
- `429`：队列已满，响应含 `Retry-After`；调用方应退避，视频场景可丢弃低价值帧；
- `503`：engine 仍在构建或服务未 ready；
- `404`：单查的 session ID 不存在；
- `410`：单查结果已经过期。

这些硬边界保证突发图片不会形成无界积压。不要通过无限增大队列掩盖算力不足。

## 6. 使用随包样例一键验收

`test_service.py` 只使用 Python 3 标准库，默认提交 `samples/` 中的 3 张图，批量轮询并打印每张图的 session ID、终态、识别结果和耗时：

```bash
python3 test_service.py --server http://127.0.0.1:8000 \
  --output actual_results.json
```

也可以指定自己的 1 至 64 张图：

```bash
python3 test_service.py --server http://SERVER:8000 \
  camera-01.jpg camera-02.jpg
```

退出码为 0 表示所有图片均为 `done`；连接失败、协议失败或超时为 1；至少一张图片进入 `error/unknown/expired` 为 2。`reference_results_rtx3080ti_fp16.json` 是 2026-08-20 在 RTX 3080 Ti FP16 engine 上得到的参考业务结果，用于人工检查字段和数量，不应在另一台 GPU 上作为逐字符精度断言。尤其车牌、阈值附近的属性和检测数量可能有合理差异。

## 7. 视频、RTSP 和摄像头输入

推理容器只接收图片，不直接接收整段视频，也不会主动连接 RTSP。视频输入由独立
`video-client` 读取、抽帧和提交：

```text
RTSP / 摄像头 / 视频文件
  ↓
video-client 按 sample_fps 抽帧并编码 JPEG
  ↓ 每批最多 64 帧
POST /v1/task-batches
  ↓ 每个帧返回一个 session_id
POST /v1/results:batch
  ↓
camera_id + frame_index + timestamp_ms + 识别结果 → JSONL
```

session ID 不是视频 ID。一个视频不会只返回一个 ID；每个被采样帧都有独立 ID，客户端
在本地维护 `session_id → camera_id/frame_index/timestamp_ms`，拿到终态后写出一行 JSON。

推理服务容器：

```bash
docker run -d --name pvr-v2 --gpus '"device=0"' \
  --restart unless-stopped -p 8000:8000 \
  -v pvr-engine-cache:/var/cache/pvr \
  person-vehicle-recognition:v2.0.0
```

另开一个容器读取 RTSP，不给它 GPU：

```bash
mkdir -p "$PWD/output"
docker run --rm --network host \
  -v "$PWD/output:/output" \
  person-vehicle-recognition:v2.0.0 \
  video-client \
  --camera-id gate-1 \
  --source 'rtsp://user:password@camera/stream' \
  --sample-fps 2 \
  --batch-size 32 \
  --max-pending 4096 \
  --poll-interval 0.05 \
  --jpeg-quality 90 \
  --server http://127.0.0.1:8000 \
  --output /output/gate-1.jsonl
```

本地视频只需替换 source，并挂载只读目录：

```bash
docker run --rm --network host \
  -v /data/videos:/videos:ro \
  -v "$PWD/output:/output" \
  person-vehicle-recognition:v2.0.0 \
  video-client --camera-id file-demo \
  --source /videos/demo.mp4 --sample-fps 1 \
  --server http://127.0.0.1:8000 \
  --output /output/demo.jsonl
```

一行输出示例：

```json
{"camera_id":"gate-1","frame_index":1250,"timestamp_ms":50000,"session_id":"...","status":"done","result":{"行人":[],"车辆":[]},"timing_ms":{"queue":2.1,"inference":113.4,"total":115.5}}
```

当前客户端每次启动处理一个 source。服务 429 时当前批次计为 dropped，不会无限积压。
基础版本尚未包含 RTSP 自动重连、断点持久化或摄像头控制面；大量实时流部署前应补充有界
最新帧缓冲、采集/提交线程分离、墙钟时间、重连和可靠结果存储，但仍不要把这些生命周期
塞入 TensorRT 推理容器。

## 8. 日常监控和调试

先按以下顺序定位问题：

```bash
docker ps -a --filter name=pvr-v2
docker inspect pvr-v2 \
  --format 'status={{.State.Status}} health={{.State.Health.Status}} restart={{.RestartCount}} policy={{.HostConfig.RestartPolicy.Name}}'
docker logs --tail 200 pvr-v2
curl -i http://127.0.0.1:8000/v1/health
curl -sS http://127.0.0.1:8000/metrics
nvidia-smi
docker system df
docker volume inspect pvr-engine-cache
```

`/v1/health` 会给出 readiness、初始化错误、GPU/显存、engine 精度和缓存键、队列水位、结果缓存、请求限制及 TensorRT context 显存池。`/metrics` 是 Prometheus 文本，包含接受、完成、拒绝、图片错误、GPU batch、队列、显存和总延迟指标。

常见故障：

| 现象 | 处理 |
|---|---|
| 首启长期 503 | 查看日志确认 engine 是否仍在逐个构建；同时检查磁盘、GPU 显存和驱动错误。 |
| 容器立即退出 | 查看 `docker logs` 和 `docker inspect`；优先修复 GPU 透传、动态库或端口冲突。 |
| 大量 429 | 输入速率超过持续完成能力；客户端指数退避、减小采样率或增加 GPU 实例。 |
| 查询 404/410/unknown | ID 发到了错误实例、结果超过 60 秒或实例重启；重新提交并修正路由。 |
| OOM | 恢复默认队列/批次参数，确认同一 GPU 没有其他大进程，并检查是否挂错旧 engine cache。 |
| 重启又构建 engine | 确认仍挂载同一个命名卷；GPU、TensorRT、模型、profile 或精度变化会有意生成新缓存键。 |
| 输出和参考略有差异 | 先确认镜像 ID、GPU 和 health 中的精度；参考结果不是跨 GPU 金标准。 |

不要进入运行容器直接修改代码或模型；那种改动无法可靠复现。开发应在 Git 工作区修改，构建新标签并使用新容器/端口预验。

## 9. 单卡、多卡和容量规划

一个容器当前只拥有一个原生 worker，并使用分配给它的一张 GPU。不要用 `--gpus all` 期待单容器自动跨多卡提速。多卡服务器应每张卡运行一个容器、一个独立端口和一个独立 engine cache：

```bash
for gpu in 0 1 2 3; do
  port=$((8000 + gpu))
  docker volume create "pvr-engine-cache-gpu${gpu}"
  docker run -d --name "pvr-v2-gpu${gpu}" \
    --gpus "device=${gpu}" --restart unless-stopped \
    -p "${port}:8000" \
    -v "pvr-engine-cache-gpu${gpu}:/var/cache/pvr" \
    person-vehicle-recognition:v2.0.0
done
```

结果缓存在各实例内存中，因此负载均衡必须保持“提交 session 的实例”和“查询该 session 的实例”一致。最简单的调用方做法是连同 session ID 保存后端端口；也可使用基于 cookie/一致性哈希的粘性路由。本版本没有跨实例共享结果存储。

现版本不会因检测到多张 A30 就自动跨卡，也不会因 CPU 核心更多就自动增加 native worker。
推荐先按一卡一容器横向扩展并实测每卡，再逐步改造：

- 网关保存或编码 `session_id → GPU 实例` 路由，保证批提交和批查询命中同一后端；
- 需要实例故障转移时，再增加有 TTL 的共享结果存储，而不是让所有容器共享 engine cache；
- 多核 CPU 优先承载分片的视频采集/RTSP 重连、HTTP 摄入和兼容格式解码；
- 双路 CPU/多卡主机再评估 NUMA 绑核、GPU/网卡亲和性和多个摄入进程；
- 每增加并行度都重新验证内存上界、锁竞争、完成吞吐和 p95，不以 CPU 使用率更高为目标。

RTX 3080 Ti 短测只得到约 154 图/s 的饱和完成能力；160 图/s 输入的短测 p95 总延迟约 580 ms。这不是 A30 正式数字，也不是连续 10 分钟发布结果。A30 必须在本机重建 engine 后实测；容量按下式留出 30% 余量：

```text
实例数 = ceil(目标图片/s ÷ (A30 单实例实测完成图片/s × 0.7))
```

## 10. 关机、恢复和持久化

镜像、容器配置、Git 工作区和命名卷不会因正常关机丢失。队列、正在推理的任务和 60 秒结果缓存只存在于内存，关机后旧 session ID 不保证可查询。

确认 Docker 开机启动、容器重启策略和队列水位：

```bash
systemctl is-enabled docker
docker inspect pvr-v2 --format '{{.HostConfig.RestartPolicy.Name}}'
curl -sS http://127.0.0.1:8000/v1/health
```

`unless-stopped` 已配置时可以正常执行 `sudo shutdown -h now`。开机后 Docker 会恢复容器；如果关机前人工执行过 `docker stop pvr-v2`，则运行：

```bash
docker start pvr-v2
curl -fsS http://127.0.0.1:8000/v1/health
```

## 11. 从源码更新并在服务器构建

推荐固定工作区 `/home/ubuntu/pvr-src`。Git 只管理代码；被忽略的 `models/` 和 `vendor/PaddleDetection` 是服务器本机构建资产，普通 `git pull` 不会删除它们。不要执行 `git clean -fdx`。

私有仓库应把服务器 SSH 公钥添加为只读 GitHub Deploy key，然后：

```bash
cd /home/ubuntu/pvr-src
git pull --ff-only origin main

DOCKER_BUILDKIT=1 docker build --progress=plain \
  -f deploy/Dockerfile \
  -t person-vehicle-recognition:v2.0.1 .
```

用新名称、端口和 cache 卷预验，确认后再切换流量；保留 `v1.0` 回滚镜像。不要在共享服务器执行无目标的 `docker system prune -a`。

## 12. 将当前服务器交给接收方验收

当前服务监听服务器 `0.0.0.0:8000`，但应用自身没有登录认证或 TLS。不要直接把 8000
开放给整个公网，也不要共享现有 `ubuntu` 密码、私人 SSH key、GitHub token、sudo、
Docker socket 或模型目录写权限。

推荐流程：

1. 接收方生成自己的 SSH key，并只把 `.pub` 公钥发给管理员；
2. 管理员创建有截止时间的 `pvr-review` 临时账号；
3. 该公钥只允许转发到 `127.0.0.1:8000`，不复用维护者账号；
4. 云安全组的 SSH 22 只允许接收方固定公网 IP；
5. 验收结束删除临时公钥/账号和安全组规则。

接收方建立本地隧道：

```bash
ssh -N -L 18000:127.0.0.1:8000 \
  pvr-review@117.50.173.181
```

在另一个终端体验，不需要登录服务器 shell：

```bash
curl -i http://127.0.0.1:18000/v1/health
python3 test_service.py \
  --server http://127.0.0.1:18000 \
  --output acceptance-results.json
```

应交付给接收方的“服务器验收小套件”：

```text
DEPLOY.md
test_service.py
samples/sample_front_vehicle.jpg
samples/sample_rear_vehicle.jpg
samples/sample_multi_vehicle.jpg
reference_results_rtx3080ti_fp16.json
服务器地址、SSH 端口、临时用户名、主机指纹
访问开始/截止时间、允许用途、问题联系人
```

接收方验收顺序：health ready → 三张样例全部 done → 自有图片测试 → 视频文件/RTSP（如被
授权）→ 查看 `timing_ms` → 记录问题。参考输出仅用于理解字段，不要求跨 GPU 逐字符相等。

仓库访问与服务器访问分开授权。私有 GitHub 仓库应把对方 GitHub 账号加入协作者；若只
需要服务器拉取代码，使用仓库只读 Deploy key。GitHub 不含模型、数据集和 Docker 镜像，
这一点必须提前说明。

管理员还应记录交接时的：Git commit、镜像 ID、外层分享包 SHA-256、GPU、health cache
key、账号创建/撤销时间。长期外部服务应另加 HTTPS、API key/OAuth、速率限制和访问审计，
不能把临时 SSH 隧道当成正式生产网关。

## 13. 使用网盘分享

建议只上传两个外层文件：

```text
pvr-v2.0.0-share-bundle.tar.zst
pvr-v2.0.0-share-bundle.tar.zst.sha256
```

推荐流程：

1. 从构建服务器把两个文件下载到电脑上的专用网盘目录；不要放进 Git 仓库。
2. 等待网盘客户端显示“同步完成”，再创建分享链接。
3. 使用指定接收人、访问密码和有效期；不要建立永久公开链接。
4. 通过另一个通信渠道把外层 SHA-256 发给接收方。接收方下载后先校验，再解包。
5. 接收方完成下载和验收后，按数据与模型授权要求撤销链接或缩短有效期。

Windows PowerShell 下载示例（先自行创建目标目录）：

```powershell
scp ubuntu@SERVER:/home/ubuntu/pvr-v2.0.0-share-bundle.tar.zst `
  'D:\BaiduNetdiskDownload\PVR\'
scp ubuntu@SERVER:/home/ubuntu/pvr-v2.0.0-share-bundle.tar.zst.sha256 `
  'D:\BaiduNetdiskDownload\PVR\'
```

如果网盘限制单文件大小，在服务器拆分并生成分片校验：

```bash
mkdir -p pvr-share-parts
split -b 1900M -d -a 3 \
  pvr-v2.0.0-share-bundle.tar.zst \
  pvr-share-parts/pvr-v2.0.0-share-bundle.tar.zst.part-
sha256sum pvr-share-parts/* > pvr-share-parts/SHA256SUMS
```

接收方先校验分片，再合并并校验外层文件：

```bash
cd pvr-share-parts
sha256sum -c SHA256SUMS
cat pvr-v2.0.0-share-bundle.tar.zst.part-* \
  > ../pvr-v2.0.0-share-bundle.tar.zst
cd ..
sha256sum -c pvr-v2.0.0-share-bundle.tar.zst.sha256
```

某些网盘拦截 `.tar.zst` 后缀时，可以上传前改名为 `.bin`，但必须同时告知接收方下载后恢复原文件名；不要重新压缩已经压缩过的镜像包。
