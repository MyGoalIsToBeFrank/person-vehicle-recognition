# 高吞吐行人/车辆识别服务

本仓库把现有微调权重部署为单实例、异步的 CUDA/TensorRT 识别服务。`finetune/` 只作为权重与历史实验来源；构建和运行不会训练或修改任何模型。

## 数据路径

1. FastAPI 校验请求大小和格式，把图片交给有界原生队列，并立即返回 `202 + session_id`。
2. 单个 C++20/CUDA worker 在最多 2ms 的窗口内跨请求聚批。
3. JPEG 由 nvJPEG 批量解码到 GPU；resize、颜色转换、归一化、ROI、NMS、DB 连通域、车牌透视裁切、属性后处理和 OCR token argmax 都在 CUDA 中完成。
4. 两个 PP-YOLOE 检测器共享预处理并使用双 CUDA stream；属性、口罩和 PP-OCRv3 det/rec 对全部 ROI 聚批。
5. 完成结果保留 60 秒，通过单个或批量结果接口获取。队列满立即返回 429，不允许请求无限积压。

FastAPI 只负责网络协议和状态码，不加载 Paddle、ONNX Runtime 或模型。进程内只有一个原生 worker，不使用 Python 多进程搬运图片。

## 模型

- 行人检测、人体属性、车辆检测、车辆属性、口罩：继续使用仓库已有权重。
- 车牌：固定为 PaddleDetection 发布的 PP-OCRv3 det + rec。
- Docker 构建从既有 checkpoint/部署权重生成动态 ONNX；不训练模型。
- TensorRT engine 不进入共享镜像。首次启动按 GPU 名称、SM、TensorRT 精确版本、精度、builder 优化级别、workspace、ONNX SHA-256 和 profile 哈希写入 `/var/cache/pvr`。
- TensorRT 在稳定输入地址上为固定批次 `1/8/16/32/64/128` 捕获 CUDA Graph；动态小批次或捕获不受支持时自动使用普通 `enqueueV3`，不牺牲正确性。
- 两个并发检测器各有独立 TensorRT 激活内存池；严格串行的属性、口罩、车牌 det/rec 共享一个按最大需求分配的常驻池，避免 7 个 context 重复占用最大 profile 显存。
- 当前默认精度为 FP16。INT8 只有通过逐模型精度闸门后才会启用，不能仅为吞吐数字牺牲结果。

每个 ONNX 的来源哈希、预处理、profile、输出语义和语义保持图重写均记录在镜像内的 `models/manifest.json`。来源与再分发提醒见 [models/MODEL_SOURCES.md](models/MODEL_SOURCES.md)。

## 快速运行

```bash
DOCKER_BUILDKIT=1 docker build \
  -f deploy/Dockerfile \
  -t person-vehicle-recognition:v2.0.0 .

docker run -d --name pvr-v2 --gpus all \
  --restart unless-stopped -p 8000:8000 \
  -v pvr-engine-cache:/var/cache/pvr \
  person-vehicle-recognition:v2.0.0
```

首次启动构建本机 engine，期间 `/v1/health` 返回 503；完成后才进入 ready。项目维护者的构建细节见 [deploy/DOCKER_TUTORIAL.md](deploy/DOCKER_TUTORIAL.md)；镜像接收方应使用包含接口、部署、样例验收和排障说明的 [DEPLOY.md](DEPLOY.md)。

## API

```bash
# 提交后立即返回 session_id
curl -F 'file=@sample.jpg;type=image/jpeg' http://127.0.0.1:8000/v1/tasks

# 独立查询结果
curl http://127.0.0.1:8000/v1/tasks/<session_id>
```

高吞吐客户端使用 `POST /v1/task-batches` 的版本化长度前缀二进制协议，一次最多 64 张；`POST /v1/results:batch` 一次查询最多 512 个 ID。完整协议、限制和状态码见 [service/README.md](service/README.md)。

## 验证原则

发布吞吐是连续 10 分钟内满足以下条件的最高完成速率：有效图片错误率为 0、完成/接受至少 99.9%、p95 总延迟不超过 1 秒、无 OOM、稳定负载不出现 429。另以两倍负载验证快速 429、内存有界和恢复能力。

RTX 3080 Ti 只用于镜像、功能、稳定性和相对优化预验。A30 24GB 的最终数字必须在 A30 上重新构建 engine 后实测；仓库不会预先宣称单卡“数千张/秒”。实例数按：

```text
ceil(目标吞吐 / (单实例实测完成吞吐 × 0.7))
```

2026-08-20 的 3080 Ti 短测得到约 154 图/s 的饱和完成能力；160 图/s 固定输入短扫的 p95 约 580ms，但 43 图严格回归仍有 7 图差异，因此它既不是 10 分钟发布值，也不能外推为 A30 精度或吞吐结论。完整记录见部署手册的“本次 RTX 3080 Ti 预验记录”。

## 目录

```text
model_export/  既有权重到动态 ONNX、checker 和 manifest
native/        C++20/CUDA/TensorRT worker
service/       FastAPI 薄接口
pvr_api/       批协议与共享类型
client/        视频抽帧批提交客户端
benchmarks/    HTTP 端到端验收工具
deploy/        唯一 Dockerfile、入口与部署文档
tests/         协议和服务边界测试
```

Python 依赖固定使用 `uv 0.11.2` 和锁文件，PyPI 默认清华源；构建过程不调用 `pip`。Ubuntu apt 使用中科大源。
