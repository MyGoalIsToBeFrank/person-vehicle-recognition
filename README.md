# Person–Vehicle Recognition（PVR）v2.0.0

面向固定摄像头抓拍与视频抽帧的行人、人体属性、口罩、车辆属性和车牌识别项目。本仓库同时保留两条边界清晰的链路：

1. **离线研发链路**：原始数据 → 模型预标注 → 人工二次标注 → 离线退化增强 → 三个模型微调 → 模型验收；
2. **生产部署链路**：既有微调权重/官方部署权重 → 7 个动态 ONNX → C++/CUDA/TensorRT worker → 异步 FastAPI → Docker 成品。

当前交付不重新训练模型。`finetune/` 是可审计、可继续使用的离线工具；生产镜像只消费已经确认的权重，构建和运行均不会读取训练数据、启动训练或改写 `finetune/`。

镜像接收、运行、接口、视频使用、服务器验收和排障请同时阅读 [DEPLOY.md](DEPLOY.md)。
公开仓库地址、克隆方式和交付边界见 [PROJECT_LINK.md](PROJECT_LINK.md)。

## 1. 当前可交付状态

| 项目 | 状态 |
|---|---|
| 生产镜像 | `person-vehicle-recognition:v2.0.0` |
| 镜像 ID | `sha256:9aebf7312a1607e07d1c184ec8bebc288488a254dd382d5c485f770b3adc7439` |
| HTTP 服务 | 单实例、异步提交/查询、有界队列、有界结果缓存 |
| 运行精度 | 当前 7 个 TensorRT engine 均为 FP16；INT8 尚未通过逐模型精度闸门，不启用 |
| 车牌模块 | PaddleDetection PP-OCRv3 det + rec，不再使用 HyperLPR |
| 目标硬件 | NVIDIA A30 24GB；必须在目标机首启重建 engine 并实测容量 |
| 离线分享包 | 镜像、`DEPLOY.md`、测试客户端、3 张样例、参考输出、双层 SHA-256 |
| GitHub 内容 | 代码、配置、文档、锁文件；不含原始数据、模型权重和第三方源码树 |

公开源码仓库：<https://github.com/MyGoalIsToBeFrank/person-vehicle-recognition>。仓库不包含
受限数据、模型权重或 Docker 镜像，公开源码不能替代完整部署交付。

项目不预先宣称单卡数千图/s。正式容量必须在目标 A30 上按本文定义连续压测后发布。

## 2. 系统全景

```text
离线数据与微调（不进入生产容器）

dataset/raw（只读）
   ├─ PRW/业务整图 ─→ 人物检测预标注 ─→ WebUI 修框 ─→ detection confirmed
   │                                      └─→ 人物裁片+属性预标注 ─→ WebUI 修属性
   └─ AIZOO ─────────→ 人脸框/口罩标签导入 ─────────────────────→ WebUI 修口罩
                                      ↓
                             离线尘土/昏黄退化增强
                                      ↓
          PP-YOLOE-S 人检 / PP-HGNet 属性 / YOLOv5s 口罩微调
                                      ↓
                         models/finetuned（被 Git 忽略）

生产构建与运行

models/finetuned + 车辆/车牌官方部署权重
               ↓ Docker model-exporter（不训练）
        7 个动态 ONNX + manifest
               ↓
  C++20/CUDA/TensorRT worker（单 GPU、跨请求聚批）
               ↓
 FastAPI：提交图片立即返回 session_id
               ↓
 调用方单查或批量查询 pending/running/done/error
```

最重要的数据所有权原则：

- `dataset/raw/` 永远只读；
- 只有人工确认的 `confirmed/gold` 和训练划分的离线增强副本进入训练；
- Docker 构建只读模型权重，不读取数据集；
- TensorRT engine 不是通用模型文件，按本机 GPU 和运行时生成并保存在命名卷；
- 视频采集、RTSP 重连和抽帧属于独立客户端，不属于推理 worker。

## 3. Git 仓库与本机资产边界

GitHub 仓库不是完整数据/权重归档。以下大文件或受授权约束的目录由 `.gitignore` 排除：

```text
dataset/raw/                 原始图片和上游数据集
dataset/processed/           候选、人工标注、裁片、增强和 COCO 导出
models/                      官方权重、微调 checkpoint、部署模型和 ONNX
vendor/                      PaddleDetection、PaddleClas、YOLOv5 源码树
easy_test/                   本机回归图片
.venv* / .torch-cu130/       本机 Python/CUDA 环境
logs/                        训练、构建、回归和临时日志
```

仓库只保留数据/模型来源索引、标注训练导出服务代码、Dockerfile、依赖锁、不含原图的训练报告和接收方说明。由此分为三种接收权限：

1. **只体验服务**：只需访问部署方提供的受控服务端点和验收资料；
2. **部署镜像**：需要离线分享包，不需要源码和训练数据；
3. **重新构建/微调**：除 Git 仓库外，还必须单独获得有权使用的 `models/`、`vendor/` 和相应数据集。

能够重建生产镜像的工作区必须另行准备 `models/` 和 `vendor/PaddleDetection`；能够重跑
完整三模型微调的工作区还需要训练数据、`vendor/PaddleClas` 和 `vendor/yolov5`。

## 4. 模型清单

| 子任务 | 当前模型 | 权重性质 | Docker 构建是否训练 | TensorRT 输入上限 |
|---|---|---|---|---|
| 行人检测 | 微调 PP-YOLOE-S | `models/finetuned/person_detector_checkpoints` | 否 | batch 64，640×640 |
| 人体属性 | 微调 PP-HGNet small，26 属性 | `models/finetuned/person_attribute` | 否 | ROI batch 128，256×192 |
| 车辆检测 | PP-Vehicle PP-YOLOE-S | 官方部署权重/已验证 ONNX | 否 | batch 64，640×640 |
| 车辆属性 | PP-Vehicle vehicle attribute | 官方部署权重/已验证 ONNX | 否 | ROI batch 128，192×256 |
| 口罩 | 微调 YOLOv5s，两类 | `models/finetuned/face_mask` | 否 | ROI batch 128，640×640 |
| 车牌检测 | PP-OCRv3 DB | PaddleDetection 发布的 CCPD 微调部署权重 | 否 | ROI batch 64，736×736 |
| 车牌识别 | PP-OCRv3 CTC | PaddleDetection 发布的 CCPD 微调部署权重 | 否 | plate batch 128，48×320 |

每个 ONNX 的源文件 SHA-256、预处理、opset、动态 profile、输出语义和语义保持图重写写入镜像内 `/opt/pvr/models/manifest.json`。来源和再分发边界见 [models/MODEL_SOURCES.md](models/MODEL_SOURCES.md)。

## 5. 原始数据和二次标注

数据源见 [dataset/DATASETS.md](dataset/DATASETS.md)。当前记录包括 PRW、AIZOO、MSP60K 和 PA-100K；实际训练只消费本机存在且经人工确认的记录。

### 5.1 数据结构

```text
dataset/
├── DATASETS.md
├── raw/                              # 永远只读
└── processed/
    ├── 1_detection/
    │   ├── candidates.json           # 模型初检人物框
    │   └── confirmed.json            # 人工确认人物框
    ├── 2_attribute/
    │   ├── images/                   # 人物裁片
    │   ├── candidates.json           # 26 属性预标注
    │   └── gold.json                 # 人工确认属性
    ├── 3_mask/
    │   ├── images/                   # 带上下文人脸裁片
    │   ├── candidates.json           # AIZOO 框和原标签
    │   └── gold.json                 # 人工确认口罩标签
    ├── 4_augmented/{detection,attribute,mask}/
    └── 5_export/person_detection_coco/
```

统一 schema 位于 `finetune/dataset_schema.py`，当前版本 3：顶层固定为 `info/images/annotations/categories`，框固定为 COCO `bbox=[x,y,width,height]`，属性固定为 26 项布尔值，口罩类别固定为 `w/o mask`/`w/ mask`。裁片保存 `source_image_id` 和 `source_bbox`；数据写入采用临时文件、`fsync` 和原子替换。

训练/验证/测试按源整图 ID 的 SHA-1 哈希划分：80% train、10% val、10% test。同一原图的所有人物、脸部和增强副本继承同一划分，避免同源泄漏。

### 5.2 生成候选

人物候选：

```powershell
python finetune/prepare_dataset.py `
  --input-dir dataset/raw/prw-download/frames `
  --device GPU --checkpoint-every 50
```

可加 `--limit N` 小规模试跑、`--shuffle-seed 42` 确定性打乱、`--replace` 重建候选。口罩候选从 AIZOO XML 导入：

```powershell
python finetune/import_aizoo.py `
  --input-dir dataset/raw/aizoo --checkpoint-every 100
```

候选只为人工核对提速，本身不是训练金标准。

### 5.3 人工复核 WebUI

```powershell
python finetune/review_server.py --device GPU --seed 42
```

默认地址 `http://127.0.0.1:8765/`。可用 `--host/--port` 修改监听，`--no-browser` 禁止自动开浏览器，`--no-prelabel` 从空属性开始人工填写。

四个阶段：

1. **人物框确认**：整图增删改人物框；确认后生成裁片和 26 项属性候选；
2. **人体属性确认**：修正帽子、眼镜、袖长、服装、鞋靴、包、手持物、年龄、性别和朝向；
3. **口罩确认**：检查带 1.6 倍上下文的人脸裁片和 AIZOO 原标签；
4. **尘土化增强**：选择阶段、副本数、强度和效果，先预览再生成。

保存、翻页和跳转都会先保存当前记录。框不变时保留裁片和标签；框变化时重裁但保留标签供复核；框删除时连带删除属性记录；“排除”只移出候选，不删 raw 原图。

### 5.4 离线退化增强

```powershell
python finetune/dust_augment.py --stage detection --variants 2 --intensity 1.0
python finetune/dust_augment.py --stage attribute --variants 2 --intensity 1.0
python finetune/dust_augment.py --stage mask --variants 2 --intensity 1.0
```

增强模拟尘土、昏黄和雾化，几何和标签不变。它是唯一训练增强来源；只有 train 源图的增强副本进入训练，val/test 保持人工确认原图。

## 6. 三模型微调

操作细节见 [finetune/README.md](finetune/README.md)，历史结果见 [finetune/TRAINING_REPORT.md](finetune/TRAINING_REPORT.md)。离线路径集中在 `finetune/config.py`，不再依赖已删除的 v1 Python 推理目录。

### 6.1 资产和环境

微调机器需另外准备：

```text
models/original/person_detector/mot_ppyoloe_s_36e_pipeline.pdparams
models/human/PPHGNet_small_person_attribute_954_infer/
models/face_mask_yolov5/face_mask_detection.onnx
vendor/PaddleDetection/  vendor/PaddleClas/  vendor/yolov5/
dataset/raw/ 和 dataset/processed/
```

参考训练环境是 Windows、Python 3.10、Paddle GPU 和 CUDA PyTorch 的独立 `.venv-train`。训练依赖清单在 `finetune/requirements.txt`，不能装入生产 `.venv` 或 Docker runtime。Paddle/PyTorch GPU wheel 与 CUDA/驱动强相关，复现前必须记录 GPU、驱动、Paddle、Torch 和 CUDA 版本。

### 6.2 训练命令

```powershell
.venv-train\Scripts\python.exe finetune/train_person_detector.py `
  --device GPU --epochs 10 --batch-size 8 --learning-rate 1e-4
.venv-train\Scripts\python.exe finetune/train_attribute.py `
  --device GPU --epochs 15 --batch-size 32 --learning-rate 1e-4
.venv-train\Scripts\python.exe finetune/train_mask.py `
  --device GPU --epochs 25 --batch-size 16 --learning-rate 1e-4
```

- 人检：confirmed + train 增强副本导出 COCO，从官方 PP-YOLOE-S 继续；
- 属性：gold 人物裁片，从官方 PP-HGNet small 继续，优化 26 维 BCE；
- 口罩：gold 人脸裁片，从官方 ONNX 反建 YOLOv5s 两类模型；
- 属性和口罩裁片启动时预解码进内存；检测整图由 PaddleDetection 多 worker 读取；
- 新训练默认从固定官方起点开始，不会悄悄接续上次微调结果。

中断续训必须显式指定 `--resume` 或 `--checkpoint`，示例见 `finetune/README.md`。输出写入：

```text
models/finetuned/
├── person_detector_checkpoints/
├── person_detector/
├── person_attribute/
└── face_mask/
```

### 6.3 已记录结果

| 环节 | 人工金标准 | 配置 | 已记录最佳指标 |
|---|---:|---|---:|
| 人物检测 | 1328 图 / 6425 框 | PP-YOLOE-S，10 epochs | mAP@0.50:0.95 0.7520；mAP@0.50 0.9450 |
| 人体属性 | 802 裁片 | PP-HGNet small，15 epochs | val macro-F1 0.4922 |
| 口罩 | 1028 裁片 | YOLOv5s，25 epochs | mAP@0.50 0.9949 |

这些是特定数据、划分和环境的历史记录，不代表所有现场或 A30 TensorRT 精度。重新标注、换权重或环境后必须生成新报告。

## 7. 从微调权重到生产 ONNX

Docker 的 model-exporter 阶段执行 `model_export/export_models.py`，不会训练，只会重建/复制 7 个动态 ONNX、运行 ONNX checker、检查动态 batch 和检测器输出、修正 TensorRT 不兼容但语义等价的 Squeeze axes，并生成 `manifest.json`。

Docker 构建要求 `models/` 和 `vendor/PaddleDetection/` 齐全。Git clone 不会自动下载或替换缺失权重。

## 8. 生产推理架构

```text
HTTP 请求校验
  ↓
有界队列（8192 图 + 1 GiB）
  ↓ 2 ms 跨请求聚批
nvJPEG GPU 解码 + 共享 CUDA 预处理
  ├─ stream A：行人检测 → 人体属性 + 头部口罩
  └─ stream B：车辆检测 → 车辆属性 + PP-OCRv3 det/rec
  ↓
GPU NMS / DB / 透视裁切 / CTC argmax
  ↓
60 秒有界结果缓存
```

JPEG 纳入吞吐 SLA；PNG/BMP/WebP 兼容。固定 batch bucket `1/8/16/32/64/128` 尝试 CUDA Graph，失败自动回退 `enqueueV3`。FastAPI 不加载 Paddle、ONNX Runtime 或模型，也没有多 Python 进程复制模型。

### 8.1 已实现的推理优化清单

这些优化共同构成当前性能基线。后续优化不能只比较某个 kernel 的离线耗时，必须同时复测完整识别结果、完成吞吐、p95 总延迟、显存、错误率和过载恢复。

| 层级 | 当前实现 | 目的与维护注意事项 | 代码入口 |
|---|---|---|---|
| HTTP 摄入 | 单图 multipart 在 ASGI middleware 中流式限长；批请求先检查 Content-Length，再按 64 MiB 硬上限读取；摄入和 Python→原生提交分别使用 semaphore | 在图片进入原生层前阻止大请求并发耗尽内存；修改限制时必须同时检查 FastAPI 和 native 两层 | `service/app.py`、`service/settings.py` |
| 批协议 | `PVRB` 版本化 little-endian 长度前缀协议，声明媒体类型必须匹配文件签名 | 避免 multipart 在高吞吐时产生大量解析/请求开销；协议字段变化必须升级 version | `pvr_api/protocol.py` |
| 原生队列 | 图片数、总字节数和结果记录数三重 admission；一个批次要么全部接收，要么整体 429；payload 移入共享 buffer，不再跨进程复制 | 提供确定的背压和内存上界；不能靠无限增大队列掩盖算力不足 | `native/src/engine.cpp` |
| 动态聚批 | worker 取到第一项后最多等待 2 ms，凑到 64 立即执行；跨多个 HTTP 请求合并 | 在低延迟和 GPU 利用率之间折中；修改窗口必须同时扫吞吐和 p95 | `Engine::take_batch` |
| 结果生命周期 | `pending→running→done/error→expired`；按 TTL、结果字节和记录数回收，过期后保留短期 tombstone 以区分 404/410 | 防止结果缓存无限增长；调用方必须在 TTL 内查询 | `Engine::store_batch/expire_locked` |
| JPEG 解码 | 先解析并验证尺寸，使用 `nvjpegDecodeBatched` 直接输出 GPU RGB；批解码失败时重建 decoder state，再逐图隔离坏 JPEG | 坏图不毒害同批有效图；逐图回退只在批调用失败时发生 | `native/src/image_decoder.cpp` |
| 兼容格式 | PNG/BMP 由 stb、WebP 由 libwebp 在 CPU 解码后异步 H2D | 保证兼容性，不属于 JPEG 千图吞吐路径 | `ImageDecoder::decode_compat_one` |
| 高水位显存复用 | 解码槽、输入、输出、ROI、NMS 和 token buffer 只在历史最大需求增长时扩容，后续批次复用地址 | 减少 cudaMalloc 抖动，也是 CUDA Graph 地址稳定的前提；压测后空闲显存不会立即回落属于预期 | `DeviceBuffer`、`decoded_pool_`、各 pipeline buffer |
| 共享预处理 | 原图只做一次 640×640 RGB resize，产生检测 tensor 和 scale factor，行人/车辆检测共同使用 | 避免两个检测器重复 resize/颜色转换；两模型预处理契约变化时不能继续盲目共享 | `Pipeline::run`、`launch_resize_normalize` |
| 双流检测 | 共享输入就绪后用 CUDA event 同步，行人检测在 primary stream、车辆检测在 secondary stream 并发 | 隐藏两个大检测器的串行时间；改变 context 内存布局或 stream 依赖时要用事件证明无数据竞争 | `Pipeline::run` |
| GPU 检测后处理 | 候选筛选和 NMS 每图一个 CUDA block，最多保留固定数量；CPU 只回收小体积框和计数 | 不把全部 anchor scores/boxes 回传 CPU | `filter_nms_kernel` |
| ROI 跨图聚批 | 同一原图批次内所有人物、头部、车辆、车牌裁片分别汇总后再跑属性/口罩/OCR；不是逐目标调用模型 | 目标多时显著减少 launch 和小 batch；任何逐框同步循环都是性能回退 | `Pipeline::attributes/masks/plates` |
| GPU ROI 与透视 | resize/颜色/归一化、车辆/人体裁片、车牌四边形单应性透视都在 CUDA 完成 | 避免 OpenCV CPU crop 和大像素回传 | `resize_normalize_kernel`、`warp_quad_normalize_kernel` |
| GPU OCR 后处理 | DB 概率图上生成车牌四边形；识别输出在 GPU 做 CTC timestep argmax/去重，只回传 token ID 和长度 | CPU 仅进行字典查表与字符串组装 | `plate_quads_kernel`、`ctc_argmax_kernel` |
| Engine 首启顺序 | 先逐个完成 7 个 engine 的 build/cache，再创建任何常驻 execution context | 防止前面 context 占用显存导致后续 builder tactic 因 workspace 不足退化 | `Pipeline::Impl` 构造函数 |
| TensorRT 构建 | builder optimization level 5；总显存 ≥20 GiB 使用 12 GiB workspace，否则 6 GiB；当前 FP16 | tactic 选择与显存相关；改 builder、workspace 或精度会改变 cache key，必须重新做精度/吞吐验收 | `TensorRtModel::load_or_build` |
| 动态 profile | 检测 min/opt/max batch 为 1/32/64；属性、口罩、OCR 最大 64 或 128，shape 固定 | profile 过宽可能损失 tactic，过窄会拒绝业务 batch；变更需同步 manifest 和客户端上限 | `model_export/export_models.py`、`native/src/pipeline.cpp` |
| Engine cache | key 包含 GPU 名称/SM、TensorRT 精确补丁版本、精度、ONNX SHA、builder level、workspace 和 profile hash；原子写入；反序列化失败删除后重建 | 禁止跨不兼容 GPU 偷复用；共享镜像不包含 engine | `native/src/tensorrt_model.cpp` |
| 用户管理 context 显存 | 行人/车辆两个并发 detector 各自一个 pool；严格串行的属性、口罩、plate det/rec 共用一个“最大需求”pool | 避免 7 个 context 各自常驻最大激活内存；若把串行模型改为并发，必须先拆 pool | `bind_context_memory`、`Pipeline::Impl` |
| CUDA Graph | 只捕获 1/8/16/32/64/128 且输入输出地址、shape、dtype 完全一致的执行；每个模型最多缓存 32 个 graph；不支持或实例化失败永久回退普通 `enqueueV3` | 防止旧地址 graph 导致错误；不能为了命中 graph 固定填充虚假图片 | `TensorRtModel::infer` |
| 可观测性 | health 暴露 GPU、精度、7 个 cache key、context pool、队列/结果水位；Prometheus 暴露 accepted/completed/rejected/error、batch、延迟和显存 | 每次性能改动必须保存 health、metrics 和 benchmark JSON，形成可比较证据 | `Engine::prometheus`、`service/app.py` |

### 8.2 后续优化的安全顺序

建议按以下顺序改动，避免优化无从归因：

1. 固定镜像、模型哈希、测试图片和 GPU，分别记录离线 pipeline 与 HTTP 端到端基线；
2. 一次只改变一个维度：batch/profile、stream、context pool、CUDA Graph、精度或 kernel；
3. 扫 batch `1/8/16/32/64`，记录完成吞吐、queue/inference/total p50/p95/p99、GPU 利用率和显存；
4. 对 43 图回归和各任务金标准运行精度闸门；
5. 连续 10 分钟稳定负载，再做两倍过载和恢复；
6. 只有完整识别结果和资源边界都通过，才替换生产镜像标签。

不要把“提交更快”“单个模型更快”“短时峰值更高”当作完整识别服务优化完成。

### 8.3 A30 多卡和更多 CPU 核心的下一步

当前容器不会自动把一个 batch 分散到多张 GPU，也不会因 CPU 核心增加而创建多个 native
worker。近期最稳妥的扩展是**一张 A30 一个容器**：每个容器固定 `device=N`、独立端口和
独立 engine cache，再由网关按实例分发；提交后必须把该 session 的查询粘到同一实例。

建议的后续代码方向是让 session ID/提交响应携带不可伪造的实例路由标识，网关据此转发
查询；需要跨实例容错时再引入有 TTL 的共享结果层。不要先写一个进程内“自动多 GPU
manager”，它会把 engine 生命周期、显存故障和背压耦合在一起，通常不如独立实例清楚。

CPU 核心增加主要可用于视频采集/重连、HTTP 解析、PNG/BMP/WebP 兼容解码、结果序列化和
多 GPU 实例摄入；JPEG/TensorRT 热路径仍以 GPU 为主。优化方向是把视频采集做成可分片的
独立 worker 池，对摄入/兼容解码做有界并行，并在双路 CPU 服务器上测试 NUMA 绑核、网卡
RSS 和 GPU 亲和性。任何加线程方案都必须证明没有增加内存上界、锁竞争和 p95 延迟。

## 9. 图片异步 API

提交单图：

```bash
curl -i -F 'file=@sample.jpg;type=image/jpeg' http://127.0.0.1:8000/v1/tasks
```

HTTP 202：

```json
{"session_id":"69548ba00d8fe6dae8109bf33b047512","status":"pending"}
```

session ID 是服务接收图片后返回的任务 ID，不是调用方提前生成。随后独立查询：

```bash
curl http://127.0.0.1:8000/v1/tasks/69548ba00d8fe6dae8109bf33b047512
```

完成结果：

```json
{
  "session_id":"69548ba00d8fe6dae8109bf33b047512",
  "status":"done",
  "result":{"行人":[],"车辆":[{"颜色":"白色","车型":"轿车","车牌":"皖D5H594"}]},
  "timing_ms":{"queue":2.1,"inference":113.4,"total":115.5}
}
```

高吞吐使用 `POST /v1/task-batches` 的 PVRB 二进制协议，一次最多 64 图；`POST /v1/results:batch` 一次查询最多 512 个 ID。可直接使用零依赖客户端：

```bash
python3 deploy/test_service.py --server http://127.0.0.1:8000 \
  image-1.jpg image-2.jpg --output actual-results.json
```

默认限制：单图 8 MiB/20 MP、批请求 64 图/64 MiB、队列 8192 图/1 GiB、结果 TTL 60 秒/1 GiB/262144 条。队列满立即 429，不会无限堆积。

## 10. 视频与 RTSP

推理服务不直接接收视频流。正确流程：

```text
RTSP / 摄像头 / 视频文件
  ↓ 独立 video-client：读取、按 sample_fps 抽帧、JPEG 编码
POST /v1/task-batches（最多 64 帧）
  ↓ 每个帧一个 session_id
POST /v1/results:batch
  ↓
camera_id + frame_index + timestamp_ms + 结果 → JSONL
```

一个视频没有一个总 session ID；每个被采样帧都有独立 ID。客户端保存 `session_id → camera_id/frame_index/timestamp_ms`。

同一镜像用不同容器分别运行服务和采集客户端：

```bash
# GPU 服务
docker run -d --name pvr-v2 --gpus '"device=0"' \
  --restart unless-stopped -p 8000:8000 \
  -v pvr-engine-cache:/var/cache/pvr \
  person-vehicle-recognition:v2.0.0

# 单摄像头采集，不申请 GPU
mkdir -p output
docker run --rm --network host -v "$PWD/output:/output" \
  person-vehicle-recognition:v2.0.0 video-client \
  --camera-id gate-1 \
  --source 'rtsp://user:password@camera/stream' \
  --sample-fps 2 --batch-size 32 --max-pending 4096 \
  --server http://127.0.0.1:8000 \
  --output /output/gate-1.jsonl
```

`--source` 可为 RTSP URL、本地视频或摄像头编号。429 时当前批次记为 dropped，不无限缓存。当前客户端是基础采样器；大量实时流前应增加 RTSP 自动重连、独立采集/提交线程、有界最新帧缓冲、真实墙钟时间、可靠结果存储和粘性路由，但仍应与 TensorRT 服务分离。

## 11. Docker 构建和启动

```bash
DOCKER_BUILDKIT=1 docker build --progress=plain \
  -f deploy/Dockerfile -t person-vehicle-recognition:v2.0.0 .
docker volume create pvr-engine-cache
docker run -d --name pvr-v2 --gpus '"device=0"' \
  --restart unless-stopped -p 8000:8000 \
  -v pvr-engine-cache:/var/cache/pvr \
  person-vehicle-recognition:v2.0.0
```

三阶段构建：Paddle 3.2.1 导出 ONNX；TensorRT 10.5/CUDA 12.6 编译 C++20/CUDA；CUDA 12.6.3/cuDNN 9.5 runtime 安装 HTTP/客户端依赖。Python 依赖使用 `uv 0.11.2` 和锁文件，清华 PyPI、中科大 apt、DaoCloud 镜像加速；Dockerfile 不调用 `pip`。

首次启动针对本机 GPU 在 `/var/cache/pvr` 构建 engine，期间 health 为 503；全部完成后才返回 HTTP 200、`ready=true`。

## 12. 验证和吞吐

```bash
uv run pytest
curl -i http://127.0.0.1:8000/v1/health
curl http://127.0.0.1:8000/metrics
python3 deploy/test_service.py --server http://127.0.0.1:8000 sample.jpg
```

正式吞吐是连续 10 分钟内同时满足：有效图错误 0、完成/接受 ≥99.9%、p95 ≤1 秒、无 OOM、稳定负载无 429 的最高完成速率；随后以两倍负载验证快速 429 和恢复。A30 数量按：

```text
ceil(目标图片/s ÷ (A30 单实例实测完成图片/s × 0.7))
```

## 13. 受控远程验收

若部署方提供远程验收，不要共享维护账号密码、个人私钥、GitHub token 或 sudo。应创建
有期限的独立验收账号，只允许 SSH 本地端口转发到服务监听地址：

```bash
ssh -N -L 18000:127.0.0.1:8000 REVIEW_USER@DEPLOY_HOST
```

接收方另开终端：

```bash
curl http://127.0.0.1:18000/v1/health
python3 test_service.py --server http://127.0.0.1:18000
```

给接收方：`DEPLOY.md`、`test_service.py`、`samples/`、参考输出，以及单独安全发送的端点、
临时用户名、SSH 主机指纹、验收期限和联系人。服务体验不需要 Docker socket、模型目录
写权限或训练数据。

源码仓库公开可读；部署端点仍必须独立授权。当前服务没有应用层认证，不能把 8000
无条件暴露公网；临时体验优先使用受限 SSH 隧道，长期部署再加 TLS、认证、限流和审计。

## 14. 离线镜像分享

```text
pvr-v2.0.0-handover.tar.zst
pvr-v2.0.0-handover.tar.zst.sha256
```

外层 SHA-256 以同目录 `.sha256` sidecar 为唯一准值；README 被收入外层包，因此不能在包内
硬编码外层包自身的摘要。接收方先校验外层 SHA，再解包并校验 `MANIFEST.sha256`，最后
`docker load`。详见 [DEPLOY.md](DEPLOY.md)。

## 15. 目录职责

```text
dataset/       数据来源；真实 raw/processed 被 Git 忽略
finetune/      二次标注、schema、增强、三模型微调和训练报告
models/        模型来源；真实权重被 Git 忽略
model_export/  既有权重到动态 ONNX、checker 和 manifest
native/        C++20/CUDA/TensorRT worker
service/       FastAPI 异步边界
pvr_api/       PVRB 批协议
client/        视频/RTSP 抽帧客户端
benchmarks/    HTTP 压测与回归
deploy/        Dockerfile、入口、测试和维护手册
tests/         协议与服务边界测试
```

## 16. 已知边界

- 单容器只使用一张 GPU；多卡是一卡一容器；
- 结果在单实例内存，提交与查询必须命中同一实例；
- TTL 默认 60 秒，容器重启后未确认结果不持久化；
- 视频客户端当前每进程一个 source，尚无自动重连和持久任务存储；
- 输出不包含检测框坐标；INT8 尚未通过精度闸门；
- 任意开发卡短测都不能替代目标 A30 正式测试；
- GitHub 不包含受限数据和权重；权重再分发必须单独审查。

## 17. 文档索引

- [DEPLOY.md](DEPLOY.md)：镜像接收、API、视频、远程验收、调试、多卡、关机和网盘；
- [deploy/DOCKER_TUTORIAL.md](deploy/DOCKER_TUTORIAL.md)：构建主机和 Docker 维护细节；
- [service/README.md](service/README.md)：HTTP/PVRB 协议；
- [finetune/README.md](finetune/README.md)：离线标注与训练；
- [finetune/TRAINING_REPORT.md](finetune/TRAINING_REPORT.md)：历史训练数据量和指标；
- [dataset/DATASETS.md](dataset/DATASETS.md)：数据来源；
- [models/MODEL_SOURCES.md](models/MODEL_SOURCES.md)：模型来源、哈希和授权边界。
