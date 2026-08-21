# 三模型人工二次标注与金标准微调

这里是完全离线的研发工具链：用既有模型生成候选，经 WebUI 人工二次标注形成
金标准，再分别微调人物检测、人物属性和口罩识别。车辆检测、车辆属性和车牌
PP-OCRv3 不在本目录训练。

离线路径集中在 `finetune/config.py`；候选预标注只使用
`finetune/prelabel_models.py`。二者不会被生产服务或 Docker runtime 导入，已经删除的
v1 Python 推理管线也不会作为训练依赖恢复。原部署模型和人物检测训练权重保存在
`models/original/`，训练输出写入 `models/finetuned/`，不会覆盖原始起点。

当前交付默认使用已经存在的微调权重，不要求也不会在 Docker 构建中重新训练。
只有需要扩充数据或产生新权重时才运行本文命令。

## 数据布局

`dataset/raw/` 只读；其余全部中间产物在 `dataset/processed/` 下按阶段编号
（详见根目录 README.md）：

- `1_detection/`：人物框候选与人工确认（整图级，无标签）
- `2_attribute/`：人物裁剪小图、属性模型预标注、人工金标准
- `3_mask/`：AIZOO 人脸框候选、带上下文人脸裁剪、人工金标准
- `4_augmented/`：尘土化/昏黄化扩充，标签与源一致
- `5_export/person_detection_coco/`：人物检测训练用 COCO

所有 JSON 为 COCO 风格（`images/annotations/categories`，`bbox=[x,y,w,h]`），
裁剪图带 `source_image_id` 以继承 train/val/test 划分。

`dataset_schema.py` 当前 schema version 为 3。ID 由源路径/源标注确定性生成；
保存采用同目录临时文件、`fsync` 和原子替换。`dataset/raw/` 永远只读，候选、裁片、
人工金标准、增强和 COCO 导出全部位于 `dataset/processed/`。

## 0. 环境和本机资产

Git 不保存训练数据、模型权重或第三方源码。完整训练机器必须具有：

```text
dataset/raw/ 与 dataset/processed/
models/original/ 与 models/finetuned/
vendor/PaddleDetection/
vendor/PaddleClas/
vendor/yolov5/
```

参考环境是 Windows、Python 3.10、Paddle GPU 与 CUDA PyTorch 的独立
`.venv-train`。依赖清单见 `finetune/requirements.txt`，不要安装进生产 `.venv`，也不
进入 Docker runtime。GPU wheel 与驱动/CUDA 强相关；复现实验时必须先记录
`nvidia-smi`、Python、Paddle、Torch、CUDA/cuDNN 和三个 vendor commit。

配置检查：

```powershell
.venv-train\Scripts\python.exe -X utf8 -c "import sys; sys.path.insert(0,'finetune'); import config; print(config.PROJECT_ROOT); print(config.TRAINING_OUTPUT_DIR)"
```

候选生成和 WebUI 默认重入项目 `.venv`，训练脚本使用 `.venv-train`。环境只由 `uv`
维护，不要直接运行系统 `pip` 改写已经验证的环境。

## 1. 生成候选

人物检测候选（只跑检测模型；`--replace` 只重建检测候选，已确认与金标准不受影响）：

```powershell
python finetune/prepare_dataset.py --input-dir dataset/raw/prw-download/frames --device GPU
```

口罩候选从 AIZOO 的原图、原框和原标签导入：

```powershell
python finetune/import_aizoo.py
```

## 2. 人工核对与尘土化（WebUI）

```powershell
python finetune/review_server.py            # 默认开启属性预标注
python finetune/review_server.py --seed 42  # 可选：固定随机出图顺序，便于按同顺序续标
```

界面四个标签页，对应管线四个阶段：

- **① 目标识别**：整图上增删改人物框（不涉及标签）。确认整图时自动生成人物裁剪，
  并用属性模型逐框预标注（`--no-prelabel` 可关闭，关闭后属性候选标签为空需全手工补全）。
- **② 属性标注**：只看人物裁剪小图；模型预标注已填好 26 项属性，检查更改后保存。
  「查看原图」可弹出整图上下文。
- **③ 口罩标注**：只看带 1.6 倍上下文的人脸裁剪；AIZOO 类别已预填，确认或改选。
- **④ 尘土化**：选择环节、副本数、强度与效果开关，可先预览再生成；产物写入
  `4_augmented/`，每次生成重建该环节整个增强集。

保存、翻页、跳转都会先保存当前记录。候选按**随机顺序**出图（避免同一摄像头连拍簇集中），
顺序在会话内稳定，`--seed` 可固定。每个核对页工具栏有「待核对 / 回改已保存」切换：
回改模式按最近保存在前排列，重新保存即原地更新（可修正标错的记录）；检测页重新确认时
按框 id 对齐重建属性裁剪——框未动标签保留，框动了重裁剪但保留标签，框删除则连同其
属性裁剪与标注一并移除。「排除」只从候选移除，不写金标准，也不删除 raw 原图。

需要从头重新标注时，优先在 WebUI 的「回改已保存」逐条修正。整体删除
`confirmed.json`、`gold.json` 或 `4_augmented/` 会丢失人工工作，执行前必须做独立备份并
明确核对绝对路径；候选文件与 raw 原图是不同的数据所有者，不能用递归清理一起删除。

## 3. 微调三个模型

训练入口只读取 confirmed/gold 与 `4_augmented/` 中源图属于训练划分的样本：

```powershell
.venv-train\Scripts\python.exe finetune/train_person_detector.py --device GPU --epochs 10
.venv-train\Scripts\python.exe finetune/train_attribute.py --device GPU --epochs 15
.venv-train\Scripts\python.exe finetune/train_mask.py --device GPU --epochs 25
```

三个脚本每次都**从官方权重重新开始**（起点固定在代码里，不会以上一轮微调结果为起点），
训练完成直接覆盖导出到 `models/finetuned/{person_detector, person_attribute, face_mask}`：

- 人物检测：人工确认整图与人物框（含增强训练集）导出为 COCO，从官方
  `mot_ppyoloe_s_36e_pipeline.pdparams` 继续训练 PP-YOLOE-S（数据量大，约 10 epochs 足够）。
- 人物属性：使用最终人物裁剪，从官方 `PPHGNet_small_person_attribute_954_infer` 继续训练（15 epochs）。
- 口罩识别：使用带上下文的人脸裁剪，从官方 `face_mask_detection.onnx` 转回 YOLOv5s 继续训练（25 epochs，
  数据量小、收敛快）。

训练侧不做任何在线增强（随机裁剪、缩放、翻转等全部移除），增强只来自离线
尘土化副本；验证集不增强。数据按源整图 id 哈希划分，同图所有目标
与增强副本不会跨训练/验证集。属性和口罩的 Dataset 在启动时把全部裁剪小图一次性解码进
内存缓存（百 MB 级），训练期不再读盘；检测数据是整图、量大，用
`worker_num=12` 的多进程 DataLoader 读盘（Windows 下多 worker 不共享内存，不做 RAM 缓存）。

中断续训（checkpoint 均已落盘）：

```powershell
# 检测：--resume 必须给 PaddleDetection 能解析的绝对路径
.venv-train\Scripts\python.exe finetune/train_person_detector.py --device GPU --epochs 10 --resume "B:/绝对路径/models/finetuned/person_detector_checkpoints/8.pdparams"
# 属性 / 口罩：--checkpoint 指向上次保存的权重
.venv-train\Scripts\python.exe finetune/train_attribute.py --device GPU --epochs 15 --checkpoint models/finetuned/person_attribute/last.pdparams
.venv-train\Scripts\python.exe finetune/train_mask.py --device GPU --epochs 25 --checkpoint models/finetuned/face_mask/last.pt
```

人物检测训练结束会自动导出部署模型（PIR 格式 `model.json` + `model.pdiparams`）。
若导出步骤报 `sigmoid(): argument must be Value`，按 `train_person_detector.py`
注释里的手动命令用 `env -u FLAGS_enable_pir_api` 重跑一次即可。

## 4. 将新权重接入生产服务

生产服务没有运行时模型目录开关，也不使用旧 `inference/run.py`。新权重必须经过完整、
可审计的导出与镜像重建：

1. 人物检测最终 checkpoint 位于
   `models/finetuned/person_detector_checkpoints/model_final.*`；
2. 人体属性最佳权重和部署图位于 `models/finetuned/person_attribute/`；
3. 口罩最佳权重及 dynamic-batch ONNX 位于 `models/finetuned/face_mask/`；
4. `model_export/export_models.py` 从这些固定路径生成动态 ONNX，并记录源 SHA-256、
   预处理、profile、opset 和输出语义；
5. `deploy/Dockerfile` 在 model-exporter 阶段运行 ONNX checker，随后编译原生服务；
6. 新镜像使用新标签和新 engine cache 卷预验，不能覆盖正在服务的 v2.0.0；
7. 通过逐模型精度闸门、43 图端到端回归、10 分钟稳定负载和两倍过载恢复后才切换。

```bash
DOCKER_BUILDKIT=1 docker build --progress=plain \
  -f deploy/Dockerfile \
  -t person-vehicle-recognition:v2.0.1 .

docker volume create pvr-engine-cache-v201
docker run -d --name pvr-v201 --gpus '"device=0"' \
  -p 8001:8000 \
  -v pvr-engine-cache-v201:/var/cache/pvr \
  person-vehicle-recognition:v2.0.1
```

TensorRT engine 不能从旧镜像或另一张 GPU 复制。新模型改变 ONNX SHA 后会自然生成新的
cache key。回滚使用旧镜像标签及其原 engine cache 卷，不通过修改运行容器内文件实现。

## 5. 训练记录和再现要求

历史数据量、曲线和最佳指标见 `TRAINING_REPORT.md`。这些数字只对应当时的数据、划分、
起点权重和环境。重新标注、改变增强、超参数或依赖后，应保存三份原始训练日志并重新运行：

```powershell
.venv-train\Scripts\python.exe finetune/training_report.py `
  --det-log logs/person-det.log `
  --attr-log logs/person-attr.log `
  --mask-log logs/mask.log `
  --output finetune/TRAINING_REPORT.md
```

报告生成不等于精度闸门通过。生产发布仍需在部署 GPU 上比较 FP32/FP16/候选 INT8 的业务
输出，人工复核所有差异，并保留模型文件 SHA-256、镜像 ID、engine cache key 和测试集版本。
