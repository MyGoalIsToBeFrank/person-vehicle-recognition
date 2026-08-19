# 二次标注、微调与推理管线

从既有数据集（AIZOO 口罩、PRW 全景）出发，经人工核对产出金标准，微调人物检测、
人物属性、口罩识别三个模型，最后在 `inference/` 推理应用端一键切换模型。

项目按职责分三段，对应三个入口文档：

| 段 | 做什么 | 代码位置 | 详细文档 |
| --- | --- | --- | --- |
| **一、二次标注** | 既有数据集 → 模型初检/预标注 → 人工核对出金标准 | `finetune/prepare_dataset.py`、`finetune/import_aizoo.py`、`finetune/review_server.py` | [finetune/README.md](finetune/README.md) |
| **二、微调** | 金标准 + 尘土化扩充 → 训练三个模型 → 训练报告 | `finetune/train_*.py`、`finetune/dust_augment.py`、`finetune/training_report.py` | [finetune/README.md](finetune/README.md) |
| **三、推理** | 用模型跑图片目录，产出结构化结果 | `inference/`（精简应用端） | [inference/README.md](inference/README.md) |

## 一、二次标注（四个阶段）

```
raw 原图 ──► ① 目标识别 ──► ② 属性标注 ──► ③ 口罩标注 ──► ④ 尘土化 ──► 训练
             整图核框        小图核属性      小图核口罩      标签不变的扩充
             （模型初检）     （模型预标注）   （AIZOO 预填）
```

- **① 目标识别**：`prepare_dataset.py` 用当前人物检测模型对整图初检；WebUI 里人工增删改框
  （不看标签）。确认整图时自动把每个人物框裁成小图，并用属性模型预标注。
- **② 属性标注**：只看人物裁剪小图，26 项属性已由模型预填，人工检查后保存。
- **③ 口罩标注**：AIZOO 人脸框与类别由 `import_aizoo.py` 直接导入；WebUI 里只看带上下文的
  人脸裁剪小图，确认或更改类别。候选基本正确时可批量批注：
  `python finetune/batch_approve_mask.py --count 800`（沿用 AIZOO 原标签）。
- **④ 尘土化**：对任一环节的金标准做昏黄化、尘土化等退化扩充，**几何与标签不变**；
  增强副本按源图哈希继承 train/val/test 划分，且只有训练划分的副本参与训练。

四个阶段都有 WebUI 标签页（`finetune/review_server.py`），全程只需检查与更改。

- **随机出图**：候选按随机顺序呈现（原始图片高度相关，顺序出图会集中在少数固定摄像头
  的连拍簇上）。顺序在一次会话内稳定；`--seed N` 可固定顺序，便于中断后按同样顺序继续。
- **回改已保存**：每个核对标签页工具栏有「待核对 / 回改已保存」切换。回改模式按
  最近保存在前排列，改完重新保存即原地更新（候选数量不再减少）。检测页重新确认整图时
  按框 id 对齐重建属性裁剪：框未动 → 裁剪与标签全保留；框动了 → 重裁剪但保留标签；
  框被删 → 对应裁剪与属性标注（含金标准）一并移除。
- **快捷键**：Q/E 或 ←/→ 保存并切换上/下一张；Enter 或 Ctrl+S 保存；检测页 A/D 切框、
  N 新增框、Delete 删框；口罩页 1/2 选未佩戴/佩戴。

### 数据量建议（参考阈值，非硬性规则）

WebUI 标签按钮实时显示各环节「待核对 / 已保存」数量，标题栏会提示还差多少到建议起步量。
按源图哈希自动划分（train 80% / val 10% / test 10%），数量太少会导致验证集为空无法训练：

- **① 人物检测**：建议先积累 **≥500 张确认整图**再首次训练（验证集约 50 张）；1000+ 更稳。
- **② 人物属性**：26 项多标签，建议 **≥800 个人物裁剪**起步；长尾属性（帽子、裙装、
  手持物品等）若样本太少，宁可先只训练有代表性的属性，或优先标注包含这些属性的图。
- **③ 口罩识别**：AIZOO 已自带标签、核对量大，**≥800 个人脸裁剪**很快可达；
  注意保持「佩戴/未佩戴」两类大致均衡。
- **④ 尘土化**：在金标准达到一定量后再做，增强只放大训练集，不能替代真实标注多样性；
  建议增强副本数（variants）不超过 2–3，避免模型过拟合增强风格。

### 标注常用命令

```powershell
# ① 生成/补充检测候选（只跑人物检测模型）
python finetune/prepare_dataset.py --input-dir dataset/raw/prw-download/frames --device GPU

# ③ 导入 AIZOO 口罩候选（只做一次）
python finetune/import_aizoo.py

# ①②③④ 人工核对与尘土化（四标签页 WebUI）
python finetune/review_server.py            # 默认开启属性预标注
python finetune/review_server.py --no-prelabel   # 无 GPU 时跳过预标注
python finetune/review_server.py --seed 42       # 固定随机出图顺序（默认每次启动随机）

# 清空全部已确认/金标准并并回候选，从零重新标注（二次确认可用 --yes 跳过）
python finetune/reset_progress.py --yes
```

## 二、微调

训练只读 confirmed/gold 与 `4_augmented/` 的训练划分；属性和口罩训练启动时把全部
裁剪小图一次性解码进内存缓存，训练期不再读盘。

```powershell
# ④ 尘土化（也可在 WebUI 第四页操作）
.venv/Scripts/python.exe finetune/dust_augment.py --stage attribute --variants 2 --intensity 1.2

# 训练三个模型（每次都从官方权重重新开始，完成后直接覆盖 models/finetuned/）
.venv-train\Scripts\python.exe finetune/train_person_detector.py --device GPU --epochs 10
.venv-train\Scripts\python.exe finetune/train_attribute.py --device GPU --epochs 15
.venv-train\Scripts\python.exe finetune/train_mask.py --device GPU --epochs 25

# 训练完成后汇总曲线与指标
.venv-train\Scripts\python.exe finetune/training_report.py --det-log ... --attr-log ... --mask-log ...
```

三个模型都是几十 MB 的小模型，数据量也小，上面 epochs 足够收敛（合计约半小时级别）。
训练侧不做任何在线增强，增强只来自离线尘土化副本。中断续训与导出注意事项见
[finetune/README.md](finetune/README.md) 第 3 节。

产物写入 `models/finetuned/{person_detector, person_attribute, face_mask}`；
曲线图与汇总见 `finetune/TRAINING_REPORT.md` 和 `finetune/report/*.png`。

### 模型清单与溯源

微调管线的三个训练对象（推理端当前部署权重 → 微调起点）：

| 环节        | 模型                      | 当前部署权重                                              | 微调起点                                                                                  | 训练框架                   |
| ----------- | ------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------------------------- | -------------------------- |
| ① 人物检测 | PP-YOLOE-S                | `models/human/mot_ppyoloe_s_36e_pipeline`               | `models/original/person_detector/mot_ppyoloe_s_36e_pipeline.pdparams`（官方可训练权重） | `vendor/PaddleDetection` |
| ② 人物属性 | PP-HGNet small（26 属性） | `models/human/PPHGNet_small_person_attribute_954_infer` | 同一部署权重转回可训练结构                                                                | `vendor/PaddleClas`      |
| ③ 口罩识别 | YOLOv5s（2 类）           | `models/face_mask_yolov5`（ONNX）                       | 由 ONNX 反建的 YOLOv5s 权重                                                               | `vendor/yolov5`          |

推理端还有三个**不参与微调**的模型：车辆检测 `models/vehicle/mot_ppyoloe_s_36e_ppvehicle`、
车辆属性 `models/vehicle/vehicle_attribute_model`、车牌 `models/vehicle/.hyperlpr3`（HyperLPR3）。

每个权重的上游地址、归档 SHA-256、本地文件哈希和授权边界记录在
[`models/MODEL_SOURCES.md`](models/MODEL_SOURCES.md) 及各目录的 `SOURCE.md`；
训练数据来源见 [`dataset/DATASETS.md`](dataset/DATASETS.md)。

## 三、推理

`inference/` 是精简的推理应用端（约四百行），与训练管线解耦：**输入**是一个图片目录，
**输出**是结构化中文业务结果（JSON + Excel），中间不含框、置信度等模型细节。
完整接口说明（配置项、命令行覆盖、模型替换、输出字段定义）见
[**inference/README.md**](inference/README.md)。

```powershell
python inference/run.py          # 读 easy_test/ → 原子写 inference/result.json
node inference/export_xlsx.mjs   # 读 result.json → 写 inference/result.xlsx（带缩略图）
```

输出速览（字段完整定义与示例在 inference/README.md）：

- `result.json`：JSON 数组，每张图片一项，含 `图片位置`、`处理耗时（毫秒）`、
  `识别内容`（`行人` 数组：性别/年龄/朝向/眼镜/帽子/口罩/包/上装/下装/鞋靴等；
  `车辆` 数组：颜色/车型/车牌）。
- `result.xlsx`：每张图片一行：缩略图、图片位置、耗时、行人识别内容、车辆识别内容。

验收微调结果后，只改 `inference/config.py` 里的 `PERSON_DETECTOR_DIR`、
`PERSON_ATTRIBUTE_DIR`、`FACE_MASK_DIR`（属性模型另把
`PERSON_ATTRIBUTE_CROP_SCALE` 改为 1.0）即完成切换；回滚时改回原值。
模型一律按目录名指定，不做哈希校验。

## 目录结构与中间产物位置

```
dataset/
  raw/                          # 原始数据集，程序只读（AIZOO、PRW 等）
  processed/
    1_detection/
      candidates.json           # ① 模型初检的人物框（待人工核对）
      confirmed.json            # ① 人工确认的整图与人物框 = 检测训练金标准
    2_attribute/
      images/                   # ② 人物裁剪小图（检测确认时生成）
      candidates.json           # ② 属性模型对小图的预标注（待人工核对）
      gold.json                 # ② 人工确认的属性金标准
    3_mask/
      images/                   # ③ 带上下文的人脸裁剪（口罩核对保存时生成）
      candidates.json           # ③ AIZOO 导入的整图与人脸框（待人工核对）
      gold.json                 # ③ 人工确认的口罩金标准
    4_augmented/
      detection/ attribute/ mask/
        images/                 # ④ 尘土化/昏黄化后的图片副本
        annotations.json        # ④ 与源完全一致的标签（每次生成重建整个环节）
    5_export/
      person_detection_coco/    # 人物检测训练用 COCO（images/ + annotations/instances_*.json）
    _legacy/                    # 旧版数据归档，不参与任何流程
finetune/                       # 微调侧：数据准备、标注 WebUI、训练、训练报告（见 finetune/README.md）
inference/                      # 推理侧：精简的推理应用端，只读模型和图片（见 inference/README.md）
models/                         # 全部模型权重（original + finetuned，来源见 MODEL_SOURCES.md）
vendor/                         # PaddleClas / PaddleDetection / yolov5 源码（仅训练用）
easy_test/                      # 推理测试图片
```

所有中间 JSON 都是 COCO 风格：`info / images / annotations / categories`，
框一律为 `bbox = [x, y, w, h]`；属性标注在 annotation 里扩展 `attributes` 字典
（26 项布尔值，固定顺序），口罩用 `category_id`（1=未佩戴，2=佩戴）区分。
每张裁剪图带 `source_image_id`，指回原图，用于 train/val/test 划分继承。

## Linux / macOS 使用方式

代码内所有路径都从项目根目录推导，没有写死的盘符，整目录拷贝即可迁移。差异只在
环境搭建和启动命令：

```bash
# 推理环境（CPU 即可；Linux 有 NVIDIA 显卡 + nvidia-container-toolkit 时可保持 GPU）
uv sync                                            # 或：uv export | .venv/bin/pip install -r /dev/stdin

# 训练环境（仅 Linux 建议 GPU；macOS 用 --device CPU）
python3 -m venv .venv-train
.venv-train/bin/pip install paddlepaddle-gpu torch --index-url <对应 CUDA 版本的索引>

# 之后的命令与 Windows 完全一致，只把 python 路径换成：
.venv/bin/python finetune/review_server.py            # 标注 WebUI
.venv-train/bin/python finetune/train_person_detector.py --device GPU
.venv/bin/python inference/run.py
node inference/export_xlsx.mjs
```

- Windows 专用的 CUDA DLL 注册（`configure_runtime_dlls`）在非 Windows 平台自动跳过，
  无需处理。
- `.torch-cu130/` 是本机 Windows 的手工 torch 目录，Linux/macOS 上删掉，正常 pip 安装即可。
- macOS 无 CUDA，Paddle/torch 都走 CPU；口罩与车牌 ONNX 模型本来就只用 CPU。
- 推理侧如需容器化：`inference/` + `models/` + `.venv` 依赖即可成镜像，训练侧不进镜像；
  训练容器化时把 `dataset/` 和 `models/` 以卷挂载进容器（`raw` 可只读），代码无需改动。
