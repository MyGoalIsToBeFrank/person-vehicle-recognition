# 三模型人工金标准微调

目标是基于现有官网权重继续微调人物检测、人物属性和口罩识别三个模型，再转成 ONNX
接回推理端（`models/onnx/`）。原 ONNX/Paddle 部署模型和人物检测可训练权重
保存在 `models/original/`，不会被训练脚本覆盖。

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

## 4. 转 ONNX 并切回推理端

推理端只消费 `models/onnx/` 下的 ONNX 文件（onnxruntime 引擎）。微调产物
（`models/finetuned/` 的 Paddle 权重）按 [deploy/DOCKER.md](../deploy/DOCKER.md) 的
「Paddle → ONNX 转换」一节转成 ONNX，覆盖 `models/onnx/` 对应目录后运行：

```powershell
python inference/run.py
```

切换前务必跑数值回归，确认 ONNX 与 Paddle 输出一致（详见 inference/README.md）：

```powershell
.venv/Scripts/python.exe -X utf8 inference/onnx_regression.py
```

回滚时把 `models/onnx/` 里对应 .onnx 换回旧文件即可。车辆、车牌流程不变。
