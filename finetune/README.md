# 三模型人工金标准微调

目标是基于现有官网权重继续微调人物检测、人物属性和口罩识别三个模型，再通过
`independent/config.py` 接回原工作流。原 ONNX/Paddle 部署模型和人物检测可训练权重
保存在 `models/original/`，不会被训练脚本覆盖。

## 数据只有三层

- `dataset/raw/`：原始 AIZOO、PRW 图片和标注，程序只读。
- `dataset/processed/candidates.json`：模型或原数据集给出的候选框、候选标签，不参与训练。
- `dataset/processed/gold_labels.json` 与 `gold_images/`：只由 WebUI 保存产生，是三个训练脚本
  唯一读取的金标准。

没有 1000/2000 张上限，也没有标签优先级或审核状态机。已保存整图由
`gold_labels.json` 的整图 id 判断，重启 WebUI 后不会再次出现；“排除整图”只从候选中
移除，不写金标准，也不删除 raw。

## 1. 生成候选

属性栏使用当前人物检测器的原始框，并在该框内运行当前属性模型。`--replace` 只重建
属性候选，人工金标准不受影响：

```powershell
python finetune/prepare_dataset.py --input-dir dataset/raw/prw-download/frames --device GPU --replace
```

口罩栏从 AIZOO 的原图、原框和原标签导入：

```powershell
python finetune/import_aizoo.py
```

## 2. 人工核对

```powershell
python finetune/review_server.py
```

界面分“属性核对”和“口罩核对”两栏。每次显示完整原图：继承框带原候选标签；可拖动
框、缩放四角、添加框或删除框。调整继承框不会清空标签；新增框标签为空，必须补全后
才能保存。保存、上一张、下一张、跳转和切换栏目都会先保存当前整图。

只有保存成功才按最终框生成训练裁剪并写入 `gold_labels.json`。删除的框不会写入；整图
保存后即从待核对队列消失。建议先把一张图里的目标框调整完，再逐框检查右侧标签。

## 3. 微调三个模型

所有训练入口只读取 `GOLD_LABELS_PATH`：

```powershell
.venv-train\Scripts\python.exe finetune/train_person_detector.py --device GPU
.venv-train\Scripts\python.exe finetune/train_attribute.py --device GPU
.venv-train\Scripts\python.exe finetune/train_mask.py --device GPU
```

- 人物检测：人工核对的整图与最终人物框导出为 COCO，从官方
  `mot_ppyoloe_s_36e_pipeline.pdparams` 继续训练 PP-YOLOE-S。
- 人物属性：使用最终人物裁剪，从当前 PP-HGNet small 部署权重继续训练。
- 口罩识别：使用最终人脸范围生成的带上下文裁剪，从当前 YOLOv5 ONNX 权重继续训练。

属性和口罩训练仅对训练集随机施加低照度、偏黄、模糊、噪声、JPEG 压缩和轻度脏污；
验证集不增强。数据按完整原图哈希划分，避免同一张图的多个目标跨训练/验证集。

## 4. 用一个配置切换回原工作流

验收微调结果后，只改 `independent/config.py`：

```python
PERSON_DETECTOR_DIR = TRAINING_OUTPUT_DIR / "person_detector"
PERSON_ATTRIBUTE_DIR = TRAINING_OUTPUT_DIR / "person_attribute"
PERSON_ATTRIBUTE_CROP_SCALE = 1.0
FACE_MASK_DIR = TRAINING_OUTPUT_DIR / "face_mask"
FACE_MASK_SHA256 = "models/finetuned/face_mask/SHA256.txt 中的哈希"
```

随后仍运行原入口：

```powershell
python independent/run.py
node independent/export_xlsx.mjs
```

回滚时把这三个模型目录和口罩哈希改回原值即可。车辆、车牌、JSON 和 Excel 流程不变。
