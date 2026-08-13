# independent

这是只读取外部模型目录和图片目录的精简结构化识别流程。代码只有三个职责：

- `run.py`：配置、命令行、逐图计时、原子写 JSON。
- `model_adapters.py`：Paddle、ONNX Runtime 和 HyperLPR3 的薄模型适配。
- `pipeline.py`：对象裁切、属性解码和中文业务结果。

## 运行

日常只编辑 [`config.py`](config.py)，所有输入、输出、环境和七个模型目录都在此配置。
切换微调前后权重只需改 `PERSON_DETECTOR_DIR`、`PERSON_ATTRIBUTE_DIR`、
`FACE_MASK_DIR` 和口罩 SHA-256。

```powershell
python independent/run.py
python independent/run.py --device CPU --limit 1
python independent/run.py --data-dir easy_test --person-attribute-dir models/finetuned/person_attribute
python independent/run.py --person-detector-dir models/finetuned/person_detector
```

命令行参数只用于临时覆盖 `config.py`。程序复用配置的推理环境；运行时不会下载模型。
结果在所有图片成功处理后才原子替换，输出路径也由 `config.py` 决定。

`result.json` 是带缩进的标准 JSON 数组。每张图片只有“图片位置”“处理耗时（毫秒）”“识别内容”三个顶层字段；识别内容只保留行人和车辆语义，不含框、置信度或模型元数据。

口罩识别使用行人原检测框上部 40% 的裁片。没有达到阈值的可靠口罩结果统一输出“未识别”，不把无脸、背面或模糊小脸自动判成未佩戴。

GPU 选项用于 Paddle 模型；当前口罩和车牌 ONNX 模型使用 CPU Execution Provider。

所有权重的上游地址、归档哈希、本地文件哈希和授权边界统一记录在 [`models/MODEL_SOURCES.md`](../models/MODEL_SOURCES.md) 及各权重目录的 `SOURCE.md` 中。

## 导出 Excel

```powershell
node independent/export_xlsx.mjs
```

导出程序只读取 `result.json` 和其中记录的原图，结果固定写到 `independent/result.xlsx`。工作簿每张图片一行，包含保持比例的原图缩略图、图片位置、耗时、行人和车辆识别内容；不引入框、置信度、完整 JSON 或额外统计。
