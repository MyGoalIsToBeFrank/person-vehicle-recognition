# inference

推理应用端：只读取外部模型目录和图片目录的精简结构化识别流程。代码只有三个职责：

- `run.py`：配置、命令行、逐图计时、原子写 JSON。
- `model_adapters.py`：Paddle、ONNX Runtime 和 HyperLPR3 的薄模型适配。
- `pipeline.py`：对象裁切、属性解码和中文业务结果。

## 运行

日常只编辑 [`config.py`](config.py)，所有输入、输出、环境和七个模型目录都在此配置。
切换微调前后权重只需改 `PERSON_DETECTOR_DIR`、`PERSON_ATTRIBUTE_DIR`、
`FACE_MASK_DIR` 和口罩 SHA-256，无需改动任何其他代码。

```bash
python inference/run.py
python inference/run.py --device CPU --limit 1
python inference/run.py --data-dir easy_test --person-attribute-dir models/finetuned/person_attribute
python inference/run.py --person-detector-dir models/finetuned/person_detector
```

命令行参数只用于临时覆盖 `config.py`。程序复用配置的推理环境；运行时不会下载模型。
结果在所有图片成功处理后才原子替换，输出路径也由 `config.py` 决定。

## 使用接口

**输入图片从哪来**：默认读 `config.py` 的 `DATA_DIR`（当前为 `easy_test/`）下全部图片；
临时换目录用 `--data-dir 路径`；只想试前几张加 `--limit N`。

**结果写到哪**：默认原子写 `inference/result.json`（`RESULT_JSON`），Excel 导出固定读它并写
`inference/result.xlsx`（`RESULT_XLSX`）。临时改输出用 `--result-json 路径`。

**如何替换模型**：改 `config.py` 里对应的目录常量即可，七个模型互不耦合：

| 配置项 | 作用 | 微调后指向 |
| --- | --- | --- |
| `PERSON_DETECTOR_DIR` | 行人检测（Paddle 部署模型目录） | `models/finetuned/person_detector` |
| `PERSON_ATTRIBUTE_DIR` | 行人 26 属性 | `models/finetuned/person_attribute` |
| `FACE_MASK_DIR` | 口罩（含 `face_mask_detection.onnx` 与 `synset.txt`） | `models/finetuned/face_mask` |
| `VEHICLE_DETECTOR_DIR` / `VEHICLE_ATTRIBUTE_DIR` / `PLATE_MODEL_DIR` | 车辆与车牌（不参与微调） | 保持不变 |

换模型只需把对应 `*_DIR` 指向新目录（按目录名指定，不做哈希校验）。换微调属性模型时把
`PERSON_ATTRIBUTE_CROP_SCALE` 从官网模型习惯的 `1.3` 改为 `1.0`（微调模型按 WebUI 红框
原尺寸训练）。以上每一项都能在命令行用同名参数（如 `--person-detector-dir`）临时覆盖，
便于不改动配置直接对比新旧模型效果。

## 输入输出格式

**输入**：一个图片目录（`DATA_DIR`，默认 `easy_test/`），程序按文件名排序逐张处理，
支持常见图片格式；`--limit N` 只取前 N 张。无网络依赖，模型全部从本地目录加载。

**输出一：`result.json`**（默认 `inference/result.json`，全部图片成功后才原子替换旧文件）。
标准 JSON 数组，每张图片一项，只有三个顶层字段：

```json
[
  {
    "图片位置": "images (1).jpg",
    "处理耗时（毫秒）": 128.4,
    "识别内容": {
      "行人": [
        {
          "性别": "男",
          "年龄": "18至60岁",
          "朝向": "正面",
          "佩戴眼镜": "否",
          "佩戴帽子": "否",
          "手持物品": "否",
          "包": "双肩包",
          "上装": { "袖长": "长袖", "款式": ["条纹"] },
          "下装": ["长裤"],
          "鞋靴": "非靴子",
          "口罩": "未佩戴口罩"
        }
      ],
      "车辆": [
        { "颜色": "白色", "车型": "轿车", "车牌": "京A12345" }
      ]
    }
  }
]
```

注意：**`图片位置` 默认存为相对于 `--data-dir` 的相对路径**，这样 result.json 在容器或
共享目录迁移时仍可用；如果原图路径不在 `--data-dir` 下，则回退为绝对路径。

字段取值约定（全部是中文枚举字符串，便于直接落业务表）：

- 行人：`性别` 男/女；`年龄` 未满18岁/18至60岁/60岁以上；`朝向` 正面/侧面/背面；
  `佩戴眼镜`、`佩戴帽子`、`手持物品` 是/否；`包` 手提包/单肩包/双肩包/无；
  `上装.袖长` 长袖/短袖，`上装.款式` 为数组（条纹/标志/格纹/拼接，可多选或为空）；
  `下装` 为数组（长外套/长裤/短裤/裙装/条纹/图案）；`鞋靴` 靴子/非靴子；
  `口罩` 佩戴口罩/未佩戴口罩（仅两类）。
- 车辆：`颜色` 黄/橙/绿/灰/红/蓝/白/金/棕/黑或“未知”；`车型` 轿车/SUV/厢式货车/
  掀背车/MPV/皮卡/公交车/卡车/旅行车或“未知”；`车牌` 识别字符串或“未识别”；
  同车牌车辆自动去重，只保留最清晰的一条。
- 没有任何识别目标时，`行人` 和 `车辆` 为空数组。

口罩识别使用行人原检测框上部 40% 的裁片，业务上只有两类：检不到可靠口罩框
（无脸、背面、模糊小脸等）统一归为“未佩戴口罩”。

**输出二：`result.xlsx`**（`node inference/export_xlsx.mjs` 生成）。默认写
`inference/result.xlsx`；工作簿“识别结果”每张图片一行，五列：
`图像`（保持比例的原图缩略图）、`图片位置`、`处理耗时（毫秒）`、`行人`、`车辆`
（识别内容的多行文本）。只读 `result.json` 和其中记录的原图，不引入框、置信度或
额外统计。

## 外部目录 / Docker 用法

推理端被设计成可以脱离项目目录，只把 `inference/` 和 `models/` 放进容器，
输入图片与结果目录通过挂载卷传入：

```bash
# 容器内运行：读取外部挂载的 /data/images，输出到外部挂载的 /output
python inference/run.py \
  --data-dir /data/images \
  --output-dir /output

# Excel 导出同样可指定输入 JSON 与输出 xlsx 位置
node inference/export_xlsx.mjs \
  --result-json /output/result.json \
  --data-dir /data/images \
  --output-xlsx /output/result.xlsx
```

要点：

- `--data-dir` 覆盖图片输入目录；`--output-dir` 指定输出目录，程序自动在目录下写
  `result.json`（优先于 `--result-json`）。
- `--result-json` 仍保留，用于直接覆盖 JSON 文件路径。
- `export_xlsx.mjs` 用 `--data-dir` 把 `result.json` 里的相对图片位置解析回绝对路径读取缩略图；
  用 `--output-xlsx` 把 Excel 也写到外部目录。
- 容器内启动 WebUI 时记得 `--host 0.0.0.0` 并映射端口。

GPU 选项用于 Paddle 模型；当前口罩和车牌 ONNX 模型使用 CPU Execution Provider。

所有权重的上游地址、归档哈希、本地文件哈希和授权边界统一记录在 [`models/MODEL_SOURCES.md`](../models/MODEL_SOURCES.md) 及各权重目录的 `SOURCE.md` 中。

## 导出 Excel

```bash
node inference/export_xlsx.mjs

# 外部目录 / Docker 场景
node inference/export_xlsx.mjs \
  --result-json /output/result.json \
  --data-dir /data/images \
  --output-xlsx /output/result.xlsx
```

导出程序只读取 `result.json` 和其中记录的原图，结果默认写到 `inference/result.xlsx`。
工作簿每张图片一行，包含保持比例的原图缩略图、图片位置、耗时、行人和车辆识别内容；
不引入框、置信度、完整 JSON 或额外统计。
