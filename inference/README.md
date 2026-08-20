# inference

推理应用端：只读取模型目录和图片目录的精简结构化识别流程。推理引擎是
**onnxruntime 单引擎**（GPU 自动探测，无卡回落 CPU），所有模型都消费 ONNX 文件。
代码只有三个职责：

- `run.py`：配置、命令行、逐图计时、原子写 JSON。
- `model_adapters.py`：onnxruntime 与 HyperLPR3 的薄模型适配（含检测器的 numpy NMS）。
- `pipeline.py`：对象裁切、属性解码和中文业务结果。

另有一个开发用校验脚本 `onnx_regression.py`（见下文「数值回归」），不参与推理。

## 运行

日常只编辑 [`config.py`](config.py)，所有输入、输出、环境和模型目录都在此配置。

```bash
python inference/run.py
python inference/run.py --device CPU --limit 1
python inference/run.py --data-dir easy_test --person-detector-dir models/onnx/person_detector
```

命令行参数只用于临时覆盖 `config.py`。程序复用配置的推理环境；运行时不会下载模型。
结果在所有图片成功处理后才原子替换，输出路径也由 `config.py` 决定。

## 使用接口

**输入图片从哪来**：默认读 `config.py` 的 `DATA_DIR`（当前为 `easy_test/`）下全部图片；
临时换目录用 `--data-dir 路径`；只想试前几张加 `--limit N`。

**结果写到哪**：默认原子写 `inference/result.json`（`RESULT_JSON`）。
临时改输出用 `--result-json 路径`，或 `--output-dir 目录`（自动写 `<目录>/result.json`）。

**如何替换模型**：改 `config.py` 里对应的目录常量即可。每个模型目录里放**恰好一个
`.onnx`** 文件（车辆检测目录另带 `nms_config.json`），换模型就是换文件：

| 配置项 | 作用 | 当前指向 |
| --- | --- | --- |
| `PERSON_DETECTOR_DIR` | 行人检测（PP-YOLOE-S，图内 NMS） | `models/onnx/person_detector` |
| `PERSON_ATTRIBUTE_DIR` | 行人 26 属性（PP-HGNet small） | `models/onnx/person_attribute` |
| `VEHICLE_DETECTOR_DIR` | 车辆检测（PP-YOLOE，图外 numpy NMS） | `models/onnx/vehicle_detector` |
| `VEHICLE_ATTRIBUTE_DIR` | 车辆颜色/车型 | `models/onnx/vehicle_attribute` |
| `FACE_MASK_DIR` | 口罩（`face_mask_detection.onnx` + `synset.txt`） | `models/finetuned/face_mask` |
| `PLATE_MODEL_DIR` | 车牌（HyperLPR3 模型缓存目录的上一级） | `models/vehicle` |

每一项都能在命令行用同名参数（如 `--person-detector-dir`）临时覆盖，
便于不改动配置直接对比新旧模型效果。按目录名指定，不做哈希校验。

`DEVICE = "GPU"` 表示优先 GPU：装有 onnxruntime-gpu 且有可用显卡时走
CUDAExecutionProvider，否则自动回落 CPU 并打印实际后端。

## 输入输出格式

**输入**：一个图片目录（`DATA_DIR`，默认 `easy_test/`），程序按文件名排序逐张处理，
支持常见图片格式；`--limit N` 只取前 N 张。无网络依赖，模型全部从本地目录加载。

**输出：`result.json`**（默认 `inference/result.json`，全部图片成功后才原子替换旧文件）。
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

## 数值回归（Paddle → ONNX 切换前的验收）

ONNX 文件由 Paddle 权重转换而来（转换工具与流程见
[deploy/DOCKER.md](../deploy/DOCKER.md)）。每次换 ONNX 后必须跑回归，确认与
Paddle 基准输出一致：

```bash
.venv/Scripts/python.exe -X utf8 inference/onnx_regression.py
```

通过标准：属性模型逐元素最大误差 ≤ 1e-4；检测器检出框数量完全一致、
框坐标/分数误差 ≤ 5e-4。当前四个模型全部通过，且 43 张测试图端到端业务结果
与 Paddle 管线 100% 一致。

## 外部目录 / Docker 用法

推理端被设计成可以脱离项目目录：镜像只含 `inference/` 代码和 `models/onnx/` 等部署
权重，输入图片与结果目录通过挂载卷传入。构建、运行、分享全流程见
[deploy/DOCKER.md](../deploy/DOCKER.md)，简版：

```bash
docker build -f deploy/Dockerfile -t person-vehicle-recognition .
docker run --rm --gpus all \
  -v /宿主机/图片目录:/data/images -v /宿主机/输出目录:/data/output \
  person-vehicle-recognition
# 结果写在 /宿主机/输出目录/result.json；无 GPU 的机器去掉 --gpus all 自动走 CPU
```

所有权重的上游地址、归档哈希和授权边界统一记录在
[`models/MODEL_SOURCES.md`](../models/MODEL_SOURCES.md) 及各权重目录的 `SOURCE.md` 中。
