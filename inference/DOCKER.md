# Docker 部署全流程手册（零基础版）

这份文档把「在本项目服务器上用 Docker 部署推理端」的完整过程、踩过的坑、以及最终
最简操作路径记录下来。跟着「最终路径」小节的命令抄一遍就能跑通；「原理与坑」
小节解释为什么这么做。

## 0. 先搞清楚三个概念

- **镜像（image）**：一个打包好的「系统 + 依赖 + 代码 + 模型」快照。别人拿到镜像
  文件，就等于拿到了一台配置好的、只装了本项目推理程序的虚拟电脑。
- **容器（container）**：镜像跑起来的一次实例。同一镜像可以跑很多个容器。
- **卷挂载（-v）**：把宿主机（真实电脑/服务器）的一个文件夹「映射」进容器。
  本项目的图片输入和结果输出都靠它，**模型和代码封死在镜像里，图片和结果不进镜像**。

一句话：我们做的事 = 把推理程序和所有模型冻进一个镜像文件 → 把文件发给任何人 →
对方一条命令跑起来，挂载自己的图片文件夹就能出结果。

## 1. 镜像里有什么（设计决策）

| 内容 | 来源 | 大小 |
| --- | --- | --- |
| Python 3.10 + paddlepaddle 3.3.0（CPU 版）+ onnxruntime + opencv + hyperlpr3 | `inference/requirements-docker.txt` | 大头，约 1 GB |
| Node 22 + sharp + @oai/artifact-tool（导出 xlsx 用） | `inference/package.json` | 约 200 MB |
| 推理代码 6 个文件 | `inference/*.py`、`export_xlsx.mjs` | KB 级 |
| 6 个部署模型 | `models/finetuned/` 的 3 个微调模型 + `models/vehicle/` 的车辆/车牌模型 | 约 200 MB |

**只放部署权重，不放训练 checkpoint**。`models/finetuned/` 本地有 1.8 GB，
其中 1.6 GB 是训练中间产物（`person_detector_checkpoints/`、`.pdopt`、`best.pt` 等），
推理根本用不到；镜像只带 `model.json/model.pdiparams`、`inference.json/inference.pdiparams`、
`face_mask_detection.onnx` 这些部署文件。根目录 `.dockerignore` 用白名单保证这一点。

**为什么用 CPU 版 Paddle**：要分享的机器大多数没有 NVIDIA 显卡，CPU 版镜像在哪都能跑。
43 张测试图在 CPU 上约几分钟，完全可接受。有 GPU 的机器跑这个镜像也没问题（只是用 CPU 算）。

## 2. 最终路径：从零到跑通

### 2.1 准备一台有 Docker 的机器

任何 Linux 服务器/电脑都行。本项目用的验证环境：Ubuntu 22.04、Docker 29、12 核 CPU、
31 GB 内存、56 GB 空闲磁盘（构建期峰值约用 6 GB）。

检查 Docker 是否就绪：

```bash
docker --version          # 有版本号即可
docker run --rm hello-world   # 能打出欢迎信息说明守护进程正常
```

没有 Docker 的话（Ubuntu 一条命令装）：

```bash
curl -fsSL https://get.docker.com | sudo bash
sudo usermod -aG docker $USER   # 免 sudo 用 docker
```

**坑 0（实际踩到）**：`usermod -aG docker` 之后**必须重新登录**（SSH 断开重连）才生效，
否则 `docker build` 直接报
`permission denied while trying to connect to the docker API at unix:///var/run/docker.sock`。
这不是镜像问题，是当前登录会话还拿着旧的用户组信息。

### 2.2 拿到镜像

两种方式（详见第 4 节「分享」），任选其一：

- 拿到 `asperastra-inference.tar.gz` 文件 → `docker load -i asperastra-inference.tar.gz`
- 或拿到仓库源码自己构建 → 在项目根目录执行
  `docker build -f inference/Dockerfile -t asperastra-inference .`

### 2.3 跑推理

```bash
mkdir -p ~/my_images ~/my_output
# 把要识别的图片放进 ~/my_images

docker run --rm \
  -v ~/my_images:/data/images \
  -v ~/my_output:/data/output \
  asperastra-inference
```

跑完后 `~/my_output/` 里有两个文件：

- `result.json`：每张图一条记录（图片位置、耗时、行人属性数组、车辆数组），
  字段定义见 [README.md](README.md#输入输出格式)。
- `result.xlsx`：同样的内容排成带缩略图的表格，给人看用。

可用环境变量（`-e NAME=value` 传给 `docker run`）：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `IMAGES_DIR` | `/data/images` | 容器内图片目录（一般不用改，改挂载点更方便） |
| `OUTPUT_DIR` | `/data/output` | 容器内输出目录 |
| `DEVICE` | `CPU` | 推理设备，镜像只带 CPU 版 Paddle，保持 CPU |

## 3. 原理与坑（构建阶段）

> 这一节记录构建镜像时实际踩到的坑，对应命令都在 `inference/Dockerfile` 里。

### 3.1 构建上下文要克制

`docker build` 会把整个上下文目录发给 Docker 守护进程。本项目根目录有好几个 GB
（数据集、训练 checkpoint、venv），直接构建会慢到怀疑人生。根目录 `.dockerignore`
用「先 `*` 全排除、再逐个 `!` 放行」的白名单写法，把上下文压到约 200 MB。

### 3.2 Node 不用 apt 装

Ubuntu 源里的 Node 太老，nodesource 又要加外部源。Dockerfile 用多阶段构建直接从
官方 `node:22-slim` 镜像里「借」二进制：

```dockerfile
FROM node:22-slim AS node
...
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
```

### 3.3 sharp 必须按平台安装

`sharp`（xlsx 里生成缩略图用的库）带平台相关的预编译二进制。**绝对不能把 Windows 上
`npm install` 出来的 `node_modules` 复制进 Linux 镜像**，必须在镜像构建时重新
`npm install`。这也是 `inference/package.json` 存在的原因（本地 node_modules 是
手动装的，仓库里原本没有依赖清单）。

### 3.4 opencv 用完整版而不是 headless 版

`hyperlpr3`（车牌）的依赖声明写的是 `opencv-python`，装 headless 版满足不了它，
pip 会再拉一个完整版。索性直接装完整版 `opencv-python==4.6.0.66`，并在系统层补
`libgl1 libglib2.0-0`（没有这两个库 import cv2 会直接崩）。

### 3.5 国内服务器的镜像加速

构建时要从 PyPI 和 npm 拉约 1.5 GB 依赖，服务器在国内，直连很慢。Dockerfile 里默认
使用清华 PyPI 镜像和 npmmirror，可用 build-arg 覆盖：

```bash
docker build -f inference/Dockerfile -t asperastra-inference . \
  --build-arg PIP_INDEX_URL=https://pypi.org/simple \
  --build-arg NPM_REGISTRY=https://registry.npmjs.org
```

<!-- 以下小节在实际构建/测试/导出完成后补充验证结论 -->

## 4. 分享镜像给别人

（待镜像导出验证后补最终命令）

## 5. 常见问题排查

（待试运行后补实际遇到的问题）
