# Docker 部署全程教学（零基础版）

**这份文档假设你完全不懂 Docker、甚至不熟 Linux。** 跟着阶段一步步走，
每一步都说了"在做什么、为什么、怎么做、怎么确认没出问题、踩过什么坑"。
所有命令都是在我们自己的服务器上真实跑通过的。

最终你会得到：一个在 GPU 服务器上常驻运行的行人/车辆识别 HTTP 服务，
以及一个可以随便发给别人的镜像文件。

---

## 阶段 0：先搞懂 5 个概念（5 分钟）

用做菜打比方：

| 概念 | 类比 | 说明 |
|---|---|---|
| **镜像（image）** | 速冻料理包 | 一个打包好的完整环境：操作系统 + Python + 深度学习框架 + 我们的代码 + 模型权重，全在里面 |
| **容器（container）** | 料理包加热后上桌的菜 | 镜像"跑起来"的实例。同一个镜像可以跑多个容器 |
| **Dockerfile** | 料理包的配方 | 一个文本文件，写着"从哪个基础环境开始、装什么、复制什么、启动什么" |
| **构建（build）** | 按配方做料理包 | `docker build` 执行 Dockerfile，产出镜像 |
| **端口映射** | 餐厅的外卖窗口 | 容器内部是隔离的，`-p 8000:8000` 把服务器的 8000 端口接到容器的 8000 端口，外面才访问得到 |

为什么用 Docker？因为"在我电脑上能跑"的环境（CUDA 版本、Python 包版本……）
手工在另一台机器上重现极其痛苦。Docker 把整个环境冻成一个文件，到哪台机器都一样的跑法。

**我们的镜像策略（为什么构建很快）**：不从零开始装环境，而是直接拿
**Paddle 官方 GPU 镜像**当底座（里面已经装好 paddlepaddle-gpu + CUDA + cuDNN，
这几 GB 的东西不用我们操心），我们只需要：
pip 装几个小包 → 复制代码 → 复制模型权重。全程几分钟。

---

## 阶段 1：连上服务器

你需要一台有 NVIDIA 显卡的 Linux 服务器（我们的目标机型是 A30，实测用的是 3080Ti，
任何 8GB 以上显存的 N 卡都行），以及它的 IP、用户名、密码。

**Windows 上**：打开 PowerShell，直接输入：

```powershell
ssh ubuntu@117.50.173.181      # 换成你的用户名@服务器IP
# 首次连接会问 yes/no，输 yes；然后输密码（输入时屏幕不显示，正常的）
```

看到类似 `ubuntu@hostname:~$` 的提示符，就说明你已经"在服务器里面"了。
后面所有没特别说明的命令，都是在 SSH 进去的这个窗口里执行。

> 名词扫盲：
> - `~` = 你的 home 目录（`/home/ubuntu`），命令里可以直接用
> - `sudo` = 以管理员身份执行这一条命令（会问密码）
> - `cd 目录` 进入目录；`ls` 看目录内容；`pwd` 看自己在哪

---

## 阶段 2：装 Docker（一次性，约 5 分钟）

```bash
# 一键安装脚本（官方）
curl -fsSL https://get.docker.com | sudo bash

# 把当前用户加入 docker 组（否则每条 docker 命令都要加 sudo）
sudo usermod -aG docker $USER
```

> **坑 1（必踩）：加完组不会立刻生效！** 必须 `exit` 退出 SSH、重新登录，
> 否则 `docker ps` 会报 `permission denied`。重新登录后验证：

```bash
docker ps
# 输出一个空表头（CONTAINER ID ...）就是成功了；
# 报 permission denied 就是没重新登录
```

---

## 阶段 3：GPU 驱动 + GPU 透传（一次性，约 10 分钟）

Docker 容器默认**看不到**显卡，需要两个东西：

**3.1 显卡驱动**（服务器商一般已装好，验证一下）：

```bash
nvidia-smi
# 能看到显卡型号表格（如 3080Ti / A30）→ 驱动 OK，跳到 3.2
# 报 command not found → 需要先装驱动：
#   sudo apt-get update && sudo apt-get install -y nvidia-driver-580
#   sudo reboot   （装驱动必须重启）
```

**3.2 nvidia-container-toolkit**（让 docker 能把显卡"递"进容器）：

```bash
# 添加 NVIDIA 的软件源并安装
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# 告诉 Docker 用它，然后重启 Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 验证：在容器里跑 nvidia-smi，能看到显卡就全通了
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

> 这里的 `--rm` 表示"跑完就删掉这个容器"（一次性测试用）；
> `--gpus all` 就是把所有显卡递给容器。

---

## 阶段 4：构建识别服务镜像（首次约 10-15 分钟）

项目提供两个配方，**默认用瘦身版**：

| 配方 | 底座 | 产物体积 | 构建速度 | 适用 |
|---|---|---|---|---|
| `deploy/Dockerfile`（默认） | python:3.10-slim + pip 装 paddle-gpu | **约 3.5-4GB** | 首次 10-15 分钟（下载 NVIDIA 轮子），之后秒级 | 要导出分享给别人 |
| `deploy/Dockerfile.full` | Paddle 官方 GPU 镜像 | 约 14GB | 底座拉过一次后构建很快 | 自己服务器内部快速迭代 |

**4.1（仅使用 Dockerfile.full 时需要）拉 Paddle 官方 GPU 底座镜像**（只拉一次）：

```bash
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.3.0-gpu-cuda12.6-cudnn9.5
```

> **坑 2（我们踩过）**：飞桨官方镜像的**旧地址 `registry.baidubce.com/...` 已失效**，
> 拉取会报 `not found`。3.x 版本要用新地址前缀
> `ccr-2vdh3abv-pub.cnc.bj.baidubce.com`，完整 tag 列表见
> [飞桨官方 Docker 安装文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/docker/linux-docker.html)。
> CUDA 12.6 + cuDNN 9.5 这个组合对 A30 / 3080Ti / 3090 等 Ampere 卡兼容性最稳。
> 默认瘦身版配方不需要这一步（python:3.10-slim 会自动拉）。

**4.2 把构建上下文弄上服务器。**"构建上下文"就是镜像里要装的全部文件
（代码 + 模型权重，约 200MB）。两种途径任选：

- 途径 A（推荐）：私有 git 仓库 clone 到服务器（见阶段 7）；
- 途径 B：本地打包上传。在**你自己电脑的 Git Bash**（项目根目录）执行：

```bash
tar -czf docker-ctx.tar.gz inference service deploy models   # 按 .dockerignore 精简过的更快
scp docker-ctx.tar.gz ubuntu@117.50.173.181:~/               # 上传
```

然后**回到服务器 SSH 窗口**解压：

```bash
tar -xzf ~/docker-ctx.tar.gz -C ~/
```

**4.3 构建**：

```bash
cd ~/docker-ctx        # 上下文根目录（里面有 inference/ service/ deploy/ models/）
docker build -f deploy/Dockerfile -t person-vehicle-recognition:latest .
```

> 读一下这条命令：`-f deploy/Dockerfile` 指定配方位置；
> `-t person-vehicle-recognition:latest` 给镜像起名；
> 结尾的 `.` 表示"上下文是当前目录"（Dockerfile 里的 COPY 路径都相对它）。

配方做了什么（`deploy/Dockerfile` 里有逐行注释）：
1. `FROM python:3.10-slim`，pip 装 paddlepaddle-gpu（飞桨官方 cu126 索引，
   自带 NVIDIA 依赖轮子）；
2. pip 装其余小包（FastAPI、opencv、onnxruntime-gpu 等，走清华镜像加速）；
3. 车牌库 `hyperlpr3` 单独用 `--no-deps` 装 —— **坑 3**：它的依赖列表会拉
   CPU 版 onnxruntime/paddle，把 GPU 环境顶掉，必须阻止它装依赖；
4. 复制代码和模型权重；
5. 入口 = 启动 uvicorn 提供 HTTP 服务。

看到 `naming to ... person-vehicle-recognition:latest` 就是构建成功。

> **坑 4（我们踩过）**：构建/拉镜像中途报 `no space left on device` = 服务器磁盘满了。
> 用 `docker system df` 看占用，`docker builder prune -af` 清构建缓存（安全），
> `docker rmi 镜像名` 删不用的旧镜像。

---

## 阶段 5：启动服务并验证（约 2 分钟）

```bash
# 启动：后台常驻 + GPU 透传 + 端口映射 + 重启策略
docker run -d --name pvr --gpus all -p 8000:8000 \
  --restart unless-stopped \
  person-vehicle-recognition:latest

# 看启动日志（首次要加载模型，等 30-60 秒；Ctrl+C 只是退出查看，不会停服务）
docker logs -f pvr
# 看到 "Uvicorn running on http://0.0.0.0:8000" 即就绪
```

> 读一下 `docker run` 的参数：`-d` 后台跑；`--name pvr` 给容器起名；
> `--restart unless-stopped` 服务器重启后自动拉起服务。

**验证三连**（在服务器上，或任何能访问服务器 8000 端口的机器上）：

```bash
# ① 健康检查
curl http://localhost:8000/v1/health
# → {"status":"ok","device":"GPU","workers":4,"backlog":512,"queue":0,"kept_results":0}

# ② 提交一张图（异步！立即返回 session_id，不等结果）
curl -F "file=@任意图片.jpg" http://localhost:8000/v1/tasks
# → {"session_id":"afd8d588...","status":"pending"}

# ③ 凭 session_id 取结果（识别是后台做的，轮询直到 status=done）
curl http://localhost:8000/v1/tasks/afd8d588...
# → {"session_id":"...","status":"done","elapsed_ms":231.5,
#    "result":{"行人":[{...26项属性+口罩...}],"车辆":[{"颜色":"白色","车型":"轿车","车牌":"鲁A66666"}]}}
```

从外部机器访问时把 `localhost` 换成服务器 IP。
**连不上先查防火墙/安全组**：云服务器要在控制台放行 8000 端口（TCP）。

> **坑 5（必知）：车牌模型的离线下载陷阱。** 车牌库 hyperlpr3 首次 import 时会
> 偷偷联网下载 11.9MB 模型包到自己的 home 目录。我们在代码里把它的 home 临时
> 指向镜像内置的模型包（`models/vehicle/.hyperlpr3/`），所以容器**完全离线可跑**。
> 验证方式：`docker logs pvr` 里不该出现任何 Pull/Download 字样。

**可调环境变量**（启动时 `-e 名字=值` 覆盖）：

| 变量 | 默认 | 干什么用 |
|---|---|---|
| `PORT` | 8000 | 服务端口 |
| `DEVICE` | GPU | 没显卡的机器可 `-e DEVICE=CPU` 勉强跑（很慢，仅调试用） |
| `WORKERS` | 4 | 并行识别路数；每路一套模型实例，显存约占 2GB/路 |
| `BACKLOG` | 512 | 排队上限（见下面的"防积压"） |
| `MAX_KEPT` | 20000 | 结果在内存里最多留多少条 |

---

## 阶段 6：理解异步与防积压（设计核心，务必看懂）

业务场景是**抓拍/视频抽帧，高峰一秒几千张**——这远超任何单卡 GPU 的识别速度。
所以服务是这样设计的：

```
调用方                服务                        GPU
  │  POST 图片         │                            │
  │ ─────────────────► │ 放进队列，立即返回 sessionId │  （不等识别！）
  │ ◄─ 202 + sessionId │                            │
  │                    │  N 个 worker 从队列取图 ──►  │ 逐张识别
  │  GET /tasks/{id}   │                            │
  │ ─────────────────► │ 识别完了 → 返回结果          │
  │                    │                            │
  │  队列满了还 POST？  │                            │
  │ ─────────────────► │  立即返回 429（拒收）        │  ← 关键！
```

- **提交和取结果是两个独立接口**，中间靠 sessionId 关联——这就是"异步"的含义；
- **队列有硬上限（BACKLOG=512）**：满了就返回 429。调用方收到 429 应当
  **丢掉这一帧**（或稍后重试）。服务宁可拒收，也绝不让图片在内存里无限堆积——
  这就是"图片再多服务也不会崩"的原理；
- 结果只在内存保留最近 20000 条，取结果要及时；
- **吞吐上限是物理事实**：这套管线每张图要过 5 个模型（行人检测→属性→口罩→
  车辆检测→车辆属性+车牌），实测单卡 3080Ti 识别吞吐约 **15-20 张/秒**
  （提交速率可达 200-450 张/秒，超出的由 429 削峰），A30 同量级。
  "每秒几千张"对单卡不可能；几千张/秒是**提交**能力，不是识别能力。
  真要更高的持续吞吐：加卡、加机器（无状态服务，多跑几个容器前面挂负载均衡即可），
  或简化模型。调 `WORKERS`（识别进程数，默认 4，每进程约占 2GB 显存 +
  1-2 个 CPU 核）和 `BATCH_SIZE`（攒批大小，默认 8）可按硬件微调。

**配套抽帧客户端**（自带 429 丢帧逻辑）：

```bash
# 任意一台能访问服务 8000 端口的机器上，只需 pip install opencv-python requests
python service/video_client.py --source video.mp4 --fps 2 \
    --server http://服务器IP:8000 --output results.jsonl
python service/video_client.py --source rtsp://摄像头地址 --fps 1 \
    --server http://服务器IP:8000 --output results.jsonl
```

`--fps` 控制每秒抽几帧；结果写 JSONL，每行一帧（帧号、时间戳、识别结果）。

---

## 阶段 7：把镜像分享给别人

镜像构建好后就是个整体，**对方不需要本项目代码、不需要 git**：

```bash
# 1. 在你的服务器上导出（gzip 后约 3GB，要等几分钟）
docker save person-vehicle-recognition:latest | gzip > pvr-image.tar.gz

# 2. 用 scp / 网盘 / U盘，任何方式把 pvr-image.tar.gz 给对方

# 3. 对方机器上（对方只需装好阶段 2、3 的 Docker + GPU 透传）：
gunzip -c pvr-image.tar.gz | docker load          # 导入镜像
docker run -d --name pvr --gpus all -p 8000:8000 \
  --restart unless-stopped person-vehicle-recognition:latest
curl http://localhost:8000/v1/health              # 验证
```

> 如果对方机器没有 N 卡，也能勉强跑起来看效果：
> `docker run -d -p 8000:8000 -e DEVICE=CPU person-vehicle-recognition:latest`
> （CPU 模式每张图几秒，仅用于联调接口，不能上生产。）

代码仓库（二次开发用）走 GitHub 私有仓库：GitHub 网页上建 Private repo →
`git remote add origin git@github.com:你的账号/仓库.git` → `git push`。
把协作者加成仓库的 Collaborator（Settings → Collaborators），对方就能 clone。

---

## 阶段 8：日常运维速查

```bash
docker ps                       # 看正在跑的容器
docker logs -f pvr              # 跟踪服务日志（Ctrl+C 退出查看，不影响服务）
docker restart pvr              # 重启服务（换了模型文件后用）
docker stop pvr                 # 停
docker start pvr                # 起
docker rm -f pvr                # 删掉容器（镜像还在，随时能重新 run）
docker images                   # 看本地有哪些镜像
docker system df                # 看磁盘占用
docker builder prune -af        # 磁盘满了先清构建缓存（安全）
```

**换模型**：替换镜像里的模型文件 → 需要重新构建（改了 models/ 后重新跑阶段 4.3 的
build 命令即可，底座和 pip 层都有缓存，几十秒完成），然后
`docker rm -f pvr` + 重新 `docker run`。

---

## 坑位速查表

| # | 坑 | 现象 | 解法 |
|---|---|---|---|
| 1 | docker 用户组不生效 | permission denied | 退出 SSH 重登录 |
| 2 | Paddle 镜像旧地址失效 | pull 报 not found | 用 `ccr-2vdh3abv-pub.cnc.bj.baidubce.com` 前缀 |
| 3 | hyperlpr3 依赖污染 | 装完 GPU 变 CPU 版 | 必须 `--no-deps` 安装 |
| 4 | 磁盘满 | no space left on device | `docker builder prune -af` + 删旧镜像 |
| 5 | 车牌库联网下载 | 离线环境卡住 | 已内置模型包 + 代码重定向 HOME，无需处理 |
| 6 | 429 被当成故障 | 高峰期部分帧返回 429 | **设计行为**：削峰，调用方丢帧即可 |
| 7 | 外部连不上服务 | curl 超时 | 云控制台安全组/防火墙放行 8000 端口 |
| 8 | 容器看不到显卡 | 服务报 GPU 不可用 | 阶段 3.2 的 toolkit 没装或没重启 docker |
| 9 | 多进程子进程报 CUDA 错 | cudaErrorInitializationError | 代码里已用 spawn 启动子进程（fork 会继承父进程 CUDA 状态），勿改 |
| 10 | 提交快但识别慢 | 大量 pending | 识别吞吐是 GPU 物理上限，靠 429 削峰；这不是故障 |
| 11 | slim 镜像里 paddle 导入失败 | libgomp.so.1 找不到 | Dockerfile 已含 `apt-get install libgomp1`（slim 系统没装 OpenMP 运行时） |
| 12 | 容器里 apt-get 失败 | exit code: 100 | 国内服务器连不上 deb.debian.org，Dockerfile 已先换成中科大镜像源 |
