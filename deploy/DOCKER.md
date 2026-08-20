# Docker 部署与分享教学（小白版）

本文档把「在服务器上用 Docker 部署本识别服务」的全流程按阶段整理，
每个阶段只记录**验证过的**命令和原理，以及踩过的坑。
目标读者：完全没用过 Docker 的人。

## 〇、30 秒理解 Docker

- **镜像（image）**：一个打包好的"系统快照"——操作系统环境 + Python + 深度学习框架
  + 我们的代码和模型，全部冻在里面。同一镜像在任何机器上跑起来行为一致。
- **容器（container）**：镜像跑起来的一次实例，像一台轻量虚拟机。
- **Dockerfile**：描述"怎么做出这个镜像"的配方文件，构建时逐条执行。
- **端口映射**：容器内部有自己的网络，`-p 8000:8000` 表示把服务器的 8000 端口
  转发到容器的 8000 端口，外部才能访问服务。
- **`--gpus all`**：把宿主机的 NVIDIA 显卡透传给容器（需要装 nvidia-container-toolkit）。

本项目的镜像策略：**不从零装环境**。直接用 Paddle 官方 GPU 镜像做底座
（里面已有 paddlepaddle-gpu + CUDA + cuDNN），再 pip 装少量增量包、
COPY 我们的代码和模型权重。这样构建只需几分钟，而不是每次下载 2-3GB 的 NVIDIA 依赖。

## 一、服务器准备（一次性）

需要的三样东西：Docker 本身、NVIDIA 驱动、nvidia-container-toolkit（GPU 透传）。

```bash
# 1. 装 Docker（Ubuntu 官方脚本，国内服务器可加 --mirror Aliyun）
curl -fsSL https://get.docker.com | sudo bash

# 2. 把当前用户加入 docker 组，免 sudo 用 docker
sudo usermod -aG docker $USER
# ★ 坑 1：加组后必须「退出 SSH 重新登录」才生效，否则还是 permission denied

# 3. 确认显卡驱动（能看到显卡型号即驱动 OK）
nvidia-smi

# 4. 装 nvidia-container-toolkit（让 docker 能用 GPU）
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 5. 验证 GPU 透传（能打印出 nvidia-smi 就成功了）
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

## 二、构建镜像

```bash
# 1. 拉取 Paddle 官方 GPU 底座镜像（约 5GB，只拉一次；以后构建会复用）
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.3.0-gpu-cuda12.6-cudnn9.5
```

> ★ 坑 2：Paddle 官方镜像的旧地址 `registry.baidubce.com/paddlepaddle/paddle:...`
> 已失效（报 not found）。3.x 版本新地址前缀是
> `ccr-2vdh3abv-pub.cnc.bj.baidubce.com`，tag 命名如 `3.3.0-gpu-cuda12.6-cudnn9.5`。
> 可用 tag 列表见飞桨官方安装文档。

```bash
# 2. 把项目构建上下文放到服务器（只需 inference/ service/ deploy/ models/ 四类文件，
#    本地打包 scp 上传，或用 git clone 私有仓库后按 .dockerignore 精简）

# 3. 在上下文根目录构建（-f 指定 Dockerfile 位置，`.` 是上下文根）
cd ~/docker-ctx
docker build -f deploy/Dockerfile -t person-vehicle-recognition:latest .
```

Dockerfile 做的事（`deploy/Dockerfile`，每行都有注释）：

1. `FROM` Paddle 官方 GPU 镜像；
2. pip 装增量包（fastapi / uvicorn / opencv / onnxruntime 等，走清华镜像加速）；
3. `hyperlpr3`（车牌识别）用 `--no-deps` 单独装——★ 坑 3：它的依赖会拉 CPU 版
   onnxruntime / paddle，把底座镜像的 GPU 环境覆盖掉；
4. COPY 代码与模型权重进镜像；
5. 入口为 `deploy/docker_entrypoint.sh`：启动 uvicorn 提供服务。

## 三、运行与验证

```bash
# 启动（后台常驻，GPU 透传，端口映射）
docker run -d --name pvr --gpus all -p 8000:8000 \
  --restart unless-stopped \
  person-vehicle-recognition:latest

# 看日志（首次启动要加载模型，约 30-60 秒）
docker logs -f pvr

# 健康检查
curl http://localhost:8000/v1/health
# → {"status":"ok","device":"GPU","backlog":512,"queue":0,"kept_results":0}

# 提交一张测试图（立即返回 session_id，异步识别）
curl -F "file=@test.jpg" http://localhost:8000/v1/tasks
# → {"session_id":"afd8...","status":"pending"}

# 凭 session_id 轮询取结果
curl http://localhost:8000/v1/tasks/afd8...
# → {"session_id":"...","status":"done","elapsed_ms":620.5,"result":{"行人":[...],"车辆":[...]}}
```

可调环境变量（`docker run -e NAME=VALUE`）：`PORT`（默认 8000）、`DEVICE`
（GPU/CPU）、`BACKLOG`（队列上限 512，满则返回 429 拒收）、`MAX_KEPT`
（内存结果保留条数 20000）。

> ★ 坑 4：429 是**设计行为**不是故障。上游（视频抽帧/抓拍流）一秒可能来几千张，
> 远超 GPU 吞吐；服务宁可拒收也不无限堆积，调用方收到 429 应丢帧或稍后重试。
> 配套抽帧客户端 `service/video_client.py` 已内置该逻辑。

> ★ 坑 5：车牌识别库 hyperlpr3 首次 import 时会往 `$HOME` 下联网下载 11.9MB
> 模型包。镜像里已内置模型包（`models/vehicle/.hyperlpr3/`），代码在 import 前
> 临时把 HOME 指过去，容器完全离线可跑。验证方式：`docker logs pvr` 里
> 不应出现任何 "Pull/Download" 字样。

## 四、视频抽帧接入

```bash
# 在能访问服务 8000 端口的任意机器上（客户端只需 cv2 + requests）
python service/video_client.py --source video.mp4 --fps 2 \
    --server http://服务器IP:8000 --output results.jsonl
python service/video_client.py --source rtsp://摄像头地址 --fps 1 \
    --server http://服务器IP:8000 --output results.jsonl
```

`--fps` 是每秒抽帧数（参数化），输出 JSONL 每行一帧的完整识别结果。

## 五、分享镜像给别人

镜像打好后是一个整体文件，不依赖 git、不依赖本项目源码：

```bash
# 1. 导出（gzip 压缩后约 2-3GB，耐心等）
docker save person-vehicle-recognition:latest | gzip > person-vehicle-recognition.tar.gz

# 2. 用 scp / 网盘 / U盘 任意方式把 tar.gz 给对方

# 3. 对方机器上（同样需 Docker + nvidia-container-toolkit + NVIDIA 驱动）：
gunzip -c person-vehicle-recognition.tar.gz | docker load
docker run -d --name pvr --gpus all -p 8000:8000 person-vehicle-recognition:latest
```

> 若对方机器没有 NVIDIA 显卡，可以 `-e DEVICE=CPU` 启动（慢，仅适合验证）：
> `docker run -d -p 8000:8000 -e DEVICE=CPU person-vehicle-recognition:latest`

## 六、常用运维命令速查

```bash
docker ps                    # 看运行中的容器
docker logs -f pvr           # 跟踪日志
docker restart pvr           # 重启
docker stop pvr && docker rm pvr   # 停止并删除容器（镜像还在）
docker images                # 看本地镜像
docker rmi 镜像名            # 删镜像
docker system df             # 看磁盘占用；docker system prune 清理无用缓存
```
