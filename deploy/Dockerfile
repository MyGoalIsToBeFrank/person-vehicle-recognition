# 行人/车辆属性识别服务镜像（瘦身版，默认推荐）
#
# python:3.10-slim + pip 安装 paddlepaddle-gpu：最终镜像约 3.5-4GB，
# 适合导出分享。首次构建需下载 2-3GB NVIDIA 轮子（约 10-15 分钟），
# 之后 Docker 层缓存命中，重建只需几十秒。
# 若想省首次构建时间、不在乎镜像体积，可用 deploy/Dockerfile.full
#（Paddle 官方镜像做底座，但产物约 14GB）。
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    INFERENCE_RUN_IN_PLACE=1 \
    DEVICE=GPU \
    PORT=8000

# libgomp1：slim 基础镜像没有 OpenMP 运行时，paddle import 会直接 ImportError。
# 服务器在国内，apt 默认源 deb.debian.org 可能连不上（exit 100），先换中科大镜像。
RUN sed -i 's|deb.debian.org|mirrors.ustc.edu.cn|g' /etc/apt/sources.list.d/debian.sources \
        2>/dev/null || sed -i 's|deb.debian.org|mirrors.ustc.edu.cn|g' /etc/apt/sources.list
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# paddlepaddle-gpu 走飞桨官方 cu126 索引（自带 nvidia-*-cu12 依赖轮子）；
# 其余小包走清华镜像加速。
RUN pip install --no-cache-dir paddlepaddle-gpu==3.3.0 \
        -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
        --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple

# onnxruntime-gpu 需要找到 pip 轮子里的 CUDA/cuDNN 动态库
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib

COPY deploy/requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
        -r /tmp/requirements-docker.txt \
    && pip install --no-cache-dir --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple \
        hyperlpr3==0.1.3 \
    && rm /tmp/requirements-docker.txt

WORKDIR /app
COPY inference/config.py inference/model_adapters.py inference/pipeline.py inference/run.py /app/inference/
COPY service/app.py /app/service/
COPY deploy/docker_entrypoint.sh /app/deploy/docker_entrypoint.sh
COPY models/ /app/models/

EXPOSE 8000
ENTRYPOINT ["bash", "/app/deploy/docker_entrypoint.sh"]
