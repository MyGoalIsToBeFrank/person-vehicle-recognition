# 推理端一体化镜像：onnxruntime 单引擎（GPU 自动探测，无卡回落 CPU），只输出 result.json。
# 构建上下文 = 项目根目录（.dockerignore 已限定只带入推理代码与部署权重）：
#   docker build -f deploy/Dockerfile -t person-vehicle-recognition .
FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    INFERENCE_RUN_IN_PLACE=1

# OpenCV 用 headless 构建（容器里不需要 GUI 库），ONNX Runtime 只需要 libgomp1。
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依赖层单独先装，代码改动不触发重装。默认走国内镜像，构建时可用
# --build-arg PIP_INDEX_URL=... 覆盖。
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
COPY deploy/requirements-docker.txt deploy/requirements-docker.txt
RUN pip install -i "${PIP_INDEX_URL}" -r deploy/requirements-docker.txt \
    && pip install -i "${PIP_INDEX_URL}" --no-deps hyperlpr3==0.1.3

# onnxruntime-gpu 的动态库来自 pip 的 nvidia-* 轮子，需要显式进入链接器搜索路径。
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/site-packages/nvidia/cufft/lib:/usr/local/lib/python3.10/site-packages/nvidia/curand/lib:/usr/local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib

# 推理代码（deploy/docker_entrypoint.sh 为容器入口）。
COPY inference/config.py \
     inference/model_adapters.py \
     inference/pipeline.py \
     inference/run.py \
     inference/
COPY deploy/docker_entrypoint.sh deploy/
# 部署权重（.dockerignore 保证只有这些进入上下文）。
COPY models/ models/

RUN chmod +x deploy/docker_entrypoint.sh

# 输入/输出都走挂载卷：/data/images 放图片，/data/output 收 result.json。
VOLUME ["/data"]
ENTRYPOINT ["bash", "deploy/docker_entrypoint.sh"]
