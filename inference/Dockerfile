# 推理端一体化镜像：Paddle 推理（CPU）+ Node Excel 导出。
# 构建上下文 = 项目根目录（.dockerignore 已限定只带入推理代码与部署权重）：
#   docker build -f inference/Dockerfile -t asperastra-inference .
FROM node:22-slim AS node

FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# OpenCV (非 headless，与本地依赖一致) 和 ONNX Runtime 需要的最小系统库。
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 从官方 node 镜像借二进制，避免再配 apt 源。
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -sf /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm

WORKDIR /app

# 依赖层单独先装，代码改动不触发重装。默认走国内镜像，构建时可用
# --build-arg PIP_INDEX_URL=... / NPM_REGISTRY=... 覆盖。
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG NPM_REGISTRY=https://registry.npmmirror.com
COPY inference/requirements-docker.txt inference/requirements-docker.txt
RUN pip install -i "${PIP_INDEX_URL}" -r inference/requirements-docker.txt
COPY inference/package.json inference/package.json
RUN cd inference && npm install --registry="${NPM_REGISTRY}"

# 推理代码（docker_entrypoint.sh 为容器入口）。
COPY inference/config.py \
     inference/model_adapters.py \
     inference/pipeline.py \
     inference/run.py \
     inference/export_xlsx.mjs \
     inference/docker_entrypoint.sh \
     inference/
# 部署权重（.dockerignore 保证只有这些进入上下文）。
COPY models/ models/

RUN chmod +x inference/docker_entrypoint.sh

# 输入/输出都走挂载卷：/data/images 放图片，/data/output 收 result.json/result.xlsx。
VOLUME ["/data"]
ENTRYPOINT ["bash", "inference/docker_entrypoint.sh"]
