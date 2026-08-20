# syntax=docker/dockerfile:1.7

ARG PADDLE_IMAGE=ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.2.1
ARG NATIVE_BUILDER_IMAGE=nvcr.io/nvidia/tensorrt:24.10-py3
ARG CUDA_RUNTIME_IMAGE=m.daocloud.io/docker.io/nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04
ARG UV_IMAGE=m.daocloud.io/ghcr.io/astral-sh/uv:0.11.2

FROM ${UV_IMAGE} AS uv-bin

# Paddle exists only in this stage. It reconstructs ONNX from the existing
# checkpoints/deployment weights and then validates every output with ONNX.
FROM ${PADDLE_IMAGE} AS model-exporter
COPY --from=uv-bin /uv /uvx /usr/local/bin/
ENV UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/export-env
WORKDIR /src
COPY model_export/pyproject.toml model_export/uv.lock /src/model_export/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python 3.10 --system-site-packages /opt/export-env && \
    uv sync --project /src/model_export --locked --no-dev --active
COPY model_export/ /src/model_export/
COPY vendor/PaddleDetection/ /src/vendor/PaddleDetection/
COPY models/ /src/models/
RUN /opt/export-env/bin/python /src/model_export/export_models.py \
      --root /src --output /opt/pvr-models

# NVIDIA TensorRT 24.10 supplies TensorRT 10.5.0.18, cuDNN 9.5 and the CUDA
# 12.6 toolchain. CUDA 12.6.2 is used only by the compiler image; the shipped
# runtime is the pinned CUDA 12.6.3 patch release.
FROM ${NATIVE_BUILDER_IMAGE} AS native-builder
COPY deploy/ustc-apt.sh /tmp/ustc-apt.sh
RUN bash /tmp/ustc-apt.sh && \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build python3.10-dev pybind11-dev \
      libssl-dev libstb-dev libpng-dev libwebp-dev && \
    rm -rf /var/lib/apt/lists/*
COPY native/ /src/native/
COPY deploy/assert_versions.sh /src/deploy/assert_versions.sh
RUN EXPECTED_CUDA=12.6.2 EXPECTED_TRT=10.5 EXPECTED_CUDNN=9.5 \
    bash /src/deploy/assert_versions.sh
RUN cmake -S /src/native -B /tmp/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/pvr-native && \
    cmake --build /tmp/build --parallel "$(nproc)" && \
    cmake --install /tmp/build && \
    mkdir -p /opt/pvr-libs && \
    trt_dir="$(dirname "$(ldconfig -p | awk '/libnvinfer.so.10 / {print $NF; exit}')")" && \
    test -d "$trt_dir" && \
    cp -a "$trt_dir"/libnvinfer.so* \
      "$trt_dir"/libnvinfer_builder_resource.so* \
      "$trt_dir"/libnvinfer_plugin.so* \
      "$trt_dir"/libnvonnxparser.so* /opt/pvr-libs/

# Small export-only target used by CI and low-disk remote hosts to compile and
# retrieve the native extension without materializing the large builder rootfs.
FROM scratch AS native-artifact
COPY --from=native-builder /opt/pvr-native/ /

# Build the small HTTP/client environment with uv. No Paddle, ORT, DALI, or pip
# enters this environment.
FROM ${CUDA_RUNTIME_IMAGE} AS python-runtime
COPY deploy/ustc-apt.sh /tmp/ustc-apt.sh
RUN bash /tmp/ustc-apt.sh && \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv libgomp1 libpng16-16 libwebp7 libssl3 tini && \
    rm -rf /var/lib/apt/lists/*
COPY --from=uv-bin /uv /uvx /usr/local/bin/
ENV UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/pvr/.venv
WORKDIR /opt/pvr
COPY pyproject.toml uv.lock /opt/pvr/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project && \
    rm -f /usr/local/bin/uv /usr/local/bin/uvx

FROM python-runtime AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/pvr/.venv/bin:$PATH \
    PYTHONPATH=/opt/pvr \
    LD_LIBRARY_PATH=/opt/pvr/lib:/usr/local/cuda/lib64 \
    PVR_MODEL_DIR=/opt/pvr/models \
    PVR_ENGINE_CACHE_DIR=/var/cache/pvr \
    PORT=8000
WORKDIR /opt/pvr
COPY --from=model-exporter /opt/pvr-models/ /opt/pvr/models/
COPY --from=native-builder /opt/pvr-native/ /opt/pvr/.venv/lib/python3.10/site-packages/
COPY --from=native-builder /opt/pvr-libs/ /opt/pvr/lib/
COPY pvr_api/ /opt/pvr/pvr_api/
COPY service/ /opt/pvr/service/
COPY client/ /opt/pvr/client/
COPY benchmarks/ /opt/pvr/benchmarks/
COPY deploy/entrypoint.sh /opt/pvr/deploy/entrypoint.sh
RUN mkdir -p /var/cache/pvr && chmod 0775 /var/cache/pvr
VOLUME ["/var/cache/pvr"]
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=120m --retries=3 \
  CMD python -c "import json,urllib.request; assert json.load(urllib.request.urlopen('http://127.0.0.1:8000/v1/health',timeout=2))['ready']"
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/pvr/deploy/entrypoint.sh"]
CMD ["serve"]
