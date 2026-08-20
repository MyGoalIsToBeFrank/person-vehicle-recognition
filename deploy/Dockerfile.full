# 行人/车辆属性识别服务镜像（Paddle GPU 版）
#
# base 用 Paddle 官方 GPU 镜像：自带 paddlepaddle-gpu + CUDA + cuDNN，
# 增量只装少量 pip 包，构建快、无需每次下载数 GB 的 NVIDIA 轮子。
FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.3.0-gpu-cuda12.6-cudnn9.5

ENV PYTHONUNBUFFERED=1 \
    INFERENCE_RUN_IN_PLACE=1 \
    DEVICE=GPU \
    PORT=8000

# 增量依赖（清华镜像加速）。hyperlpr3 必须 --no-deps，见 requirements 注释。
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
