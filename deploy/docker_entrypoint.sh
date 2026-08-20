#!/bin/bash
# 容器入口：对外部图片目录跑推理，把 result.json 写进输出目录。
# 可用环境变量覆盖：IMAGES_DIR、OUTPUT_DIR、DEVICE（CPU/GPU，默认 GPU 自动回落 CPU）。
set -euo pipefail

IMAGES_DIR="${IMAGES_DIR:-/data/images}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/output}"
DEVICE="${DEVICE:-GPU}"

mkdir -p "${OUTPUT_DIR}"
python inference/run.py \
    --device "${DEVICE}" \
    --data-dir "${IMAGES_DIR}" \
    --output-dir "${OUTPUT_DIR}"
echo "完成: ${OUTPUT_DIR}/result.json"
