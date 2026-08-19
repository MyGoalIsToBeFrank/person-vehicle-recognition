#!/bin/bash
# 容器入口：对外部图片目录跑推理，把 result.json / result.xlsx 写进输出目录。
# 可用环境变量覆盖：IMAGES_DIR、OUTPUT_DIR、DEVICE（CPU/GPU，默认 CPU）。
set -euo pipefail

IMAGES_DIR="${IMAGES_DIR:-/data/images}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/output}"
DEVICE="${DEVICE:-CPU}"

mkdir -p "${OUTPUT_DIR}"
python inference/run.py \
    --device "${DEVICE}" \
    --data-dir "${IMAGES_DIR}" \
    --output-dir "${OUTPUT_DIR}"
node inference/export_xlsx.mjs \
    --result-json "${OUTPUT_DIR}/result.json" \
    --data-dir "${IMAGES_DIR}" \
    --output-xlsx "${OUTPUT_DIR}/result.xlsx"
echo "完成: ${OUTPUT_DIR}/result.json 与 result.xlsx"
