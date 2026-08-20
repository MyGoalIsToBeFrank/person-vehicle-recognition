#!/usr/bin/env bash
set -euo pipefail

expected_cuda="${EXPECTED_CUDA:?EXPECTED_CUDA is required}"
expected_trt="${EXPECTED_TRT:?EXPECTED_TRT is required}"
expected_cudnn="${EXPECTED_CUDNN:?EXPECTED_CUDNN is required}"

if [[ -n "${CUDA_VERSION:-}" ]]; then
  cuda_version="$CUDA_VERSION"
elif [[ -f /usr/local/cuda/version.json ]]; then
  cuda_version="$(python3 - <<'PY'
import json
from pathlib import Path

data = json.loads(Path('/usr/local/cuda/version.json').read_text())
value = data.get('cuda', data)
print(value['version'] if isinstance(value, dict) else value)
PY
)"
else
  cuda_version="$(nvcc --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p' | head -n1)"
fi
case "$cuda_version" in
  "$expected_cuda"*) ;;
  *) echo "expected CUDA $expected_cuda, found $cuda_version" >&2; exit 1 ;;
esac

trtexec_path="$(command -v trtexec || true)"
if [[ -z "$trtexec_path" && -x /opt/tensorrt/bin/trtexec ]]; then
  trtexec_path=/opt/tensorrt/bin/trtexec
fi
test -n "$trtexec_path" || { echo "trtexec is missing" >&2; exit 1; }
if [[ -n "${TENSORRT_VERSION:-}" ]]; then
  trt_version="$TENSORRT_VERSION"
else
  trt_header="$(find /usr/include /usr/local/cuda/include -maxdepth 3 \
    -type f -name NvInferVersion.h -print -quit 2>/dev/null)"
  test -n "$trt_header" || { echo "TensorRT version metadata is missing" >&2; exit 1; }
  trt_version="$(awk '
    /#define NV_TENSORRT_MAJOR/ {major=$3}
    /#define NV_TENSORRT_MINOR/ {minor=$3}
    /#define NV_TENSORRT_PATCH/ {patch=$3}
    END {print major "." minor "." patch}
  ' "$trt_header")"
fi
case "$trt_version" in
  "$expected_trt"*) ;;
  *) echo "expected TensorRT $expected_trt, found ${trt_version:-unknown}" >&2; exit 1 ;;
esac

if [[ -n "${CUDNN_VERSION:-}" ]]; then
  cudnn_version="$CUDNN_VERSION"
else
  cudnn_header="$(find /usr/include /usr/local/cuda/include -maxdepth 3 \
    -type f \( -name 'cudnn_version.h' -o -name 'cudnn_version_v*.h' \) \
    -print -quit 2>/dev/null)"
  test -n "$cudnn_header" || { echo "cuDNN version metadata is missing" >&2; exit 1; }
  cudnn_version="$(awk '
    /#define CUDNN_MAJOR/ {major=$3}
    /#define CUDNN_MINOR/ {minor=$3}
    /#define CUDNN_PATCHLEVEL/ {patch=$3}
    END {print major "." minor "." patch}
  ' "$cudnn_header")"
fi
case "$cudnn_version" in
  "$expected_cudnn"*) ;;
  *) echo "expected cuDNN $expected_cudnn, found ${cudnn_version:-unknown}" >&2; exit 1 ;;
esac

printf 'CUDA=%s TensorRT=%s cuDNN=%s\n' "$cuda_version" "$trt_version" "$cudnn_version"
