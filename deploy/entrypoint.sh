#!/usr/bin/env bash
set -euo pipefail

command_name="${1:-serve}"
if [ "$#" -gt 0 ]; then shift; fi

case "$command_name" in
  serve)
    exec python -m uvicorn service.app:app \
      --host 0.0.0.0 \
      --port "${PORT:-8000}" \
      --limit-concurrency "${PVR_HTTP_CONCURRENCY:-64}" \
      --backlog "${PVR_HTTP_BACKLOG:-2048}" \
      --timeout-keep-alive 5 \
      "$@"
    ;;
  video-client)
    exec python -m client.video_submit "$@"
    ;;
  benchmark)
    exec python -m benchmarks.http_load "$@"
    ;;
  regression)
    exec python -m benchmarks.regression "$@"
    ;;
  *)
    echo "usage: entrypoint.sh {serve|video-client|benchmark|regression} [arguments...]" >&2
    exit 64
    ;;
esac
