#!/usr/bin/env python3
"""Sample a file/camera/RTSP source and use the binary batch API."""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
import json
import time
from pathlib import Path
from typing import Any

import cv2
import httpx

from pvr_api.protocol import MediaType, encode_batch


class VideoSampler:
    def __init__(self, source: str, sample_fps: float, jpeg_quality: int):
        self.source: str | int = int(source) if source.isdigit() else source
        self.sample_fps = sample_fps
        self.jpeg_quality = jpeg_quality
        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            raise RuntimeError(f"cannot open video source: {source}")
        source_fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.interval = max(1, round((source_fps or 25.0) / sample_fps))
        self.frame_index = 0

    def next_jpeg(self) -> tuple[int, int, bytes] | None:
        while True:
            ok, frame = self.capture.read()
            if not ok:
                return None
            index = self.frame_index
            self.frame_index += 1
            if index % self.interval:
                continue
            timestamp_ms = int(self.capture.get(cv2.CAP_PROP_POS_MSEC))
            encoded, data = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            )
            if not encoded:
                continue
            return index, timestamp_ms, data.tobytes()

    def close(self) -> None:
        self.capture.release()


async def query_results(
    client: httpx.AsyncClient,
    pending: dict[str, dict[str, Any]],
    output,
) -> int:
    completed = 0
    session_ids = list(pending)[:512]
    if not session_ids:
        return completed
    response = await client.post("/v1/results:batch", json={"session_ids": session_ids})
    response.raise_for_status()
    for record in response.json()["results"]:
        if record["status"] not in {"done", "error", "expired"}:
            continue
        metadata = pending.pop(record["session_id"])
        output.write(
            json.dumps({**metadata, **record}, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        )
        output.flush()
        completed += 1
    return completed


async def run(args: argparse.Namespace) -> None:
    sampler = VideoSampler(args.source, args.sample_fps, args.jpeg_quality)
    pending: dict[str, dict[str, Any]] = {}
    batch: deque[tuple[int, int, bytes]] = deque()
    accepted = 0
    dropped = 0
    completed = 0
    output_path = Path(args.output)
    started = time.monotonic()
    limits = httpx.Limits(max_connections=4, max_keepalive_connections=4)
    async with httpx.AsyncClient(
        base_url=args.server.rstrip("/"), timeout=20.0, limits=limits
    ) as client, output_path.open("w", encoding="utf-8") as output:
        health = (await client.get("/v1/health")).json()
        if not health.get("ready"):
            raise RuntimeError(f"service is not ready: {health}")
        try:
            exhausted = False
            while not exhausted or batch or pending:
                while not exhausted and len(batch) < args.batch_size:
                    item = await asyncio.to_thread(sampler.next_jpeg)
                    if item is None:
                        exhausted = True
                        break
                    batch.append(item)
                if batch:
                    payload = encode_batch(
                        (MediaType.JPEG, item[2]) for item in batch
                    )
                    response = await client.post(
                        "/v1/task-batches",
                        content=payload,
                        headers={"content-type": "application/vnd.pvr.tasks-v1"},
                    )
                    if response.status_code == 202:
                        ids = response.json()["session_ids"]
                        for session_id, (frame, timestamp_ms, _) in zip(
                            ids, batch, strict=True
                        ):
                            pending[session_id] = {
                                "camera_id": args.camera_id,
                                "frame_index": frame,
                                "timestamp_ms": timestamp_ms,
                            }
                        accepted += len(batch)
                    elif response.status_code == 429:
                        dropped += len(batch)
                    else:
                        response.raise_for_status()
                    batch.clear()
                completed += await query_results(client, pending, output)
                while len(pending) > args.max_pending:
                    await asyncio.sleep(0.01)
                    completed += await query_results(client, pending, output)
                if exhausted and pending:
                    await asyncio.sleep(args.poll_interval)
        finally:
            sampler.close()
    elapsed = time.monotonic() - started
    print(
        f"camera={args.camera_id} accepted={accepted} dropped={dropped} "
        f"completed={completed} elapsed={elapsed:.2f}s"
    )


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="video sampler for the PVR batch API")
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--batch-size", type=int, choices=range(1, 65), default=32)
    parser.add_argument("--max-pending", type=int, default=4096)
    parser.add_argument("--poll-interval", type=float, default=0.05)
    parser.add_argument("--jpeg-quality", type=int, choices=range(1, 101), default=90)
    parser.add_argument("--output", default="results.jsonl")
    args = parser.parse_args()
    if args.sample_fps <= 0 or args.max_pending <= 0 or args.poll_interval <= 0:
        parser.error("sample-fps, max-pending, and poll-interval must be positive")
    return args


def main() -> int:
    asyncio.run(run(arguments()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
