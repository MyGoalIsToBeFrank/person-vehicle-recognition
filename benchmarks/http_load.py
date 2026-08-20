#!/usr/bin/env python3
"""Drive the binary batch API and report accepted and completed throughput."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
import json
from pathlib import Path
import time

import httpx

from pvr_api.protocol import MediaType, encode_batch


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


@dataclass(slots=True)
class Stats:
    requested: int = 0
    accepted: int = 0
    completed: int = 0
    completed_in_window: int = 0
    inference_errors: int = 0
    expired: int = 0
    rejected_429: int = 0
    http_errors: int = 0
    client_latency_ms: list[float] = field(default_factory=list)
    server_total_ms: list[float] = field(default_factory=list)


def load_jpegs(directory: Path) -> list[bytes]:
    paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    )
    if not paths:
        raise RuntimeError(f"no JPEG files found under {directory}")
    images = [path.read_bytes() for path in paths]
    invalid = [path for path, data in zip(paths, images, strict=True) if not data.startswith(b"\xff\xd8\xff")]
    if invalid:
        raise RuntimeError(f"invalid JPEG signature: {invalid[0]}")
    return images


async def run(args: argparse.Namespace) -> dict[str, object]:
    images = load_jpegs(args.input_dir)
    stats = Stats()
    pending: dict[str, float] = {}
    submit_tasks: set[asyncio.Task[None]] = set()
    submit_slots = asyncio.Semaphore(args.submit_concurrency)
    stop_polling = asyncio.Event()
    loop = asyncio.get_running_loop()
    started = loop.time()
    submit_deadline = started + args.duration
    cursor = 0

    limits = httpx.Limits(
        max_connections=args.submit_concurrency + args.poll_concurrency,
        max_keepalive_connections=args.submit_concurrency + args.poll_concurrency,
    )
    async with httpx.AsyncClient(
        base_url=args.server.rstrip("/"), timeout=args.timeout, limits=limits
    ) as client:
        health_response = await client.get("/v1/health")
        health_response.raise_for_status()
        health = health_response.json()
        if not health.get("ready"):
            raise RuntimeError(f"service is not ready: {health}")

        async def submit(payload: bytes, count: int) -> None:
            async with submit_slots:
                accepted_at = loop.time()
                try:
                    response = await client.post(
                        "/v1/task-batches",
                        content=payload,
                        headers={"content-type": "application/vnd.pvr.tasks-v1"},
                    )
                except httpx.HTTPError:
                    stats.http_errors += count
                    return
                if response.status_code == 202:
                    session_ids = response.json().get("session_ids", [])
                    if len(session_ids) != count:
                        stats.http_errors += count
                        return
                    for session_id in session_ids:
                        pending[session_id] = accepted_at
                    stats.accepted += count
                elif response.status_code == 429:
                    stats.rejected_429 += count
                else:
                    stats.http_errors += count

        async def poll_once(ids: list[str]) -> None:
            if not ids:
                return
            try:
                response = await client.post(
                    "/v1/results:batch", json={"session_ids": ids}
                )
                response.raise_for_status()
            except httpx.HTTPError:
                return
            now = loop.time()
            for record in response.json().get("results", []):
                status = record.get("status")
                if status not in {"done", "error", "expired"}:
                    continue
                accepted_at = pending.pop(record["session_id"], None)
                if accepted_at is None:
                    continue
                stats.completed += 1
                if now <= submit_deadline:
                    stats.completed_in_window += 1
                stats.client_latency_ms.append((now - accepted_at) * 1000.0)
                timing = record.get("timing_ms") or {}
                if "total" in timing:
                    stats.server_total_ms.append(float(timing["total"]))
                if status == "error":
                    stats.inference_errors += 1
                elif status == "expired":
                    stats.expired += 1

        async def poller() -> None:
            while not stop_polling.is_set():
                ids = list(pending)[: 512 * args.poll_concurrency]
                await asyncio.gather(*(
                    poll_once(ids[offset : offset + 512])
                    for offset in range(0, len(ids), 512)
                ))
                await asyncio.sleep(args.poll_interval)

        poll_task = asyncio.create_task(poller())
        next_send = started
        try:
            while loop.time() < submit_deadline:
                while (
                    len(pending) >= args.max_pending
                    or len(submit_tasks) >= args.submit_concurrency
                ):
                    await asyncio.sleep(0.001)
                    submit_tasks = {task for task in submit_tasks if not task.done()}
                batch = [
                    (MediaType.JPEG, images[(cursor + index) % len(images)])
                    for index in range(args.batch_size)
                ]
                cursor = (cursor + args.batch_size) % len(images)
                payload = encode_batch(batch)
                stats.requested += args.batch_size
                task = asyncio.create_task(submit(payload, args.batch_size))
                submit_tasks.add(task)
                task.add_done_callback(submit_tasks.discard)
                if args.rate > 0:
                    next_send += args.batch_size / args.rate
                    delay = next_send - loop.time()
                    if delay > 0:
                        await asyncio.sleep(delay)
            if submit_tasks:
                await asyncio.gather(*submit_tasks)
            drain_deadline = loop.time() + args.drain_timeout
            while pending and loop.time() < drain_deadline:
                await asyncio.sleep(args.poll_interval)
        finally:
            stop_polling.set()
            await poll_task

    finished = loop.time()
    result: dict[str, object] = {
        "input_images": len(images),
        "batch_size": args.batch_size,
        "target_images_per_second": args.rate,
        "submit_window_seconds": args.duration,
        "wall_seconds_including_drain": finished - started,
        "requested": stats.requested,
        "accepted": stats.accepted,
        "completed": stats.completed,
        "completed_in_submit_window": stats.completed_in_window,
        "pending_after_drain": len(pending),
        "rejected_429": stats.rejected_429,
        "http_errors": stats.http_errors,
        "inference_errors": stats.inference_errors,
        "expired": stats.expired,
        "accepted_images_per_second": stats.accepted / args.duration,
        "completed_images_per_second": stats.completed_in_window / args.duration,
        "completion_over_acceptance": (
            stats.completed / stats.accepted if stats.accepted else 0.0
        ),
        "client_latency_ms": {
            "p50": percentile(stats.client_latency_ms, 0.50),
            "p95": percentile(stats.client_latency_ms, 0.95),
            "p99": percentile(stats.client_latency_ms, 0.99),
            "max": max(stats.client_latency_ms, default=None),
        },
        "server_total_ms": {
            "p50": percentile(stats.server_total_ms, 0.50),
            "p95": percentile(stats.server_total_ms, 0.95),
            "p99": percentile(stats.server_total_ms, 0.99),
            "max": max(stats.server_total_ms, default=None),
        },
    }
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--rate", type=float, default=0.0, help="0 means saturation")
    parser.add_argument("--batch-size", type=int, choices=(1, 8, 16, 32, 64), default=64)
    parser.add_argument("--submit-concurrency", type=int, default=8)
    parser.add_argument("--poll-concurrency", type=int, default=2)
    parser.add_argument("--max-pending", type=int, default=8192)
    parser.add_argument("--poll-interval", type=float, default=0.01)
    parser.add_argument("--drain-timeout", type=float, default=60.0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--enforce-sla", action="store_true")
    args = parser.parse_args()
    for name in (
        "duration",
        "submit_concurrency",
        "poll_concurrency",
        "max_pending",
        "poll_interval",
        "drain_timeout",
        "timeout",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.rate < 0:
        parser.error("--rate cannot be negative")
    return args


def main() -> int:
    args = arguments()
    result = asyncio.run(run(args))
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    if not args.enforce_sla:
        return 0
    p95 = result["server_total_ms"]["p95"]
    passed = (
        result["accepted"] > 0
        and result["rejected_429"] == 0
        and result["http_errors"] == 0
        and result["inference_errors"] == 0
        and result["expired"] == 0
        and result["completion_over_acceptance"] >= 0.999
        and p95 is not None
        and p95 <= 1000.0
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
