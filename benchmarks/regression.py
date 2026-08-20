#!/usr/bin/env python3
"""Run an asynchronous end-to-end regression against a saved v1 result set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import httpx

from pvr_api.protocol import encode_batch, sniff_media_type


TERMINAL = {"done", "error", "expired"}


def load_images(directory: Path) -> list[tuple[str, Any, bytes]]:
    images: list[tuple[str, Any, bytes]] = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        payload = path.read_bytes()
        media_type = sniff_media_type(payload)
        if media_type is not None:
            images.append((path.name, media_type, payload))
    if not images:
        raise RuntimeError(f"no supported images found in {directory}")
    return images


def without_plate(result: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(result, ensure_ascii=False))
    for vehicle in normalized.get("车辆", []):
        vehicle.pop("车牌", None)
    return normalized


def run(args: argparse.Namespace) -> dict[str, Any]:
    images = load_images(args.input_dir)
    baseline_rows = json.loads(args.baseline.read_text(encoding="utf-8"))
    baseline = {row["图片位置"]: row["识别内容"] for row in baseline_rows}
    missing = sorted(name for name, _, _ in images if name not in baseline)
    if missing:
        raise RuntimeError(f"images absent from baseline: {missing[:3]}")

    pending: dict[str, str] = {}
    records: dict[str, dict[str, Any]] = {}
    with httpx.Client(base_url=args.server.rstrip("/"), timeout=args.http_timeout) as client:
        health = client.get("/v1/health")
        health.raise_for_status()
        if not health.json().get("ready"):
            raise RuntimeError(f"service is not ready: {health.json()}")
        for offset in range(0, len(images), args.batch_size):
            batch = images[offset : offset + args.batch_size]
            response = client.post(
                "/v1/task-batches",
                content=encode_batch((media_type, payload) for _, media_type, payload in batch),
                headers={"content-type": "application/vnd.pvr.tasks-v1"},
            )
            response.raise_for_status()
            ids = response.json()["session_ids"]
            for session_id, (name, _, _) in zip(ids, batch, strict=True):
                pending[session_id] = name

        deadline = time.monotonic() + args.result_timeout
        while pending and time.monotonic() < deadline:
            session_ids = list(pending)[:512]
            response = client.post(
                "/v1/results:batch", json={"session_ids": session_ids}
            )
            response.raise_for_status()
            for record in response.json()["results"]:
                if record.get("status") not in TERMINAL:
                    continue
                name = pending.pop(record["session_id"])
                records[name] = record
            if pending:
                time.sleep(args.poll_interval)

    errors: list[dict[str, str]] = []
    mismatches: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for name, _, _ in images:
        record = records.get(name)
        if record is None:
            errors.append({"image": name, "error": "result timeout"})
            continue
        if record["status"] != "done":
            errors.append({"image": name, "error": record.get("error", record["status"])})
            continue
        actual = record["result"]
        expected = baseline[name]
        if without_plate(actual) != without_plate(expected):
            mismatches.append(
                {
                    "image": name,
                    "expected_without_plate": without_plate(expected),
                    "actual_without_plate": without_plate(actual),
                }
            )
        results.append(
            {
                "图片位置": name,
                "识别内容": actual,
                "timing_ms": record.get("timing_ms", {}),
            }
        )

    report = {
        "images": len(images),
        "completed": len(records),
        "errors": errors,
        "non_plate_mismatch_count": len(mismatches),
        "non_plate_mismatches": mismatches,
        "results": results,
    }
    if args.output:
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, choices=range(1, 65), default=43)
    parser.add_argument("--poll-interval", type=float, default=0.05)
    parser.add_argument("--result-timeout", type=float, default=120.0)
    parser.add_argument("--http-timeout", type=float, default=20.0)
    args = parser.parse_args()
    if args.poll_interval <= 0 or args.result_timeout <= 0 or args.http_timeout <= 0:
        parser.error("timeouts and poll interval must be positive")
    return args


def main() -> int:
    args = arguments()
    report = run(args)
    summary = {key: report[key] for key in ("images", "completed", "errors", "non_plate_mismatch_count")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not report["errors"] and not report["non_plate_mismatches"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
