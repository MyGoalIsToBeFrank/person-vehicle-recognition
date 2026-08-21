#!/usr/bin/env python3
"""Submit a batch to PVR v2 and poll its asynchronous results.

This client intentionally uses only the Python standard library so recipients
can test an unpacked image without installing project dependencies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


HEADER = struct.Struct("<4sHHI")
ITEM_HEADER = struct.Struct("<BBHI")
SIGNATURES = (
    (b"\xff\xd8\xff", 1, "JPEG"),
    (b"\x89PNG\r\n\x1a\n", 2, "PNG"),
    (b"BM", 3, "BMP"),
)
TERMINAL_STATUSES = {"done", "error", "unknown", "expired"}


class ClientError(RuntimeError):
    """A readable command-line error."""


def media_type(payload: bytes) -> tuple[int, str]:
    for signature, value, name in SIGNATURES:
        if payload.startswith(signature):
            return value, name
    if len(payload) >= 12 and payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
        return 4, "WebP"
    raise ClientError("unsupported image signature (expected JPEG/PNG/BMP/WebP)")


def encode_batch(images: list[tuple[int, bytes]]) -> bytes:
    chunks = [HEADER.pack(b"PVRB", 1, 0, len(images))]
    for kind, payload in images:
        chunks.extend((ITEM_HEADER.pack(kind, 0, 0, len(payload)), payload))
    return b"".join(chunks)


def request_json(
    method: str,
    url: str,
    *,
    body: bytes | None = None,
    content_type: str | None = None,
    timeout: float = 30.0,
) -> tuple[int, dict[str, Any]]:
    headers = {"Accept": "application/json"}
    if content_type:
        headers["Content-Type"] = content_type
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
            return response.status, json.loads(raw)
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise ClientError(f"HTTP {exc.code} {url}: {raw}") from exc
    except URLError as exc:
        raise ClientError(f"cannot reach {url}: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise ClientError(f"server returned non-JSON data from {url}") from exc


def load_images(paths: list[Path]) -> tuple[list[tuple[int, bytes]], list[dict[str, Any]]]:
    if not 1 <= len(paths) <= 64:
        raise ClientError("provide between 1 and 64 images")
    images: list[tuple[int, bytes]] = []
    metadata: list[dict[str, Any]] = []
    total = 0
    for path in paths:
        if not path.is_file():
            raise ClientError(f"image does not exist: {path}")
        payload = path.read_bytes()
        kind, name = media_type(payload)
        if not 1 <= len(payload) <= 8 * 1024 * 1024:
            raise ClientError(f"image is empty or exceeds 8 MiB: {path}")
        total += len(payload)
        images.append((kind, payload))
        metadata.append({"file": str(path), "format": name, "bytes": len(payload)})
    if total + HEADER.size + ITEM_HEADER.size * len(images) > 64 * 1024 * 1024:
        raise ClientError("encoded batch exceeds 64 MiB")
    return images, metadata


def poll_results(
    server: str,
    session_ids: list[str],
    *,
    deadline: float,
    interval: float,
) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    while time.monotonic() < deadline:
        query = json.dumps({"session_ids": session_ids}).encode("utf-8")
        _, response = request_json(
            "POST",
            f"{server}/v1/results:batch",
            body=query,
            content_type="application/json",
        )
        records = response.get("results")
        if not isinstance(records, list) or len(records) != len(session_ids):
            raise ClientError("batch result response has an unexpected shape")
        latest = {record["session_id"]: record for record in records}
        if all(latest.get(item, {}).get("status") in TERMINAL_STATUSES for item in session_ids):
            return [latest[item] for item in session_ids]
        time.sleep(interval)
    pending = [
        item
        for item in session_ids
        if latest.get(item, {}).get("status") not in TERMINAL_STATUSES
    ]
    raise ClientError(f"timed out waiting for {len(pending)} result(s): {pending}")


def parse_args() -> argparse.Namespace:
    package_samples = sorted((Path(__file__).resolve().parent / "samples").glob("*"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", type=Path, help="1-64 image files")
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--poll-interval", type=float, default=0.05)
    parser.add_argument("--output", type=Path, help="also write formatted JSON here")
    args = parser.parse_args()
    if not args.images:
        args.images = [path for path in package_samples if path.is_file()]
    if args.timeout <= 0 or args.poll_interval <= 0:
        parser.error("timeout and poll interval must be positive")
    return args


def main() -> int:
    args = parse_args()
    server = args.server.rstrip("/")
    try:
        _, health = request_json("GET", f"{server}/v1/health")
        if not health.get("ready"):
            raise ClientError(f"service is not ready: {health}")

        images, metadata = load_images(args.images)
        started = time.monotonic()
        status, accepted = request_json(
            "POST",
            f"{server}/v1/task-batches",
            body=encode_batch(images),
            content_type="application/vnd.pvr.tasks-v1",
        )
        if status != 202:
            raise ClientError(f"expected HTTP 202, received HTTP {status}")
        session_ids = accepted.get("session_ids")
        if not isinstance(session_ids, list) or len(session_ids) != len(images):
            raise ClientError("submit response has an unexpected session_ids list")

        records = poll_results(
            server,
            session_ids,
            deadline=time.monotonic() + args.timeout,
            interval=args.poll_interval,
        )
        report = {
            "server": server,
            "client_elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            "items": [
                {**info, "session_id": session_id, "response": record}
                for info, session_id, record in zip(
                    metadata, session_ids, records, strict=True
                )
            ],
        }
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        print(rendered)
        if args.output:
            args.output.write_text(rendered + "\n", encoding="utf-8")
        return 0 if all(item["status"] == "done" for item in records) else 2
    except ClientError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
