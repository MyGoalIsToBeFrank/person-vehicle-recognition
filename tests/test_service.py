from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from pvr_api.protocol import MediaType, encode_batch
from service.app import create_app
from service.settings import Settings


JPEG = b"\xff\xd8\xff" + b"valid-enough-for-api-test"


class QueueFullError(RuntimeError):
    pass


class FakeEngine:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.closed = False
        self.ready = True
        self.reject = False

    def submit(self, payload: bytes, media_type: int) -> str:
        return self.submit_many([payload], [media_type])[0]

    def submit_many(self, payloads: list[bytes], media_types: list[int]) -> list[str]:
        if self.reject:
            raise QueueFullError("full")
        ids = []
        for _ in payloads:
            session_id = f"id-{len(self.records)}"
            self.records[session_id] = {"status": "pending"}
            ids.append(session_id)
        return ids

    def get(self, session_id: str) -> dict[str, Any] | None:
        return self.records.get(session_id)

    def get_many(self, session_ids: list[str]) -> list[dict[str, Any] | None]:
        return [self.get(session_id) for session_id in session_ids]

    def health(self) -> dict[str, Any]:
        return {"ready": self.ready, "status": "ready" if self.ready else "initializing"}

    def prometheus(self) -> str:
        return "pvr_images_accepted_total 0\n"

    def close(self) -> None:
        self.closed = True


def settings() -> Settings:
    return replace(
        Settings.from_env(),
        model_dir=Path("models"),
        engine_cache_dir=Path("cache"),
        max_image_bytes=1024,
        max_batch_bytes=4096,
    )


def test_submit_and_query_are_decoupled() -> None:
    engine = FakeEngine()
    with TestClient(create_app(settings=settings(), engine=engine)) as client:
        response = client.post(
            "/v1/tasks", files={"file": ("frame.jpg", JPEG, "image/jpeg")}
        )
        assert response.status_code == 202
        assert response.json() == {"session_id": "id-0", "status": "pending"}
        assert client.get("/v1/tasks/id-0").json()["status"] == "pending"
    assert engine.closed


def test_batch_order_and_batch_results() -> None:
    engine = FakeEngine()
    payload = encode_batch([(MediaType.JPEG, JPEG), (MediaType.JPEG, JPEG + b"2")])
    with TestClient(create_app(settings=settings(), engine=engine)) as client:
        response = client.post(
            "/v1/task-batches",
            content=payload,
            headers={"content-type": "application/vnd.pvr.tasks-v1"},
        )
        assert response.status_code == 202
        assert response.json()["session_ids"] == ["id-0", "id-1"]
        queried = client.post(
            "/v1/results:batch", json={"session_ids": ["id-1", "missing", "id-0"]}
        ).json()["results"]
        assert [item["session_id"] for item in queried] == ["id-1", "missing", "id-0"]
        assert queried[1]["status"] == "unknown"


def test_queue_full_has_retry_after() -> None:
    engine = FakeEngine()
    engine.reject = True
    with TestClient(create_app(settings=settings(), engine=engine)) as client:
        response = client.post(
            "/v1/tasks", files={"file": ("frame.jpg", JPEG, "image/jpeg")}
        )
        assert response.status_code == 429
        assert response.headers["retry-after"] == "1"


def test_health_is_503_until_native_engine_is_ready() -> None:
    engine = FakeEngine()
    engine.ready = False
    with TestClient(create_app(settings=settings(), engine=engine)) as client:
        response = client.get("/v1/health")
        assert response.status_code == 503
        body = response.json()
        assert body["queue"]["max_images"] == settings().max_queue_images
        assert body["result_cache"]["max_records"] == settings().max_result_records
        assert body["request_limits"]["max_batch_images"] == 64


def test_limits_and_media_type_are_enforced_before_enqueue() -> None:
    engine = FakeEngine()
    with TestClient(create_app(settings=settings(), engine=engine)) as client:
        assert client.post(
            "/v1/tasks", files={"file": ("x.bin", b"not-image", "application/octet-stream")}
        ).status_code == 415
        assert client.post(
            "/v1/tasks", files={"file": ("large.jpg", b"\xff\xd8\xff" + b"x" * 1024)}
        ).status_code == 413
        assert client.post(
            "/v1/tasks",
            files={"file": ("huge.jpg", JPEG + b"x" * ((1 << 20) + 2048))},
        ).status_code == 413
    assert not engine.records


def test_known_oversized_dimensions_are_rejected_before_enqueue() -> None:
    engine = FakeEngine()
    constrained = replace(settings(), max_image_pixels=100)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\x0dIHDR"
        + (20).to_bytes(4, "big")
        + (20).to_bytes(4, "big")
        + b"rest"
    )
    with TestClient(create_app(settings=constrained, engine=engine)) as client:
        response = client.post(
            "/v1/tasks", files={"file": ("large.png", png, "image/png")}
        )
        assert response.status_code == 413
    assert not engine.records
