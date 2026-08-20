"""Thin asynchronous HTTP boundary around the single native GPU engine."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable, Protocol

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from pvr_api.protocol import (
    ProtocolError,
    decode_batch,
    exceeds_pixel_limit,
    sniff_media_type,
)
from service.settings import Settings


class Engine(Protocol):
    def submit(self, payload: bytes, media_type: int) -> str: ...

    def submit_many(
        self, payloads: list[bytes], media_types: list[int]
    ) -> list[str]: ...

    def get(self, session_id: str) -> dict[str, Any] | None: ...

    def get_many(self, session_ids: list[str]) -> list[dict[str, Any] | None]: ...

    def health(self) -> dict[str, Any]: ...

    def prometheus(self) -> str: ...

    def close(self) -> None: ...


class BatchResultRequest(BaseModel):
    session_ids: list[str] = Field(min_length=1, max_length=512)


class SingleImageBodyLimitMiddleware:
    """Bound multipart bytes before Starlette creates a temporary upload file."""

    def __init__(
        self, app: Any, *, limit: int, slots: asyncio.Semaphore
    ) -> None:
        self.app = app
        self.limit = limit
        self.slots = slots

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        if scope.get("type") != "http" or scope.get("method") != "POST" or scope.get(
            "path"
        ) != "/v1/tasks":
            await self.app(scope, receive, send)
            return

        async with self.slots:
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                if len(body) + len(chunk) > self.limit:
                    payload = b'{"detail":"multipart body limit exceeded"}'
                    await send(
                        {
                            "type": "http.response.start",
                            "status": 413,
                            "headers": [
                                (b"content-type", b"application/json"),
                                (b"content-length", str(len(payload)).encode("ascii")),
                                (b"connection", b"close"),
                            ],
                        }
                    )
                    await send({"type": "http.response.body", "body": payload})
                    return
                body.extend(chunk)
                if not message.get("more_body", False):
                    break

            delivered = False

            async def replay() -> dict[str, Any]:
                nonlocal delivered
                if delivered:
                    return {"type": "http.disconnect"}
                delivered = True
                return {
                    "type": "http.request",
                    "body": bytes(body),
                    "more_body": False,
                }

            await self.app(scope, replay, send)


async def _read_limited_body(request: Request, limit: int) -> bytes:
    """Read an ASGI body without allowing an omitted length to bypass the cap."""
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > limit:
            raise HTTPException(status_code=413, detail="batch byte limit exceeded")
        body.extend(chunk)
    return bytes(body)


def _load_native_engine(settings: Settings) -> Engine:
    try:
        import pvr_native
    except ImportError as exc:
        raise RuntimeError("pvr_native extension is not installed") from exc
    return pvr_native.Engine(settings.native_config())


def _translate_native_error(exc: Exception) -> HTTPException:
    name = type(exc).__name__
    if name == "QueueFullError":
        return HTTPException(
            status_code=429,
            detail="recognition queue is full",
            headers={"Retry-After": "1"},
        )
    if name == "NotReadyError":
        return HTTPException(status_code=503, detail="native engine is not ready")
    if name == "PayloadTooLargeError":
        return HTTPException(status_code=413, detail=str(exc))
    raise exc


def create_app(
    *, settings: Settings | None = None, engine: Engine | None = None
) -> FastAPI:
    settings = settings or Settings.from_env()
    ingest_slots = asyncio.Semaphore(settings.ingest_concurrency)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine or _load_native_engine(settings)
        app.state.submit_slots = asyncio.Semaphore(settings.submit_concurrency)
        try:
            yield
        finally:
            app.state.engine.close()

    application = FastAPI(
        title="person-vehicle-recognition",
        version="2.0.0",
        lifespan=lifespan,
    )
    # One MiB covers multipart boundaries and headers while the image itself
    # remains governed by max_image_bytes in the route and native engine.
    application.add_middleware(
        SingleImageBodyLimitMiddleware,
        limit=settings.max_image_bytes + (1 << 20),
        slots=ingest_slots,
    )

    @application.post("/v1/tasks", status_code=202)
    async def submit(request: Request, file: UploadFile = File(...)) -> dict[str, str]:
        payload = await file.read(settings.max_image_bytes + 1)
        if len(payload) > settings.max_image_bytes:
            raise HTTPException(status_code=413, detail="image byte limit exceeded")
        media_type = sniff_media_type(payload)
        if media_type is None:
            raise HTTPException(status_code=415, detail="unsupported image format")
        if exceeds_pixel_limit(payload, media_type, settings.max_image_pixels):
            raise HTTPException(status_code=413, detail="image pixel limit exceeded")
        try:
            async with request.app.state.submit_slots:
                session_id = await asyncio.to_thread(
                    request.app.state.engine.submit, payload, int(media_type)
                )
        except Exception as exc:
            raise _translate_native_error(exc) from exc
        return {"session_id": session_id, "status": "pending"}

    @application.post("/v1/task-batches", status_code=202)
    async def submit_batch(request: Request) -> dict[str, object]:
        content_type = request.headers.get("content-type", "").split(";", 1)[0]
        if content_type != "application/vnd.pvr.tasks-v1":
            raise HTTPException(status_code=415, detail="unsupported batch content type")
        declared = request.headers.get("content-length")
        if declared:
            try:
                declared_size = int(declared)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="invalid content-length") from exc
            if declared_size > settings.max_batch_bytes:
                raise HTTPException(status_code=413, detail="batch byte limit exceeded")
        async with ingest_slots:
            payload = await _read_limited_body(request, settings.max_batch_bytes)
            try:
                images = decode_batch(
                    payload,
                    max_images=settings.max_batch_images,
                    max_image_bytes=settings.max_image_bytes,
                    max_batch_bytes=settings.max_batch_bytes,
                    max_image_pixels=settings.max_image_pixels,
                )
            except ProtocolError as exc:
                message = str(exc)
                status = 413 if "limit exceeded" in message else 415
                raise HTTPException(status_code=status, detail=message) from exc
            try:
                async with request.app.state.submit_slots:
                    ids = await asyncio.to_thread(
                        request.app.state.engine.submit_many,
                        [image for _, image in images],
                        [int(media_type) for media_type, _ in images],
                    )
            except Exception as exc:
                raise _translate_native_error(exc) from exc
        return {"version": 1, "session_ids": ids, "status": "pending"}

    @application.get("/v1/tasks/{session_id}")
    async def query(request: Request, session_id: str) -> dict[str, Any]:
        record = await asyncio.to_thread(request.app.state.engine.get, session_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown session_id")
        if record.get("status") == "expired":
            raise HTTPException(status_code=410, detail="result expired")
        return {"session_id": session_id, **record}

    @application.post("/v1/results:batch")
    async def query_batch(
        request: Request, body: BatchResultRequest
    ) -> dict[str, object]:
        records = await asyncio.to_thread(
            request.app.state.engine.get_many, body.session_ids
        )
        results: list[dict[str, Any]] = []
        for session_id, record in zip(body.session_ids, records, strict=True):
            if record is None:
                results.append({"session_id": session_id, "status": "unknown"})
            else:
                results.append({"session_id": session_id, **record})
        return {"results": results}

    @application.get("/v1/health")
    async def health(request: Request, response: Response) -> dict[str, Any]:
        state = await asyncio.to_thread(request.app.state.engine.health)
        state.setdefault("queue", {}).update(
            {
                "max_images": settings.max_queue_images,
                "max_bytes": settings.max_queue_bytes,
            }
        )
        state.setdefault("result_cache", {}).update(
            {
                "max_bytes": settings.max_result_bytes,
                "max_records": settings.max_result_records,
                "ttl_seconds": settings.result_ttl_seconds,
            }
        )
        state["request_limits"] = {
            "max_image_bytes": settings.max_image_bytes,
            "max_image_pixels": settings.max_image_pixels,
            "max_batch_images": settings.max_batch_images,
            "max_batch_bytes": settings.max_batch_bytes,
        }
        if not state.get("ready", False):
            response.status_code = 503
        return state

    @application.get("/metrics", response_class=PlainTextResponse)
    async def metrics(request: Request) -> str:
        return await asyncio.to_thread(request.app.state.engine.prometheus)

    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("service.app:app", host="0.0.0.0", port=Settings.from_env().port)
