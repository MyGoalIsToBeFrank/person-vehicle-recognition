"""Validated runtime limits. Every memory-growing dimension has a hard cap."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def _positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    model_dir: Path
    engine_cache_dir: Path
    max_image_bytes: int
    max_image_pixels: int
    max_batch_images: int
    max_batch_bytes: int
    max_queue_images: int
    max_queue_bytes: int
    batch_wait_us: int
    result_ttl_seconds: int
    max_result_bytes: int
    max_result_records: int
    ingest_concurrency: int
    submit_concurrency: int
    port: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_dir=Path(os.environ.get("PVR_MODEL_DIR", "/opt/pvr/models")),
            engine_cache_dir=Path(
                os.environ.get("PVR_ENGINE_CACHE_DIR", "/var/cache/pvr")
            ),
            max_image_bytes=_positive_int("PVR_MAX_IMAGE_BYTES", 8 << 20),
            max_image_pixels=_positive_int("PVR_MAX_IMAGE_PIXELS", 20_000_000),
            max_batch_images=_positive_int("PVR_MAX_BATCH_IMAGES", 64),
            max_batch_bytes=_positive_int("PVR_MAX_BATCH_BYTES", 64 << 20),
            max_queue_images=_positive_int("PVR_MAX_QUEUE_IMAGES", 8192),
            max_queue_bytes=_positive_int("PVR_MAX_QUEUE_BYTES", 1 << 30),
            batch_wait_us=_positive_int("PVR_BATCH_WAIT_US", 2000),
            result_ttl_seconds=_positive_int("PVR_RESULT_TTL_SECONDS", 60),
            max_result_bytes=_positive_int("PVR_MAX_RESULT_BYTES", 1 << 30),
            max_result_records=_positive_int("PVR_MAX_RESULT_RECORDS", 262_144),
            ingest_concurrency=_positive_int("PVR_INGEST_CONCURRENCY", 2),
            submit_concurrency=_positive_int("PVR_SUBMIT_CONCURRENCY", 8),
            port=_positive_int("PORT", 8000),
        )

    def native_config(self) -> dict[str, object]:
        return {
            "model_dir": str(self.model_dir),
            "engine_cache_dir": str(self.engine_cache_dir),
            "max_image_bytes": self.max_image_bytes,
            "max_image_pixels": self.max_image_pixels,
            "max_batch_images": self.max_batch_images,
            "max_queue_images": self.max_queue_images,
            "max_queue_bytes": self.max_queue_bytes,
            "batch_wait_us": self.batch_wait_us,
            "result_ttl_seconds": self.result_ttl_seconds,
            "max_result_bytes": self.max_result_bytes,
            "max_result_records": self.max_result_records,
        }
