"""Versioned binary request protocol used by the high-throughput endpoint.

All integers are little-endian. A request is::

    magic[4] = "PVRB"
    version:u16
    flags:u16
    image_count:u32
    repeated image_count times:
        media_type:u8, reserved:u8, reserved:u16, payload_size:u32, payload[...]

Keeping parsing independent from image decoding lets FastAPI reject oversized
requests before the GPU worker owns any memory.
"""

from __future__ import annotations

import enum
import struct
from collections.abc import Iterable


MAGIC = b"PVRB"
VERSION = 1
HEADER = struct.Struct("<4sHHI")
ITEM_HEADER = struct.Struct("<BBHI")


class ProtocolError(ValueError):
    pass


class MediaType(enum.IntEnum):
    JPEG = 1
    PNG = 2
    BMP = 3
    WEBP = 4


MAGIC_TO_MEDIA = {
    b"\xff\xd8\xff": MediaType.JPEG,
    b"\x89PNG\r\n\x1a\n": MediaType.PNG,
    b"BM": MediaType.BMP,
    b"RIFF": MediaType.WEBP,
}


def sniff_media_type(payload: bytes | memoryview) -> MediaType | None:
    view = memoryview(payload)
    for signature, media_type in MAGIC_TO_MEDIA.items():
        if view[: len(signature)] == signature:
            if media_type is MediaType.WEBP and (
                len(view) < 12 or view[8:12] != b"WEBP"
            ):
                return None
            return media_type
    return None


def image_dimensions(
    payload: bytes | bytearray | memoryview, media_type: MediaType
) -> tuple[int, int] | None:
    """Read dimensions from a supported container header without decoding pixels."""
    view = memoryview(payload)
    if media_type is MediaType.PNG:
        if len(view) >= 24 and view[12:16] == b"IHDR":
            return int.from_bytes(view[16:20], "big"), int.from_bytes(
                view[20:24], "big"
            )
        return None
    if media_type is MediaType.BMP:
        if len(view) >= 26:
            width = int.from_bytes(view[18:22], "little", signed=True)
            height = int.from_bytes(view[22:26], "little", signed=True)
            return abs(width), abs(height)
        return None
    if media_type is MediaType.WEBP:
        if len(view) < 30:
            return None
        chunk = bytes(view[12:16])
        if chunk == b"VP8X":
            return 1 + int.from_bytes(view[24:27], "little"), 1 + int.from_bytes(
                view[27:30], "little"
            )
        if chunk == b"VP8L" and view[20] == 0x2F:
            bits = int.from_bytes(view[21:25], "little")
            return 1 + (bits & 0x3FFF), 1 + ((bits >> 14) & 0x3FFF)
        if chunk == b"VP8 " and view[23:26] == b"\x9d\x01\x2a":
            return (
                int.from_bytes(view[26:28], "little") & 0x3FFF,
                int.from_bytes(view[28:30], "little") & 0x3FFF,
            )
        return None
    if media_type is not MediaType.JPEG or len(view) < 4:
        return None

    offset = 2
    start_of_frame = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    while offset + 3 < len(view):
        if view[offset] != 0xFF:
            offset += 1
            continue
        while offset < len(view) and view[offset] == 0xFF:
            offset += 1
        if offset >= len(view):
            return None
        marker = view[offset]
        offset += 1
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:
            continue
        if offset + 2 > len(view):
            return None
        segment_size = int.from_bytes(view[offset : offset + 2], "big")
        if segment_size < 2 or offset + segment_size > len(view):
            return None
        if marker in start_of_frame and segment_size >= 7:
            height = int.from_bytes(view[offset + 3 : offset + 5], "big")
            width = int.from_bytes(view[offset + 5 : offset + 7], "big")
            return width, height
        offset += segment_size
    return None


def exceeds_pixel_limit(
    payload: bytes | bytearray | memoryview,
    media_type: MediaType,
    max_image_pixels: int,
) -> bool:
    dimensions = image_dimensions(payload, media_type)
    return bool(
        dimensions
        and dimensions[0] > 0
        and dimensions[1] > 0
        and dimensions[0] * dimensions[1] > max_image_pixels
    )


def encode_batch(images: Iterable[tuple[MediaType, bytes]]) -> bytes:
    items = list(images)
    chunks = [HEADER.pack(MAGIC, VERSION, 0, len(items))]
    for media_type, payload in items:
        chunks.append(ITEM_HEADER.pack(int(media_type), 0, 0, len(payload)))
        chunks.append(payload)
    return b"".join(chunks)


def decode_batch(
    payload: bytes,
    *,
    max_images: int,
    max_image_bytes: int,
    max_batch_bytes: int,
    max_image_pixels: int | None = None,
) -> list[tuple[MediaType, bytes]]:
    if len(payload) > max_batch_bytes:
        raise ProtocolError("batch byte limit exceeded")
    if len(payload) < HEADER.size:
        raise ProtocolError("truncated batch header")
    magic, version, flags, count = HEADER.unpack_from(payload)
    if magic != MAGIC:
        raise ProtocolError("invalid batch magic")
    if version != VERSION:
        raise ProtocolError(f"unsupported batch version: {version}")
    if flags != 0:
        raise ProtocolError("unsupported batch flags")
    if not 1 <= count <= max_images:
        raise ProtocolError(f"image count must be between 1 and {max_images}")

    offset = HEADER.size
    images: list[tuple[MediaType, bytes]] = []
    for _ in range(count):
        if len(payload) - offset < ITEM_HEADER.size:
            raise ProtocolError("truncated image header")
        raw_type, reserved8, reserved16, size = ITEM_HEADER.unpack_from(payload, offset)
        offset += ITEM_HEADER.size
        if reserved8 or reserved16:
            raise ProtocolError("reserved image header fields must be zero")
        try:
            media_type = MediaType(raw_type)
        except ValueError as exc:
            raise ProtocolError(f"unsupported media type: {raw_type}") from exc
        if not 1 <= size <= max_image_bytes:
            raise ProtocolError("image byte limit exceeded")
        end = offset + size
        if end > len(payload):
            raise ProtocolError("truncated image payload")
        image = payload[offset:end]
        actual = sniff_media_type(image)
        if actual is None or actual is not media_type:
            raise ProtocolError("declared media type does not match image signature")
        if max_image_pixels is not None and exceeds_pixel_limit(
            image, media_type, max_image_pixels
        ):
            raise ProtocolError("image pixel limit exceeded")
        images.append((media_type, image))
        offset = end
    if offset != len(payload):
        raise ProtocolError("trailing bytes after batch")
    return images
