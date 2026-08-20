from __future__ import annotations

import pytest

from pvr_api.protocol import (
    MediaType,
    ProtocolError,
    decode_batch,
    encode_batch,
    image_dimensions,
)


JPEG = b"\xff\xd8\xff" + b"payload"


def test_binary_batch_round_trip_preserves_order() -> None:
    payload = encode_batch([(MediaType.JPEG, JPEG), (MediaType.JPEG, JPEG + b"2")])
    decoded = decode_batch(
        payload, max_images=64, max_image_bytes=1024, max_batch_bytes=4096
    )
    assert decoded == [(MediaType.JPEG, JPEG), (MediaType.JPEG, JPEG + b"2")]


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        encode_batch([]),
        encode_batch([(MediaType.PNG, JPEG)]),
        encode_batch([(MediaType.JPEG, JPEG)]) + b"trailing",
    ],
)
def test_binary_batch_rejects_malformed_payload(payload: bytes) -> None:
    with pytest.raises(ProtocolError):
        decode_batch(
            payload, max_images=64, max_image_bytes=1024, max_batch_bytes=4096
        )


def test_dimensions_are_read_without_decode_and_pixel_limit_is_enforced() -> None:
    png = (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\x0dIHDR"
        + (20).to_bytes(4, "big")
        + (30).to_bytes(4, "big")
        + b"rest"
    )
    assert image_dimensions(png, MediaType.PNG) == (20, 30)
    with pytest.raises(ProtocolError, match="pixel limit"):
        decode_batch(
            encode_batch([(MediaType.PNG, png)]),
            max_images=64,
            max_image_bytes=1024,
            max_batch_bytes=4096,
            max_image_pixels=500,
        )


def test_dimensions_cover_every_accepted_container() -> None:
    jpeg = (
        b"\xff\xd8\xff\xc0\x00\x11\x08\x00\x1e\x00\x14" + b"\x00" * 10
    )
    bmp = bytearray(26)
    bmp[:2] = b"BM"
    bmp[18:22] = (20).to_bytes(4, "little", signed=True)
    bmp[22:26] = (-30).to_bytes(4, "little", signed=True)
    webp = bytearray(30)
    webp[:4] = b"RIFF"
    webp[8:12] = b"WEBP"
    webp[12:16] = b"VP8X"
    webp[24:27] = (19).to_bytes(3, "little")
    webp[27:30] = (29).to_bytes(3, "little")

    assert image_dimensions(jpeg, MediaType.JPEG) == (20, 30)
    assert image_dimensions(bmp, MediaType.BMP) == (20, 30)
    assert image_dimensions(webp, MediaType.WEBP) == (20, 30)
