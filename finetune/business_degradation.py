"""贴近固定监控画面的轻量光学和压缩退化。"""

from __future__ import annotations

import random

import cv2
import numpy as np


def business_degradation(image: np.ndarray) -> np.ndarray:
    value = image.astype(np.float32)
    if random.random() < 0.35:
        value *= random.uniform(0.65, 1.15)
    if random.random() < 0.30:
        value[:, :, 2] *= random.uniform(1.05, 1.25)
        value[:, :, 0] *= random.uniform(0.75, 0.98)
    value = np.clip(value, 0, 255).astype(np.uint8)
    if random.random() < 0.20:
        value = cv2.GaussianBlur(value, (3, 3), random.uniform(0.3, 1.2))
    if random.random() < 0.20:
        noise = np.random.normal(0, random.uniform(2, 8), value.shape)
        value = np.clip(value.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if random.random() < 0.20:
        ok, encoded = cv2.imencode(".jpg", value, [cv2.IMWRITE_JPEG_QUALITY, random.randint(45, 85)])
        if ok:
            value = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if random.random() < 0.15:
        overlay = value.copy()
        height, width = value.shape[:2]
        center = (random.randrange(width), random.randrange(height))
        radius = max(2, int(min(width, height) * random.uniform(0.03, 0.12)))
        color = random.choice(((40, 55, 70), (70, 80, 80), (25, 35, 45)))
        cv2.circle(overlay, center, radius, color, -1)
        value = cv2.addWeighted(overlay, random.uniform(0.15, 0.35), value, 0.75, 0)
    return value
