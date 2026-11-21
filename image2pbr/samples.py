"""Procedural sample textures (used by selftest and tests, no downloads needed)."""
from __future__ import annotations

import cv2
import numpy as np


def make_wood(size: int = 1024, seed: int = 3) -> np.ndarray:
    """Wooden planks: growth rings, fibres, pores and seams."""
    rng = np.random.default_rng(seed)

    def noise(scale, sx=1.0, sy=1.0):
        n = rng.random((size, size)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), sigmaX=scale * sx, sigmaY=scale * sy)
        return (n - n.min()) / (n.max() - n.min() + 1e-6)

    yy = np.mgrid[0:size, 0:size][0].astype(np.float32)
    img = np.zeros((size, size, 3), np.float32)
    light, dark = np.array([0.78, 0.57, 0.36]), np.array([0.46, 0.28, 0.14])
    planks = 4
    ph = size // planks
    for i in range(planks):
        y0, y1 = i * ph, (i + 1) * ph if i < planks - 1 else size
        v = (yy[y0:y1] - y0) / ph
        warp = noise(30, 3, 1)[y0:y1] * 1.4 + noise(8, 4, 1)[y0:y1] * 0.3
        rings = (0.5 + 0.5 * np.sin((v * (5 + i) + warp * 3 + i * 1.7) * 2 * np.pi)) ** 1.5
        fibres = noise(1.2, 14, 0.7)[y0:y1]
        t = np.clip(0.65 * rings + 0.35 * fibres, 0, 1)[..., None]
        img[y0:y1] = (light * (1 - t) + dark * t) * (1.0 + (i - 1.5) * 0.04)
    pores = (noise(0.8, 6, 0.6) < 0.12).astype(np.float32)[..., None]
    img *= 1 - 0.35 * pores
    for i in range(1, planks):
        img[i * ph - 2:i * ph + 2] *= 0.35
    img *= 0.92 + 0.16 * noise(60)[..., None]
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def make_rusty_metal(size: int = 1024, seed: int = 1) -> np.ndarray:
    """Grey steel with rust blotches and scratches."""
    rng = np.random.default_rng(seed)
    n = cv2.GaussianBlur(rng.random((size, size)).astype(np.float32), (0, 0), size / 25)
    n = (n - n.min()) / (n.max() - n.min())
    rust = cv2.GaussianBlur((n > 0.55).astype(np.float32), (0, 0), 3)
    metal = np.dstack([np.full((size, size), 0.62)] * 3) + (rng.random((size, size, 1)) - 0.5) * 0.08
    rustc = np.dstack([np.full((size, size), 0.45), np.full((size, size), 0.22), np.full((size, size), 0.1)])
    img = metal * (1 - rust[..., None]) + rustc * rust[..., None]
    for _ in range(60):
        x, y = rng.integers(0, size, 2)
        cv2.line(img, (int(x), int(y)), (int(x + rng.integers(-200, 200)), int(y + rng.integers(-200, 200))),
                 (0.85, 0.85, 0.85), 1)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)
