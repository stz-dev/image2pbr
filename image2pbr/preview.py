"""Lightweight software "3D" preview: a lit sphere or plane shaded with the generated maps."""
from __future__ import annotations

import cv2
import numpy as np

TEX_RES = 512


def _to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def _norm(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


def _sample(tex, u, v):
    h, w = tex.shape[:2]
    mx = ((u % 1.0) * w).astype(np.float32)
    my = ((v % 1.0) * h).astype(np.float32)
    return cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def _tex(arr, channels):
    a = cv2.resize(arr, (TEX_RES, TEX_RES), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    if channels == 3 and a.ndim == 2:
        a = np.stack([a] * 3, -1)
    return a


ALL_LAYERS = ("basecolor", "roughness", "metallic", "normal", "height", "ao")


def _height_slopes(height_u8: np.ndarray, strength: float) -> np.ndarray:
    """Tangent-space slopes (-dh/dx, +dh/dy) from a height map, as a 2-channel texture."""
    h = cv2.resize(height_u8, (TEX_RES, TEX_RES), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    hp = np.pad(h, 1, mode="wrap")
    gx = cv2.Sobel(hp, cv2.CV_32F, 1, 0, ksize=3)[1:-1, 1:-1] / 8.0
    gy = cv2.Sobel(hp, cv2.CV_32F, 0, 1, ksize=3)[1:-1, 1:-1] / 8.0
    k = strength * 4.0 * TEX_RES / 1024.0 * 2.0
    return np.stack([-gx * k, gy * k], -1)


def render_preview(albedo: np.ndarray, maps: dict, shape: str = "Sphere", tile: float = 2.0,
                   light_az: float = 40.0, light_el: float = 45.0, size: int = 512,
                   use: tuple | list | set | None = None, height_strength: float = 2.0) -> np.ndarray:
    """Shade a sphere/plane with the generated maps.

    albedo: RGB uint8. maps: name -> uint8 arrays. use: which layers to apply (default: all that exist);
    layers that are not used or missing fall back to neutral values. "height" is applied as extra bump.
    """
    use = set(ALL_LAYERS if use is None else use)
    flat = lambda v: np.full((TEX_RES, TEX_RES), v, np.float32)       # noqa: E731

    if "basecolor" in use:
        alb = _to_linear(_tex(albedo, 3))
    else:
        alb = np.full((TEX_RES, TEX_RES, 3), 0.5, np.float32)         # neutral grey (linear ~0.21 in sRGB 0.5)
        alb = _to_linear(alb)
    rough = _tex(maps["roughness"], 1) if "roughness" in use and "roughness" in maps else flat(0.6)
    metal = _tex(maps["metallic"], 1) if "metallic" in use and "metallic" in maps else flat(0.0)
    ao = _tex(maps["ao"], 1) if "ao" in use and "ao" in maps else flat(1.0)

    nrm = np.zeros((TEX_RES, TEX_RES, 3), np.float32)
    nrm[..., 2] = 1.0
    if "normal" in use and "normal" in maps:
        nrm = _tex(maps["normal"], 3) * 2.0 - 1.0
    if "height" in use and "height" in maps:
        slope = nrm[..., :2] / np.maximum(nrm[..., 2:3], 1e-3) + _height_slopes(maps["height"], height_strength)
        v3 = np.concatenate([slope, np.ones_like(slope[..., :1])], -1)
        nrm = _norm(v3)

    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    x = (xs + 0.5) / size * 2 - 1
    y = 1 - (ys + 0.5) / size * 2

    if shape == "Sphere":
        r2 = x * x + y * y
        mask = r2 <= 1.0
        z = np.sqrt(np.maximum(1 - r2, 0))
        N = np.stack([x, y, z], -1)
        phi = np.arctan2(x, z)
        theta = np.arccos(np.clip(y, -1, 1))
        u = (phi / (2 * np.pi) + 0.5) * tile * 2
        v = theta / np.pi * tile
        T = np.stack([z, np.zeros_like(z), -x], -1)
        bad = np.linalg.norm(T, axis=-1) < 1e-3
        T[bad] = (1.0, 0.0, 0.0)
        T = _norm(T)
        B = np.cross(N, T)
    else:
        mask = np.ones((size, size), bool)
        N = np.zeros((size, size, 3), np.float32)
        N[..., 2] = 1
        T = np.zeros_like(N)
        T[..., 0] = 1
        B = np.zeros_like(N)
        B[..., 1] = 1
        u = xs / size * tile
        v = ys / size * tile

    A = _sample(alb, u, v)
    R = np.clip(_sample(rough, u, v), 0.04, 1.0)
    M = np.clip(_sample(metal, u, v), 0, 1)
    O = _sample(ao, u, v)
    nt = _norm(_sample(nrm, u, v))
    n = _norm(T * nt[..., :1] + B * nt[..., 1:2] + N * nt[..., 2:3])

    az, el = np.radians(light_az), np.radians(light_el)
    L = np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)], np.float32)
    V = np.array([0, 0, 1], np.float32)
    Hh = _norm(L + V)
    ndl = np.clip((n * L).sum(-1), 0, 1)[..., None]
    ndh = np.clip((n * Hh).sum(-1), 0, 1)[..., None]
    ndv = np.clip((n * V).sum(-1), 0, 1)[..., None]

    Rm, Mm = R[..., None], M[..., None]
    alpha = np.maximum(Rm * Rm, 0.02)
    shin = 2.0 / (alpha * alpha) - 2.0
    spec = np.power(ndh, shin) * (shin + 8.0) / (8.0 * np.pi)
    F0 = 0.04 * (1 - Mm) + A * Mm
    F = F0 + (1 - F0) * np.power(1 - ndv, 5)

    sky, ground = np.array([0.55, 0.68, 0.9], np.float32), np.array([0.14, 0.11, 0.09], np.float32)
    refl = 2 * ndv * n - V
    t = np.clip((refl[..., 1:2] + 0.3) / 0.9, 0, 1)
    env = ground * (1 - t) + sky * t
    env = env * (1 - Rm) + (sky + ground) * 0.5 * Rm
    amb = ground * (1 - (0.5 + 0.5 * n[..., 1:2])) + sky * (0.5 + 0.5 * n[..., 1:2])

    light_i = 3.0
    O3 = O[..., None]
    diffuse = A * (1 - Mm) * (ndl * light_i + amb * 0.6) * O3
    specular = (F * spec * ndl * light_i * 0.6) + F * env * (1 - 0.7 * Rm) * O3
    col = diffuse + specular
    col = 1.0 - np.exp(-col * 1.3)                       # simple tone-mapping
    img = (_to_srgb(col) * 255).astype(np.uint8)
    bg = np.full_like(img, 28)
    return np.where(mask[..., None], img, bg)
