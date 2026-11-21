"""Core of Image2PBR: builds PBR maps from an ordinary texture image.

The AI never "paints" a map. It only chooses PARAMETERS, and the map itself is computed
per pixel from the source image, so the result always matches the picture.

Two groups of parameters:
  * detail  - taken from the image itself (channel, blur, contrast, ...)
  * regions - a palette of dominant colours; the AI assigns a value to each colour
              (e.g. rust = roughness 0.9, bare metal = 0.3)
"""
from __future__ import annotations

import numpy as np
import cv2

MAPS = ["roughness", "metallic", "normal", "height", "ao"]
CHANNELS = ["luminance", "red", "green", "blue", "saturation", "value"]
SIGNED_MAPS = ("normal", "height", "ao")      # maps whose palette list is a height offset (-1..1)


# --------------------------------------------------------------------------
# Parameter specification
# --------------------------------------------------------------------------
def P(key, kind, default, label, rng=None, choices=None, hint=""):
    return dict(key=key, kind=kind, default=default, label=label,
                rng=rng, choices=choices, hint=hint)


TONAL = [
    P("channel", "choice", "luminance", "Detail source channel", choices=CHANNELS,
      hint="which image channel carries the surface detail"),
    P("invert", "bool", False, "Invert detail", hint="flip light/dark of the detail"),
    P("blur", "float", 0.0, "Blur / denoise (px)", (0, 20, 0.1), hint="gaussian blur sigma, removes noise"),
    P("delight", "float", 0.0, "Remove baked lighting (radius px)", (0, 200, 1),
      hint="high-pass radius, removes large shading/vignette; 0 = off"),
    P("clahe", "float", 0.0, "Local contrast (CLAHE)", (0, 1, 0.05), hint="local contrast amount 0..1"),
    P("black_point", "float", 0.0, "Black point", (0, 1, 0.01), hint="levels: values below become 0"),
    P("white_point", "float", 1.0, "White point", (0, 1, 0.01), hint="levels: values above become 1"),
    P("gamma", "float", 1.0, "Gamma (>1 = darker)", (0.2, 4, 0.05), hint="gamma curve"),
    P("contrast", "float", 1.0, "Contrast", (0.2, 4, 0.05), hint="contrast around 0.5"),
    P("sharpen", "float", 0.0, "Sharpen", (0, 3, 0.05), hint="unsharp mask amount"),
]

PALETTE_SOFT = P("palette_softness", "float", 14.0, "Region edge softness (Lab)", (1, 60, 1),
                 hint="small = hard borders between colour regions, large = smooth")
HEIGHT_REGIONS = [
    P("palette_heights", "list", [], "Height per palette colour (-1..1)",
      hint="one value -1..1 per palette colour index; positive = raised area, negative = recessed"),
    PALETTE_SOFT,
    P("palette_height_amount", "float", 0.0, "Palette height influence", (-2, 2, 0.05),
      hint="how much the palette heights shape the broad relief"),
]
SMOOTH = P("smooth_out", "float", 0.0, "Smooth result (px)", (0, 20, 0.1), hint="final blur")

ROUGH_EXTRA = [
    P("palette_values", "list", [], "Roughness per palette colour (0..1)",
      hint="one value 0..1 per palette colour index (0 = mirror, 1 = fully rough)"),
    PALETTE_SOFT,
    P("out_min", "float", 0.2, "Detail min roughness", (0, 1, 0.01), hint="lower bound of the detail range"),
    P("out_max", "float", 0.9, "Detail max roughness", (0, 1, 0.01), hint="upper bound of the detail range"),
    P("detail_amount", "float", 1.0, "Detail strength from image", (0, 2, 0.05),
      hint="0 = only region values, 1 = full detail from the image"),
    SMOOTH,
]

METAL_EXTRA = [
    P("palette_values", "list", [], "Metallic per palette colour (0..1)",
      hint="one value per palette colour index; bare metal ~1, rust/paint/dirt/non-metal ~0"),
    PALETTE_SOFT,
    P("out_min", "float", 0.0, "Detail min metallic", (0, 1, 0.01), hint="lower bound of detail range"),
    P("out_max", "float", 1.0, "Detail max metallic", (0, 1, 0.01), hint="upper bound of detail range"),
    P("detail_amount", "float", 0.25, "Detail strength from image", (0, 2, 0.05), hint="0 = only region values"),
    P("threshold", "float", 0.5, "Binarize threshold", (0, 1, 0.01), hint="binarize threshold"),
    P("edge_softness", "float", 0.1, "Threshold edge softness", (0.01, 0.5, 0.01), hint="smoothstep half width"),
    P("binarize", "float", 0.0, "Binarize (0 = off, 1 = pure black/white)", (0, 1, 0.05),
      hint="metallic is usually a mask: 1.0 gives a clean 0/1 mask"),
    P("cleanup", "float", 0.0, "Remove small specks (median, px)", (0, 10, 0.5), hint="median filter radius"),
    P("grow", "float", 0.0, "Grow / shrink mask (px)", (-10, 10, 0.5), hint="dilate(+)/erode(-)"),
    SMOOTH,
]

NORMAL_EXTRA = [
    P("strength", "float", 2.0, "Bump strength", (0, 20, 0.1), hint="overall bump strength"),
    P("scale_fine", "float", 1.0, "Fine detail weight", (0, 2, 0.05), hint="weight of fine scale"),
    P("scale_medium", "float", 0.5, "Medium detail weight", (0, 2, 0.05), hint="weight of medium scale"),
    P("scale_coarse", "float", 0.0, "Coarse shape weight", (0, 2, 0.05), hint="weight of coarse scale"),
    *HEIGHT_REGIONS,
    P("flip_green", "bool", False, "Flip Y (DirectX). Off for Blender", hint="OpenGL (Blender) = false"),
    P("tileable", "bool", True, "Wrap borders (tileable)", hint="wrap borders for tiling textures"),
]

HEIGHT_EXTRA = [
    *HEIGHT_REGIONS,
    P("normalize", "bool", True, "Auto-stretch to full 0..1 range", hint="percentile stretch of the result"),
    SMOOTH,
]

AO_EXTRA = [
    P("ao_radius", "float", 12.0, "Occlusion radius (px)", (1, 120, 1), hint="size of crevices that get darkened"),
    P("ao_strength", "float", 2.0, "Occlusion strength", (0, 6, 0.1), hint="darkness of crevices"),
    P("ao_color_mix", "float", 0.25, "Darken by dark colours", (0, 1, 0.05),
      hint="how much dark image areas also get occluded"),
    P("ao_floor", "float", 0.2, "Minimum AO value", (0, 1, 0.01), hint="darkest allowed AO value"),
    *HEIGHT_REGIONS,
    SMOOTH,
]

_EXTRA = {"roughness": ROUGH_EXTRA, "metallic": METAL_EXTRA, "normal": NORMAL_EXTRA,
          "height": HEIGHT_EXTRA, "ao": AO_EXTRA}


def spec(map_name: str) -> list[dict]:
    return TONAL + _EXTRA[map_name]


def defaults(map_name: str) -> dict:
    return {sp["key"]: (list(sp["default"]) if sp["kind"] == "list" else sp["default"])
            for sp in spec(map_name)}


def schema_text(map_name: str, k: int) -> str:
    lines = []
    for sp in spec(map_name):
        if sp["kind"] == "float":
            t = f"number {sp['rng'][0]}..{sp['rng'][1]}"
        elif sp["kind"] == "bool":
            t = "boolean"
        elif sp["kind"] == "choice":
            t = "one of " + "|".join(sp["choices"])
        else:
            t = f"list of exactly {k} numbers"
        lines.append(f'- "{sp["key"]}" ({t}, default {sp["default"]}): {sp["hint"] or sp["label"]}')
    return "\n".join(lines)


def sanitize(map_name: str, raw: dict, k: int) -> dict:
    """Clamp an (AI) answer to valid values; missing keys fall back to defaults."""
    out = defaults(map_name)
    pad = 0.6 if map_name == "roughness" else 0.0
    lim = (-1.0, 1.0) if map_name in SIGNED_MAPS else (0.0, 1.0)
    for sp in spec(map_name):
        key = sp["key"]
        if key not in raw:
            continue
        v = raw[key]
        try:
            if sp["kind"] == "float":
                lo, hi = sp["rng"][0], sp["rng"][1]
                out[key] = float(min(max(float(v), lo), hi))
            elif sp["kind"] == "bool":
                out[key] = bool(v)
            elif sp["kind"] == "choice":
                if v in sp["choices"]:
                    out[key] = v
            else:
                vals = [float(x) for x in v][:k]
                vals += [pad] * (k - len(vals))
                out[key] = [min(max(x, lim[0]), lim[1]) for x in vals]
        except (TypeError, ValueError):
            pass
    return out


# Presets that need no AI: {preset: {map: {param: value}}}
PRESETS = {
    "Clean metal": {
        "roughness": dict(out_min=0.15, out_max=0.45),
        "metallic": dict(out_min=0.9, out_max=1.0, detail_amount=0.2),
        "normal": dict(strength=1.2, scale_fine=1.0, scale_medium=0.3),
        "height": dict(blur=1.0, contrast=0.8),
        "ao": dict(ao_strength=1.0, ao_radius=8),
    },
    "Rusty metal": {
        "roughness": dict(out_min=0.55, out_max=0.95),
        "metallic": dict(binarize=1.0, threshold=0.5, detail_amount=1.0, cleanup=1.5),
        "normal": dict(strength=3.0, scale_fine=1.0, scale_medium=0.7),
        "height": dict(blur=0.5),
        "ao": dict(ao_strength=2.5, ao_radius=12, ao_color_mix=0.4),
    },
    "Wood": {
        "roughness": dict(out_min=0.5, out_max=0.85),
        "metallic": dict(out_min=0.0, out_max=0.0, detail_amount=0.0),
        "normal": dict(strength=2.5, scale_fine=1.0, scale_medium=0.6),
        "height": dict(blur=0.6),
        "ao": dict(ao_strength=2.0, ao_radius=10, ao_color_mix=0.3),
    },
    "Stone": {
        "roughness": dict(out_min=0.6, out_max=1.0),
        "metallic": dict(out_min=0.0, out_max=0.0, detail_amount=0.0),
        "normal": dict(strength=4.0, scale_fine=1.0, scale_medium=1.0, scale_coarse=0.3),
        "height": dict(blur=0.5),
        "ao": dict(ao_strength=3.0, ao_radius=20),
    },
    "Fabric": {
        "roughness": dict(out_min=0.8, out_max=1.0),
        "metallic": dict(out_min=0.0, out_max=0.0, detail_amount=0.0),
        "normal": dict(strength=3.0, scale_fine=1.0, scale_medium=0.2),
        "height": dict(blur=0.4),
        "ao": dict(ao_strength=1.5, ao_radius=5),
    },
    "Plastic": {
        "roughness": dict(out_min=0.25, out_max=0.6),
        "metallic": dict(out_min=0.0, out_max=0.0, detail_amount=0.0),
        "normal": dict(strength=0.8, scale_fine=1.0, scale_medium=0.2),
        "height": dict(blur=1.5, contrast=0.7),
        "ao": dict(ao_strength=1.0, ao_radius=8),
    },
}


def preset_params(name: str, map_name: str, k: int) -> dict:
    return sanitize(map_name, PRESETS[name].get(map_name, {}), k)


# --------------------------------------------------------------------------
# Basic operations
# --------------------------------------------------------------------------
def fit(rgb: np.ndarray, max_side: int | None) -> np.ndarray:
    """Downscale so the longest side is <= max_side (None = keep original)."""
    if not max_side:
        return rgb
    h, w = rgb.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return rgb
    sc = max_side / m
    return cv2.resize(rgb, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA)


def make_seamless(rgb: np.ndarray, blend: float = 0.5) -> np.ndarray:
    """Make an image tile without visible seams.

    Cross-fades the image with a copy shifted by half its size: the original keeps the
    centre, the shifted copy (whose seams lie in the middle) takes over the edges.
    `blend` (0.05..1) is the width of the cross-fade zone as a fraction of the half size.
    """
    h, w = rgb.shape[:2]
    blend = float(min(max(blend, 0.05), 1.0))
    f = rgb.astype(np.float32)
    shifted = np.roll(f, (h // 2, w // 2), axis=(0, 1))

    def profile(n):
        t = 1.0 - np.abs(2.0 * (np.arange(n, dtype=np.float32) + 0.5) / n - 1.0)   # 0 at edges, 1 centre
        t = np.clip(t / blend, 0, 1)
        return t * t * (3 - 2 * t)

    m1 = (profile(h)[:, None] * profile(w)[None, :])[..., None]
    m2 = 1.0 - m1
    # variance-preserving blend: a plain cross-fade of two different patterns loses contrast
    # in the transition zone and leaves visible patches
    mean = f.mean(axis=(0, 1), keepdims=True)
    norm = np.sqrt(m1 * m1 + m2 * m2)
    out = mean + (m1 * (f - mean) + m2 * (shifted - mean)) / norm
    return np.clip(out, 0, 255).round().astype(np.uint8)


def _gauss(x: np.ndarray, sigma: float, wrap: bool) -> np.ndarray:
    if sigma < 0.05:
        return x
    pad = int(np.ceil(sigma * 3)) + 1
    xp = np.pad(x, pad, mode="wrap" if wrap else "reflect")
    xp = cv2.GaussianBlur(xp, (0, 0), sigma)
    return xp[pad:-pad, pad:-pad]


def get_channel(f: np.ndarray, name: str) -> np.ndarray:
    if name == "red":
        return f[..., 0]
    if name == "green":
        return f[..., 1]
    if name == "blue":
        return f[..., 2]
    if name in ("saturation", "value"):
        hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
        return hsv[..., 1] if name == "saturation" else hsv[..., 2]
    return f @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def tonal(f: np.ndarray, p: dict, s: float, wrap: bool) -> np.ndarray:
    """Extract a 0..1 detail map from the image using the tonal parameters."""
    d = get_channel(f, p["channel"]).astype(np.float32)
    if p["invert"]:
        d = 1.0 - d
    d = _gauss(d, p["blur"] * s, wrap)
    if p["delight"] > 0:
        d = d - _gauss(d, p["delight"] * s, wrap)
        lo, hi = np.percentile(d, [1, 99])
        d = np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)
    if p["clahe"] > 0:
        u = (np.clip(d, 0, 1) * 255).astype(np.uint8)
        c = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(u).astype(np.float32) / 255.0
        d = d * (1 - p["clahe"]) + c * p["clahe"]
    bp, wp = p["black_point"], max(p["white_point"], p["black_point"] + 0.01)
    d = np.clip((d - bp) / (wp - bp), 0, 1)
    d = np.power(d, p["gamma"])
    d = np.clip((d - 0.5) * p["contrast"] + 0.5, 0, 1)
    if p["sharpen"] > 0:
        d = np.clip(d + p["sharpen"] * (d - _gauss(d, 2 * s, wrap)), 0, 1)
    return d.astype(np.float32)


# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------
def extract_palette(rgb: np.ndarray, k: int = 6) -> dict:
    """k-means over the colours (in Lab). Returns centres, shares and display colours."""
    small = fit(rgb, 160)
    lab = cv2.cvtColor(small.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    k = int(max(1, min(k, len(lab))))
    cv2.setRNGSeed(0)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(lab, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k).astype(np.float32)
    order = np.argsort(-counts)
    centers, counts = centers[order], counts[order]
    rgb_c = cv2.cvtColor(centers.reshape(-1, 1, 3).astype(np.float32), cv2.COLOR_LAB2RGB).reshape(-1, 3)
    rgb_c = np.clip(rgb_c, 0, 1)
    hsv_c = cv2.cvtColor(rgb_c.reshape(-1, 1, 3).astype(np.float32), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    return dict(lab=centers.astype(np.float32), rgb=(rgb_c * 255).round().astype(np.uint8),
                hsv=hsv_c, share=counts / counts.sum())


def palette_info(pal: dict) -> list[dict]:
    out = []
    for i, (c, hsv, sh) in enumerate(zip(pal["rgb"], pal["hsv"], pal["share"])):
        out.append(dict(index=i, hex="#%02x%02x%02x" % tuple(int(x) for x in c),
                        hue=int(hsv[0]), saturation=round(float(hsv[1]), 2),
                        brightness=round(float(hsv[2]), 2), share_percent=round(float(sh) * 100, 1)))
    return out


def palette_html(pal: dict) -> str:
    cells = []
    for it in palette_info(pal):
        txt = "#000" if it["brightness"] > 0.55 else "#fff"
        cells.append(
            f'<div style="display:inline-block;width:92px;margin:3px;border-radius:8px;overflow:hidden;'
            f'border:1px solid #8884;text-align:center;font:12px sans-serif">'
            f'<div style="background:{it["hex"]};color:{txt};padding:14px 0;font-size:18px"><b>{it["index"]}</b></div>'
            f'<div style="padding:3px">{it["hex"]}<br>{it["share_percent"]}%</div></div>')
    return "".join(cells)


def palette_weights(f: np.ndarray, pal: dict, softness: float) -> np.ndarray:
    """Soft membership of every pixel in every palette colour: HxWxK, sums to 1 over K."""
    lab = cv2.cvtColor(f, cv2.COLOR_RGB2LAB)
    d2 = np.stack([((lab - c) ** 2).sum(-1) for c in pal["lab"]], axis=-1)
    logits = -d2 / (2.0 * max(softness, 0.5) ** 2)
    logits -= logits.max(-1, keepdims=True)
    w = np.exp(logits)
    return w / w.sum(-1, keepdims=True)


def _region_map(f, pal, vals, softness, fallback):
    """sum_k weight_k * value_k, computed in row chunks to keep memory low on big images."""
    if pal is None or not vals:
        return None
    k = len(pal["share"])
    v = np.array((list(vals) + [fallback] * k)[:k], dtype=np.float32)
    h, w = f.shape[:2]
    out = np.empty((h, w), np.float32)
    step = max(32, int(2_000_000 // (w * max(k, 1))))
    for y in range(0, h, step):
        out[y:y + step] = palette_weights(f[y:y + step], pal, softness) @ v
    return out


# --------------------------------------------------------------------------
# Map generation (all return float32 0..1 except normal -> uint8)
# --------------------------------------------------------------------------
def _smoothstep(a, b, x):
    t = np.clip((x - a) / max(b - a, 1e-6), 0, 1)
    return t * t * (3 - 2 * t)


def _scale(rgb):
    return max(rgb.shape[:2]) / 1024.0          # pixel parameters are defined for a 1024 px image


def render_scalar(rgb, kind, p, pal):
    s = _scale(rgb)
    f = rgb.astype(np.float32) / 255.0
    d = tonal(f, p, s, wrap=True)
    lo, hi = p["out_min"], max(p["out_max"], p["out_min"])
    amp = hi - lo

    region = _region_map(f, pal, p.get("palette_values"), p["palette_softness"],
                         fallback=0.6 if kind == "roughness" else 0.0)
    if region is None:
        region = (lo + hi) / 2.0
    out = np.clip(region + (d - 0.5) * amp * p["detail_amount"], 0, 1).astype(np.float32)

    if kind == "metallic":
        if p["binarize"] > 0:
            e = p["edge_softness"]
            m = _smoothstep(p["threshold"] - e, p["threshold"] + e, out)
            out = out * (1 - p["binarize"]) + m * p["binarize"]
        r = int(round(p["cleanup"] * s))
        if r >= 1:
            out = cv2.medianBlur((out * 255).astype(np.uint8), 2 * r + 1).astype(np.float32) / 255.0
        g = int(round(p["grow"] * s))
        if g != 0:
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(g) + 1, 2 * abs(g) + 1))
            out = cv2.dilate(out, ker) if g > 0 else cv2.erode(out, ker)
    return np.clip(_gauss(out, p["smooth_out"] * s, True), 0, 1)


def compute_height(rgb, p, pal, wrap=True):
    """Height field shared by normal / height / ao. Returns (height, detail)."""
    s = _scale(rgb)
    f = rgb.astype(np.float32) / 255.0
    d = tonal(f, p, s, wrap)
    height = d
    if abs(p.get("palette_height_amount", 0.0)) > 1e-6:
        region = _region_map(f, pal, p.get("palette_heights"), p["palette_softness"], fallback=0.0)
        if region is not None:
            height = d + p["palette_height_amount"] * region
    return height.astype(np.float32), d


def render_height(rgb, p, pal):
    s = _scale(rgb)
    h, _ = compute_height(rgb, p, pal)
    if p["normalize"]:
        lo, hi = np.percentile(h, [0.5, 99.5])
        h = (h - lo) / (hi - lo + 1e-6)
    return np.clip(_gauss(h, p["smooth_out"] * s, True), 0, 1)


def render_ao(rgb, p, pal):
    s = _scale(rgb)
    h, d = compute_height(rgb, p, pal)
    r = p["ao_radius"] * s
    cav = sum(np.maximum(_gauss(h, r * m, True) - h, 0) for m in (1.0, 2.0, 4.0)) / 3.0
    c = np.clip(cav / (np.percentile(cav, 99.5) + 1e-6), 0, 1)
    ao = 1.0 - np.clip(c * p["ao_strength"] * 0.5, 0, 1) * (1.0 - p["ao_floor"])
    ao = ao * ((1.0 - p["ao_color_mix"]) + p["ao_color_mix"] * d)
    return np.clip(_gauss(ao, p["smooth_out"] * s, True), 0, 1)


def _gradient(hb, wrap):
    hp = np.pad(hb, 1, mode="wrap" if wrap else "reflect")
    gx = cv2.Sobel(hp, cv2.CV_32F, 1, 0, ksize=3)[1:-1, 1:-1] / 8.0
    gy = cv2.Sobel(hp, cv2.CV_32F, 0, 1, ksize=3)[1:-1, 1:-1] / 8.0
    return gx, gy


def render_normal(rgb, p, pal):
    s = _scale(rgb)
    wrap = bool(p["tileable"])
    height, _ = compute_height(rgb, p, pal, wrap)

    gx = np.zeros_like(height)
    gy = np.zeros_like(height)
    for sigma, wt in ((0.8, p["scale_fine"]), (3.0, p["scale_medium"]), (9.0, p["scale_coarse"])):
        if wt <= 0:
            continue
        a, b = _gradient(_gauss(height, sigma * s, wrap), wrap)
        gx += wt * a
        gy += wt * b

    k = p["strength"] * 4.0 * s          # s keeps preview and full-size output equal
    nx = -gx * k
    ny = gy * k * (-1.0 if p["flip_green"] else 1.0)
    nz = np.ones_like(nx)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz)
    n = np.stack([nx / ln, ny / ln, nz / ln], axis=-1)
    return ((n * 0.5 + 0.5) * 255).round().astype(np.uint8)


def render(rgb: np.ndarray, map_name: str, params: dict, pal: dict | None,
           max_side: int | None = None, bit16: bool = False) -> np.ndarray:
    """Render one map. uint8 HxW (or HxWx3 for normal); uint16 for height when bit16."""
    rgb = fit(rgb, max_side)
    if map_name == "normal":
        return render_normal(rgb, params, pal)
    if map_name == "height":
        x = render_height(rgb, params, pal)
    elif map_name == "ao":
        x = render_ao(rgb, params, pal)
    else:
        x = render_scalar(rgb, map_name, params, pal)
    if bit16 and map_name == "height":
        return (x * 65535).round().astype(np.uint16)
    return (x * 255).round().astype(np.uint8)
