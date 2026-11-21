"""AI clients: Qwen (via Groq) describes the texture, Gemini chooses the parameters."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import time
from pathlib import Path

import requests
from PIL import Image

from . import pbr, settings

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200"

SERVICE_NAMES = {"gemini": "Gemini", "groq": "Groq (Qwen)"}
CACHE_DIR = Path(os.environ.get("IMAGE2PBR_CACHE_DIR") or settings.ROOT / ".cache")


class AIError(RuntimeError):
    """kind: key | limit | model | network | response | other"""

    def __init__(self, message: str, service: str | None = None, kind: str = "other"):
        super().__init__(message)
        self.service = service
        self.kind = kind


# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------
def _check(resp: requests.Response, service: str):
    who = SERVICE_NAMES[service]
    text = resp.text.lower()
    if resp.status_code == 429:
        raise AIError(f"{who}: rate limit reached (429). Wait a minute or pick another model.", service, "limit")
    if resp.status_code in (401, 403) or (resp.status_code == 400 and "api key" in text):
        raise AIError(f"{who}: the API key was rejected ({resp.status_code}).", service, "key")
    if resp.status_code in (400, 404) and "model" in text:
        raise AIError(f"{who}: model not found or not suitable ({resp.status_code}). "
                      f"Choose another one in Settings. Server said: {resp.text[:200]}", service, "model")
    if not resp.ok:
        raise AIError(f"{who}: error {resp.status_code}: {resp.text[:300]}", service, "other")


def _request(method: str, url: str, service: str, **kw) -> requests.Response:
    """HTTP call with one retry on 429/5xx and a clear error when the network is down."""
    for attempt in (1, 2):
        try:
            r = requests.request(method, url, timeout=90, **kw)
        except requests.RequestException as e:
            raise AIError(f"{SERVICE_NAMES[service]}: cannot reach the server ({e.__class__.__name__}). "
                          "Check your internet connection / VPN.", service, "network")
        if attempt == 1 and (r.status_code == 429 or r.status_code >= 500):
            time.sleep(6)
            continue
        return r
    return r


def _need_key(service: str) -> str:
    key = settings.get_key(service)
    if not key:
        raise AIError(f"{SERVICE_NAMES[service]}: no API key. Add one in the Settings tab.", service, "key")
    return key


def _jpeg_b64(rgb, max_side=1024) -> str:
    im = Image.fromarray(pbr.fit(rgb, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------
# Cache (so identical requests do not burn free-tier limits)
# --------------------------------------------------------------------------
def _cache_file(ns: str, *parts: str) -> Path:
    h = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return CACHE_DIR / ns / f"{h}.json"


def cache_get(ns: str, *parts: str):
    if not settings.cache_enabled():
        return None
    try:
        return json.loads(_cache_file(ns, *parts).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def cache_put(ns: str, value, *parts: str) -> None:
    if not settings.cache_enabled():
        return
    try:
        f = _cache_file(ns, *parts)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def cache_clear() -> int:
    n = 0
    for f in CACHE_DIR.glob("*/*.json"):
        try:
            f.unlink()
            n += 1
        except OSError:
            pass
    return n


# --------------------------------------------------------------------------
# Key / model checks
# --------------------------------------------------------------------------
def _is_vision_candidate(model_id: str) -> bool:
    m = model_id.lower()
    return any(t in m for t in ("qwen", "llama-4", "scout", "maverick", "vision", "-vl"))


def list_groq_models(key: str) -> list[str]:
    r = _request("GET", GROQ_MODELS_URL, "groq", headers={"Authorization": f"Bearer {key}"})
    _check(r, "groq")
    ids = sorted(m["id"] for m in r.json().get("data", []))
    best = [i for i in ids if _is_vision_candidate(i)]
    return best + [i for i in ids if i not in best]


def list_gemini_models(key: str) -> list[str]:
    r = _request("GET", GEMINI_MODELS_URL, "gemini", headers={"x-goog-api-key": key})
    _check(r, "gemini")
    bad = ("tts", "image", "embedding", "live", "robotics", "computer", "audio", "veo", "imagen", "aqa")
    ids = []
    for m in r.json().get("models", []):
        name = m.get("name", "").split("/")[-1]
        if "generateContent" in m.get("supportedGenerationMethods", []) and name.startswith("gemini") \
                and not any(b in name for b in bad):
            ids.append(name)
    flash = sorted(i for i in ids if "flash" in i)
    return flash + sorted(i for i in ids if i not in flash)


def check_service(service: str) -> dict:
    """Returns {ok, message, models}. Never raises."""
    key = settings.get_key(service)
    if not key:
        return dict(ok=False, missing=True, message="no key", models=[])
    try:
        models = list_gemini_models(key) if service == "gemini" else list_groq_models(key)
        return dict(ok=True, missing=False, message=f"OK, {len(models)} models available", models=models)
    except AIError as e:
        return dict(ok=False, missing=False, message=str(e), models=[], kind=e.kind)


def find_groq_qwen(key: str) -> str | None:
    """Newest Qwen model in the Groq list (used when the configured name was retired)."""
    try:
        ids = sorted(i for i in list_groq_models(key) if "qwen" in i.lower())
    except AIError:
        return None
    return ids[-1] if ids else None


# --------------------------------------------------------------------------
# Qwen (Groq): describe the texture
# --------------------------------------------------------------------------
def describe_with_qwen(rgb, pal: dict, hint: str = "", model: str | None = None) -> str:
    key = _need_key("groq")
    model = model or settings.qwen_model()

    colors = "\n".join(
        f'  #{c["index"]}: {c["hex"]} ({c["share_percent"]}% of the area)' for c in pbr.palette_info(pal))
    prompt = (
        "You are a PBR material artist. Analyse this texture image for creating roughness, metallic, "
        "normal, height and ambient-occlusion maps in Blender. Answer in English, concise, structured, "
        "no preamble.\n\n"
        f"The dominant colours of the image are numbered:\n{colors}\n\n"
        "Describe:\n"
        "1. MATERIAL: what the surface is (e.g. rusty painted steel, brushed aluminium, oak wood, stone).\n"
        "2. REGIONS: for EVERY numbered colour say what it physically is (bare metal, rust, paint, dirt, "
        "wood, stone, ...), whether it is metallic, and whether it looks glossy or matte.\n"
        "3. RELIEF: what is raised and what is recessed (scratches, rivets, grain, bricks, seams), "
        "and the scale of details (fine / medium / coarse).\n"
        "4. LIGHTING: is there baked-in shadow, vignette or gradient that is NOT part of the material?\n"
        "5. PATTERN: is it tileable/repeating or a single object?\n"
    )
    if hint.strip():
        prompt += f"\nUser hint about the material: {hint.strip()}\n"

    b64 = _jpeg_b64(rgb)
    cached = cache_get("qwen", model, prompt, hashlib.sha256(b64.encode()).hexdigest())
    if cached:
        return cached["text"]

    def body_for(m):
        return {"model": m, "temperature": 0.2, "max_tokens": 2000,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}]}]}

    headers = {"Authorization": f"Bearer {key}"}
    r = _request("POST", GROQ_URL, "groq", json=body_for(model), headers=headers)
    if r.status_code in (400, 404) and "model" in r.text.lower():
        alt = find_groq_qwen(key)                  # the model was renamed: use the current Qwen
        if alt and alt != model:
            model = alt
            os.environ["QWEN_MODEL"] = alt
            r = _request("POST", GROQ_URL, "groq", json=body_for(model), headers=headers)
    _check(r, "groq")
    text = (r.json()["choices"][0]["message"].get("content") or "")
    text = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S).strip()
    if not text:
        raise AIError("Groq (Qwen): empty answer (the model may have spent all tokens thinking). "
                      "Try again or choose another model in Settings.", "groq", "response")
    cache_put("qwen", {"text": text}, model, prompt, hashlib.sha256(b64.encode()).hexdigest())
    return text


# --------------------------------------------------------------------------
# Gemini: choose the parameters
# --------------------------------------------------------------------------
GUIDE = {
    "roughness": (
        "ROUGHNESS guide: polished/clean metal 0.1-0.35, brushed metal 0.3-0.5, plastic 0.3-0.6, painted "
        "surfaces 0.4-0.7, wood 0.5-0.85, stone 0.6-0.95, rust/dirt/fabric 0.8-1.0. Use palette_values to "
        "give each colour region its own value, and out_min/out_max + detail_amount for variation INSIDE "
        "a region. Choose `channel`/`invert` so that the detail really follows the surface (e.g. scratches "
        "and wear are usually rougher; dark grooves are usually rougher)."),
    "metallic": (
        "METALLIC guide: metallic is almost binary. Bare metal = 1.0 (0.9-1.0), rust, paint, dirt, "
        "oxide, wood, stone, plastic, fabric = 0.0 (0-0.1). Put these into palette_values per colour. "
        "Use a small detail_amount (0-0.3) unless the image brightness itself separates metal from "
        "non-metal, in which case use binarize close to 1 with a suitable threshold. Use cleanup to "
        "remove speckles. Fully non-metal materials: palette_values all 0, out_min=out_max=0, detail_amount=0."),
    "normal": (
        "NORMAL guide: the normal map is computed from a height map derived from the image. Pick the "
        "channel/invert so that raised features are bright. Use `delight` to remove baked shading when "
        "the description mentions it. Fine detail (grain, scratches) -> scale_fine high; bricks/stones/"
        "panels -> scale_medium/scale_coarse higher. strength 1-2 subtle, 3-5 strong. Use palette_heights "
        "(+ palette_height_amount) only when colour regions are at clearly different heights."),
    "height": (
        "HEIGHT guide: a greyscale displacement map (white = raised, black = recessed). Pick channel/invert "
        "so raised features are bright, use `delight` if baked shading exists, light blur to remove noise, "
        "and palette_heights/palette_height_amount only when colour regions sit at clearly different heights "
        "(e.g. mortar lower than bricks)."),
    "ao": (
        "AO guide: ambient occlusion darkens crevices. ao_radius = size of the crevices in pixels (grain/"
        "scratches 3-8, bricks/panels 15-40), ao_strength 1-3 normal, ao_floor keeps AO from going fully "
        "black (0.15-0.4), ao_color_mix darkens dark colours too (0.1-0.4). The height used for AO follows "
        "channel/invert (raised features bright)."),
}


def _extract_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise AIError("Gemini returned something that is not JSON: " + text[:200], "gemini", "response")
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            raise AIError("Gemini returned broken JSON: " + text[:200], "gemini", "response")


def params_with_gemini(map_name: str, description: str, pal: dict, hint: str = "",
                       rgb=None, model: str | None = None) -> tuple[dict, str]:
    key = _need_key("gemini")
    model = model or settings.gemini_model()
    k = len(pal["share"])

    colors = "\n".join(
        f'  #{c["index"]}: {c["hex"]}, hue={c["hue"]}, saturation={c["saturation"]}, '
        f'brightness={c["brightness"]}, area={c["share_percent"]}%' for c in pbr.palette_info(pal))
    prompt = (
        f"You configure a procedural generator that builds a Blender PBR '{map_name}' map from a texture "
        "image. The map is computed per pixel from the image itself, so your job is only to choose "
        "parameters that make the result physically correct for THIS material and consistent with the "
        "picture (no random-looking patterns).\n\n"
        f"MATERIAL DESCRIPTION (from a vision model):\n{description}\n\n"
        f"PALETTE ({k} colours, index = position in the list-type parameters):\n{colors}\n\n"
        + (f"USER HINT: {hint.strip()}\n\n" if hint.strip() else "")
        + f"{GUIDE[map_name]}\n\n"
        f"AVAILABLE PARAMETERS (use only these keys, respect ranges):\n{pbr.schema_text(map_name, k)}\n\n"
        'Return ONE JSON object with these keys and an extra string key "notes" (1-3 sentences in '
        "English explaining the main choices). Return JSON only."
    )

    img_b64 = _jpeg_b64(rgb, 768) if rgb is not None else ""
    ck = (model, prompt, hashlib.sha256(img_b64.encode()).hexdigest())
    raw = cache_get("gemini", *ck)
    if raw is None:
        parts = [{"text": prompt}]
        if img_b64:
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_b64}})
        body = {"contents": [{"parts": parts}],
                "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}}
        r = _request("POST", GEMINI_URL.format(model=model), "gemini", json=body,
                     headers={"x-goog-api-key": key})
        _check(r, "gemini")
        try:
            rparts = r.json()["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in rparts if not p.get("thought"))
        except (KeyError, IndexError, TypeError):
            raise AIError("Gemini: empty answer (a safety filter may have fired).", "gemini", "response")
        raw = _extract_json(text)
        if isinstance(raw, list):
            raw = next((x for x in raw if isinstance(x, dict)), {})
        if not isinstance(raw, dict):
            raise AIError("Gemini returned JSON of an unexpected shape.", "gemini", "response")
        cache_put("gemini", raw, *ck)
    notes = str(raw.get("notes", ""))
    return pbr.sanitize(map_name, raw, k), notes
