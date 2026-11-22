"""Gradio interface for Image2PBR."""
from __future__ import annotations

import copy
import functools
import io
import os
import tempfile
import time
import traceback
import zipfile

import gradio as gr
import numpy as np
from PIL import Image

from . import ai, export, pbr, presets, settings, viewer

OUT_DIR = str(settings.ROOT / "output")
PREVIEW_SIDE = 1024
SIZE_CHOICES = ["1024", "2048", "4096", "Original"]
HIST_LIMIT = 40
DEFAULT_MAPS = ["roughness", "metallic", "normal"]

# flat list of every parameter widget: (map, spec)
FLAT = [(m, sp) for m in pbr.MAPS for sp in pbr.spec(m)]

_last_error: dict = {"err": None}


# --------------------------------------------------------------------------
# Messages
# --------------------------------------------------------------------------
def keys_help_md() -> str:
    return (
        "**How to get the free API keys**\n\n"
        f"1. **Gemini** (chooses the parameters) - [create a key in Google AI Studio]({settings.KEY_LINKS['gemini']})"
        " (sign in with a Google account, press *Create API key*).\n"
        f"2. **Groq / Qwen** (describes the texture) - [create a key in the Groq console]({settings.KEY_LINKS['groq']})"
        " (sign up, press *Create API Key*).\n"
        "3. Paste both keys in the **Settings** tab and press **Save & test keys**.\n\n"
        "Both services have a free tier. Without keys you can still use the presets and every slider.")


def status_md(results: dict | None = None) -> str:
    """Banner text: what works, what is missing."""
    if results is None:
        results = {s: ai.check_service(s) for s in ("gemini", "groq")}
    missing = [s for s, r in results.items() if r.get("missing")]
    lines = []
    for s, r in results.items():
        name = ai.SERVICE_NAMES[s]
        if r.get("missing"):
            lines.append(f"- ⚠️ **{name}**: no API key")
        elif r["ok"]:
            lines.append(f"- ✅ **{name}**: {r['message']}")
        else:
            lines.append(f"- ❌ **{name}**: {r['message']}")
    head = ""
    if len(missing) == 2:
        head = ("### 👋 Welcome! Two free API keys are needed for the AI features\n"
                + keys_help_md() + "\n\n")
    elif missing or any(not r["ok"] for r in results.values()):
        head = "### ⚠️ Some AI services are not working\nOpen the **Settings** tab to fix the keys.\n\n"
    return head + "\n".join(lines)


def explain_error(e: ai.AIError | Exception) -> str:
    if not isinstance(e, ai.AIError):
        return f"❌ **Unexpected error:** {e}\n\nDetails are in the console. You can still use presets and sliders."
    svc = ai.SERVICE_NAMES.get(e.service or "", "AI service")
    advice = {
        "key": f"Open **Settings** and paste a valid key. Create one: "
               f"{settings.KEY_LINKS.get(e.service or '', '')}",
        "limit": "Free-tier limit reached. Wait about a minute, enable the cache, or pick another model in Settings.",
        "model": "Pick another model in **Settings** (press *Refresh model lists*).",
        "network": "Check your internet connection or VPN. Presets and sliders still work offline.",
        "response": "Try again. If it keeps happening, try another model in Settings.",
    }.get(e.kind, "Try again in a moment.")
    return f"❌ **{svc} failed.** {e}\n\n➡️ {advice}"


def guarded(fn):
    """Convert errors into a toast and remember the message for the persistent error box."""
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except gr.Error:
            raise
        except Exception as e:                      # noqa: BLE001 - UI boundary
            if not isinstance(e, ai.AIError):
                traceback.print_exc()
            _last_error["err"] = e
            raise gr.Error(str(e), print_exception=False)
    return wrapper


def show_error():
    e, _last_error["err"] = _last_error["err"], None
    return explain_error(e) if e else ""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _to_rgb(img) -> np.ndarray:
    if img is None:
        raise gr.Error("Upload an image first.")
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    return a[..., :3].astype(np.uint8)


def _prep(img, seamless, blend) -> np.ndarray:
    rgb = _to_rgb(img)
    return pbr.make_seamless(rgb, blend) if seamless else rgb


def _ensure_pal(rgb, pal, k):
    k = int(k)
    if pal is None or len(pal["share"]) != k:
        pal = pbr.extract_palette(rgb, k)
    return pal


def _size(choice) -> int | None:
    return None if choice == "Original" else int(choice)


def _show(arr: np.ndarray) -> np.ndarray:
    return np.stack([arr] * 3, -1) if arr.ndim == 2 else arr


def _to_u8(arr: np.ndarray) -> np.ndarray:
    return (arr >> 8).astype(np.uint8) if arr.dtype == np.uint16 else arr


def _params_from_widgets(map_name: str, vals) -> dict:
    p = {}
    for (m, sp), v in zip(FLAT, vals):
        if m != map_name:
            continue
        if sp["kind"] == "list":
            try:
                p[sp["key"]] = [float(x) for x in str(v or "").replace(";", ",").split(",") if x.strip()]
            except ValueError:
                raise gr.Error(f"Could not read the list '{sp['label']}'. Use numbers separated by commas.")
        else:
            p[sp["key"]] = v
    return p


def _all_params(vals, maps, strip_lists=False) -> dict:
    out = {}
    for m in maps:
        p = _params_from_widgets(m, vals)
        if strip_lists:
            p = {k: v for k, v in p.items() if not isinstance(v, list)}
            p["palette_values" if m in ("roughness", "metallic") else "palette_heights"] = []
        out[m] = pbr.sanitize(m, p, 0) if strip_lists else p
    return out


def _widget_updates(params_by_map: dict) -> list:
    ups = []
    for m, sp in FLAT:
        if m not in params_by_map:
            ups.append(gr.update())
            continue
        v = params_by_map[m].get(sp["key"], sp["default"])
        if sp["kind"] == "list":
            v = ", ".join(f"{x:.2f}" for x in v)
        ups.append(gr.update(value=v))
    return ups


def _push(hist, map_name, params):
    hist = dict(hist or {})
    lst = list(hist.get(map_name, []))
    if not lst or lst[-1] != params:
        lst.append(copy.deepcopy(params))
    hist[map_name] = lst[-HIST_LIMIT:]
    return hist


# --------------------------------------------------------------------------
# Handlers: image, palette, rendering
# --------------------------------------------------------------------------
def on_upload(img, k):
    if img is None:
        return None, "", {}, {}, None
    rgb = _to_rgb(img)
    pal = pbr.extract_palette(rgb, int(k))
    return pal, pbr.palette_html(pal), {}, {}, None


def on_k_change(img, k):
    if img is None:
        return None, ""
    pal = pbr.extract_palette(_to_rgb(img), int(k))
    return pal, pbr.palette_html(pal)


def _render_impl(img, pal, k, seamless, blend, map_name, maps, hist, vals, push):
    if img is None:
        return None, maps, hist
    rgb = _prep(img, seamless, blend)
    pal = _ensure_pal(rgb, pal, k)
    params = _params_from_widgets(map_name, vals)
    out = pbr.render(rgb, map_name, params, pal, max_side=PREVIEW_SIDE)
    maps = dict(maps or {})
    maps[map_name] = out
    if push:
        hist = _push(hist, map_name, params)
    return _show(out), maps, hist


def do_render(img, pal, k, seamless, blend, map_name, maps, hist, *vals):
    return _render_impl(img, pal, k, seamless, blend, map_name, maps, hist, vals, True)


def do_render_nopush(img, pal, k, seamless, blend, map_name, maps, hist, *vals):
    return _render_impl(img, pal, k, seamless, blend, map_name, maps, hist, vals, False)


def do_undo(map_name, hist):
    lst = list((hist or {}).get(map_name, []))
    if len(lst) < 2:
        gr.Info("Nothing to undo for this map.")
        return [hist] + [gr.update()] * len(FLAT)
    lst.pop()
    hist = dict(hist)
    hist[map_name] = lst
    return [hist] + _widget_updates({map_name: lst[-1]})


def do_reset(map_name, pal, k):
    kk = int(k)
    return _widget_updates({map_name: pbr.defaults(map_name)})


def do_preset(img, k, map_name, preset):
    if not preset:
        return [gr.update()] * len(FLAT)
    return _widget_updates({map_name: presets.get_params(preset, map_name, int(k))})


def do_preset_all(k, preset):
    if not preset:
        return [gr.update()] * len(FLAT)
    return _widget_updates({m: presets.get_params(preset, m, int(k)) for m in pbr.MAPS})


def do_save_preset(name, include_lists, *vals):
    """Save the current slider values of ALL maps as a named user preset."""
    try:
        label = presets.save_user(name, {m: _params_from_widgets(m, vals) for m in pbr.MAPS}, include_lists)
    except ValueError as e:
        raise gr.Error(str(e))
    gr.Info(f"Preset saved: {label}")
    choices = presets.names()
    return gr.update(choices=choices, value=label), gr.update(choices=choices), ""


def do_delete_preset(label):
    if not presets.is_user(label):
        raise gr.Error("Select one of YOUR presets (they start with 'My: '). Built-in presets cannot be deleted.")
    presets.delete_user(label)
    gr.Info(f"Preset deleted: {label}")
    choices = presets.names()
    return gr.update(choices=choices, value=None), gr.update(choices=choices, value=None)


def do_switch(map_name):
    return [gr.update(visible=(m == map_name)) for m in pbr.MAPS]


# --------------------------------------------------------------------------
# Handlers: AI
# --------------------------------------------------------------------------
@guarded
def do_describe(img, pal, k, seamless, blend, hint):
    rgb = _prep(img, seamless, blend)
    pal = _ensure_pal(rgb, pal, k)
    return ai.describe_with_qwen(rgb, pal, hint), ""


@guarded
def do_params(img, pal, k, seamless, blend, map_name, desc, hint, send_img):
    rgb = _prep(img, seamless, blend)
    pal = _ensure_pal(rgb, pal, k)
    if not desc.strip():
        raise gr.Error("The description is empty: press 'Describe (Qwen)' first or type a description.")
    params, notes = ai.params_with_gemini(map_name, desc, pal, hint, rgb if send_img else None)
    return [notes, ""] + _widget_updates({map_name: params})


@guarded
def do_all(img, pal, k, seamless, blend, hint, send_img, desc, sel_maps, size, map_choice, maps, hist):
    """Describe once, ask Gemini for every selected map, render everything and build the ZIP."""
    if not sel_maps:
        raise gr.Error("Select at least one map to generate.")
    rgb = _prep(img, seamless, blend)
    pal = _ensure_pal(rgb, pal, k)
    if not desc.strip():
        desc = ai.describe_with_qwen(rgb, pal, hint)

    wanted = [x for x in pbr.MAPS if x in sel_maps]
    results = ai.params_for_maps(wanted, desc, pal, hint, rgb if send_img else None)   # ONE Gemini request
    params_by_map, notes, warnings = {}, [], []
    maps = dict(maps or {})
    for m in wanted:
        if m in results:
            params, note = results[m]
            notes.append(f"**{m}:** {note}")
        else:
            params = pbr.sanitize(m, {}, len(pal["share"]))
            warnings.append(f"{m}: not in the AI answer - default parameters used")
        params_by_map[m] = params
        maps[m] = pbr.render(rgb, m, params, pal, max_side=PREVIEW_SIDE)
        hist = _push(hist, m, params)

    zip_path = _make_zip(rgb, pal, params_by_map, _size(size), "image2pbr")
    shown = map_choice if map_choice in maps else next(iter(params_by_map))
    err = ("⚠️ " + "; ".join(warnings)) if warnings else ""
    return ([desc, "\n\n".join(notes), err, _show(maps[shown]), maps, hist, zip_path]
            + _widget_updates(params_by_map))


def _make_zip(rgb, pal, params_by_map, max_side, name) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        export.write_bundle(zf, "", rgb, pal, params_by_map, max_side)
    return path


def do_save(img, pal, k, seamless, blend, map_name, size, *vals):
    rgb = _prep(img, seamless, blend)
    pal = _ensure_pal(rgb, pal, k)
    params = _params_from_widgets(map_name, vals)
    arr = pbr.render(rgb, map_name, params, pal, max_side=_size(size), bit16=(map_name == "height"))
    path = os.path.join(OUT_DIR, f"texture_{map_name}_{time.strftime('%Y%m%d_%H%M%S')}.png")
    return export.save_png(path, arr)


def do_zip_current(img, pal, k, seamless, blend, sel_maps, size, *vals):
    """ZIP with the CURRENT slider settings of every selected map (no AI calls)."""
    if not sel_maps:
        raise gr.Error("Select at least one map.")
    rgb = _prep(img, seamless, blend)
    pal = _ensure_pal(rgb, pal, k)
    params = {m: _params_from_widgets(m, vals) for m in pbr.MAPS if m in sel_maps}
    return _make_zip(rgb, pal, params, _size(size), "image2pbr")


# --------------------------------------------------------------------------
# 3D preview
# --------------------------------------------------------------------------
def do_preview(img, seamless, blend, maps):
    """Build the interactive WebGL viewer (HTML) from the image and the generated maps."""
    if img is None:
        return "", "Upload an image first.", None
    maps = maps or {}
    albedo = pbr.fit(_prep(img, seamless, blend), 1024)
    doc = viewer.build_document(albedo, maps, 512)
    have = [m for m in pbr.MAPS if m in maps]
    info = "**Maps in the viewer:** base colour" + "".join(f", {m}" for m in have)
    missing = [m for m in pbr.MAPS if m not in maps]
    if missing:
        info += f"  \n**Not generated yet:** {', '.join(missing)} (their sliders are disabled)."
    path = os.path.join(OUT_DIR, "viewer.html")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return viewer.iframe(doc, 720), info, path


# --------------------------------------------------------------------------
# Batch
# --------------------------------------------------------------------------
def do_batch(files, sel_maps, mode, preset, k, seamless, blend, size, hint, send_img, *vals):
    if not files:
        raise gr.Error("Add some images first.")
    if not sel_maps:
        raise gr.Error("Select at least one map.")
    if mode == "Preset (no AI)" and not preset:
        raise gr.Error("Choose a preset for the batch.")
    maps_sel = [m for m in pbr.MAPS if m in sel_maps]
    os.makedirs(OUT_DIR, exist_ok=True)
    zpath = os.path.join(OUT_DIR, f"batch_{time.strftime('%Y%m%d_%H%M%S')}.zip")
    log, fatal = [], False
    paths = [f if isinstance(f, str) else getattr(f, "name", str(f)) for f in files]
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, path in enumerate(paths, 1):
            name = os.path.splitext(os.path.basename(path))[0]
            log.append(f"[{i}/{len(paths)}] {name} ...")
            yield "\n".join(log), None
            try:
                rgb = np.array(Image.open(path).convert("RGB"))
                if seamless:
                    rgb = pbr.make_seamless(rgb, blend)
                pal = pbr.extract_palette(rgb, int(k))
                kk = len(pal["share"])
                if mode == "Preset (no AI)":
                    params = {m: presets.get_params(preset, m, kk) for m in maps_sel}
                elif mode == "Current slider settings":
                    params = _all_params(vals, maps_sel, strip_lists=True)
                else:
                    if fatal:
                        raise ai.AIError("AI skipped (key/network problem earlier)", None, "other")
                    desc = ai.describe_with_qwen(rgb, pal, hint)
                    res = ai.params_for_maps(maps_sel, desc, pal, hint, rgb if send_img else None)
                    params = {}
                    for m in maps_sel:
                        if m in res:
                            params[m] = res[m][0]
                        else:
                            params[m] = pbr.sanitize(m, {}, kk)
                            log.append(f"    {m}: not in the AI answer, defaults used")
                export.write_bundle(zf, f"{name}/", rgb, pal, params, _size(size))
                log[-1] += " done"
            except ai.AIError as e:
                if e.kind in ("key", "network"):
                    fatal = True
                log[-1] += f" FAILED: {e}"
            except Exception as e:                  # noqa: BLE001
                log[-1] += f" FAILED: {e}"
    log.append(f"Finished. ZIP: {zpath}")
    yield "\n".join(log), zpath


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
def _model_updates(results):
    q = results["groq"].get("models") or []
    g = results["gemini"].get("models") or []
    qv, gv = settings.qwen_model(), settings.gemini_model()
    return (gr.update(choices=list(dict.fromkeys(q + [qv])), value=qv),
            gr.update(choices=list(dict.fromkeys(g + [gv])), value=gv))


def do_startup():
    results = {s: ai.check_service(s) for s in ("gemini", "groq")}
    q, g = _model_updates(results)
    return status_md(results), q, g


def do_save_keys(gem, groq, remember):
    updates = {}
    if gem and gem.strip():
        settings.set_key("gemini", gem)
        updates[settings.ENV_KEYS["gemini"]] = gem.strip()
    if groq and groq.strip():
        settings.set_key("groq", groq)
        updates[settings.ENV_KEYS["groq"]] = groq.strip()
    saved = ""
    if updates and remember:
        try:
            settings.save_env(updates)
            saved = f"\n\n💾 Saved to `{settings.ENV_PATH.name}` (stays on this computer, ignored by git)."
        except OSError as e:
            saved = f"\n\n⚠️ Could not write {settings.ENV_PATH}: {e}"
    results = {s: ai.check_service(s) for s in ("gemini", "groq")}
    q, g = _model_updates(results)
    banner = status_md(results)
    return banner + saved, banner, q, g, "", ""


def do_refresh_models():
    results = {s: ai.check_service(s) for s in ("gemini", "groq")}
    q, g = _model_updates(results)
    return status_md(results), q, g


def do_models_changed(qwen, gem, remember):
    if qwen:
        os.environ["QWEN_MODEL"] = qwen
    if gem:
        os.environ["GEMINI_MODEL"] = gem
    if remember:
        try:
            settings.save_env({"QWEN_MODEL": qwen or None, "GEMINI_MODEL": gem or None})
        except OSError:
            pass


def do_cache_toggle(on):
    os.environ["IMAGE2PBR_CACHE"] = "1" if on else "0"


def do_cache_clear():
    n = ai.cache_clear()
    gr.Info(f"Cache cleared ({n} entries).")


DARK_JS = """() => {
  const d = document.body.classList.toggle('dark');
  try { localStorage.setItem('image2pbr_dark', d ? '1' : '0'); } catch (e) {}
}"""
DARK_LOAD_JS = """() => {
  try { if (localStorage.getItem('image2pbr_dark') === '1') document.body.classList.add('dark'); } catch (e) {}
}"""


# --------------------------------------------------------------------------
# UI construction
# --------------------------------------------------------------------------
def make_widget(sp):
    if sp["kind"] == "float":
        lo, hi, st = sp["rng"]
        return gr.Slider(lo, hi, value=sp["default"], step=st, label=sp["label"])
    if sp["kind"] == "bool":
        return gr.Checkbox(value=sp["default"], label=sp["label"])
    if sp["kind"] == "choice":
        return gr.Dropdown(sp["choices"], value=sp["default"], label=sp["label"])
    return gr.Textbox(value="", label=sp["label"], placeholder="e.g. 0.9, 0.3, 0.6, ...")


def build() -> gr.Blocks:
    with gr.Blocks(title="Image2PBR") as demo:
        gr.Markdown("# Image2PBR\n"
                    "AI-assisted **PBR map generator for Blender**: texture image → roughness, metallic, normal, "
                    "height and AO maps. Qwen describes the material, Gemini chooses the settings, the maps are "
                    "computed from your image's pixels.")
        banner = gr.Markdown("Checking API keys...")
        pal_state, maps_state, hist_state = gr.State(None), gr.State({}), gr.State({})

        with gr.Tabs():
            # ------------------------------------------------------------------ Generate
            with gr.Tab("Generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        img = gr.Image(label="Source texture", type="numpy", image_mode="RGB", height=300)
                        map_choice = gr.Radio(pbr.MAPS, value="roughness", label="Map to edit / preview")
                        sel_maps = gr.CheckboxGroup(pbr.MAPS, value=DEFAULT_MAPS,
                                                    label="Maps for “Generate all + ZIP”")
                        hint = gr.Textbox(label="Material hint (optional)",
                                          placeholder="e.g. rusty painted steel plate")
                        with gr.Row():
                            k_slider = gr.Slider(2, 10, value=6, step=1, label="Palette colours")
                            size_dd = gr.Dropdown(SIZE_CHOICES, value="2048", label="Output size (px)")
                        with gr.Row():
                            seamless = gr.Checkbox(value=False, label="Make seamless (tileable)")
                            blend = gr.Slider(0.05, 1.0, value=0.5, step=0.05, label="Seamless blend width")
                        send_img = gr.Checkbox(value=False, label="Also send the image to Gemini (more accurate; "
                                               "free-tier data may be used by Google)")
                        btn_all = gr.Button("⚡ Generate ALL selected maps (Qwen + Gemini) + ZIP", variant="primary")
                        with gr.Row():
                            btn_auto = gr.Button("Auto for this map")
                            btn_desc = gr.Button("1. Describe (Qwen)")
                            btn_gem = gr.Button("2. Settings (Gemini)")
                        with gr.Row():
                            preset = gr.Dropdown(presets.names(), value=None, label="Preset (no AI)")
                            btn_preset_all = gr.Button("Apply preset to ALL maps")
                        with gr.Accordion("Save / delete my presets", open=False):
                            gr.Markdown("Saves the current slider values of **all** maps under a name. "
                                        "Your presets appear in the dropdown above as *My: name* "
                                        "(stored in `user_presets.json` next to the app).")
                            with gr.Row():
                                preset_name = gr.Textbox(label="Preset name", placeholder="e.g. Brushed steel",
                                                         scale=2)
                                preset_lists = gr.Checkbox(value=False, scale=1,
                                                           label="Include per-colour values (only useful for "
                                                                 "images with the same colours)")
                            with gr.Row():
                                btn_psave = gr.Button("💾 Save preset")
                                btn_pdel = gr.Button("🗑 Delete selected preset")
                        err_box = gr.Markdown("")
                        palette_html = gr.HTML()
                        desc = gr.Textbox(label="Texture description (editable)", lines=8)
                        notes = gr.Textbox(label="Gemini notes", lines=3, interactive=False)

                    with gr.Column(scale=1):
                        result = gr.Image(label="Preview of the selected map", interactive=False, height=420)
                        with gr.Row():
                            btn_render = gr.Button("🖼 Refresh")
                            btn_undo = gr.Button("↩ Undo")
                            btn_reset = gr.Button("Reset map")
                        with gr.Row():
                            btn_save = gr.Button("💾 Save this map (PNG)")
                            btn_zip = gr.Button("📦 ZIP with current settings")
                        saved_file = gr.File(label="Saved PNG")
                        zip_file = gr.File(label="ZIP bundle (maps + Blender script)")
                        gr.Markdown("**In Blender:** run `blender_setup.py` from the ZIP (Scripting tab → Open → "
                                    "Run Script), or load the maps manually: *Non-Color* for everything except the "
                                    "base colour; normal map through a *Normal Map* node (OpenGL).")

                comps = []
                groups = {}
                with gr.Accordion("Parameters of the selected map", open=True):
                    for m in pbr.MAPS:
                        with gr.Group(visible=(m == "roughness")) as grp:
                            for mm, sp in FLAT:
                                if mm == m:
                                    comps.append(make_widget(sp))
                        groups[m] = grp

            # ------------------------------------------------------------------ 3D preview
            with gr.Tab("3D preview") as tab_prev:
                gr.Markdown("**Real-time 3D viewer** (WebGL, runs in your browser). Drag to rotate, wheel to zoom. "
                            "Every map has its own **strength slider** inside the viewer (0 = off); height can act "
                            "as bump and as real displacement. It refreshes after **Generate ALL**, when you open "
                            "this tab, or with the button.")
                with gr.Row():
                    btn_prev = gr.Button("Refresh viewer", variant="primary", scale=1)
                    prev_file = gr.File(label="Standalone viewer (open in any browser)", interactive=False, scale=2)
                prev_info = gr.Markdown("")
                prev_html = gr.HTML()

            # ------------------------------------------------------------------ Batch
            with gr.Tab("Batch"):
                gr.Markdown("Process many images at once. Result: one ZIP with a folder per image "
                            "(maps + Blender script). AI mode uses 1 Qwen + 1 Gemini request per image, "
                            "so mind the free-tier limits (the cache helps when you re-run).")
                b_files = gr.File(label="Images", file_count="multiple", file_types=["image"])
                with gr.Row():
                    b_maps = gr.CheckboxGroup(pbr.MAPS, value=DEFAULT_MAPS, label="Maps")
                    b_mode = gr.Radio(["AI for every image", "Preset (no AI)", "Current slider settings"],
                                      value="AI for every image", label="Settings source")
                    b_preset = gr.Dropdown(presets.names(), value=None, label="Preset (for preset mode)")
                b_run = gr.Button("Run batch", variant="primary")
                b_log = gr.Textbox(label="Log", lines=10, interactive=False)
                b_zip = gr.File(label="Batch ZIP")

            # ------------------------------------------------------------------ Settings
            with gr.Tab("Settings"):
                gr.Markdown("## API keys")
                gr.Markdown(keys_help_md())
                with gr.Row():
                    gem_key = gr.Textbox(label="Gemini API key", type="password",
                                         placeholder="leave empty to keep the current key")
                    groq_key = gr.Textbox(label="Groq API key", type="password",
                                          placeholder="leave empty to keep the current key")
                remember = gr.Checkbox(value=True, label="Remember on this computer (saved in the local .env file, "
                                                          "never uploaded to git)")
                with gr.Row():
                    btn_keys = gr.Button("Save & test keys", variant="primary")
                    btn_models = gr.Button("Refresh model lists")
                keys_status = gr.Markdown("")
                gr.Markdown("## Models")
                with gr.Row():
                    qwen_dd = gr.Dropdown([settings.qwen_model()], value=settings.qwen_model(),
                                          label="Vision model (Groq)", allow_custom_value=True)
                    gem_dd = gr.Dropdown([settings.gemini_model()], value=settings.gemini_model(),
                                         label="Gemini model", allow_custom_value=True)
                gr.Markdown("## Other")
                with gr.Row():
                    cache_on = gr.Checkbox(value=settings.cache_enabled(),
                                           label="Cache AI answers (saves free-tier requests)")
                    btn_cache = gr.Button("Clear cache")
                    btn_dark = gr.Button("🌓 Toggle dark theme")

        # ------------------------------------------------------------------ events
        render_in = [img, pal_state, k_slider, seamless, blend, map_choice, maps_state, hist_state] + comps
        render_out = [result, maps_state, hist_state]

        img.upload(on_upload, [img, k_slider], [pal_state, palette_html, maps_state, hist_state, result])
        k_slider.release(on_k_change, [img, k_slider], [pal_state, palette_html])
        map_choice.change(do_switch, map_choice, [groups[m] for m in pbr.MAPS]).then(
            do_render, render_in, render_out)

        prev_in = [img, seamless, blend, maps_state]
        prev_out = [prev_html, prev_info, prev_file]
        ai_in = [img, pal_state, k_slider, seamless, blend]
        # single-step buttons
        e = btn_desc.click(do_describe, ai_in + [hint], [desc, err_box])
        e.failure(show_error, None, err_box)
        e = btn_gem.click(do_params, ai_in + [map_choice, desc, hint, send_img], [notes, err_box] + comps)
        e.failure(show_error, None, err_box)
        e.success(do_render, render_in, render_out)

        # auto for the current map: describe -> params -> render
        e1 = btn_auto.click(on_k_change, [img, k_slider], [pal_state, palette_html])
        e2 = e1.success(do_describe, ai_in + [hint], [desc, err_box])
        e2.failure(show_error, None, err_box)
        e3 = e2.success(do_params, ai_in + [map_choice, desc, hint, send_img], [notes, err_box] + comps)
        e3.failure(show_error, None, err_box)
        e3.success(do_render, render_in, render_out)

        # all maps at once
        e = btn_all.click(do_all, ai_in + [hint, send_img, desc, sel_maps, size_dd, map_choice, maps_state, hist_state],
                          [desc, notes, err_box, result, maps_state, hist_state, zip_file] + comps)
        e.failure(show_error, None, err_box)
        e.success(do_preview, prev_in, prev_out)

        preset.input(do_preset, [img, k_slider, map_choice, preset], comps).then(do_render, render_in, render_out)
        btn_preset_all.click(do_preset_all, [k_slider, preset], comps).then(do_render, render_in, render_out)
        btn_psave.click(do_save_preset, [preset_name, preset_lists] + comps, [preset, b_preset, preset_name])
        btn_pdel.click(do_delete_preset, preset, [preset, b_preset])

        btn_render.click(do_render, render_in, render_out)
        btn_undo.click(do_undo, [map_choice, hist_state], [hist_state] + comps).then(
            do_render_nopush, render_in, render_out)
        btn_reset.click(do_reset, [map_choice, pal_state, k_slider], comps).then(do_render, render_in, render_out)
        btn_save.click(do_save, [img, pal_state, k_slider, seamless, blend, map_choice, size_dd] + comps, saved_file)
        btn_zip.click(do_zip_current, [img, pal_state, k_slider, seamless, blend, sel_maps, size_dd] + comps, zip_file)
        seamless.input(do_render, render_in, render_out)
        blend.release(do_render, render_in, render_out)

        triggers = []
        for c in comps:
            if isinstance(c, gr.Slider):
                triggers.append(c.release)
            elif isinstance(c, (gr.Checkbox, gr.Dropdown)):
                triggers.append(c.input)
            else:
                triggers.append(c.submit)
        gr.on(triggers=triggers, fn=do_render, inputs=render_in, outputs=render_out)

        btn_prev.click(do_preview, prev_in, prev_out)
        tab_prev.select(do_preview, prev_in, prev_out)

        b_run.click(do_batch, [b_files, b_maps, b_mode, b_preset, k_slider, seamless, blend, size_dd, hint, send_img]
                    + comps, [b_log, b_zip])

        btn_keys.click(do_save_keys, [gem_key, groq_key, remember],
                       [keys_status, banner, qwen_dd, gem_dd, gem_key, groq_key])
        btn_models.click(do_refresh_models, None, [keys_status, qwen_dd, gem_dd]).then(
            lambda s: s, keys_status, banner)
        qwen_dd.input(do_models_changed, [qwen_dd, gem_dd, remember], None)
        gem_dd.input(do_models_changed, [qwen_dd, gem_dd, remember], None)
        cache_on.input(do_cache_toggle, cache_on, None)
        btn_cache.click(do_cache_clear, None, None)
        btn_dark.click(None, None, None, js=DARK_JS)

        demo.load(do_startup, None, [banner, qwen_dd, gem_dd])
        demo.load(None, None, None, js=DARK_LOAD_JS)
    return demo


def startup_message() -> str:
    missing = [ai.SERVICE_NAMES[s] for s in ("gemini", "groq") if not settings.get_key(s)]
    if not missing:
        return ""
    return ("\n" + "=" * 70 + "\n"
            f"  Missing API key(s): {', '.join(missing)}\n"
            "  The app starts anyway. Add free keys in the Settings tab:\n"
            f"    Gemini: {settings.KEY_LINKS['gemini']}\n"
            f"    Groq  : {settings.KEY_LINKS['groq']}\n" + "=" * 70 + "\n")


def main():
    settings.load_env()
    msg = startup_message()
    if msg:
        print(msg)
    demo = build()
    os.makedirs(OUT_DIR, exist_ok=True)
    demo.queue().launch(inbrowser=True, allowed_paths=[OUT_DIR, tempfile.gettempdir()])


if __name__ == "__main__":
    main()
