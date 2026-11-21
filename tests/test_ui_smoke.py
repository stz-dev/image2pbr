import zipfile

import pytest

gr = pytest.importorskip("gradio")

from image2pbr import ai, pbr, samples, ui  # noqa: E402


def test_ui_builds():
    assert ui.build() is not None


def test_status_banner_without_keys(monkeypatch):
    for s in ("GEMINI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(s, raising=False)
    md = ui.status_md()
    assert "aistudio.google.com/apikey" in md and "console.groq.com/keys" in md


def test_error_explanations():
    e = ai.AIError("Gemini: the API key was rejected (403).", "gemini", "key")
    txt = ui.explain_error(e)
    assert "Gemini" in txt and "Settings" in txt
    assert "Unexpected" in ui.explain_error(ValueError("boom"))


def test_generate_all_flow_with_fake_ai(monkeypatch):
    monkeypatch.setattr(ai, "describe_with_qwen", lambda rgb, pal, hint="", model=None: "wood")
    monkeypatch.setattr(ai, "params_with_gemini",
                        lambda m, d, pal, hint="", rgb=None, model=None: (pbr.sanitize(m, {}, len(pal["share"])), "n"))
    img = samples.make_wood(256)
    pal, *_ = ui.on_upload(img, 4)
    out = ui.do_all(img, pal, 4, False, 0.5, "", False, "", ["roughness", "normal", "ao"], "1024",
                    "roughness", {}, {})
    assert len(out) == 7 + len(ui.FLAT)
    names = zipfile.ZipFile(out[6]).namelist()
    assert {"roughness.png", "normal.png", "ao.png", "blender_setup.py"} <= set(names)
    assert "metallic.png" not in names


def test_undo_restores_previous_values():
    img = samples.make_wood(128)
    pal, _, maps, hist, _ = ui.on_upload(img, 4)
    vals = [(sp["default"] if sp["kind"] != "list" else "") for _, sp in ui.FLAT]
    _, maps, hist = ui.do_render(img, pal, 4, False, 0.5, "roughness", maps, hist, *vals)
    idx = next(i for i, (m, sp) in enumerate(ui.FLAT) if m == "roughness" and sp["key"] == "blur")
    vals2 = list(vals)
    vals2[idx] = 3.0
    _, maps, hist = ui.do_render(img, pal, 4, False, 0.5, "roughness", maps, hist, *vals2)
    res = ui.do_undo("roughness", hist)
    assert res[0]["roughness"][-1]["blur"] == 0.0


def test_preview_handler_reports_missing_maps():
    img = samples.make_wood(128)
    arr, info = ui.do_preview(img, False, 0.5, {"roughness": pbr.render(img, "roughness", pbr.defaults("roughness"), None)},
                              ["Base colour", "Roughness", "Normal"], "Sphere", 2, 40, 45, 2.0)
    assert arr.shape == (512, 512, 3)
    assert "roughness" in info and "normal" in info and "Not generated" in info
    arr, info = ui.do_preview(None, False, 0.5, {}, [], "Sphere", 2, 40, 45, 2.0)
    assert arr is None
