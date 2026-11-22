import io
import zipfile

import cv2
import numpy as np
import pytest

from image2pbr import export, pbr, samples, viewer


@pytest.fixture(scope="module")
def wood():
    return samples.make_wood(512)


@pytest.fixture(scope="module")
def palette(wood):
    return pbr.extract_palette(wood, 5)


def test_palette_shares_sum_to_one(palette):
    assert len(palette["share"]) == 5
    assert abs(float(palette["share"].sum()) - 1.0) < 1e-5
    assert all(0 <= info["share_percent"] <= 100 for info in pbr.palette_info(palette))


@pytest.mark.parametrize("name", pbr.MAPS)
def test_every_map_renders(wood, palette, name):
    out = pbr.render(wood, name, pbr.defaults(name), palette)
    assert out.shape[:2] == wood.shape[:2]
    assert out.dtype == np.uint8
    assert (out.ndim == 3) == (name == "normal")


def test_height_16bit(wood, palette):
    out = pbr.render(wood, "height", pbr.defaults("height"), palette, bit16=True)
    assert out.dtype == np.uint16 and out.max() > 255


def test_palette_values_are_used(wood, palette):
    k = len(palette["share"])
    p = pbr.sanitize("roughness", {"palette_values": [0.9] * k, "detail_amount": 0.0}, k)
    out = pbr.render(wood, "roughness", p, palette)
    assert abs(float(out.mean()) / 255 - 0.9) < 0.03


def test_resolution_independent_normal(wood, palette):
    p = pbr.sanitize("normal", {"strength": 3}, 5)
    a = pbr.render(wood, "normal", p, palette, max_side=256).astype(float)
    b = pbr.render(wood, "normal", p, palette, max_side=512).astype(float)
    assert abs(a.std() - b.std()) / b.std() < 0.15


def test_metal_binarize_gives_mask(wood, palette):
    p = pbr.sanitize("metallic", {"binarize": 1.0, "detail_amount": 1.0, "edge_softness": 0.01}, 5)
    out = pbr.render(wood, "metallic", p, palette)
    assert ((out < 40) | (out > 215)).mean() > 0.9


def test_sanitize_clamps_and_pads():
    p = pbr.sanitize("roughness", {"blur": 999, "channel": "bogus", "palette_values": [5, -2], "invert": 1}, 4)
    assert p["blur"] == 20
    assert p["channel"] == "luminance"
    assert p["palette_values"] == [1.0, 0.0, 0.6, 0.6]
    assert p["invert"] is True


def test_sanitize_ignores_garbage():
    p = pbr.sanitize("normal", {"strength": "abc", "palette_heights": 5}, 3)
    assert p["strength"] == pbr.defaults("normal")["strength"]
    assert p["palette_heights"] == []


def test_presets_cover_all_maps(wood, palette):
    for name in pbr.PRESETS:
        for m in pbr.MAPS:
            p = pbr.preset_params(name, m, 5)
            pbr.render(wood, m, p, palette, max_side=128)


def test_seamless_reduces_edge_mismatch(wood):
    def edge(a):
        a = a.astype(float)
        return np.abs(a[:, 0] - a[:, -1]).mean() + np.abs(a[0] - a[-1]).mean()

    sm = pbr.make_seamless(wood, 0.5)
    assert sm.shape == wood.shape and sm.dtype == np.uint8
    assert edge(sm) < edge(wood) * 0.5


def test_chunked_region_matches_full(wood, palette):
    vals = [0.1, 0.4, 0.7, 0.9, 0.5]
    f = wood.astype(np.float32) / 255
    full = pbr.palette_weights(f, palette, 14.0) @ np.array(vals, np.float32)
    chunked = pbr._region_map(f, palette, vals, 14.0, 0.0)
    assert np.allclose(full, chunked, atol=1e-5)


def test_bundle_contents_and_blender_script(wood, palette):
    params = {m: pbr.defaults(m) for m in pbr.MAPS}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        export.write_bundle(zf, "pack/", wood, palette, params, 256)
    z = zipfile.ZipFile(buf)
    names = set(z.namelist())
    for m in pbr.MAPS:
        assert f"pack/{m}.png" in names
    assert {"pack/basecolor.png", "pack/blender_setup.py", "pack/params.json"} <= names
    compile(z.read("pack/blender_setup.py").decode(), "blender_setup.py", "exec")
    h = cv2.imdecode(np.frombuffer(z.read("pack/height.png"), np.uint8), cv2.IMREAD_UNCHANGED)
    assert h.dtype == np.uint16 and h.shape == (256, 256)


def test_save_png_roundtrip(tmp_path, wood):
    path = export.save_png(str(tmp_path / "sub" / "x.png"), wood)
    back = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    assert np.array_equal(back, wood)


def test_user_presets_roundtrip(tmp_path, monkeypatch, wood, palette):
    from image2pbr import presets
    monkeypatch.setenv("IMAGE2PBR_PRESETS", str(tmp_path / "p.json"))
    assert presets.names() == list(pbr.PRESETS)
    params = {m: pbr.defaults(m) for m in pbr.MAPS}
    params["roughness"]["out_min"] = 0.33
    params["roughness"]["palette_values"] = [0.1, 0.2]
    label = presets.save_user("  Brushed   steel ", params)
    assert label == "My: Brushed steel" and label in presets.names()
    got = presets.get_params(label, "roughness", 5)
    assert abs(got["out_min"] - 0.33) < 1e-9
    assert got["palette_values"] == []                      # lists are dropped unless requested
    presets.save_user("Keep lists", params, include_lists=True)
    assert presets.get_params("My: Keep lists", "roughness", 4)["palette_values"] == [0.1, 0.2, 0.6, 0.6]
    for m in pbr.MAPS:
        pbr.render(wood, m, presets.get_params(label, m, 5), palette, max_side=64)
    assert presets.get_params("Wood", "normal", 5)["strength"] == 2.5     # built-ins still work
    with pytest.raises(ValueError):
        presets.save_user("Wood", params)                    # cannot shadow a built-in
    with pytest.raises(ValueError):
        presets.save_user("   ", params)
    assert presets.delete_user(label) and label not in presets.names()
    assert not presets.delete_user(label)


def test_viewer_document(wood, palette):
    maps = {m: pbr.render(wood, m, pbr.defaults(m), palette, max_side=64) for m in ("roughness", "normal", "height")}
    doc = viewer.build_document(wood, maps, 64)
    assert doc.startswith("<!doctype html>") and "__CONFIG__" not in doc
    assert doc.count("data:image/png;base64,") == 3 and "data:image/jpeg;base64," in doc
    assert "kNormal" in doc and "kBump" in doc and "kDisp" in doc
    assert 'srcdoc="' in viewer.iframe(doc, 300)
    assert "metallic" not in viewer.build_document(wood, {}, 32).split('"maps":')[1].split("}")[0]
