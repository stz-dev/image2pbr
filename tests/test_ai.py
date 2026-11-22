"""AI layer tests with mocked HTTP (no network, no keys needed)."""
import json

import pytest
import requests

from image2pbr import ai, pbr, samples, settings


class FakeResp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text or json.dumps(payload or {})
        self.ok = status < 400

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    monkeypatch.setenv("GROQ_API_KEY", "q-test")
    monkeypatch.setenv("IMAGE2PBR_CACHE", "1")
    monkeypatch.setattr(ai, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(ai.time, "sleep", lambda s: None)


@pytest.fixture(scope="module")
def img_pal():
    rgb = samples.make_wood(128)
    return rgb, pbr.extract_palette(rgb, 4)


def gemini_ok(raw):
    return FakeResp(200, {"candidates": [{"content": {"parts": [{"text": json.dumps(raw)}]}}]})


def test_missing_key_is_clear_error(monkeypatch, img_pal):
    monkeypatch.delenv("GEMINI_API_KEY")
    with pytest.raises(ai.AIError) as e:
        ai.params_with_gemini("roughness", "wood", img_pal[1])
    assert e.value.kind == "key" and e.value.service == "gemini"


@pytest.mark.parametrize("status,kind", [(429, "limit"), (401, "key"), (403, "key")])
def test_http_errors_are_classified(monkeypatch, img_pal, status, kind):
    monkeypatch.setattr(requests, "request", lambda *a, **k: FakeResp(status, text="nope"))
    with pytest.raises(ai.AIError) as e:
        ai.params_with_gemini("roughness", "wood", img_pal[1])
    assert e.value.kind == kind


def test_network_error(monkeypatch, img_pal):
    def boom(*a, **k):
        raise requests.ConnectionError("down")
    monkeypatch.setattr(requests, "request", boom)
    with pytest.raises(ai.AIError) as e:
        ai.describe_with_qwen(*img_pal)
    assert e.value.kind == "network"


def test_gemini_parses_sanitizes_and_caches(monkeypatch, img_pal):
    calls = []

    def fake(method, url, **kw):
        calls.append(url)
        return gemini_ok({"normal": {"strength": 99, "palette_heights": [0.5], "notes": "ok", "bogus": 1}})

    monkeypatch.setattr(requests, "request", fake)
    params, notes = ai.params_with_gemini("normal", "wood", img_pal[1])
    assert params["strength"] == 20 and params["palette_heights"] == [0.5, 0.0, 0.0, 0.0]
    assert notes == "ok" and "bogus" not in params
    ai.params_with_gemini("normal", "wood", img_pal[1])
    assert len(calls) == 1                                  # second call served from cache


def test_cache_can_be_disabled(monkeypatch, img_pal):
    monkeypatch.setenv("IMAGE2PBR_CACHE", "0")
    calls = []
    monkeypatch.setattr(requests, "request", lambda *a, **k: calls.append(1) or gemini_ok({"ao": {"notes": "x"}}))
    ai.params_with_gemini("ao", "wood", img_pal[1])
    ai.params_with_gemini("ao", "wood", img_pal[1])
    assert len(calls) == 2


def test_gemini_list_response_and_code_fence(monkeypatch, img_pal):
    text = "```json\n[{\"ao\": {\"ao_strength\": 3}}]\n```"
    resp = FakeResp(200, {"candidates": [{"content": {"parts": [{"text": text}]}}]})
    monkeypatch.setattr(requests, "request", lambda *a, **k: resp)
    params, _ = ai.params_with_gemini("ao", "wood", img_pal[1])
    assert params["ao_strength"] == 3


def test_qwen_strips_thinking(monkeypatch, img_pal):
    payload = {"choices": [{"message": {"content": "<think>hmm</think>Oak wood, matte."}}]}
    monkeypatch.setattr(requests, "request", lambda *a, **k: FakeResp(200, payload))
    assert ai.describe_with_qwen(*img_pal) == "Oak wood, matte."


def test_qwen_empty_answer(monkeypatch, img_pal):
    payload = {"choices": [{"message": {"content": "<think>endless"}}]}
    monkeypatch.setattr(requests, "request", lambda *a, **k: FakeResp(200, payload))
    with pytest.raises(ai.AIError) as e:
        ai.describe_with_qwen(*img_pal)
    assert e.value.kind == "response"


def test_qwen_retired_model_falls_back(monkeypatch, img_pal):
    seen = []

    def fake(method, url, **kw):
        if method == "GET":
            return FakeResp(200, {"data": [{"id": "qwen/qwen9-1b"}, {"id": "other"}]})
        seen.append(kw["json"]["model"])
        if kw["json"]["model"] == "old/model":
            return FakeResp(404, text='{"error":{"message":"The model `old/model` does not exist"}}')
        return FakeResp(200, {"choices": [{"message": {"content": "ok text"}}]})

    monkeypatch.setattr(requests, "request", fake)
    assert ai.describe_with_qwen(*img_pal, model="old/model") == "ok text"
    assert seen == ["old/model", "qwen/qwen9-1b"]


def test_check_service(monkeypatch):
    models = {"models": [
        {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-tts", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/embedding-001", "supportedGenerationMethods": ["embedContent"]}]}
    monkeypatch.setattr(requests, "request", lambda *a, **k: FakeResp(200, models))
    r = ai.check_service("gemini")
    assert r["ok"] and r["models"] == ["gemini-2.5-flash"]
    monkeypatch.delenv("GROQ_API_KEY")
    assert ai.check_service("groq")["missing"]


def test_env_file_roundtrip(tmp_path):
    p = tmp_path / ".env"
    p.write_text("# comment\nFOO=bar\nGEMINI_API_KEY=old\n")
    settings.save_env({"GEMINI_API_KEY": "new", "GROQ_API_KEY": "q"}, p)
    text = p.read_text()
    assert "FOO=bar" in text and "GEMINI_API_KEY=new" in text and "GROQ_API_KEY=q" in text
    assert "old" not in text


def test_many_maps_use_one_request_and_partial_cache(monkeypatch, img_pal):
    prompts = []

    def fake(method, url, **kw):
        prompts.append(kw["json"]["contents"][0]["parts"][0]["text"])
        asked = [m for m in pbr.MAPS if f'=== MAP "{m}" ===' in prompts[-1]]
        return gemini_ok({m: {"notes": f"n-{m}", "strength": 3} for m in asked})

    monkeypatch.setattr(requests, "request", fake)
    wanted = ["roughness", "metallic", "normal", "height", "ao"]
    res = ai.params_for_maps(wanted, "wood", img_pal[1])
    assert len(prompts) == 1 and set(res) == set(wanted)          # 5 maps, ONE request
    assert res["normal"][0]["strength"] == 3 and res["ao"][1] == "n-ao"
    ai.params_for_maps(wanted, "wood", img_pal[1])
    assert len(prompts) == 1                                      # everything cached
    ai.params_for_maps(["roughness"], "wood", img_pal[1])
    assert len(prompts) == 1                                      # single map reuses the same cache
    ai.params_for_maps(["roughness"], "metal", img_pal[1])        # different description -> new request
    assert len(prompts) == 2


def test_missing_map_in_answer_is_reported(monkeypatch, img_pal):
    monkeypatch.setattr(requests, "request", lambda *a, **k: gemini_ok({"roughness": {"notes": "x"}}))
    res = ai.params_for_maps(["roughness", "ao"], "wood", img_pal[1])
    assert "roughness" in res and "ao" not in res
    with pytest.raises(ai.AIError):
        ai.params_with_gemini("ao", "wood2", img_pal[1])


def test_answer_without_requested_maps_is_error(monkeypatch, img_pal):
    monkeypatch.setattr(requests, "request", lambda *a, **k: gemini_ok({"other": {}}))
    with pytest.raises(ai.AIError) as e:
        ai.params_for_maps(["normal"], "wood", img_pal[1])
    assert e.value.kind == "response"
