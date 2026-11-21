"""Keys, model names and the local .env file (never committed - see .gitignore)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"

DEFAULT_QWEN = "qwen/qwen3.8-27b"
DEFAULT_GEMINI = "gemini-2.5-flash"

KEY_LINKS = {
    "gemini": "https://aistudio.google.com/apikey",
    "groq": "https://console.groq.com/keys",
}
ENV_KEYS = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}


def _parse_env(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def load_env(path: Path | None = None) -> None:
    """Load .env into os.environ without overriding variables that are already set."""
    path = path or ENV_PATH
    try:
        values = _parse_env(path.read_text(encoding="utf-8"))
    except OSError:
        return
    for k, v in values.items():
        if v and not os.environ.get(k):
            os.environ[k] = v


def save_env(values: dict, path: Path | None = None) -> None:
    """Create/update KEY=VALUE lines in .env, keeping everything else."""
    path = path or ENV_PATH
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    pending = {k: v for k, v in values.items() if v is not None}
    out = []
    for line in lines:
        key = line.split("=", 1)[0].strip()
        if "=" in line and key in pending:
            out.append(f"{key}={pending.pop(key)}")
        else:
            out.append(line)
    out += [f"{k}={v}" for k, v in pending.items()]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def get_key(service: str) -> str:
    return os.environ.get(ENV_KEYS[service], "").strip()


def set_key(service: str, value: str) -> None:
    os.environ[ENV_KEYS[service]] = value.strip()


def qwen_model() -> str:
    return os.environ.get("QWEN_MODEL", "").strip() or DEFAULT_QWEN


def gemini_model() -> str:
    return os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_GEMINI


def cache_enabled() -> bool:
    return os.environ.get("IMAGE2PBR_CACHE", "1").strip() not in ("0", "false", "no", "off")
