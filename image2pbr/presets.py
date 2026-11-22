"""Built-in presets (see pbr.PRESETS) plus user presets saved to user_presets.json."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from . import pbr, settings

USER_PREFIX = "My: "
MAX_NAME = 40


def _path() -> Path:
    return Path(os.environ.get("IMAGE2PBR_PRESETS") or settings.ROOT / "user_presets.json")


def load_user() -> dict:
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(data: dict) -> None:
    p = _path()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def clean_name(name: str) -> str:
    name = re.sub(r"\s+", " ", (name or "").strip())
    if name.startswith(USER_PREFIX):
        name = name[len(USER_PREFIX):].strip()
    return name[:MAX_NAME]


def names() -> list[str]:
    """Dropdown choices: built-in presets first, then the user's own."""
    return list(pbr.PRESETS) + [USER_PREFIX + n for n in sorted(load_user(), key=str.lower)]


def is_user(name: str | None) -> bool:
    return bool(name) and name.startswith(USER_PREFIX)


def save_user(name: str, params_by_map: dict, include_lists: bool = False) -> str:
    """Store the settings of every map under a name. Returns the dropdown label."""
    name = clean_name(name)
    if not name:
        raise ValueError("Give the preset a name.")
    if name in pbr.PRESETS:
        raise ValueError(f"'{name}' is a built-in preset name. Choose another name.")
    stored = {}
    for m, p in params_by_map.items():
        if m not in pbr.MAPS:
            continue
        stored[m] = {k: v for k, v in p.items() if include_lists or not isinstance(v, list)}
    data = load_user()
    data[name] = stored
    _write(data)
    return USER_PREFIX + name


def delete_user(label: str) -> bool:
    name = clean_name(label)
    data = load_user()
    if name not in data:
        return False
    del data[name]
    _write(data)
    return True


def get_params(label: str, map_name: str, k: int) -> dict:
    """Settings of a built-in or user preset for one map (sanitized for k palette colours)."""
    if is_user(label):
        raw = load_user().get(clean_name(label), {}).get(map_name, {})
        return pbr.sanitize(map_name, raw, k)
    return pbr.preset_params(label, map_name, k)
