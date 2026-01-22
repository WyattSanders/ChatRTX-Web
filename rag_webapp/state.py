from __future__ import annotations

import json
from pathlib import Path


STATE_FILE = "state.json"


def _state_path(data_dir: str) -> Path:
    p = Path(data_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / STATE_FILE


def get_active_model(data_dir: str) -> str | None:
    p = _state_path(data_dir)
    if not p.exists():
        return None
    try:
        o = json.loads(p.read_text(encoding="utf-8"))
        m = o.get("active_model")
        if isinstance(m, str) and m.strip():
            return m.strip()
    except Exception:
        return None
    return None


def set_active_model(data_dir: str, model: str) -> None:
    p = _state_path(data_dir)
    p.write_text(json.dumps({"active_model": model}, ensure_ascii=False) + "\n", encoding="utf-8")

