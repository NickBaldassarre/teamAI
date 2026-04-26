from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

GLOBAL_STATE_DIR = Path("~/.teamai")
RUNTIME_SETTINGS_FILE = "runtime_settings.json"


def _runtime_settings_path() -> Path:
    return GLOBAL_STATE_DIR.expanduser() / RUNTIME_SETTINGS_FILE


def load_runtime_settings() -> dict[str, object]:
    path = _runtime_settings_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to read runtime settings file: %s", path)
        return {}
    if not isinstance(raw, dict):
        return {}
    return raw


def save_runtime_settings(state: dict[str, object]) -> None:
    path = _runtime_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
