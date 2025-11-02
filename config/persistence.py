"""Helpers for persisting runtime configuration without Qt dependencies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def _normalise_drop_params(value: Optional[object]) -> Optional[list[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",") if part.strip()]
    else:
        items = []
        for item in value:  # type: ignore[arg-type]
            items.append(str(item).strip())
    cleaned: list[str] = []
    for text in items:
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned or None


def persist_settings_to_config(updates: dict[str, Optional[object]]) -> None:
    """Persist selected settings to ``config/config.json``.

    Secrets (e.g. API keys) are never written to disk. ``litellm_drop_params`` values
    are normalised to de-duplicated lists so changes survive across restarts.
    """

    config_path = Path("config/config.json")
    config_data: dict[str, object] = {}
    if config_path.exists():
        try:
            config_data = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            config_data = {}

    secret_keys = {"litellm_api_key"}
    for key, value in updates.items():
        if key in secret_keys:
            config_data.pop(key, None)
            continue
        if key == "litellm_drop_params":
            value = _normalise_drop_params(value)
        if value is not None:
            config_data[key] = value
        else:
            config_data.pop(key, None)

    config_data.pop("catalog_path", None)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config_data, indent=2, ensure_ascii=False), encoding="utf-8")


__all__ = ["persist_settings_to_config"]
