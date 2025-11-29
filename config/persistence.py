"""Helpers for persisting runtime configuration without Qt dependencies.

Updates: v0.2.5 - 2025-12-06 - Persist embedding backend/model selections and drop defaults.
Updates: v0.2.4 - 2025-12-03 - Normalise chat palette overrides and skip default colours.
Updates: v0.2.3 - 2025-11-05 - Persist theme mode and chat appearance overrides.
Updates: v0.2.2 - 2025-11-05 - Persist chat appearance overrides.
Updates: v0.2.1 - 2025-11-05 - Persist LiteLLM workflow routing selections.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

from config.settings import (
    DEFAULT_CHAT_ASSISTANT_BUBBLE_COLOR,
    DEFAULT_CHAT_USER_BUBBLE_COLOR,
    DEFAULT_EMBEDDING_BACKEND,
    DEFAULT_EMBEDDING_MODEL,
)
from prompt_templates import (
    DEFAULT_PROMPT_TEMPLATES,
    PROMPT_TEMPLATE_KEYS,
)


def _normalise_drop_params(value: object | None) -> list[str] | None:
    if value is None:
        return None
    items: list[str]
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, Iterable):
        iterable = cast("Iterable[object]", value)
        items = [str(item).strip() for item in iterable if str(item).strip()]
    else:
        return None
    cleaned: list[str] = []
    for text in items:
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned or None


_CHAT_PALETTE_KEYS = {"user", "assistant"}
_CHAT_COLOR_PATTERN = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def _normalise_chat_palette(value: object | None) -> dict[str, str] | None:
    if value is None or not isinstance(value, Mapping):
        return None
    mapping = cast("Mapping[str, object]", value)
    cleaned: dict[str, str] = {}
    for role, colour in mapping.items():
        if role not in _CHAT_PALETTE_KEYS or not isinstance(colour, str):
            continue
        normalised = colour.strip().lower()
        if not _CHAT_COLOR_PATTERN.match(normalised):
            continue
        cleaned[role] = normalised
    return cleaned or None


def _is_default_chat_palette(palette: Mapping[str, str] | None) -> bool:
    if not palette:
        return True
    defaults = {
        "user": DEFAULT_CHAT_USER_BUBBLE_COLOR.lower(),
        "assistant": DEFAULT_CHAT_ASSISTANT_BUBBLE_COLOR.lower(),
    }
    for role, default_hex in defaults.items():
        current = palette.get(role)
        if current is None:
            continue
        if current != default_hex:
            return False
    return True


def _normalise_prompt_templates(value: object | None) -> dict[str, str] | None:
    if value is None or not isinstance(value, Mapping):
        return None
    template_mapping = cast("Mapping[str, object]", value)
    cleaned: dict[str, str] = {}
    for key, template in template_mapping.items():
        if key not in PROMPT_TEMPLATE_KEYS:
            continue
        if not isinstance(template, str):
            continue
        stripped = template.strip()
        if stripped:
            cleaned[key] = stripped
    return cleaned or None


def persist_settings_to_config(updates: dict[str, object | None]) -> None:
    """Persist selected settings to ``config/config.json``.

    Secrets (e.g. API keys) are never written to disk. ``litellm_drop_params`` values
    are normalised to de-duplicated lists so changes survive across restarts. Workflow
    routing entries set to ``fast`` are omitted so only explicit ``inference``
    selections persist.
    """

    config_path = Path("config/config.json")
    config_data: dict[str, Any] = {}
    if config_path.exists():
        try:
            parsed = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(parsed, Mapping):
                parsed_mapping = cast("Mapping[object, Any]", parsed)
                config_data = {str(key): value for key, value in parsed_mapping.items()}
        except json.JSONDecodeError:
            config_data = {}

    secret_keys = {"litellm_api_key"}
    for key, value in updates.items():
        if key in secret_keys:
            config_data.pop(key, None)
            continue
        if key == "litellm_drop_params":
            value = _normalise_drop_params(value)
        if key == "theme_mode":
            if isinstance(value, str):
                choice = value.strip().lower()
                if choice not in {"light", "dark"} or choice == "light":
                    value = None
                else:
                    value = choice
            else:
                value = None
        if key == "embedding_backend":
            if value == DEFAULT_EMBEDDING_BACKEND:
                value = None
        if key == "embedding_model":
            if value == DEFAULT_EMBEDDING_MODEL:
                value = None
        if key == "chat_colors":
            palette = _normalise_chat_palette(value)
            value = None if _is_default_chat_palette(palette) else palette
        if key == "prompt_templates":
            templates = _normalise_prompt_templates(value)
            if templates:
                defaults = cast("Mapping[str, str]", DEFAULT_PROMPT_TEMPLATES)
                filtered: dict[str, str] = {}
                for workflow, text in templates.items():
                    if text != defaults.get(workflow):
                        filtered[workflow] = text
                value = filtered or None
            else:
                value = None
        if key == "litellm_workflow_models" and isinstance(value, Mapping):
            cleaned: dict[str, str] = {}
            for route_key, route_value in cast("Mapping[object, object]", value).items():
                choice = str(route_value).strip().lower()
                if choice == "inference":
                    cleaned[str(route_key)] = "inference"
            value = cleaned or None
        if value is not None:
            config_data[key] = value
        else:
            config_data.pop(key, None)

    config_data.pop("catalog_path", None)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config_data, indent=2, ensure_ascii=False), encoding="utf-8")


__all__ = ["persist_settings_to_config"]
