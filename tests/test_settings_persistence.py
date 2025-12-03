"""Verify settings persistence helpers without requiring PySide runtime."""

from __future__ import annotations

import json
from pathlib import Path

from config.settings import DEFAULT_CHAT_ASSISTANT_BUBBLE_COLOR, DEFAULT_CHAT_USER_BUBBLE_COLOR


def test_persist_settings_to_config(tmp_path, monkeypatch):
    """Persist configured settings overrides to config.json."""
    monkeypatch.chdir(tmp_path)

    from config.persistence import persist_settings_to_config

    persist_settings_to_config(
        {
            "litellm_drop_params": ["max_tokens", "temperature"],
            "litellm_api_base": "https://example.com",
            "litellm_tts_model": "openai/tts-1",
        }
    )

    config_path = Path("config/config.json")
    assert config_path.exists()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data["litellm_drop_params"] == ["max_tokens", "temperature"]
    assert data["litellm_tts_model"] == "openai/tts-1"


def test_persist_settings_to_config_persists_chat_palette_override(tmp_path, monkeypatch):
    """Write non-default chat palette overrides to the config file."""
    monkeypatch.chdir(tmp_path)

    from config.persistence import persist_settings_to_config

    persist_settings_to_config(
        {
            "chat_colors": {
                "assistant": "#123456",
            }
        }
    )

    data = json.loads(Path("config/config.json").read_text(encoding="utf-8"))
    assert data["chat_colors"] == {"assistant": "#123456"}


def test_persist_settings_to_config_ignores_default_palette(tmp_path, monkeypatch):
    """Avoid writing chat colors when they match the defaults."""
    monkeypatch.chdir(tmp_path)

    from config.persistence import persist_settings_to_config

    persist_settings_to_config(
        {
            "chat_colors": {
                "user": DEFAULT_CHAT_USER_BUBBLE_COLOR,
                "assistant": DEFAULT_CHAT_ASSISTANT_BUBBLE_COLOR,
            }
        }
    )

    data = json.loads(Path("config/config.json").read_text(encoding="utf-8"))
    assert "chat_colors" not in data
