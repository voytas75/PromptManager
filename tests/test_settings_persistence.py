"""Verify settings persistence helpers without requiring PySide runtime."""

from __future__ import annotations

import json
from pathlib import Path


def test_persist_settings_to_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    from config.persistence import persist_settings_to_config

    persist_settings_to_config(
        {
            "litellm_drop_params": ["max_tokens", "temperature"],
            "litellm_api_base": "https://example.com",
        }
    )

    config_path = Path("config/config.json")
    assert config_path.exists()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data["litellm_drop_params"] == ["max_tokens", "temperature"]
