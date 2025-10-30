"""Tests for configuration loading and validation logic.

Updates: v0.1.0 - 2025-11-03 - Cover JSON/env precedence and validation errors.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import PromptManagerSettings, SettingsError, load_settings


def test_load_settings_reads_json_and_env(monkeypatch, tmp_path) -> None:
    """Ensure JSON configuration is loaded and environment overrides take precedence."""
    config_payload = {
        "database_path": str(tmp_path / "from_json.db"),
        "chroma_path": str(tmp_path / "chromadb"),
        "cache_ttl_seconds": 900,
    }
    config_path = tmp_path / "settings.json"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    monkeypatch.setenv("PROMPT_MANAGER_CONFIG_JSON", str(config_path))
    monkeypatch.setenv("PROMPT_MANAGER_CACHE_TTL_SECONDS", "120")
    monkeypatch.setenv("PROMPT_MANAGER_REDIS_DSN", "redis://localhost:6379/1")

    settings = load_settings()

    assert isinstance(settings, PromptManagerSettings)
    assert settings.db_path == Path(config_payload["database_path"])
    assert settings.chroma_path == Path(config_payload["chroma_path"])
    assert settings.cache_ttl_seconds == 120
    assert settings.redis_dsn == "redis://localhost:6379/1"


def test_load_settings_raises_when_json_missing(monkeypatch, tmp_path) -> None:
    """Raise SettingsError when a JSON file path variable points to a missing file."""
    missing_path = tmp_path / "absent.json"
    monkeypatch.setenv("PROMPT_MANAGER_CONFIG_JSON", str(missing_path))
    with pytest.raises(SettingsError):
        load_settings()


def test_load_settings_rejects_invalid_cache_ttl(monkeypatch) -> None:
    """Reject non-positive cache TTL values sourced from environment variables."""
    monkeypatch.setenv("PROMPT_MANAGER_CACHE_TTL_SECONDS", "0")
    with pytest.raises(SettingsError):
        load_settings()
