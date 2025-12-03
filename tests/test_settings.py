"""Tests for configuration loading and validation logic.

Updates:
  v0.1.7 - 2025-11-30 - Add docstrings to tests for lint compliance.
  v0.1.6 - 2025-11-30 - Cover routing for category suggestion workflow.
  v0.1.5 - 2025-11-29 - Wrap embedding backend tests for Ruff line length.
  v0.1.4 - 2025-11-05 - Cover LiteLLM inference model configuration.
  v0.1.3 - 2025-11-15 - Warn and ignore LiteLLM API secrets supplied via JSON configuration.
  v0.1.2 - 2025-11-14 - Cover LiteLLM API key loading from JSON configuration.
  v0.1.1 - 2025-11-03 - Add precedence test using example template.
  v0.1.0 - 2025-11-03 - Cover JSON/env precedence and validation errors.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from pytest import LogCaptureFixture, MonkeyPatch

from config import PromptManagerSettings, SettingsError, load_settings
from config.settings import DEFAULT_EMBEDDING_BACKEND, DEFAULT_EMBEDDING_MODEL


def _clear_litellm_env(monkeypatch: MonkeyPatch) -> None:
    vars_to_clear = [
        "PROMPT_MANAGER_LITELLM_MODEL",
        "PROMPT_MANAGER_LITELLM_INFERENCE_MODEL",
        "PROMPT_MANAGER_LITELLM_API_KEY",
        "PROMPT_MANAGER_LITELLM_API_BASE",
        "PROMPT_MANAGER_LITELLM_API_VERSION",
        "PROMPT_MANAGER_LITELLM_WORKFLOW_MODELS",
        "PROMPT_MANAGER_LITELLM_TTS_MODEL",
        "PROMPT_MANAGER_LITELLM_TTS_STREAM",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "LITELLM_MODEL",
        "LITELLM_INFERENCE_MODEL",
        "LITELLM_API_KEY",
        "LITELLM_API_BASE",
        "LITELLM_API_VERSION",
        "LITELLM_WORKFLOW_MODELS",
        "LITELLM_TTS_MODEL",
        "LITELLM_TTS_STREAM",
    ]
    for var in vars_to_clear:
        monkeypatch.delenv(var, raising=False)


def test_load_settings_reads_json_and_env(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Ensure JSON configuration is loaded and environment fills missing values."""
    _clear_litellm_env(monkeypatch)
    db_path_value = str(tmp_path / "from_json.db")
    chroma_path_value = str(tmp_path / "chromadb")
    config_payload = {
        "database_path": db_path_value,
        "chroma_path": chroma_path_value,
        "cache_ttl_seconds": 900,
    }
    config_path = tmp_path / "settings.json"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    monkeypatch.setenv("PROMPT_MANAGER_CONFIG_JSON", str(config_path))
    monkeypatch.setenv("PROMPT_MANAGER_CACHE_TTL_SECONDS", "120")
    monkeypatch.setenv("PROMPT_MANAGER_REDIS_DSN", "redis://localhost:6379/1")

    settings = load_settings()

    assert isinstance(settings, PromptManagerSettings)
    assert settings.db_path == Path(db_path_value)
    assert settings.chroma_path == Path(chroma_path_value)
    # JSON is higher precedence for overlapping keys
    assert settings.cache_ttl_seconds == 900
    assert settings.redis_dsn == "redis://localhost:6379/1"
    assert settings.litellm_model is None
    assert settings.litellm_inference_model is None
    assert settings.litellm_drop_params is None
    assert settings.litellm_tts_model is None
    assert settings.litellm_tts_stream is True


def test_json_precedes_env_when_both_provided(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Application JSON settings override environment variables for overlapping keys."""
    _clear_litellm_env(monkeypatch)
    # Use the repo's template as a base JSON.
    template_path = Path("config/config.template.json").resolve()
    assert template_path.exists(), "template config should exist"

    # Point loader at the template
    monkeypatch.setenv("PROMPT_MANAGER_CONFIG_JSON", str(template_path))
    # Override two values via environment
    monkeypatch.setenv("PROMPT_MANAGER_DATABASE_PATH", str(tmp_path / "override.db"))
    monkeypatch.setenv("PROMPT_MANAGER_CACHE_TTL_SECONDS", "42")

    settings = load_settings()
    assert settings.db_path == Path("data/prompt_manager.db").resolve()
    assert settings.cache_ttl_seconds == 600
    assert settings.litellm_drop_params == [
        "max_tokens",
        "max_output_tokens",
        "temperature",
        "timeout",
    ]


def test_json_with_litellm_api_key_is_ignored(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    caplog: LogCaptureFixture,
) -> None:
    """Ensure LiteLLM API keys present in JSON configs are ignored."""
    _clear_litellm_env(monkeypatch)
    config_payload = {
        "litellm_model": "azure/gpt-4o",
        "litellm_api_key": "from-json",
        "litellm_api_base": "https://azure.example.com",
    }
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "config.json"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with caplog.at_level(logging.WARNING, logger="prompt_manager.settings"):
        settings = load_settings()

    assert settings.litellm_model == "azure/gpt-4o"
    assert settings.litellm_inference_model is None
    assert settings.litellm_api_key is None
    assert settings.litellm_api_base == "https://azure.example.com"
    assert settings.litellm_drop_params is None
    assert settings.litellm_tts_model is None
    assert "Ignoring LiteLLM secret key" in caplog.text


def test_litellm_settings_from_env(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Load LiteLLM configuration entirely from environment variables."""
    _clear_litellm_env(monkeypatch)
    tmp_config = tmp_path / "config.json"
    tmp_config.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("PROMPT_MANAGER_CONFIG_JSON", str(tmp_config))
    monkeypatch.setenv("PROMPT_MANAGER_LITELLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("PROMPT_MANAGER_LITELLM_INFERENCE_MODEL", "gpt-4.1")
    monkeypatch.setenv("PROMPT_MANAGER_LITELLM_API_KEY", "secret-key")
    monkeypatch.setenv("PROMPT_MANAGER_LITELLM_API_BASE", "https://proxy.example.com")
    monkeypatch.setenv("PROMPT_MANAGER_LITELLM_TTS_MODEL", "openai/tts-1")
    monkeypatch.setenv("PROMPT_MANAGER_LITELLM_TTS_STREAM", "false")
    monkeypatch.setenv(
        "PROMPT_MANAGER_LITELLM_WORKFLOW_MODELS",
        json.dumps({"prompt_execution": "inference"}),
    )

    settings = load_settings()

    assert settings.litellm_model == "gpt-4o-mini"
    assert settings.litellm_inference_model == "gpt-4.1"
    assert settings.litellm_api_key == "secret-key"
    assert settings.litellm_api_base == "https://proxy.example.com"
    assert settings.litellm_drop_params is None
    assert settings.litellm_stream is False
    assert settings.litellm_workflow_models == {"prompt_execution": "inference"}
    assert settings.litellm_tts_model == "openai/tts-1"
    assert settings.litellm_tts_stream is False


def test_litellm_tts_model_from_json(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Load LiteLLM TTS model selection from JSON configs."""
    _clear_litellm_env(monkeypatch)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"litellm_tts_model": "azure/voice", "litellm_tts_stream": False}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROMPT_MANAGER_CONFIG_JSON", str(config_path))

    settings = load_settings()

    assert settings.litellm_tts_model == "azure/voice"
    assert settings.litellm_tts_stream is False


def test_litellm_inference_model_from_json(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Respect inference model overrides sourced from JSON config files."""
    _clear_litellm_env(monkeypatch)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "litellm_inference_model": "gpt-4.1",
                "litellm_workflow_models": {"prompt_engineering": "inference"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROMPT_MANAGER_CONFIG_JSON", str(config_path))

    settings = load_settings()

    assert settings.litellm_inference_model == "gpt-4.1"
    assert settings.litellm_model is None
    assert settings.litellm_workflow_models == {"prompt_engineering": "inference"}


def test_litellm_workflow_models_strip_fast_entries(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Remove redundant fast-tier entries when persisting workflow models."""
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "litellm_model": "gpt-4o-mini",
                "litellm_workflow_models": {
                    "prompt_execution": "fast",
                    "scenario_generation": "inference",
                    "prompt_structure_refinement": "inference",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROMPT_MANAGER_CONFIG_JSON", str(config_path))

    settings = load_settings()

    assert settings.litellm_workflow_models == {
        "scenario_generation": "inference",
        "prompt_structure_refinement": "inference",
    }


def test_litellm_workflow_models_include_category_generation(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ensure category generation routing survives JSON round-trips."""
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "litellm_model": "gpt-4o-mini",
                "litellm_workflow_models": {
                    "category_generation": "inference",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROMPT_MANAGER_CONFIG_JSON", str(config_path))

    settings = load_settings()

    assert settings.litellm_workflow_models == {"category_generation": "inference"}


def test_reasoning_effort_normalised(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Normalise reasoning effort strings before storing settings."""
    config_payload = {"litellm_reasoning_effort": "Medium"}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROMPT_MANAGER_CONFIG_JSON", str(config_path))

    settings = load_settings()

    assert settings.litellm_reasoning_effort == "medium"


def test_reasoning_effort_rejects_invalid(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Reject reasoning effort values that fall outside the supported enum."""
    config_payload = {"litellm_reasoning_effort": "fast"}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROMPT_MANAGER_CONFIG_JSON", str(config_path))

    with pytest.raises(SettingsError):
        load_settings()


def test_litellm_settings_accept_azure_aliases(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Allow Azure-specific LiteLLM identifiers when loading from env."""
    tmp_config = tmp_path / "config.json"
    tmp_config.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("PROMPT_MANAGER_CONFIG_JSON", str(tmp_config))
    monkeypatch.delenv("PROMPT_MANAGER_LITELLM_API_BASE", raising=False)
    monkeypatch.delenv("LITELLM_API_BASE", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://azure.example.com")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
    settings = load_settings(litellm_model="azure/gpt")
    assert settings.litellm_api_key == "azure-key"
    assert str(settings.litellm_api_base) == "https://azure.example.com"
    assert settings.litellm_api_version == "2024-05-01-preview"
    assert settings.litellm_drop_params is None


def test_litellm_drop_params_from_json_not_overridden_by_empty_env(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep JSON drop params even if env variables are empty."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"litellm_drop_params": ["max_tokens"]}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROMPT_MANAGER_CONFIG_JSON", str(config_path))
    monkeypatch.setenv("PROMPT_MANAGER_LITELLM_DROP_PARAMS", " ")

    settings = load_settings()

    assert settings.litellm_drop_params == ["max_tokens"]


def test_litellm_stream_flag_from_env(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Read the LiteLLM streaming flag from environment variables."""
    tmp_config = tmp_path / "config.json"
    tmp_config.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("PROMPT_MANAGER_CONFIG_JSON", str(tmp_config))
    monkeypatch.setenv("PROMPT_MANAGER_LITELLM_STREAM", "true")

    settings = load_settings()

    assert settings.litellm_stream is True


def test_embedding_backend_defaults_to_litellm(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Fallback to LiteLLM embeddings when no backend overrides are provided."""
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("PROMPT_MANAGER_CONFIG_JSON", str(config_path))
    monkeypatch.delenv("PROMPT_MANAGER_LITELLM_MODEL", raising=False)
    settings = load_settings()
    assert settings.embedding_backend == DEFAULT_EMBEDDING_BACKEND
    assert settings.embedding_model == DEFAULT_EMBEDDING_MODEL


def test_embedding_backend_requires_model(monkeypatch: MonkeyPatch) -> None:
    """Require a model name when selecting the sentence-transformers backend."""
    monkeypatch.setenv("PROMPT_MANAGER_EMBEDDING_BACKEND", "sentence-transformers")
    with pytest.raises(SettingsError):
        load_settings()


def test_embedding_backend_defaults_to_configured_constant_when_missing(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Use the configured constant when embedding settings are omitted."""
    tmp_config = tmp_path / "config.json"
    tmp_config.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("PROMPT_MANAGER_CONFIG_JSON", str(tmp_config))
    monkeypatch.setenv("PROMPT_MANAGER_LITELLM_MODEL", "azure/gpt-4.1")
    monkeypatch.setenv("PROMPT_MANAGER_EMBEDDING_BACKEND", "litellm")
    settings = load_settings()
    assert settings.embedding_backend == "litellm"
    assert settings.embedding_model == DEFAULT_EMBEDDING_MODEL


def test_embedding_backend_respects_explicit_embedding_model(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Honor explicit embedding model overrides regardless of defaults."""
    tmp_config = tmp_path / "config.json"
    tmp_config.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("PROMPT_MANAGER_CONFIG_JSON", str(tmp_config))
    monkeypatch.setenv("PROMPT_MANAGER_EMBEDDING_BACKEND", "litellm")
    monkeypatch.setenv("PROMPT_MANAGER_EMBEDDING_MODEL", "text-embedding-3-small")
    settings = load_settings()
    assert settings.embedding_backend == "litellm"
    assert settings.embedding_model == "text-embedding-3-small"


def test_embedding_model_auto_switches_backend(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Auto-switch the backend when a hosted embedding model is selected."""
    config_path = tmp_path / "config.json"
    payload = {"embedding_model": "text-embedding-3-small"}
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROMPT_MANAGER_CONFIG_JSON", str(config_path))

    settings = load_settings()

    assert settings.embedding_backend == "litellm"
    assert settings.embedding_model == "text-embedding-3-small"


def test_embedding_backend_rejects_unknown(monkeypatch: MonkeyPatch) -> None:
    """Raise SettingsError when an unsupported backend is configured."""
    monkeypatch.setenv("PROMPT_MANAGER_EMBEDDING_BACKEND", "unsupported")
    with pytest.raises(SettingsError):
        load_settings()


def test_json_settings_override_env(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Ensure JSON config can override environment-provided values when desired."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "config.json"
    config_path.write_text(json.dumps({"litellm_model": "file-model"}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROMPT_MANAGER_LITELLM_MODEL", "env-model")
    settings = load_settings()
    assert settings.litellm_model == "file-model"


def test_load_settings_raises_when_json_missing(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Raise SettingsError when the referenced JSON file cannot be found."""
    """Raise SettingsError when a JSON file path variable points to a missing file."""
    missing_path = tmp_path / "absent.json"
    monkeypatch.setenv("PROMPT_MANAGER_CONFIG_JSON", str(missing_path))
    with pytest.raises(SettingsError):
        load_settings()


def test_load_settings_rejects_invalid_cache_ttl(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Reject invalid cache TTL values loaded from JSON configuration."""
    """Reject non-positive cache TTL values sourced from environment variables."""
    tmp_config = tmp_path / "config.json"
    tmp_config.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("PROMPT_MANAGER_CONFIG_JSON", str(tmp_config))
    monkeypatch.setenv("PROMPT_MANAGER_CACHE_TTL_SECONDS", "0")
    with pytest.raises(SettingsError):
        load_settings()


def test_load_settings_rejects_missing_path() -> None:
    """Require filesystem paths when loading settings without defaults."""
    """Ensure db/chroma path validators reject None values."""
    with pytest.raises(SettingsError):
        load_settings(db_path=None)


def test_load_settings_raises_on_invalid_json(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Raise when the JSON config file cannot be parsed successfully."""
    config_path = tmp_path / "invalid.json"
    config_path.write_text("{invalid", encoding="utf-8")
    monkeypatch.setenv("PROMPT_MANAGER_CONFIG_JSON", str(config_path))
    with pytest.raises(SettingsError) as excinfo:
        load_settings()
    assert "Invalid JSON" in str(excinfo.value)


def test_load_settings_requires_json_object(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Reject JSON configs that do not contain a top-level object."""
    config_path = tmp_path / "array.json"
    config_path.write_text(json.dumps(["not", "object"]), encoding="utf-8")
    monkeypatch.setenv("PROMPT_MANAGER_CONFIG_JSON", str(config_path))
    with pytest.raises(SettingsError) as excinfo:
        load_settings()
    assert "must contain a JSON object" in str(excinfo.value)
