"""Tests for CLI settings summary rendering.

Updates:
  v0.1.0 - 2026-04-25 - Cover effective-state labels for models, routing, and embeddings.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from cli.settings_summary import print_settings_summary
from config.settings import DEFAULT_EMBEDDING_BACKEND, DEFAULT_EMBEDDING_MODEL

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture


class _SummarySettings(SimpleNamespace):
    """Minimal settings object for summary rendering tests."""

    def __init__(self) -> None:
        super().__init__(
            db_path=Path("/tmp/db.sqlite"),
            chroma_path=Path("/tmp/chroma"),
            redis_dsn=None,
            cache_ttl_seconds=300,
            litellm_model=None,
            litellm_inference_model=None,
            litellm_api_key=None,
            litellm_api_base=None,
            litellm_api_version=None,
            litellm_reasoning_effort=None,
            litellm_drop_params=None,
            litellm_tts_model=None,
            litellm_tts_stream=True,
            litellm_stream=False,
            litellm_logging_enabled=False,
            litellm_workflow_models=None,
            embedding_backend=DEFAULT_EMBEDDING_BACKEND,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            web_search_provider="exa",
            exa_api_key=None,
            tavily_api_key=None,
            serper_api_key=None,
            serpapi_api_key=None,
            google_api_key=None,
            google_cse_id=None,
            auto_open_share_links=True,
            privatebin_url="https://privatebin.net/",
            privatebin_expiration="1week",
            privatebin_format="markdown",
            privatebin_compression="zlib",
            privatebin_burn_after_reading=False,
            privatebin_open_discussion=False,
        )


def _render(settings: _SummarySettings, capsys: CaptureFixture[str]) -> str:
    print_settings_summary(cast("Any", settings))
    captured = capsys.readouterr()
    return captured.out


def test_print_settings_summary_marks_default_and_explicit_states(
    capsys: CaptureFixture[str],
) -> None:
    settings = _SummarySettings()
    settings.litellm_model = "azure/gpt-4o-mini"
    settings.embedding_model = DEFAULT_EMBEDDING_MODEL
    settings.litellm_workflow_models = {"scenario_generation": "inference"}

    output = _render(settings, capsys)

    assert "Overall status: FAIL" in output
    assert "OK | Fast model: azure/gpt-4o-mini" in output
    assert "FAIL | API key: missing" in output
    assert "Next steps:" in output
    assert "- Add a LiteLLM API key so model calls can authenticate successfully." in output
    assert "Fast model: azure/gpt-4o-mini (explicit)" in output
    assert "Inference model: not set" in output
    assert f"Backend: {DEFAULT_EMBEDDING_BACKEND} (default)" in output
    assert f"Model: {DEFAULT_EMBEDDING_MODEL} (default)" in output
    assert "Scenario drafting: Inference (explicit)" in output
    assert "Prompt execution: Fast (default)" in output


def test_print_settings_summary_marks_derived_embedding_model(
    capsys: CaptureFixture[str],
) -> None:
    settings = _SummarySettings()
    settings.litellm_model = "azure/gpt-4o-mini"
    settings.embedding_model = settings.litellm_model

    output = _render(settings, capsys)

    assert f"Model: {settings.litellm_model} (derived from fast model)" in output


def test_print_settings_summary_marks_explicit_embedding_backend_and_model(
    capsys: CaptureFixture[str],
) -> None:
    settings = _SummarySettings()
    settings.embedding_backend = "sentence-transformers"
    settings.embedding_model = "all-MiniLM-L6-v2"

    output = _render(settings, capsys)

    assert "Backend: sentence-transformers (explicit)" in output
    assert "Model: all-MiniLM-L6-v2 (explicit)" in output


def test_print_settings_summary_marks_fully_configured_stack_as_ok(
    capsys: CaptureFixture[str],
) -> None:
    settings = _SummarySettings()
    settings.litellm_model = "azure/gpt-4o-mini"
    settings.litellm_inference_model = "azure/gpt-5"
    settings.litellm_api_key = "secret-key"
    settings.litellm_tts_model = "azure/tts-1"
    settings.redis_dsn = "redis://localhost:6379/0"

    output = _render(settings, capsys)

    assert "Overall status: OK" in output
    assert "OK | Redis cache: Redis caching configured via redis://localhost:6379/0" in output
    assert "Next steps:" not in output
