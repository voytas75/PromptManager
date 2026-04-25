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


def test_print_settings_summary_starts_with_diagnostics_block(
    capsys: CaptureFixture[str],
) -> None:
    settings = _SummarySettings()
    settings.litellm_model = "azure/gpt-4.1"

    output = _render(settings, capsys)
    lines = output.splitlines()

    assert lines[0] == "Prompt Manager configuration summary"
    assert lines[1] == "------------------------------------"
    assert lines[2] == "Diagnostics"
    assert lines[3] == "-----------"
    assert lines[4].startswith("Overall status: ")
    assert output.index("Diagnostics\n-----------") < output.index("Database path:")


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
    assert "Routing summary: inference for: Scenario drafting [custom]" in output
    assert (
        f"Embeddings summary: {DEFAULT_EMBEDDING_BACKEND} / {DEFAULT_EMBEDDING_MODEL} [custom]"
        in output
    )


def test_print_settings_summary_marks_derived_embedding_model(
    capsys: CaptureFixture[str],
) -> None:
    settings = _SummarySettings()
    settings.litellm_model = "azure/gpt-4o-mini"
    settings.embedding_model = settings.litellm_model

    output = _render(settings, capsys)

    assert f"Model: {settings.litellm_model} (derived from fast model)" in output
    assert (
        f"Embeddings summary: {DEFAULT_EMBEDDING_BACKEND} / {settings.litellm_model} "
        "[derived from fast model]" in output
    )


def test_print_settings_summary_uses_user_labels_in_routing_summary(
    capsys: CaptureFixture[str],
) -> None:
    settings = _SummarySettings()
    settings.litellm_workflow_models = {"scenario_generation": "inference"}

    output = _render(settings, capsys)

    assert "Routing summary: inference for: Scenario drafting [custom]" in output


def test_print_settings_summary_marks_all_fast_routes_as_custom_when_explicitly_set(
    capsys: CaptureFixture[str],
) -> None:
    settings = _SummarySettings()
    settings.litellm_workflow_models = {
        workflow_key: "fast" for workflow_key in ("scenario_generation", "prompt_execution")
    }

    output = _render(settings, capsys)

    assert "Scenario drafting: Fast (default)" in output
    assert "Prompt execution: Fast (default)" in output
    assert "Routing summary: all workflows use the fast model [custom]" in output


def test_print_settings_summary_marks_backend_only_embedding_summary_as_custom(
    capsys: CaptureFixture[str],
) -> None:
    settings = _SummarySettings()
    settings.embedding_backend = "sentence-transformers"
    settings.embedding_model = None

    output = _render(settings, capsys)

    assert "Backend: sentence-transformers (explicit)" in output
    assert "Model: not set" in output
    assert "Embeddings summary: sentence-transformers / not set [custom]" in output


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


def test_print_settings_summary_shows_precedence_explanation_for_env_override(
    capsys: CaptureFixture[str],
) -> None:
    settings = _SummarySettings()
    settings.litellm_model = "azure/gpt-4.1-env"
    settings.litellm_api_key = "secret-key"
    settings._diagnostics_sources = {
        "litellm_model": "env",
        "litellm_model_config": "azure/gpt-4.1-config",
        "litellm_api_key": "env",
        "litellm_api_key_config": "configured in config.json",
        "embedding_model": "derived",
        "redis_dsn": "runtime",
    }

    output = _render(settings, capsys)

    assert (
        "OK | Fast model: azure/gpt-4.1-env [env; env overrides config (azure/gpt-4.1-config)]"
        in output
    )
    assert (
        "OK | API key: configured [env; env overrides config (configured in config.json)]" in output
    )


def test_print_settings_summary_shows_default_and_derived_precedence_labels(
    capsys: CaptureFixture[str],
) -> None:
    settings = _SummarySettings()
    settings.litellm_model = "azure/gpt-4.1"
    settings.litellm_api_key = "secret-key"
    settings.embedding_model = settings.litellm_model
    settings._diagnostics_sources = {
        "litellm_model": "config",
        "litellm_inference_model": "default",
        "litellm_api_key": "config",
        "embedding_model": "derived",
        "litellm_tts_model": "default",
        "redis_dsn": "runtime",
    }

    output = _render(settings, capsys)

    assert "WARN | Inference model: not configured [default; default value in effect]" in output
    assert "OK | Embeddings: litellm / azure/gpt-4.1 [derived; derived from fast model]" in output


def test_print_settings_summary_keeps_next_steps_inside_diagnostics_block(
    capsys: CaptureFixture[str],
) -> None:
    settings = _SummarySettings()

    output = _render(settings, capsys)

    diagnostics_start = output.index("Diagnostics\n-----------")
    next_steps_start = output.index("Next steps:\n")
    database_start = output.index("Database path:")

    fast_model_step = (
        "- Set a LiteLLM fast model to unlock chat, prompt runs, and derived defaults."
    )
    api_key_step = "- Add a LiteLLM API key so model calls can authenticate successfully."

    assert diagnostics_start < next_steps_start < database_start
    assert output.index(fast_model_step) < database_start
    assert output.index(api_key_step) < database_start
