"""Tests for GUI diagnostics status helpers in Prompt Manager.

Updates:
  v0.1.0 - 2026-04-25 - Cover OK/WARN/FAIL diagnostics summarisation and banner styling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from gui.runtime_settings_service import (
    DiagnosticsItem,
    build_config_diagnostics_items,
    summarise_diagnostics_severity,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
else:
    Mapping = Any


def test_summarise_diagnostics_severity_prefers_fail() -> None:
    items = [
        DiagnosticsItem(label="Fast model", status="OK", detail="configured"),
        DiagnosticsItem(label="API key", status="FAIL", detail="missing"),
    ]

    assert summarise_diagnostics_severity(items) == "FAIL"


def test_summarise_diagnostics_severity_falls_back_to_warn() -> None:
    items = [
        DiagnosticsItem(label="Fast model", status="OK", detail="configured"),
        DiagnosticsItem(label="TTS", status="WARN", detail="not configured"),
    ]

    assert summarise_diagnostics_severity(items) == "WARN"


def test_build_config_diagnostics_items_returns_operational_statuses() -> None:
    diagnostics = build_config_diagnostics_items(
        litellm_model="azure/gpt-4.1",
        litellm_inference_model=None,
        litellm_api_key="secret-key",
        embedding_backend="litellm",
        embedding_model="azure/UDTEMBED3L",
        litellm_tts_model=None,
        redis_status="Redis caching disabled (no DSN configured).",
        sources={
            "litellm_model": "env",
            "litellm_inference_model": "default",
            "litellm_api_key": "env",
            "embedding_model": "derived",
            "litellm_tts_model": "default",
            "redis_dsn": "runtime",
        },
    )

    assert diagnostics["summary_status"] == "WARN"
    items = diagnostics["items"]
    assert items == [
        {
            "label": "Fast model",
            "status": "OK",
            "detail": "azure/gpt-4.1",
            "source": "env",
            "precedence": "env override in effect",
        },
        {
            "label": "Inference model",
            "status": "WARN",
            "detail": "not configured",
            "source": "default",
            "precedence": "default value in effect",
        },
        {
            "label": "API key",
            "status": "OK",
            "detail": "configured",
            "source": "env",
            "precedence": "env override in effect",
        },
        {
            "label": "Embeddings",
            "status": "OK",
            "detail": "litellm / azure/UDTEMBED3L",
            "source": "derived",
            "precedence": "derived from fast model",
        },
        {
            "label": "TTS",
            "status": "WARN",
            "detail": "not configured",
            "source": "default",
            "precedence": "default value in effect",
        },
        {
            "label": "Redis cache",
            "status": "WARN",
            "detail": "Redis caching disabled (no DSN configured).",
            "source": "runtime",
        },
    ]
    assert diagnostics["next_steps"] == [
        "Optional: set an inference model before routing heavier workflows to inference.",
        "Optional: configure a TTS model only if you plan to use voice output.",
        "Optional: configure Redis only if you want shared caching or faster repeat lookups.",
    ]


def test_build_config_diagnostics_items_marks_missing_fast_model_as_fail() -> None:
    diagnostics = build_config_diagnostics_items(
        litellm_model=None,
        litellm_inference_model=None,
        litellm_api_key=None,
        embedding_backend="litellm",
        embedding_model="text-embedding-3-large",
        litellm_tts_model=None,
        redis_status=None,
    )

    assert diagnostics["summary_status"] == "FAIL"
    items = cast("list[Mapping[str, object]]", diagnostics["items"])
    assert items[0] == {
        "label": "Fast model",
        "status": "FAIL",
        "detail": "missing",
        "source": "default",
        "precedence": "default value in effect",
    }
