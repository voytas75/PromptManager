"""Tests for runtime settings service web search reconfiguration.

Updates:
  v0.1.1 - 2026-04-24 - Cover compact GUI diagnostics summary fields.
  v0.1.0 - 2025-12-07 - Ensure runtime settings rewire web search providers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
from pytest import MonkeyPatch

from config import DEFAULT_CHAT_USER_BUBBLE_COLOR, DEFAULT_THEME_MODE
from config.settings import DEFAULT_EMBEDDING_BACKEND, DEFAULT_EMBEDDING_MODEL
from core.web_search import (
    SerpApiWebSearchProvider,
    TavilyWebSearchProvider,
    WebSearchService,
)
from gui.runtime_settings_service import RuntimeSettingsService

if TYPE_CHECKING:
    from core import PromptManager
else:  # pragma: no cover - runtime fallback for typing-only import
    PromptManager = Any


class _DummyPromptManager:
    def __init__(self) -> None:
        self.executor = object()
        self.web_search_service = WebSearchService()
        self.web_search = self.web_search_service
        self.set_name_generator_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.redis_unavailable_reason: str | None = None
        self._redis_client = None

    def set_name_generator(self, *args: Any, **kwargs: Any) -> None:
        self.set_name_generator_calls.append((args, kwargs))


class _DummyChatColors:
    def __init__(self) -> None:
        self.user = DEFAULT_CHAT_USER_BUBBLE_COLOR
        self.assistant = "#f4f4f5"


class _DummySettings:
    def __init__(self) -> None:
        self.litellm_model = "azure/gpt-4.1"
        self.litellm_inference_model = "azure/gpt-5.4"
        self.litellm_api_key = "secret-key"
        self.litellm_api_base = "https://example.azure.com"
        self.litellm_api_version = "2025-04-01-preview"
        self.litellm_reasoning_effort = None
        self.litellm_tts_model = "azure/tts-1"
        self.litellm_tts_stream = True
        self.litellm_stream = False
        self.litellm_workflow_models = None
        self.litellm_drop_params = None
        self.embedding_backend = "litellm"
        self.embedding_model = "azure/UDTEMBED3L"
        self.quick_actions = None
        self.chat_user_bubble_color = DEFAULT_CHAT_USER_BUBBLE_COLOR
        self.theme_mode = DEFAULT_THEME_MODE
        self.prompt_output_font_family = None
        self.prompt_output_font_size = None
        self.prompt_output_font_color = None
        self.chat_font_family = None
        self.chat_font_size = None
        self.chat_font_color = None
        self.chat_colors = _DummyChatColors()
        self.prompt_templates = None
        self.web_search_provider = None
        self.exa_api_key = None
        self.tavily_api_key = None
        self.serper_api_key = None
        self.serpapi_api_key = None
        self.google_api_key = None
        self.google_cse_id = None
        self.auto_open_share_links = True
        self.redis_dsn = None


def _base_runtime_settings() -> dict[str, object | None]:
    return {
        "litellm_model": None,
        "litellm_inference_model": None,
        "litellm_api_key": None,
        "litellm_api_base": None,
        "litellm_api_version": None,
        "litellm_reasoning_effort": None,
        "litellm_tts_model": None,
        "litellm_tts_stream": True,
        "litellm_workflow_models": None,
        "litellm_drop_params": None,
        "litellm_stream": False,
        "embedding_backend": DEFAULT_EMBEDDING_BACKEND,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "quick_actions": None,
        "chat_user_bubble_color": DEFAULT_CHAT_USER_BUBBLE_COLOR,
        "chat_colors": None,
        "theme_mode": DEFAULT_THEME_MODE,
        "prompt_templates": None,
        "web_search_provider": None,
        "exa_api_key": None,
        "tavily_api_key": None,
        "serper_api_key": None,
        "serpapi_api_key": None,
        "google_api_key": None,
        "google_cse_id": None,
        "auto_open_share_links": True,
    }


@pytest.fixture(autouse=True)
def _persist_settings_stub(monkeypatch: MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    def _noop_persist(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        "gui.runtime_settings_service.persist_settings_to_config",
        _noop_persist,
    )


def test_apply_updates_reconfigures_web_search_provider() -> None:
    """Switching to Tavily updates the manager's configured provider."""
    runtime = _base_runtime_settings()
    runtime.update(
        {
            "web_search_provider": "serpapi",
            "serpapi_api_key": "serpapi-secret",
        }
    )
    manager = _DummyPromptManager()
    initial_service = manager.web_search_service
    initial_service.configure(SerpApiWebSearchProvider(api_key="serpapi-secret"))
    service = RuntimeSettingsService(cast("PromptManager", manager), None)

    service.apply_updates(
        runtime,
        {
            "web_search_provider": "tavily",
            "tavily_api_key": "tavily-secret",
        },
    )

    provider = manager.web_search_service.provider
    assert provider is not None
    assert provider.slug == "tavily"
    assert runtime["web_search_provider"] == "tavily"
    assert runtime["tavily_api_key"] == "tavily-secret"
    assert manager.web_search_service is initial_service
    assert manager.web_search is initial_service


def test_apply_updates_clears_web_search_provider_when_disabled() -> None:
    """Clearing the provider disables the manager's web search service."""
    runtime = _base_runtime_settings()
    runtime.update(
        {
            "web_search_provider": "tavily",
            "tavily_api_key": "tavily-secret",
        }
    )
    manager = _DummyPromptManager()
    manager.web_search_service.configure(TavilyWebSearchProvider(api_key="tavily-secret"))
    service = RuntimeSettingsService(cast("PromptManager", manager), None)

    service.apply_updates(runtime, {"web_search_provider": None})

    assert runtime["web_search_provider"] is None
    assert manager.web_search_service.provider is None


def test_apply_updates_returns_user_facing_model_routing_summary() -> None:
    """Applying model/routing changes should return a concise toast summary."""
    runtime = _base_runtime_settings()
    manager = _DummyPromptManager()
    service = RuntimeSettingsService(cast("PromptManager", manager), None)

    result = service.apply_updates(
        runtime,
        {
            "litellm_model": "azure/gpt-4.1-mini",
            "litellm_inference_model": "azure/gpt-5.4",
            "litellm_workflow_models": {
                "prompt_generation": "inference",
                "chat_title": "fast",
            },
        },
    )

    assert result.summary_message is not None
    assert "Fast model: azure/gpt-4.1-mini" in result.summary_message
    assert "Inference model: azure/gpt-5.4" in result.summary_message
    assert "Routing: inference for: prompt_generation" in result.summary_message


def test_build_initial_runtime_settings_includes_compact_diagnostics_summary() -> None:
    """Initial runtime snapshot should expose a compact user-facing diagnostics summary."""
    manager = _DummyPromptManager()
    settings = _DummySettings()
    service = RuntimeSettingsService(cast("PromptManager", manager), cast("Any", settings))

    runtime = service.build_initial_runtime_settings()

    diagnostics = runtime["config_diagnostics"]
    assert diagnostics == {
        "models_configured": True,
        "inference_model_configured": True,
        "api_key_configured": True,
        "embedding_backend": "litellm",
        "embedding_model": "azure/UDTEMBED3L",
        "tts_configured": True,
        "redis_status": "Redis caching disabled (no DSN configured).",
    }
