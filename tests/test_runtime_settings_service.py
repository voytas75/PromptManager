"""Tests for runtime settings service web search reconfiguration.

Updates:
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

    def set_name_generator(self, *args: Any, **kwargs: Any) -> None:
        self.set_name_generator_calls.append((args, kwargs))


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
        "auto_open_share_links": True,
    }


@pytest.fixture(autouse=True)
def _stub_persistence(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "gui.runtime_settings_service.persist_settings_to_config",
        lambda *_args, **_kwargs: None,
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
