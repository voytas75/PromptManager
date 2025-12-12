"""Workspace web search tooltip tests.

Updates:
  v0.1.0 - 2025-12-12 - Cover MainWindow web search tooltip messaging for providers and Random.
"""

from __future__ import annotations

from types import SimpleNamespace

from gui.main_window import MainWindow


def _build_tooltip(runtime: dict[str, object | None]) -> str:
    runtime_settings = {
        "web_search_provider": runtime.get("web_search_provider"),
        "exa_api_key": runtime.get("exa_api_key"),
        "tavily_api_key": runtime.get("tavily_api_key"),
        "serper_api_key": runtime.get("serper_api_key"),
        "serpapi_api_key": runtime.get("serpapi_api_key"),
        "google_api_key": runtime.get("google_api_key"),
        "google_cse_id": runtime.get("google_cse_id"),
    }
    window = SimpleNamespace(_runtime_settings=runtime_settings, _settings=None)
    return MainWindow._build_web_search_tooltip_text(window)  # type: ignore[arg-type]


def test_workspace_tooltip_for_specific_provider() -> None:
    """SerpApi provider name should be reflected in the tooltip."""

    tooltip = _build_tooltip(
        {
            "web_search_provider": "serpapi",
            "serpapi_api_key": "serpapi-key",
            "exa_api_key": None,
            "tavily_api_key": None,
            "serper_api_key": None,
            "google_api_key": None,
            "google_cse_id": None,
        }
    )
    assert tooltip == "Include live web search context via SerpApi before executing prompts."


def test_workspace_tooltip_for_random_provider_lists_available() -> None:
    """Random provider should enumerate available downstream providers."""

    tooltip = _build_tooltip(
        {
            "web_search_provider": "random",
            "exa_api_key": "exa-key",
            "tavily_api_key": "tvly-key",
            "serper_api_key": None,
            "serpapi_api_key": None,
            "google_api_key": None,
            "google_cse_id": None,
        }
    )
    assert tooltip == (
        "Include live web search context via the Random provider, rotating between "
        "Exa and Tavily before executing prompts."
    )


def test_workspace_tooltip_defaults_when_provider_missing() -> None:
    """Unset provider should prompt the user to configure web search."""

    tooltip = _build_tooltip(
        {
            "web_search_provider": None,
            "exa_api_key": None,
            "tavily_api_key": None,
            "serper_api_key": None,
            "serpapi_api_key": None,
            "google_api_key": None,
            "google_cse_id": None,
        }
    )
    assert tooltip == (
        "Include live web search context. Configure Exa, Tavily, Serper, SerpApi, or Google "
        "under Settings to enable provider-specific routing."
    )
