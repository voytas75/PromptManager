"""Tests for the web search provider/service scaffolding.

Updates:
  v0.1.0 - 2025-12-04 - Cover Exa provider and service scaffolding.
"""

from __future__ import annotations

import pytest
import httpx

from core.exceptions import WebSearchProviderError, WebSearchUnavailable
from core.web_search import ExaWebSearchProvider, WebSearchDocument, WebSearchResult, WebSearchService


def _build_mock_client(response: httpx.Response) -> httpx.AsyncClient:
    """Return an AsyncClient that always responds with *response*."""

    def handler(_: httpx.Request) -> httpx.Response:
        return response

    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport, base_url="https://api.exa.ai")


@pytest.mark.asyncio()
async def test_exa_provider_parses_results() -> None:
    """Ensure the Exa provider maps API payloads into canonical documents."""
    mock_client = _build_mock_client(
        httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Example Article",
                        "url": "https://example.com/article",
                        "summary": "Key findings",
                        "highlights": ["Highlight line"],
                        "highlightScores": [0.9],
                        "publishedDate": "2025-12-04T00:00:00Z",
                        "author": "Example Author",
                    }
                ]
            },
        )
    )
    provider = ExaWebSearchProvider(api_key="test-key", client_factory=lambda: mock_client)
    result = await provider.search("latest news", limit=5)
    await mock_client.aclose()

    assert result.provider == "exa"
    assert len(result.documents) == 1
    doc = result.documents[0]
    assert isinstance(doc, WebSearchDocument)
    assert doc.title == "Example Article"
    assert doc.highlights == ["Highlight line"]
    assert doc.score == pytest.approx(0.9)
    assert doc.published_at is not None


@pytest.mark.asyncio()
async def test_exa_provider_raises_on_error() -> None:
    """Raise a provider error when Exa rejects the request."""
    mock_client = _build_mock_client(
        httpx.Response(401, json={"message": "bad"})
    )
    provider = ExaWebSearchProvider(api_key="test-key", client_factory=lambda: mock_client)

    with pytest.raises(WebSearchProviderError):
        await provider.search("latest news")
    await mock_client.aclose()


@pytest.mark.asyncio()
async def test_web_search_service_requires_provider() -> None:
    """WebSearchService should guard against missing providers."""
    service = WebSearchService()
    with pytest.raises(WebSearchUnavailable):
        await service.search("anything")


@pytest.mark.asyncio()
async def test_web_search_service_delegates_to_provider() -> None:
    """WebSearchService forwards calls to the configured provider."""

    class DummyProvider:
        slug = "dummy"
        display_name = "Dummy"

        async def search(self, query: str, *, limit: int = 5, **_: object) -> WebSearchResult:  # type: ignore[override]
            return WebSearchResult(provider=self.slug, query=query, documents=[])

    service = WebSearchService(provider=DummyProvider())
    result = await service.search("q", limit=2)

    assert result.provider == "dummy"
    assert result.query == "q"
