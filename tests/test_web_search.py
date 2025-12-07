"""Tests for the web search provider/service scaffolding.

Updates:
  v0.1.5 - 2025-12-07 - Cover SerpApi provider success/error flows.
  v0.1.4 - 2025-12-07 - Cover Serper provider success/error flows.
  v0.1.3 - 2025-12-07 - Cover random provider fan-out behaviour.
  v0.1.2 - 2025-12-07 - Cover Tavily provider success/error flows.
  v0.1.1 - 2025-12-05 - Reorder imports and wrap long lines for lint compliance.
  v0.1.0 - 2025-12-04 - Cover Exa provider and service scaffolding.
"""

from __future__ import annotations

import httpx
import pytest

from core.exceptions import WebSearchProviderError, WebSearchUnavailable
from core.web_search import (
    ExaWebSearchProvider,
    RandomWebSearchProvider,
    SerpApiWebSearchProvider,
    SerperWebSearchProvider,
    TavilyWebSearchProvider,
    WebSearchDocument,
    WebSearchResult,
    WebSearchService,
)


def _build_mock_client(
    response: httpx.Response,
    base_url: str = "https://api.exa.ai",
) -> httpx.AsyncClient:
    """Return an AsyncClient that always responds with *response*."""

    def handler(_: httpx.Request) -> httpx.Response:
        return response

    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport, base_url=base_url)


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
    mock_client = _build_mock_client(httpx.Response(401, json={"message": "bad"}))
    provider = ExaWebSearchProvider(api_key="test-key", client_factory=lambda: mock_client)

    with pytest.raises(WebSearchProviderError):
        await provider.search("latest news")
    await mock_client.aclose()


@pytest.mark.asyncio()
async def test_tavily_provider_parses_results() -> None:
    """Ensure the Tavily provider maps API payloads into canonical documents."""
    mock_client = _build_mock_client(
        httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Example Insight",
                        "url": "https://example.com/insight",
                        "content": "Key snippet",
                        "score": 0.77,
                        "raw_content": "Key snippet\nMore text",
                        "favicon": "https://example.com/favicon.png",
                    }
                ],
                "answer": "summary",
            },
        ),
        base_url="https://api.tavily.com",
    )
    provider = TavilyWebSearchProvider(api_key="tvly-test", client_factory=lambda: mock_client)
    result = await provider.search("latest news", limit=5, topic="general")
    await mock_client.aclose()

    assert result.provider == "tavily"
    assert len(result.documents) == 1
    doc = result.documents[0]
    assert doc.summary == "Key snippet"
    assert doc.score == pytest.approx(0.77)
    assert doc.highlights == []


@pytest.mark.asyncio()
async def test_tavily_provider_raises_on_error() -> None:
    """Raise a provider error when Tavily rejects the request."""
    mock_client = _build_mock_client(
        httpx.Response(500, json={"message": "error"}),
        base_url="https://api.tavily.com",
    )
    provider = TavilyWebSearchProvider(api_key="tvly-test", client_factory=lambda: mock_client)

    with pytest.raises(WebSearchProviderError):
        await provider.search("latest news")
    await mock_client.aclose()


@pytest.mark.asyncio()
async def test_serper_provider_parses_results() -> None:
    """Ensure the Serper provider maps API payloads into canonical documents."""
    mock_client = _build_mock_client(
        httpx.Response(
            200,
            json={
                "organic": [
                    {
                        "title": "Example Result",
                        "link": "https://example.com/result",
                        "snippet": "Snippet text",
                        "date": "2025-12-07",
                        "source": "Example Source",
                        "snippetHighlighted": ["Snippet text"],
                    }
                ],
                "relatedSearches": [],
            },
        ),
        base_url="https://google.serper.dev",
    )
    provider = SerperWebSearchProvider(api_key="serper-test", client_factory=lambda: mock_client)
    result = await provider.search("latest news", limit=5)
    await mock_client.aclose()

    assert result.provider == "serper"
    assert len(result.documents) == 1
    doc = result.documents[0]
    assert doc.summary == "Snippet text"
    assert doc.highlights == ["Snippet text"]
    assert doc.author == "Example Source"


@pytest.mark.asyncio()
async def test_serper_provider_raises_on_error() -> None:
    """Raise a provider error when Serper rejects the request."""
    mock_client = _build_mock_client(
        httpx.Response(429, json={"message": "error"}),
        base_url="https://google.serper.dev",
    )
    provider = SerperWebSearchProvider(api_key="serper-test", client_factory=lambda: mock_client)

    with pytest.raises(WebSearchProviderError):
        await provider.search("latest news")
    await mock_client.aclose()


@pytest.mark.asyncio()
async def test_serpapi_provider_parses_results() -> None:
    """Ensure the SerpApi provider maps API payloads into canonical documents."""
    mock_client = _build_mock_client(
        httpx.Response(
            200,
            json={
                "organic_results": [
                    {
                        "title": "SerpApi Result",
                        "link": "https://example.com/serpapi",
                        "snippet": "SerpApi snippet",
                        "snippet_highlighted_words": ["SerpApi snippet"],
                        "position": 1,
                        "date": "2025-12-07",
                        "source": "Example Source",
                    }
                ]
            },
        ),
        base_url="https://serpapi.com",
    )
    provider = SerpApiWebSearchProvider(api_key="serpapi-test", client_factory=lambda: mock_client)
    result = await provider.search("latest news", limit=3, gl="us")
    await mock_client.aclose()

    assert result.provider == "serpapi"
    assert len(result.documents) == 1
    doc = result.documents[0]
    assert doc.summary == "SerpApi snippet"
    assert doc.highlights == ["SerpApi snippet"]
    assert doc.score == pytest.approx(1.0)
    assert doc.author == "Example Source"


@pytest.mark.asyncio()
async def test_serpapi_provider_raises_on_error() -> None:
    """Raise a provider error when SerpApi rejects the request."""
    mock_client = _build_mock_client(
        httpx.Response(503, json={"error": "upstream"}),
        base_url="https://serpapi.com",
    )
    provider = SerpApiWebSearchProvider(api_key="serpapi-test", client_factory=lambda: mock_client)

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


@pytest.mark.asyncio()
async def test_random_provider_delegates_to_selected_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Random provider should funnel requests to the randomly selected provider."""

    calls: list[tuple[str, str, int]] = []

    class DummyProvider:
        def __init__(self, slug: str) -> None:
            self.slug = slug
            self.display_name = slug

        async def search(self, query: str, *, limit: int = 5, **_: object) -> WebSearchResult:  # type: ignore[override]
            calls.append((self.slug, query, limit))
            return WebSearchResult(provider=self.slug, query=query, documents=[])

    providers = [DummyProvider("exa"), DummyProvider("tavily")]
    random_provider = RandomWebSearchProvider(providers)
    monkeypatch.setattr("core.web_search.providers.random.choice", lambda seq: seq[1])

    result = await random_provider.search("topic", limit=3)

    assert result.provider == "tavily"
    assert calls == [("tavily", "topic", 3)]


def test_random_provider_requires_candidates() -> None:
    """Random provider cannot be constructed without downstream providers."""

    with pytest.raises(WebSearchUnavailable):
        RandomWebSearchProvider([])
