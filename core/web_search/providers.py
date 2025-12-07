"""Provider protocol definitions and concrete provider implementations.

Updates:
  v0.1.2 - 2025-12-07 - Add Tavily provider and generalise document parsing.
  v0.1.1 - 2025-12-05 - Move typing-only imports behind TYPE_CHECKING and document __post_init__.
  v0.1.0 - 2025-12-04 - Introduce provider abstraction and Exa HTTP client.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import httpx

from ..exceptions import WebSearchProviderError
from .models import WebSearchDocument, WebSearchResult

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

MAX_RESULTS = 25


@runtime_checkable
class WebSearchProvider(Protocol):
    """Protocol implemented by every external web search provider."""

    slug: str
    display_name: str

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        **kwargs: Any,
    ) -> WebSearchResult:
        """Return provider-specific web search results."""


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _extract_documents(payload: Mapping[str, Any], provider_slug: str) -> list[WebSearchDocument]:
    documents: list[WebSearchDocument] = []
    results = payload.get("results") or []
    if not isinstance(results, list):
        return documents
    for entry in results:
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title") or "").strip()
        url = str(entry.get("url") or "").strip()
        if not title or not url:
            continue
        highlights = entry.get("highlights") or []
        highlight_list = [str(item).strip() for item in highlights if str(item).strip()]
        scores = entry.get("highlightScores") or []
        score_value = None
        if isinstance(scores, list) and scores:
            try:
                score_value = float(scores[0])
            except (ValueError, TypeError):
                score_value = None
        if score_value is None:
            raw_score = entry.get("score")
            if raw_score is not None:
                try:
                    score_value = float(raw_score)
                except (ValueError, TypeError):
                    score_value = None
        summary_text = (
            str(entry.get("summary") or "").strip()
            or str(entry.get("content") or "").strip()
            or None
        )
        published_value = entry.get("publishedDate") or entry.get("published_date")
        documents.append(
            WebSearchDocument(
                title=title,
                url=url,
                summary=summary_text,
                highlights=highlight_list,
                author=str(entry.get("author") or "").strip() or None,
                published_at=_parse_datetime(published_value),
                score=score_value,
                raw=entry,
            )
        )
    return documents


@dataclass(slots=True)
class ExaWebSearchProvider:
    """HTTPX-backed Exa web search provider."""

    api_key: str
    base_url: str = "https://api.exa.ai"
    timeout: float = 15.0
    slug: str = "exa"
    display_name: str = "Exa Web Search"
    client_factory: Callable[[], httpx.AsyncClient] | None = None

    def __post_init__(self) -> None:
        """Validate provided API key and normalise spacing."""
        if not self.api_key or not self.api_key.strip():
            raise ValueError("Exa API key is required")
        self.api_key = self.api_key.strip()

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        search_type: str | None = None,
        livecrawl: str | None = None,
    ) -> WebSearchResult:
        """Perform a search request against Exa's REST API."""
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("query must be a non-empty string")
        clamped_limit = max(1, min(limit, MAX_RESULTS))
        payload: dict[str, Any] = {
            "query": cleaned_query,
            "numResults": clamped_limit,
            "type": (search_type or "auto"),
            "contents": {
                "text": True,
                "highlights": {
                    "numSentences": 2,
                    "highlightsPerUrl": 1,
                },
            },
        }
        if include_domains:
            payload["includeDomains"] = include_domains
        if exclude_domains:
            payload["excludeDomains"] = exclude_domains
        if livecrawl:
            payload["contents"]["livecrawl"] = livecrawl

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        manage_client = self.client_factory is None
        if self.client_factory is None:
            client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
            )
        else:
            client = self.client_factory()
        try:
            response = await client.post("/search", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - exercised via tests
            raise WebSearchProviderError("Exa search request failed") from exc
        finally:
            if manage_client:
                await client.aclose()
        try:
            data = response.json()
        except ValueError as exc:  # pragma: no cover - invalid provider payload
            raise WebSearchProviderError("Exa returned invalid JSON") from exc
        documents = _extract_documents(data, self.slug)
        return WebSearchResult(
            provider=self.slug,
            query=cleaned_query,
            documents=documents,
            raw=data,
        )


@dataclass(slots=True)
class TavilyWebSearchProvider:
    """HTTPX-backed Tavily web search provider."""

    api_key: str
    base_url: str = "https://api.tavily.com"
    timeout: float = 15.0
    slug: str = "tavily"
    display_name: str = "Tavily Web Search"
    client_factory: Callable[[], httpx.AsyncClient] | None = None

    def __post_init__(self) -> None:
        """Validate provided API key and normalise spacing."""
        if not self.api_key or not self.api_key.strip():
            raise ValueError("Tavily API key is required")
        self.api_key = self.api_key.strip()

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        topic: str | None = None,
        search_depth: str | None = None,
        time_range: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        include_answer: str | bool | None = None,
        include_raw_content: str | bool | None = None,
        include_images: bool | None = None,
        include_image_descriptions: bool | None = None,
        include_favicon: bool | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        country: str | None = None,
        auto_parameters: bool | None = None,
    ) -> WebSearchResult:
        """Perform a search request against Tavily's REST API."""
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("query must be a non-empty string")
        clamped_limit = max(1, min(limit, min(MAX_RESULTS, 20)))
        payload: dict[str, Any] = {
            "query": cleaned_query,
            "max_results": clamped_limit,
        }

        def _assign(key: str, value: Any | None) -> None:
            if value in (None, "", []):
                return
            payload[key] = value

        _assign("topic", (topic or "").strip() or None)
        _assign("search_depth", (search_depth or "").strip() or None)
        _assign("time_range", (time_range or "").strip() or None)
        _assign("start_date", (start_date or "").strip() or None)
        _assign("end_date", (end_date or "").strip() or None)
        _assign("include_answer", include_answer)
        _assign("include_raw_content", include_raw_content)
        _assign("include_images", include_images)
        _assign("include_image_descriptions", include_image_descriptions)
        _assign("include_favicon", include_favicon)
        _assign("include_domains", include_domains)
        _assign("exclude_domains", exclude_domains)
        _assign("country", (country or "").strip() or None)
        _assign("auto_parameters", auto_parameters)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        manage_client = self.client_factory is None
        if self.client_factory is None:
            client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
            )
        else:
            client = self.client_factory()
        try:
            response = await client.post("/search", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - exercised via tests
            raise WebSearchProviderError("Tavily search request failed") from exc
        finally:
            if manage_client:
                await client.aclose()

        try:
            data = response.json()
        except ValueError as exc:  # pragma: no cover - invalid provider payload
            raise WebSearchProviderError("Tavily returned invalid JSON") from exc

        documents = _extract_documents(data, self.slug)
        return WebSearchResult(
            provider=self.slug,
            query=cleaned_query,
            documents=documents,
            raw=data,
        )


__all__ = ["ExaWebSearchProvider", "TavilyWebSearchProvider", "WebSearchProvider"]
