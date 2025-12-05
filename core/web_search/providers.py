"""Provider protocol definitions and concrete Exa implementation.

Updates:
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
        documents.append(
            WebSearchDocument(
                title=title,
                url=url,
                summary=str(entry.get("summary") or "").strip() or None,
                highlights=highlight_list,
                author=str(entry.get("author") or "").strip() or None,
                published_at=_parse_datetime(entry.get("publishedDate")),
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


__all__ = ["ExaWebSearchProvider", "WebSearchProvider"]
