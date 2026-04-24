"""Provider protocol definitions and concrete provider implementations.

Updates:
  v0.1.8 - 2025-12-12 - Retry transient HTTP failures with exponential backoff.
  v0.1.7 - 2025-12-07 - Add Google Programmable Search provider and HTML snippet parsing.
  v0.1.6 - 2025-12-07 - Add resolve_web_search_provider helper for runtime wiring.
  v0.1.5 - 2025-12-07 - Add SerpApi provider and organic result normalisation helper.
  v0.1.4 - 2025-12-07 - Add Serper provider and response normalisation helpers.
  v0.1.3 - 2025-12-07 - Add RandomWebSearchProvider to rotate between configured services.
  v0.1.2 - 2025-12-07 - Add Tavily provider and generalise document parsing.
  v0.1.1 - 2025-12-05 - Move typing-only imports behind TYPE_CHECKING and document __post_init__.
  v0.1.0 - 2025-12-04 - Introduce provider abstraction and Exa HTTP client.
"""

from __future__ import annotations

import html
import random
import re
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import httpx

from ..exceptions import WebSearchProviderError, WebSearchUnavailable
from ..retry import async_retry, is_retryable_httpx_error
from .models import WebSearchDocument, WebSearchResult

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

MAX_RESULTS = 25
GOOGLE_MAX_RESULTS = 10
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


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
        ...


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


def _mapping_str_any(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return cast("dict[str, Any]", value)


def _list_objects(value: object) -> list[object]:
    if isinstance(value, list):
        return cast("list[object]", value)
    return []


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float | str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _string_or_none(value: object) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _mapping_value(mapping: Mapping[str, Any], key: str) -> object | None:
    return mapping.get(key)


def _first_present(mapping: Mapping[str, Any], *keys: str) -> object | None:
    for key in keys:
        value = _mapping_value(mapping, key)
        if value not in (None, ""):
            return value
    return None


def _extract_documents(payload: Mapping[str, Any], provider_slug: str) -> list[WebSearchDocument]:
    documents: list[WebSearchDocument] = []
    results = _list_objects(_mapping_value(payload, "results"))
    for raw_entry in results:
        entry = _mapping_str_any(raw_entry)
        if entry is None:
            continue
        title = _string_or_none(_mapping_value(entry, "title"))
        url = _string_or_none(_mapping_value(entry, "url"))
        if not title or not url:
            continue
        highlights = _list_objects(_mapping_value(entry, "highlights"))
        highlight_list = [str(item).strip() for item in highlights if str(item).strip()]
        scores = _list_objects(_mapping_value(entry, "highlightScores"))
        score_value = None
        if scores:
            score_value = _float_or_none(scores[0])
        if score_value is None:
            raw_score = _mapping_value(entry, "score")
            if raw_score is not None:
                score_value = _float_or_none(raw_score)
        summary_text = (
            _string_or_none(_mapping_value(entry, "summary"))
            or _string_or_none(_mapping_value(entry, "content"))
            or None
        )
        published_value = _string_or_none(_first_present(entry, "publishedDate", "published_date"))
        raw_candidate = _mapping_str_any(_mapping_value(entry, "raw"))
        raw_entry_payload = raw_candidate or dict(entry)
        documents.append(
            WebSearchDocument(
                title=title,
                url=url,
                summary=summary_text,
                highlights=highlight_list,
                author=_string_or_none(_mapping_value(entry, "author")),
                published_at=_parse_datetime(published_value),
                score=score_value,
                raw=raw_entry_payload,
            )
        )
    return documents


def _normalise_serper_results(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    r"""Normalise Serper "organic" entries into the canonical result schema."""
    organic_entries = _list_objects(_mapping_value(payload, "organic"))
    normalised: list[dict[str, Any]] = []
    for raw_entry in organic_entries:
        entry = _mapping_str_any(raw_entry)
        if entry is None:
            continue
        title = _string_or_none(_mapping_value(entry, "title"))
        url = _string_or_none(_mapping_value(entry, "link"))
        if not title or not url:
            continue
        snippet = _string_or_none(_mapping_value(entry, "snippet"))
        highlight_source = _list_objects(
            _first_present(entry, "snippetHighlighted", "snippet_highlighted")
        )
        highlights = [str(item).strip() for item in highlight_source if str(item).strip()]
        normalised.append(
            {
                "title": title,
                "url": url,
                "summary": snippet,
                "highlights": highlights,
                "score": _mapping_value(entry, "score"),
                "publishedDate": _mapping_value(entry, "date"),
                "author": _mapping_value(entry, "source"),
                "raw": dict(entry),
            }
        )
    return normalised


def _normalise_serpapi_results(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalise SerpApi organic results into the canonical result schema."""
    organic_entries = _list_objects(
        _first_present(payload, "organic_results", "organic", "search_results")
    )
    normalised: list[dict[str, Any]] = []
    for raw_entry in organic_entries:
        entry = _mapping_str_any(raw_entry)
        if entry is None:
            continue
        title = _string_or_none(_mapping_value(entry, "title"))
        url = _string_or_none(_first_present(entry, "link", "url"))
        if not title or not url:
            continue
        snippet = _string_or_none(_mapping_value(entry, "snippet"))
        highlights_source = _list_objects(
            _first_present(
                entry,
                "snippet_highlighted_words",
                "snippetHighlightedWords",
                "snippetHighlighted",
            )
        )
        highlights = [str(item).strip() for item in highlights_source if str(item).strip()]
        score_value = None
        position_value = _first_present(entry, "position", "rank")
        if position_value is not None:
            score_value = _float_or_none(position_value)
        normalised.append(
            {
                "title": title,
                "url": url,
                "summary": snippet,
                "highlights": highlights,
                "score": score_value,
                "publishedDate": _mapping_value(entry, "date"),
                "author": _first_present(entry, "source", "displayed_link"),
                "raw": dict(entry),
            }
        )
    return normalised


def _normalise_google_results(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalise Google Programmable Search entries into the canonical schema."""
    items = _list_objects(_mapping_value(payload, "items"))
    normalised: list[dict[str, Any]] = []

    def _clean_html(text: str | None) -> str | None:
        if not text:
            return None
        stripped = _HTML_TAG_PATTERN.sub("", html.unescape(text))
        collapsed = " ".join(stripped.split())
        return collapsed or None

    for index, raw_entry in enumerate(items, start=1):
        entry_dict = _mapping_str_any(raw_entry)
        if entry_dict is None:
            continue
        title = _string_or_none(_mapping_value(entry_dict, "title"))
        url = _string_or_none(_mapping_value(entry_dict, "link"))
        if not title or not url:
            continue
        snippet = _string_or_none(_mapping_value(entry_dict, "snippet"))
        highlight_text = _clean_html(
            _string_or_none(_first_present(entry_dict, "htmlSnippet", "html_snippet"))
        )
        highlights = [highlight_text] if highlight_text else []
        pagemap = _mapping_str_any(_mapping_value(entry_dict, "pagemap")) or {}
        metatags = _list_objects(_mapping_value(pagemap, "metatags"))
        published_value: str | None = None
        for raw_tag in metatags:
            tag = _mapping_str_any(raw_tag)
            if tag is None:
                continue
            for key in (
                "article:published_time",
                "article:modified_time",
                "og:updated_time",
                "og:published_time",
                "pubdate",
            ):
                candidate = _string_or_none(_mapping_value(tag, key))
                if candidate:
                    published_value = candidate
                    break
            if published_value:
                break
        normalised.append(
            {
                "title": title,
                "url": url,
                "summary": snippet,
                "highlights": highlights,
                "author": _mapping_value(entry_dict, "displayLink"),
                "publishedDate": published_value,
                "score": float(index),
                "raw": dict(entry_dict),
            }
        )
    return normalised


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
        **_: Any,
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

            async def _send_request() -> httpx.Response:
                response = await client.post("/search", json=payload)
                response.raise_for_status()
                return response

            response = await async_retry(_send_request, should_retry=is_retryable_httpx_error)
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
class GoogleWebSearchProvider:
    """HTTPX-backed Google Programmable Search provider."""

    api_key: str
    cse_id: str
    base_url: str = "https://www.googleapis.com"
    endpoint: str = "/customsearch/v1"
    timeout: float = 15.0
    slug: str = "google"
    display_name: str = "Google Programmable Search"
    client_factory: Callable[[], httpx.AsyncClient] | None = None

    def __post_init__(self) -> None:
        """Ensure credentials are present and trimmed."""
        if not self.api_key or not self.api_key.strip():
            raise ValueError("Google API key is required")
        if not self.cse_id or not self.cse_id.strip():
            raise ValueError("Google Custom Search Engine ID is required")
        self.api_key = self.api_key.strip()
        self.cse_id = self.cse_id.strip()

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        safe: str | None = None,
        start_index: int | None = None,
        lr: str | None = None,
        cr: str | None = None,
        gl: str | None = None,
        cx: str | None = None,
        **extra: Any,
    ) -> WebSearchResult:
        """Perform a search request against Google's Custom Search JSON API."""
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("query must be a non-empty string")
        clamped_limit = max(1, min(limit, min(MAX_RESULTS, GOOGLE_MAX_RESULTS)))
        params: dict[str, Any] = {
            "key": self.api_key,
            "cx": (cx or self.cse_id),
            "q": cleaned_query,
            "num": clamped_limit,
        }

        def _assign_text(key: str, value: str | None) -> None:
            if value is None:
                return
            stripped = value.strip()
            if stripped:
                params[key] = stripped

        _assign_text("safe", safe)
        _assign_text("lr", lr)
        _assign_text("cr", cr)
        _assign_text("gl", gl)
        if start_index is not None:
            try:
                params["start"] = max(1, int(start_index))
            except (TypeError, ValueError) as exc:
                raise ValueError("start_index must be an integer when provided") from exc

        if extra:
            for key, value in extra.items():
                if value in (None, "", []):
                    continue
                params[str(key)] = value

        manage_client = self.client_factory is None
        if self.client_factory is None:
            client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        else:
            client = self.client_factory()
        try:

            async def _send_request() -> httpx.Response:
                response = await client.get(self.endpoint, params=params)
                response.raise_for_status()
                return response

            response = await async_retry(_send_request, should_retry=is_retryable_httpx_error)
        except httpx.HTTPError as exc:  # pragma: no cover - exercised via tests
            raise WebSearchProviderError("Google search request failed") from exc
        finally:
            if manage_client:
                await client.aclose()

        try:
            data = response.json()
        except ValueError as exc:  # pragma: no cover - invalid provider payload
            raise WebSearchProviderError("Google returned invalid JSON") from exc
        documents = _extract_documents(
            {"results": _normalise_google_results(data)},
            self.slug,
        )
        return WebSearchResult(
            provider=self.slug,
            query=cleaned_query,
            documents=documents,
            raw=data,
        )


@dataclass(slots=True)
class SerperWebSearchProvider:
    """HTTPX-backed Serper web search provider."""

    api_key: str
    base_url: str = "https://google.serper.dev"
    timeout: float = 15.0
    slug: str = "serper"
    display_name: str = "Serper Web Search"
    client_factory: Callable[[], httpx.AsyncClient] | None = None

    def __post_init__(self) -> None:
        """Validate provided API key and normalise spacing."""
        if not self.api_key or not self.api_key.strip():
            raise ValueError("Serper API key is required")
        self.api_key = self.api_key.strip()

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        gl: str | None = None,
        hl: str | None = None,
        location: str | None = None,
        tbs: str | None = None,
        search_type: str | None = None,
        **_: Any,
    ) -> WebSearchResult:
        """Perform a search request against Serper's REST API."""
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("query must be a non-empty string")
        clamped_limit = max(1, min(limit, MAX_RESULTS))
        payload: dict[str, Any] = {
            "q": cleaned_query,
            "num": clamped_limit,
        }

        def _assign_text(key: str, value: str | None) -> None:
            if value is None:
                return
            stripped = value.strip()
            if stripped:
                payload[key] = stripped

        _assign_text("gl", gl)
        _assign_text("hl", hl)
        _assign_text("location", location)
        _assign_text("tbs", tbs)

        headers = {
            "X-API-KEY": self.api_key,
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
        endpoint = f"/{(search_type or 'search').strip().lstrip('/') or 'search'}"
        try:

            async def _send_request() -> httpx.Response:
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
                return response

            response = await async_retry(_send_request, should_retry=is_retryable_httpx_error)
        except httpx.HTTPError as exc:  # pragma: no cover - exercised via tests
            raise WebSearchProviderError("Serper search request failed") from exc
        finally:
            if manage_client:
                await client.aclose()

        try:
            data = response.json()
        except ValueError as exc:  # pragma: no cover - invalid provider payload
            raise WebSearchProviderError("Serper returned invalid JSON") from exc
        documents = _extract_documents(
            {"results": _normalise_serper_results(data)},
            self.slug,
        )
        return WebSearchResult(
            provider=self.slug,
            query=cleaned_query,
            documents=documents,
            raw=data,
        )


@dataclass(slots=True)
class SerpApiWebSearchProvider:
    """HTTPX-backed SerpApi web search provider."""

    api_key: str
    base_url: str = "https://serpapi.com"
    timeout: float = 15.0
    slug: str = "serpapi"
    display_name: str = "SerpApi Web Search"
    client_factory: Callable[[], httpx.AsyncClient] | None = None

    def __post_init__(self) -> None:
        """Validate provided API key and normalise spacing."""
        if not self.api_key or not self.api_key.strip():
            raise ValueError("SerpApi API key is required")
        self.api_key = self.api_key.strip()

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        engine: str | None = None,
        google_domain: str | None = None,
        gl: str | None = None,
        hl: str | None = None,
        location: str | None = None,
        uule: str | None = None,
        tbs: str | None = None,
        tbm: str | None = None,
        start: int | None = None,
        safe: str | None = None,
        device: str | None = None,
        no_cache: bool | None = None,
        additional_params: Mapping[str, Any] | None = None,
        **extra: Any,
    ) -> WebSearchResult:
        """Perform a search request against SerpApi's REST API."""
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("query must be a non-empty string")
        clamped_limit = max(1, min(limit, MAX_RESULTS))
        params: dict[str, Any] = {
            "q": cleaned_query,
            "num": clamped_limit,
            "api_key": self.api_key,
            "engine": (engine or "google").strip() or "google",
        }

        def _assign_text(key: str, value: str | None) -> None:
            if value is None:
                return
            stripped = value.strip()
            if stripped:
                params[key] = stripped

        _assign_text("google_domain", google_domain)
        _assign_text("gl", gl)
        _assign_text("hl", hl)
        _assign_text("location", location)
        _assign_text("uule", uule)
        _assign_text("tbs", tbs)
        _assign_text("tbm", tbm)
        _assign_text("safe", safe)
        _assign_text("device", device)

        if start is not None:
            try:
                params["start"] = max(0, int(start))
            except (TypeError, ValueError) as exc:
                raise ValueError("start must be an integer when provided") from exc
        if no_cache is not None:
            params["no_cache"] = bool(no_cache)

        def _assign_mapping(values: Mapping[str, Any] | None) -> None:
            if not values:
                return
            for key, value in values.items():
                if value in (None, "", []):
                    continue
                params[str(key)] = value

        _assign_mapping(additional_params)
        if extra:
            _assign_mapping(extra)

        manage_client = self.client_factory is None
        if self.client_factory is None:
            client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        else:
            client = self.client_factory()
        try:

            async def _send_request() -> httpx.Response:
                response = await client.get("/search", params=params)
                response.raise_for_status()
                return response

            response = await async_retry(_send_request, should_retry=is_retryable_httpx_error)
        except httpx.HTTPError as exc:  # pragma: no cover - exercised via tests
            raise WebSearchProviderError("SerpApi search request failed") from exc
        finally:
            if manage_client:
                await client.aclose()

        try:
            data = response.json()
        except ValueError as exc:  # pragma: no cover - invalid provider payload
            raise WebSearchProviderError("SerpApi returned invalid JSON") from exc
        documents = _extract_documents(
            {"results": _normalise_serpapi_results(data)},
            self.slug,
        )
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
        **_: Any,
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

            async def _send_request() -> httpx.Response:
                response = await client.post("/search", json=payload)
                response.raise_for_status()
                return response

            response = await async_retry(_send_request, should_retry=is_retryable_httpx_error)
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


@dataclass(slots=True)
class RandomWebSearchProvider:
    """Provider that randomly fans out requests across configured providers."""

    providers: Sequence[WebSearchProvider]
    slug: str = "random"
    display_name: str = "Random Web Search"

    def __post_init__(self) -> None:
        """Ensure at least one downstream provider is configured."""
        cleaned = tuple(self.providers)
        if not cleaned:
            raise WebSearchUnavailable(
                "Random web search requires at least one configured provider"
            )
        self.providers = cleaned

    async def search(self, query: str, *, limit: int = 5, **kwargs: Any) -> WebSearchResult:
        """Delegate searches to a randomly selected downstream provider."""
        selected = random.choice(self.providers)
        return await selected.search(query, limit=limit, **kwargs)


def resolve_web_search_provider(
    provider_slug: str | None,
    *,
    exa_api_key: str | None = None,
    tavily_api_key: str | None = None,
    serper_api_key: str | None = None,
    serpapi_api_key: str | None = None,
    google_api_key: str | None = None,
    google_cse_id: str | None = None,
) -> WebSearchProvider | None:
    """Return a provider instance that matches *provider_slug* and available API keys."""

    def _clean(value: str | None) -> str | None:
        if not value:
            return None
        stripped = value.strip()
        return stripped or None

    slug = (provider_slug or "").strip().lower() or None
    exa_key = _clean(exa_api_key)
    tavily_key = _clean(tavily_api_key)
    serper_key = _clean(serper_api_key)
    serpapi_key = _clean(serpapi_api_key)
    google_key = _clean(google_api_key)
    google_cse = _clean(google_cse_id)

    exa_provider = ExaWebSearchProvider(api_key=exa_key) if exa_key else None
    tavily_provider = TavilyWebSearchProvider(api_key=tavily_key) if tavily_key else None
    serper_provider = SerperWebSearchProvider(api_key=serper_key) if serper_key else None
    serpapi_provider = SerpApiWebSearchProvider(api_key=serpapi_key) if serpapi_key else None
    google_provider = (
        GoogleWebSearchProvider(api_key=google_key, cse_id=google_cse)
        if google_key and google_cse
        else None
    )

    if slug == "exa":
        return exa_provider
    if slug == "tavily":
        return tavily_provider
    if slug == "serper":
        return serper_provider
    if slug == "serpapi":
        return serpapi_provider
    if slug == "google":
        return google_provider
    if slug == "random":
        available = [
            candidate
            for candidate in (
                exa_provider,
                tavily_provider,
                serper_provider,
                serpapi_provider,
                google_provider,
            )
            if candidate
        ]
        if len(available) == 1:
            return available[0]
        if len(available) > 1:
            return RandomWebSearchProvider(available)
    return None


__all__ = [
    "ExaWebSearchProvider",
    "GoogleWebSearchProvider",
    "RandomWebSearchProvider",
    "SerpApiWebSearchProvider",
    "SerperWebSearchProvider",
    "TavilyWebSearchProvider",
    "resolve_web_search_provider",
    "WebSearchProvider",
]
