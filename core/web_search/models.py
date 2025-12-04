"""Shared data models for external web search integrations.

Updates:
  v0.1.0 - 2025-12-04 - Introduce provider-agnostic result dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class WebSearchDocument:
    """Canonical representation of a single search result."""

    title: str
    url: str
    summary: str | None = None
    highlights: list[str] = field(default_factory=list)
    author: str | None = None
    published_at: datetime | None = None
    score: float | None = None
    raw: dict[str, Any] | None = None


@dataclass(slots=True)
class WebSearchResult:
    """Aggregated search results returned by a provider."""

    provider: str
    query: str
    documents: list[WebSearchDocument]
    raw: dict[str, Any] | None = None


__all__ = ["WebSearchDocument", "WebSearchResult"]
