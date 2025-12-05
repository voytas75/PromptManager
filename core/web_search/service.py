"""Coordinator for configured web search providers.

Updates:
  v0.1.1 - 2025-12-05 - Limit type-only imports to TYPE_CHECKING and document __init__.
  v0.1.0 - 2025-12-04 - Add provider-aware service with availability checks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..exceptions import WebSearchUnavailable

if TYPE_CHECKING:
    from .models import WebSearchResult
    from .providers import WebSearchProvider


class WebSearchService:
    """High-level interface that delegates to the configured provider."""

    def __init__(self, provider: WebSearchProvider | None = None) -> None:
        """Initialise the service with an optional provider."""
        self._provider = provider

    @property
    def provider(self) -> WebSearchProvider | None:
        """Return the currently configured provider."""
        return self._provider

    def configure(self, provider: WebSearchProvider | None) -> None:
        """Swap in a new provider implementation."""
        self._provider = provider

    def is_available(self) -> bool:
        """Return ``True`` when a provider is configured."""
        return self._provider is not None

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        **kwargs: Any,
    ) -> WebSearchResult:
        """Invoke the underlying provider."""
        if self._provider is None:
            raise WebSearchUnavailable("No web search provider is configured")
        return await self._provider.search(query, limit=limit, **kwargs)


__all__ = ["WebSearchService"]
