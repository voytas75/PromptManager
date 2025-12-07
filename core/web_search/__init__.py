"""Web search provider abstractions and service wiring.

Updates:
  v0.1.1 - 2025-12-07 - Export Tavily provider alongside Exa implementation.
  v0.1.0 - 2025-12-04 - Introduce Exa provider, models, and service exports.
"""

from .models import WebSearchDocument, WebSearchResult
from .providers import ExaWebSearchProvider, TavilyWebSearchProvider, WebSearchProvider
from .service import WebSearchService

__all__ = [
    "ExaWebSearchProvider",
    "TavilyWebSearchProvider",
    "WebSearchDocument",
    "WebSearchProvider",
    "WebSearchResult",
    "WebSearchService",
]
