"""Web search provider abstractions and service wiring.

Updates:
  v0.1.5 - 2025-12-07 - Re-export resolve_web_search_provider helper.
  v0.1.4 - 2025-12-07 - Export SerpApi provider alongside existing implementations.
  v0.1.3 - 2025-12-07 - Export Serper provider alongside existing implementations.
  v0.1.2 - 2025-12-07 - Export random provider wrapper alongside Exa/Tavily.
  v0.1.1 - 2025-12-07 - Export Tavily provider alongside Exa implementation.
  v0.1.0 - 2025-12-04 - Introduce Exa provider, models, and service exports.
"""

from .models import WebSearchDocument, WebSearchResult
from .providers import (
    ExaWebSearchProvider,
    RandomWebSearchProvider,
    SerpApiWebSearchProvider,
    SerperWebSearchProvider,
    TavilyWebSearchProvider,
    WebSearchProvider,
    resolve_web_search_provider,
)
from .service import WebSearchService

__all__ = [
    "ExaWebSearchProvider",
    "RandomWebSearchProvider",
    "SerpApiWebSearchProvider",
    "SerperWebSearchProvider",
    "TavilyWebSearchProvider",
    "WebSearchDocument",
    "WebSearchProvider",
    "WebSearchResult",
    "WebSearchService",
    "resolve_web_search_provider",
]
