"""Web search provider abstractions and service wiring.

Updates:
  v0.1.2 - 2025-12-07 - Export random provider wrapper alongside Exa/Tavily.
  v0.1.1 - 2025-12-07 - Export Tavily provider alongside Exa implementation.
  v0.1.0 - 2025-12-04 - Introduce Exa provider, models, and service exports.
"""

from .models import WebSearchDocument, WebSearchResult
from .providers import (
    ExaWebSearchProvider,
    RandomWebSearchProvider,
    TavilyWebSearchProvider,
    WebSearchProvider,
)
from .service import WebSearchService

__all__ = [
    "ExaWebSearchProvider",
    "RandomWebSearchProvider",
    "TavilyWebSearchProvider",
    "WebSearchDocument",
    "WebSearchProvider",
    "WebSearchResult",
    "WebSearchService",
]
