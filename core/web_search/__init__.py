"""Web search provider abstractions and service wiring.

Updates:
  v0.1.0 - 2025-12-04 - Introduce Exa provider, models, and service exports.
"""

from .models import WebSearchDocument, WebSearchResult
from .providers import ExaWebSearchProvider, WebSearchProvider
from .service import WebSearchService

__all__ = [
    "ExaWebSearchProvider",
    "WebSearchDocument",
    "WebSearchProvider",
    "WebSearchResult",
    "WebSearchService",
]
