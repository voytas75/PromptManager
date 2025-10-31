"""Factories for constructing PromptManager instances from validated settings.

Updates: v0.1.0 - 2025-11-03 - Added build_prompt_manager helper for GUI/bootstrap reuse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, cast

from config import PromptManagerSettings

from .prompt_manager import (
    PromptCacheError,
    PromptManager,
)
from .embedding import EmbeddingProvider, EmbeddingSyncWorker
from .repository import PromptRepository

try:  # pragma: no cover - redis optional dependency
    import redis
except ImportError:  # pragma: no cover - redis optional dependency
    redis = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from chromadb.api import ClientAPI
    from redis import Redis
else:  # pragma: no cover - typing only
    ClientAPI = Any  # type: ignore[misc,assignment]
    Redis = Any  # type: ignore[misc,assignment]


def _resolve_redis_client(
    redis_dsn: Optional[str],
    redis_client: Optional["Redis"],
) -> Optional["Redis"]:
    """Create a Redis client when a DSN is provided but no client supplied."""
    if redis_client is not None or not redis_dsn:
        return redis_client
    if redis is None:
        raise PromptCacheError("Redis DSN provided but redis package is not installed")
    from_url = cast(Callable[[str], "Redis"], getattr(redis, "from_url"))
    return from_url(redis_dsn)

def build_prompt_manager(
    settings: PromptManagerSettings,
    *,
    redis_client: Optional["Redis"] = None,
    chroma_client: Optional["ClientAPI"] = None,
    embedding_function: Optional[Any] = None,
    repository: Optional[PromptRepository] = None,
    embedding_provider: Optional[EmbeddingProvider] = None,
    embedding_worker: Optional[EmbeddingSyncWorker] = None,
    enable_background_sync: bool = True,
) -> PromptManager:
    """Return a PromptManager configured from validated settings."""
    resolved_redis = _resolve_redis_client(settings.redis_dsn, redis_client)
    return PromptManager(
        chroma_path=str(settings.chroma_path),
        db_path=str(settings.db_path),
        cache_ttl_seconds=settings.cache_ttl_seconds,
        redis_client=resolved_redis,
        chroma_client=chroma_client,
        embedding_function=embedding_function,
        repository=repository,
        embedding_provider=embedding_provider,
        embedding_worker=embedding_worker,
        enable_background_sync=enable_background_sync,
    )


__all__ = ["build_prompt_manager"]
