"""Factories for constructing PromptManager instances from validated settings.

Updates: v0.6.1 - 2025-11-07 - Add configurable embedding backends via settings.
Updates: v0.6.0 - 2025-11-06 - Wire intent classifier for hybrid retrieval suggestions.
Updates: v0.5.0 - 2025-11-05 - Add LiteLLM name generator wiring.
Updates: v0.1.0 - 2025-11-03 - Added build_prompt_manager helper for GUI/bootstrap reuse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, cast

from config import PromptManagerSettings

from .embedding import EmbeddingProvider, EmbeddingSyncWorker, create_embedding_function
from .intent_classifier import IntentClassifier
from .prompt_manager import NameGenerationError, PromptCacheError, PromptManager
from .name_generation import (
    LiteLLMDescriptionGenerator,
    LiteLLMNameGenerator,
    NameGenerationError,
)
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


def _resolve_embedding_components(
    settings: PromptManagerSettings,
    embedding_function: Optional[Any],
    embedding_provider: Optional[EmbeddingProvider],
) -> Tuple[Optional[Any], EmbeddingProvider]:
    """Return embedding function/provider pair respecting overrides and settings."""

    resolved_function = embedding_function
    if resolved_function is None:
        try:
            resolved_function = create_embedding_function(
                settings.embedding_backend,
                model=settings.embedding_model,
                api_key=settings.litellm_api_key,
                api_base=settings.litellm_api_base,
                device=settings.embedding_device,
            )
        except Exception as exc:
            raise RuntimeError(f"Unable to configure embedding backend: {exc}") from exc

    if embedding_provider is not None:
        return resolved_function, embedding_provider
    return resolved_function, EmbeddingProvider(resolved_function)

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
    resolved_embedding_function, resolved_embedding_provider = _resolve_embedding_components(
        settings,
        embedding_function,
        embedding_provider,
    )
    name_generator = None
    description_generator = None
    if settings.litellm_model:
        try:
            name_generator = LiteLLMNameGenerator(
                model=settings.litellm_model,
                api_key=settings.litellm_api_key,
                api_base=settings.litellm_api_base,
            )
            description_generator = LiteLLMDescriptionGenerator(
                model=settings.litellm_model,
                api_key=settings.litellm_api_key,
                api_base=settings.litellm_api_base,
            )
        except RuntimeError as exc:
            raise NameGenerationError(
                "LiteLLM is required for prompt name generation. Install litellm and configure credentials."
            ) from exc
    intent_classifier = IntentClassifier()

    manager = PromptManager(
        chroma_path=str(settings.chroma_path),
        db_path=str(settings.db_path),
        cache_ttl_seconds=settings.cache_ttl_seconds,
        redis_client=resolved_redis,
        chroma_client=chroma_client,
        embedding_function=resolved_embedding_function,
        repository=repository,
        embedding_provider=resolved_embedding_provider,
        embedding_worker=embedding_worker,
        enable_background_sync=enable_background_sync,
        name_generator=name_generator,
        description_generator=description_generator,
        intent_classifier=intent_classifier,
    )
    return manager


__all__ = ["build_prompt_manager"]
