"""High-level CRUD manager for prompt records backed by SQLite, ChromaDB, and Redis.

Updates: v0.4.1 - 2025-11-05 - Add lifecycle shutdown hooks and mute Chroma telemetry.
Updates: v0.4.0 - 2025-11-05 - Add embedding provider with background sync and retry handling.
Updates: v0.3.0 - 2025-11-03 - Require explicit DB path; accept resolved settings inputs.
Updates: v0.2.0 - 2025-10-31 - Add SQLite repository integration with ChromaDB/Redis sync.
Updates: v0.1.0 - 2025-10-30 - Initial PromptManager with CRUD and search support.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, cast

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.errors import ChromaError

os.environ.setdefault("CHROMA_TELEMETRY_IMPLEMENTATION", "none")
os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "0")


def _mute_posthog_capture() -> None:
    try:
        import posthog  # type: ignore
    except Exception:
        return

    def _noop_capture(*_: Any, **__: Any) -> None:
        return None

    try:
        posthog.capture = _noop_capture  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        posthog.disabled = True  # type: ignore[attr-defined]
    except Exception:
        pass


_mute_posthog_capture()

try:
    import redis
    from redis.exceptions import RedisError
except ImportError:  # pragma: no cover - redis optional during development
    redis = None  # type: ignore
    RedisError = Exception  # type: ignore[misc,assignment]

from models.prompt_model import Prompt

from .embedding import EmbeddingGenerationError, EmbeddingProvider, EmbeddingSyncWorker
from .repository import PromptRepository, RepositoryError, RepositoryNotFoundError

logger = logging.getLogger(__name__)


class PromptManagerError(Exception):
    """Base exception for PromptManager failures."""


class PromptNotFoundError(PromptManagerError):
    """Raised when a prompt cannot be located in the backing store."""


class PromptStorageError(PromptManagerError):
    """Raised when interactions with persistent backends fail."""


class PromptCacheError(PromptManagerError):
    """Raised when Redis cache lookups or writes fail."""


class PromptManager:
    """Manage prompt persistence, caching, and semantic search."""

    def __init__(
        self,
        chroma_path: str,
        db_path: Union[str, Path, None] = None,
        collection_name: str = "prompt_manager",
        cache_ttl_seconds: int = 300,
        redis_client: Optional["redis.Redis"] = None,
        chroma_client: Optional[ClientAPI] = None,
        embedding_function: Optional[Any] = None,
        repository: Optional[PromptRepository] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        embedding_worker: Optional[EmbeddingSyncWorker] = None,
        enable_background_sync: bool = True,
    ) -> None:
        """Initialise the manager with data backends.

        Args:
            chroma_path: Filesystem path where ChromaDB stores data.
            db_path: Filesystem path to the SQLite database file.
            collection_name: Name of the ChromaDB collection for prompts.
            cache_ttl_seconds: Expiration for Redis cache entries.
            redis_client: Optional Redis client instance for caching.
            chroma_client: Optional preconfigured Chroma client.
            embedding_function: Optional embedding function for Chroma.
            repository: Optional preconfigured repository instance (e.g. for testing).
        """
        # Allow tests to supply repository directly; only require db_path when building one.
        if repository is None and db_path is None:
            raise ValueError("db_path must be provided when no repository is supplied")

        self._closed = False
        self._cache_ttl_seconds = cache_ttl_seconds
        self._redis_client = redis_client
        self._chroma_client: Any
        self._collection: Any = cast(Any, None)
        resolved_db_path = Path(db_path).expanduser() if db_path is not None else None
        resolved_chroma_path = Path(chroma_path).expanduser()
        try:
            self._repository = repository or PromptRepository(str(resolved_db_path))
        except RepositoryError as exc:
            raise PromptStorageError("Unable to initialise SQLite repository") from exc
        # Disable Chroma anonymized telemetry to avoid noisy PostHog errors in some environments
        # and respect privacy defaults. Use persistent client at the configured path.
        if chroma_client is None:
            try:
                from chromadb.config import Settings as ChromaSettings

                chroma_settings = ChromaSettings(
                    anonymized_telemetry=False,
                    is_persistent=True,
                    persist_directory=str(resolved_chroma_path),
                )
                self._chroma_client = cast(Any, chromadb.Client(chroma_settings))
            except Exception:
                # Fallback to legacy PersistentClient signature if settings import or usage fails
                self._chroma_client = cast(
                    Any,
                    chromadb.PersistentClient(path=str(resolved_chroma_path)),
                )
        else:
            self._chroma_client = cast(Any, chroma_client)
        try:
            collection = self._chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=embedding_function,
            )
            self._collection = cast(Any, collection)
        except ChromaError as exc:
            raise PromptStorageError("Unable to initialiase ChromaDB collection") from exc

        self._embedding_provider = embedding_provider or EmbeddingProvider(embedding_function)
        if enable_background_sync:
            worker_logger = logger.getChild("embedding_sync")
            self._embedding_worker = embedding_worker or EmbeddingSyncWorker(
                provider=self._embedding_provider,
                fetch_prompt=self._repository.get,
                persist_callback=self._persist_embedding_from_worker,
                logger=worker_logger,
            )
        else:
            self._embedding_worker = embedding_worker or _NullEmbeddingWorker()

    @property
    def collection(self) -> Collection:
        """Expose the underlying Chroma collection."""
        return cast(Collection, self._collection)

    @property
    def repository(self) -> PromptRepository:
        """Expose the SQLite repository."""
        return self._repository

    def close(self) -> None:
        """Release background workers and backend connections."""
        if self._closed:
            return

        self._closed = True

        worker = getattr(self, "_embedding_worker", None)
        stop = getattr(worker, "stop", None)
        if callable(stop):
            try:
                stop()
            except Exception:
                logger.debug("Failed to stop embedding worker cleanly", exc_info=True)

        if self._redis_client is not None:
            redis_close = getattr(self._redis_client, "close", None)
            if callable(redis_close):
                try:
                    redis_close()
                except Exception:
                    logger.debug(
                        "Failed to close Redis client cleanly",
                        exc_info=True,
                    )
            pool = getattr(self._redis_client, "connection_pool", None)
            disconnect = getattr(pool, "disconnect", None)
            if callable(disconnect):
                try:
                    disconnect()
                except Exception:
                    logger.debug(
                        "Failed to disconnect Redis connection pool cleanly",
                        exc_info=True,
                    )

        chroma_client = getattr(self, "_chroma_client", None)
        if chroma_client is not None:
            close_fn = getattr(chroma_client, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    logger.debug(
                        "Failed to close Chroma client cleanly",
                        exc_info=True,
                    )

    def __enter__(self) -> "PromptManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001 - dynamic signature
        self.close()

    def create_prompt(
        self,
        prompt: Prompt,
        embedding: Optional[Sequence[float]] = None,
    ) -> Prompt:
        """Persist a new prompt in SQLite/ChromaDB and prime the cache."""
        generated_embedding: Optional[List[float]] = None
        if embedding is not None:
            generated_embedding = list(embedding)
        else:
            try:
                generated_embedding = self._embedding_provider.embed(prompt.document)
            except EmbeddingGenerationError:
                logger.warning(
                    "Falling back to background embedding for prompt",
                    extra={"prompt_id": str(prompt.id)},
                )
        try:
            if generated_embedding is not None:
                prompt.ext4 = list(generated_embedding)
            stored_prompt = self._repository.add(prompt)
        except RepositoryError as exc:
            raise PromptStorageError(f"Failed to persist prompt {prompt.id}") from exc
        if generated_embedding is not None:
            try:
                self._persist_embedding(stored_prompt, generated_embedding, is_new=True)
            except PromptStorageError as exc:
                try:
                    self._repository.delete(prompt.id)
                except RepositoryError:
                    logger.error(
                        "Unable to roll back SQLite insert after Chroma failure",
                        extra={"prompt_id": str(prompt.id)},
                    )
                raise exc
        else:
            self._embedding_worker.schedule(stored_prompt.id)
            try:
                self._cache_prompt(stored_prompt)
            except PromptCacheError:
                logger.warning(
                    "Prompt created but not cached",
                    extra={"prompt_id": str(prompt.id)},
                )
        return stored_prompt

    def get_prompt(self, prompt_id: uuid.UUID) -> Prompt:
        """Retrieve a prompt from cache or SQLite."""
        prompt = self._get_cached_prompt(prompt_id)
        if prompt:
            return prompt
        try:
            prompt_obj = self._repository.get(prompt_id)
        except RepositoryNotFoundError as exc:
            raise PromptNotFoundError(f"Prompt {prompt_id} not found") from exc
        except RepositoryError as exc:
            raise PromptStorageError(f"Failed to fetch prompt {prompt_id} from SQLite") from exc
        try:
            self._cache_prompt(prompt_obj)
        except PromptCacheError:
            logger.debug(
                "Fetched prompt could not be cached",
                extra={"prompt_id": str(prompt_id)},
            )
        return prompt_obj

    def update_prompt(
        self,
        prompt: Prompt,
        embedding: Optional[Sequence[float]] = None,
    ) -> Prompt:
        """Update an existing prompt with new metadata."""
        generated_embedding: Optional[List[float]] = None
        if embedding is not None:
            generated_embedding = list(embedding)
        else:
            try:
                generated_embedding = self._embedding_provider.embed(prompt.document)
            except EmbeddingGenerationError:
                logger.warning(
                    "Scheduling background embedding refresh",
                    extra={"prompt_id": str(prompt.id)},
                )
        try:
            if generated_embedding is not None:
                prompt.ext4 = list(generated_embedding)
            updated_prompt = self._repository.update(prompt)
        except RepositoryNotFoundError as exc:
            raise PromptNotFoundError(f"Prompt {prompt.id} not found") from exc
        except RepositoryError as exc:
            raise PromptStorageError(f"Failed to update prompt {prompt.id} in SQLite") from exc
        if generated_embedding is not None:
            self._persist_embedding(updated_prompt, generated_embedding, is_new=False)
        else:
            self._embedding_worker.schedule(updated_prompt.id)
            try:
                self._cache_prompt(updated_prompt)
            except PromptCacheError:
                logger.warning(
                    "Prompt updated but cache refresh failed",
                    extra={"prompt_id": str(prompt.id)},
                )
        return updated_prompt

    def delete_prompt(self, prompt_id: uuid.UUID) -> None:
        """Remove a prompt from all data stores."""
        try:
            self._repository.delete(prompt_id)
        except RepositoryNotFoundError as exc:
            raise PromptNotFoundError(f"Prompt {prompt_id} not found") from exc
        except RepositoryError as exc:
            raise PromptStorageError(f"Failed to delete prompt {prompt_id} from SQLite") from exc
        try:
            self._collection.delete(ids=[str(prompt_id)])
        except ChromaError as exc:
            raise PromptStorageError(f"Failed to delete prompt {prompt_id}") from exc
        try:
            self._evict_cached_prompt(prompt_id)
        except PromptCacheError:
            logger.debug(
                "Prompt deleted but cache eviction failed",
                extra={"prompt_id": str(prompt_id)},
            )

    def search_prompts(
        self,
        query_text: str,
        limit: int = 5,
        where: Optional[Dict[str, Any]] = None,
        embedding: Optional[Sequence[float]] = None,
    ) -> List[Prompt]:
        """Search prompts semantically using a text query or embedding."""
        if not query_text and embedding is None:
            raise ValueError("query_text or embedding must be provided")

        query_embedding: Optional[List[float]]
        if embedding is not None:
            query_embedding = list(embedding)
        else:
            try:
                query_embedding = self._embedding_provider.embed(query_text)
            except EmbeddingGenerationError as exc:
                raise PromptStorageError("Failed to generate query embedding") from exc

        try:
            results = cast(
                Dict[str, Any],
                self._collection.query(
                    query_texts=None,
                    query_embeddings=[query_embedding] if query_embedding is not None else None,
                    n_results=limit,
                    where=where,
                ),
            )
        except ChromaError as exc:
            raise PromptStorageError("Failed to query prompts") from exc

        prompts: List[Prompt] = []
        ids = cast(List[str], results.get("ids", [[]])[0])
        documents = cast(List[str], results.get("documents", [[]])[0])
        metadatas = cast(List[Dict[str, Any]], results.get("metadatas", [[]])[0])

        for prompt_id, document, metadata in zip(ids, documents, metadatas):
            try:
                prompt_uuid = uuid.UUID(prompt_id)
            except ValueError:
                logger.warning(
                    "Invalid prompt UUID in Chroma results",
                    extra={"prompt_id": prompt_id},
                )
                continue
            try:
                prompt_record = self._repository.get(prompt_uuid)
            except RepositoryNotFoundError:
                record = {"id": prompt_id, "document": document, "metadata": metadata}
                prompt_record = Prompt.from_chroma(record)
            except RepositoryError as exc:
                raise PromptStorageError(
                    f"Failed to hydrate prompt {prompt_id} from SQLite"
                ) from exc
            prompts.append(prompt_record)
        return prompts

    def increment_usage(self, prompt_id: uuid.UUID) -> None:
        """Increment usage counter for a prompt."""
        prompt = self.get_prompt(prompt_id)
        prompt.usage_count += 1
        self.update_prompt(prompt)

    # Cache helpers ----------------------------------------------------- #

    def _cache_prompt(self, prompt: Prompt) -> None:
        """Store prompt representation in Redis."""
        if self._redis_client is None:
            return
        payload = json.dumps(prompt.to_record(), ensure_ascii=False)
        try:
            self._redis_client.setex(
                self._cache_key(prompt.id),
                self._cache_ttl_seconds,
                payload,
            )
        except RedisError as exc:  # pragma: no cover - redis not in CI
            raise PromptCacheError("Failed to write prompt to Redis") from exc

    def _get_cached_prompt(self, prompt_id: uuid.UUID) -> Optional[Prompt]:
        """Fetch prompt from Redis cache when available."""
        if self._redis_client is None:
            return None
        try:
            cached = self._redis_client.get(self._cache_key(prompt_id))
        except RedisError as exc:  # pragma: no cover - redis not in CI
            raise PromptCacheError("Failed to read prompt from Redis") from exc
        if not cached:
            return None
        cached_text = cached.decode("utf-8") if isinstance(cached, bytes) else str(cached)
        try:
            record = json.loads(cached_text)
        except json.JSONDecodeError as exc:
            logger.warning("Cannot decode cached prompt", extra={"prompt_id": str(prompt_id)})
            raise PromptCacheError("Invalid JSON cached value") from exc
        return Prompt.from_record(record)

    def _evict_cached_prompt(self, prompt_id: uuid.UUID) -> None:
        """Remove cached prompt from Redis."""
        if self._redis_client is None:
            return
        try:
            self._redis_client.delete(self._cache_key(prompt_id))
        except RedisError as exc:  # pragma: no cover - redis not in CI
            raise PromptCacheError("Failed to evict prompt from Redis") from exc

    @staticmethod
    def _cache_key(prompt_id: uuid.UUID) -> str:
        """Format cache key for prompt entries."""
        return f"prompt:{prompt_id}"

    def _persist_embedding(self, prompt: Prompt, embedding: Sequence[float], *, is_new: bool) -> None:
        """Persist embeddings to Chroma and refresh caches."""

        payload: Dict[str, Any] = {
            "ids": [str(prompt.id)],
            "documents": [prompt.document],
            "metadatas": [prompt.to_metadata()],
            "embeddings": [list(embedding)],
        }
        try:
            if is_new:
                self._collection.add(**payload)
            else:
                self._collection.upsert(**payload)
        except ChromaError as exc:
            raise PromptStorageError(f"Failed to persist embedding for prompt {prompt.id}") from exc
        try:
            self._cache_prompt(prompt)
        except PromptCacheError:
            logger.warning(
                "Prompt cached embedding refresh failed",
                extra={"prompt_id": str(prompt.id)},
            )

    def _persist_embedding_from_worker(self, prompt: Prompt, embedding: Sequence[float]) -> None:
        """Callback invoked by background worker once embedding is generated."""

        prompt.ext4 = list(embedding)
        try:
            self._repository.update(prompt)
        except RepositoryError as exc:
            raise PromptStorageError(f"Failed to persist embedding for prompt {prompt.id}") from exc
        self._persist_embedding(prompt, embedding, is_new=False)


class _NullEmbeddingWorker:
    """Embedding worker placeholder used when background sync is disabled."""

    def schedule(self, _: uuid.UUID) -> None:  # pragma: no cover - trivial noop
        return

    def stop(self) -> None:  # pragma: no cover - trivial noop
        return


__all__ = [
    "PromptManager",
    "PromptManagerError",
    "PromptNotFoundError",
    "PromptStorageError",
    "PromptCacheError",
]
