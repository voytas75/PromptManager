"""High-level CRUD manager for prompt records backed by SQLite, ChromaDB, and Redis.

Updates: v0.2.0 - 2025-10-31 - Added SQLite repository integration with ChromaDB/Redis sync.
Updates: v0.1.0 - 2025-10-30 - Initial PromptManager with CRUD and search support.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.errors import ChromaError

try:
    import redis
    from redis.exceptions import RedisError
except ImportError:  # pragma: no cover - redis optional during development
    redis = None  # type: ignore
    RedisError = Exception  # type: ignore[misc,assignment]

from models.prompt_model import Prompt
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
        db_path: Optional[str] = None,
        collection_name: str = "prompt_manager",
        cache_ttl_seconds: int = 300,
        redis_client: Optional["redis.Redis[Any]"] = None,
        chroma_client: Optional[ClientAPI] = None,
        embedding_function: Optional[Any] = None,
        repository: Optional[PromptRepository] = None,
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
        self._cache_ttl_seconds = cache_ttl_seconds
        self._redis_client = redis_client
        resolved_db_path = Path(db_path).expanduser() if db_path else Path("data") / "prompt_manager.db"
        try:
            self._repository = repository or PromptRepository(str(resolved_db_path))
        except RepositoryError as exc:
            raise PromptStorageError("Unable to initialise SQLite repository") from exc
        self._chroma_client = chroma_client or chromadb.PersistentClient(path=chroma_path)
        try:
            self._collection = self._chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=embedding_function,
            )
        except ChromaError as exc:
            raise PromptStorageError("Unable to initialiase ChromaDB collection") from exc

    @property
    def collection(self) -> Collection:
        """Expose the underlying Chroma collection."""
        return self._collection

    @property
    def repository(self) -> PromptRepository:
        """Expose the SQLite repository."""
        return self._repository

    def create_prompt(
        self,
        prompt: Prompt,
        embedding: Optional[Sequence[float]] = None,
    ) -> Prompt:
        """Persist a new prompt in SQLite/ChromaDB and prime the cache."""
        if embedding is not None:
            prompt.ext4 = list(embedding)
        try:
            stored_prompt = self._repository.add(prompt)
        except RepositoryError as exc:
            raise PromptStorageError(f"Failed to persist prompt {prompt.id}") from exc
        payload = {
            "ids": [str(prompt.id)],
            "documents": [stored_prompt.document],
            "metadatas": [stored_prompt.to_metadata()],
        }
        if embedding is not None:
            payload["embeddings"] = [list(embedding)]
        try:
            self._collection.add(**payload)
        except ChromaError as exc:
            try:
                self._repository.delete(prompt.id)
            except RepositoryError:
                logger.error(
                    "Unable to roll back SQLite insert after Chroma failure",
                    extra={"prompt_id": str(prompt.id)},
                )
            raise PromptStorageError(f"Failed to create prompt {prompt.id}") from exc
        try:
            self._cache_prompt(stored_prompt)
        except PromptCacheError:
            logger.warning("Prompt created but not cached", extra={"prompt_id": str(prompt.id)})
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
        if embedding is not None:
            prompt.ext4 = list(embedding)
        try:
            updated_prompt = self._repository.update(prompt)
        except RepositoryNotFoundError as exc:
            raise PromptNotFoundError(f"Prompt {prompt.id} not found") from exc
        except RepositoryError as exc:
            raise PromptStorageError(f"Failed to update prompt {prompt.id} in SQLite") from exc
        payload = {
            "ids": [str(prompt.id)],
            "documents": [updated_prompt.document],
            "metadatas": [updated_prompt.to_metadata()],
        }
        if embedding is not None:
            payload["embeddings"] = [list(embedding)]
        try:
            self._collection.upsert(**payload)
        except ChromaError as exc:
            raise PromptStorageError(f"Failed to update prompt {prompt.id}") from exc
        try:
            self._cache_prompt(updated_prompt)
        except PromptCacheError:
            logger.warning("Prompt updated but cache refresh failed", extra={"prompt_id": str(prompt.id)})
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

        try:
            results = self._collection.query(
                query_texts=[query_text] if query_text else None,
                query_embeddings=[list(embedding)] if embedding is not None else None,
                n_results=limit,
                where=where,
            )
        except ChromaError as exc:
            raise PromptStorageError("Failed to query prompts") from exc

        prompts: List[Prompt] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for prompt_id, document, metadata in zip(ids, documents, metadatas):
            try:
                prompt_uuid = uuid.UUID(prompt_id)
            except ValueError:
                logger.warning("Invalid prompt UUID in Chroma results", extra={"prompt_id": prompt_id})
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
        try:
            record = json.loads(cached)
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


__all__ = [
    "PromptManager",
    "PromptManagerError",
    "PromptNotFoundError",
    "PromptStorageError",
    "PromptCacheError",
]
