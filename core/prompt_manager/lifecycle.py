"""Prompt lifecycle, caching, and embedding helpers for Prompt Manager.

Updates:
  v0.1.0 - 2025-12-03 - Extract prompt CRUD, caching, and embedding APIs into mixin.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from chromadb.errors import ChromaError

from models.prompt_model import Prompt

from ..embedding import EmbeddingGenerationError
from ..exceptions import (
    PromptCacheError,
    PromptManagerError,
    PromptNotFoundError,
    PromptStorageError,
)
from ..repository import RepositoryError, RepositoryNotFoundError

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from models.prompt_model import PromptVersion

    from ..repository import PromptRepository

logger = logging.getLogger(__name__)

RedisValue = str | bytes | memoryview

__all__ = ["PromptLifecycleMixin"]


class PromptLifecycleMixin:
    """Prompt CRUD orchestration with caching and embedding persistence."""

    _repository: PromptRepository
    _embedding_provider: Any
    _embedding_worker: Any
    _redis_client: Any | None
    _cache_ttl_seconds: int

    def create_prompt(
        self,
        prompt: Prompt,
        embedding: Sequence[float] | None = None,
        *,
        commit_message: str | None = None,
    ) -> Prompt:
        """Persist a new prompt in SQLite/ChromaDB and prime the cache."""
        prompt = self._apply_category_metadata(prompt)
        self._update_category_insight(prompt, previous_prompt=None)
        generated_embedding: list[float] | None = None
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
        version = self._commit_prompt_version(stored_prompt, commit_message=commit_message)
        logger.debug(
            "Prompt version committed",
            extra={
                "prompt_id": str(stored_prompt.id),
                "version_id": version.id,
                "version_number": version.version_number,
            },
        )
        return stored_prompt

    def get_prompt(self, prompt_id: UUID) -> Prompt:
        """Retrieve a prompt from cache or SQLite."""
        prompt_id = self._ensure_uuid(prompt_id)
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
        embedding: Sequence[float] | None = None,
        *,
        commit_message: str | None = None,
    ) -> Prompt:
        """Update an existing prompt with new metadata."""
        prompt = self._apply_category_metadata(prompt)
        try:
            previous_prompt = self._repository.get(prompt.id)
        except RepositoryNotFoundError as exc:
            raise PromptNotFoundError(f"Prompt {prompt.id} not found") from exc
        except RepositoryError as exc:
            raise PromptStorageError(
                f"Failed to load prompt {prompt.id} for version comparison"
            ) from exc

        latest_version: PromptVersion | None = None
        has_version_history = True
        try:
            latest_version = self._repository.get_prompt_latest_version(prompt.id)
            has_version_history = latest_version is not None
        except RepositoryError:
            logger.debug(
                "Unable to determine existing prompt versions; assuming history present",
                extra={"prompt_id": str(prompt.id)},
                exc_info=True,
            )

        self._update_category_insight(prompt, previous_prompt=previous_prompt)

        body_changed = self._normalise_body(previous_prompt.context) != self._normalise_body(
            prompt.context
        )
        should_commit_version = body_changed or not has_version_history
        current_version_number = latest_version.version_number if latest_version else 0
        target_version_number: int | None
        if should_commit_version:
            target_version_number = current_version_number + 1 if current_version_number >= 0 else 1
        elif current_version_number > 0:
            target_version_number = current_version_number
        else:
            target_version_number = None
        if target_version_number is not None:
            target_label = str(target_version_number)
            if prompt.version != target_label:
                prompt.version = target_label

        generated_embedding: list[float] | None = None
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
        if should_commit_version:
            version = self._commit_prompt_version(updated_prompt, commit_message=commit_message)
            logger.debug(
                "Prompt version committed",
                extra={
                    "prompt_id": str(updated_prompt.id),
                    "version_id": version.id,
                    "version_number": version.version_number,
                },
            )
        else:
            logger.debug(
                "Prompt body unchanged and version history already present; skipping version "
                "commit",
                extra={"prompt_id": str(updated_prompt.id)},
            )
        return updated_prompt

    def delete_prompt(self, prompt_id: UUID) -> None:
        """Remove a prompt from all data stores."""
        prompt_id = self._ensure_uuid(prompt_id)
        try:
            self._repository.delete(prompt_id)
        except RepositoryNotFoundError as exc:
            raise PromptNotFoundError(f"Prompt {prompt_id} not found") from exc
        except RepositoryError as exc:
            raise PromptStorageError(f"Failed to delete prompt {prompt_id} from SQLite") from exc
        try:
            self.collection.delete(ids=[str(prompt_id)])
        except ChromaError as exc:
            raise PromptStorageError(f"Failed to delete prompt {prompt_id}") from exc
        try:
            self._evict_cached_prompt(prompt_id)
        except PromptCacheError:
            logger.debug(
                "Prompt deleted but cache eviction failed",
                extra={"prompt_id": str(prompt_id)},
            )

    def increment_usage(self, prompt_id: UUID) -> None:
        """Increment usage counter for a prompt."""
        prompt_id = self._ensure_uuid(prompt_id)
        prompt = self.get_prompt(prompt_id)
        prompt.usage_count += 1
        self.update_prompt(prompt)
        self._record_prompt_usage(prompt)

    # Internal utilities ------------------------------------------------ #

    @staticmethod
    def _normalise_body(body: str | None) -> str:
        return body.replace("\r\n", "\n") if body else ""

    @staticmethod
    def _ensure_uuid(value: UUID) -> UUID:
        if not isinstance(value, UUID):
            raise TypeError("prompt_id must be a uuid.UUID instance.")
        return value

    def _record_prompt_usage(self, prompt: Prompt) -> None:
        """Persist prompt usage into the single-user profile when possible."""
        try:
            profile = self._repository.record_user_prompt_usage(prompt)
        except RepositoryError:
            logger.debug(
                "Failed to update user profile preferences",
                extra={"prompt_id": str(prompt.id)},
                exc_info=True,
            )
            return
        self._user_profile = profile

    def _apply_rating(self, prompt_id: UUID, rating: float) -> None:
        """Update prompt aggregates from a new rating."""
        prompt_id = self._ensure_uuid(prompt_id)
        try:
            prompt = self.get_prompt(prompt_id)
        except PromptManagerError:
            logger.warning(
                "Unable to fetch prompt for rating update",
                extra={"prompt_id": str(prompt_id)},
                exc_info=True,
            )
            return

        prompt.rating_count += 1
        prompt.rating_sum += float(rating)
        if prompt.rating_count > 0:
            prompt.quality_score = round(prompt.rating_sum / prompt.rating_count, 2)
        prompt.last_modified = datetime.now(UTC)

        try:
            self.update_prompt(prompt)
        except PromptManagerError:
            logger.warning(
                "Unable to persist rating update for prompt",
                extra={"prompt_id": str(prompt_id)},
                exc_info=True,
            )

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
        except Exception as exc:  # noqa: BLE001 - redis optional
            raise PromptCacheError("Failed to write prompt to Redis") from exc

    def _get_cached_prompt(self, prompt_id: UUID) -> Prompt | None:
        """Fetch prompt from Redis cache when available."""
        prompt_id = self._ensure_uuid(prompt_id)
        if self._redis_client is None:
            return None
        try:
            cached_value: RedisValue | None = self._redis_client.get(self._cache_key(prompt_id))
        except Exception as exc:  # noqa: BLE001 - redis optional
            raise PromptCacheError("Failed to read prompt from Redis") from exc
        if not cached_value:
            return None
        if isinstance(cached_value, memoryview):
            cached_text = cached_value.tobytes().decode("utf-8")
        elif isinstance(cached_value, bytes):
            cached_text = cached_value.decode("utf-8")
        else:
            cached_text = str(cached_value)
        try:
            record = json.loads(cached_text)
        except json.JSONDecodeError as exc:
            logger.warning("Cannot decode cached prompt", extra={"prompt_id": str(prompt_id)})
            raise PromptCacheError("Invalid JSON cached value") from exc
        return Prompt.from_record(record)

    def _evict_cached_prompt(self, prompt_id: UUID) -> None:
        """Remove cached prompt from Redis."""
        prompt_id = self._ensure_uuid(prompt_id)
        if self._redis_client is None:
            return
        try:
            self._redis_client.delete(self._cache_key(prompt_id))
        except Exception as exc:  # noqa: BLE001 - redis optional
            raise PromptCacheError("Failed to evict prompt from Redis") from exc

    @staticmethod
    def _cache_key(prompt_id: UUID) -> str:
        """Format cache key for prompt entries."""
        return f"prompt:{prompt_id}"

    def _persist_embedding(
        self,
        prompt: Prompt,
        embedding: Sequence[float],
        *,
        is_new: bool,
    ) -> None:
        """Persist embeddings to Chroma and refresh caches."""
        if not isinstance(embedding, Sequence):
            raise TypeError("Embedding payload must be a sequence.")
        payload: dict[str, Any] = {
            "ids": [str(prompt.id)],
            "documents": [prompt.document],
            "metadatas": [prompt.to_metadata()],
            "embeddings": [list(embedding)],
        }
        collection = self.collection
        try:
            collection.upsert(**payload)
        except ChromaError as exc:
            raise PromptStorageError(f"Failed to persist embedding for prompt {prompt.id}") from exc
        try:
            self._cache_prompt(prompt)
        except PromptCacheError:
            logger.warning(
                "Prompt cached embedding refresh failed",
                extra={"prompt_id": str(prompt.id)},
            )
        if is_new:
            logger.debug(
                "Stored embedding for prompt",
                extra={"prompt_id": str(prompt.id)},
            )

    def _persist_embedding_from_worker(
        self,
        prompt: Prompt,
        embedding: Sequence[float],
    ) -> None:
        """Callback invoked by background worker once embedding is generated."""
        prompt.ext4 = list(embedding)
        try:
            self._repository.update(prompt)
        except RepositoryError as exc:
            raise PromptStorageError(f"Failed to persist embedding for prompt {prompt.id}") from exc
        self._persist_embedding(prompt, embedding, is_new=False)
