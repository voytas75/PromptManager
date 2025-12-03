"""Runtime lifecycle helpers for Prompt Manager.

Updates:
  v0.1.0 - 2025-12-03 - Extract backend and lifecycle utilities into dedicated mixin.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from ..exceptions import (
    CategoryError,
    CategoryStorageError,
    PromptManagerError,
    PromptStorageError,
)
from ..repository import RepositoryError
from .backends import CollectionProtocol, RedisClientProtocol, build_chroma_client

if TYPE_CHECKING:  # pragma: no cover - typing only
    from types import TracebackType

    from models.prompt_model import Prompt, UserProfile

    from ..category_registry import CategoryRegistry
    from ..execution import CodexExecutor
    from ..history_tracker import HistoryTracker
    from ..intent_classifier import IntentClassifier
    from ..notifications import NotificationCenter
    from ..repository import PromptRepository

logger = logging.getLogger(__name__)

__all__ = ["PromptRuntimeMixin"]


class PromptRuntimeMixin:
    """Mixin that encapsulates backend helpers and runtime lifecycle hooks."""

    _category_registry: CategoryRegistry
    _repository: PromptRepository
    _collection: CollectionProtocol | None
    _chroma_client: Any
    _collection_name: str
    _chroma_path: str
    _embedding_function: Any | None
    _db_path: Path | None
    _notification_center: NotificationCenter
    _logs_path: Path
    _redis_client: RedisClientProtocol | None
    _embedding_worker: Any
    _closed: bool
    _history_tracker: HistoryTracker | None
    _executor: CodexExecutor | None

    def _apply_category_metadata(self, prompt: Prompt) -> Prompt:
        """Ensure prompt categories map to registry entries."""
        category_value = (prompt.category or "").strip()
        slug_candidate = prompt.category_slug or category_value
        if not slug_candidate:
            prompt.category = ""
            prompt.category_slug = None
            return prompt
        try:
            category = self._category_registry.ensure(
                slug=slug_candidate,
                label=category_value or slug_candidate,
                description=prompt.description,
            )
        except CategoryStorageError as exc:
            raise PromptStorageError("Unable to resolve prompt category") from exc
        except CategoryError as exc:
            raise PromptManagerError(str(exc)) from exc
        prompt.category = category.label
        prompt.category_slug = category.slug
        return prompt

    def _initialise_chroma_collection(self) -> None:
        """Create or refresh the Chroma collection backing prompt embeddings."""
        self._chroma_client, collection = build_chroma_client(
            self._chroma_path,
            self._collection_name,
            self._embedding_function,
            chroma_client=self._chroma_client,
        )
        self._collection = collection

    def _persist_chroma_client(self) -> None:
        """Flush the Chroma client to disk when supported."""
        persist = getattr(self._chroma_client, "persist", None)
        if not callable(persist):
            return
        try:
            persist()
        except Exception:  # noqa: BLE001
            logger.warning("Unable to flush Chroma client persistence state", exc_info=True)

    def _resolve_repository_path(self) -> Path:
        """Return the path to the SQLite repository, ensuring it exists."""
        candidate = self._db_path or getattr(self._repository, "_db_path", None)
        if candidate is None:
            raise PromptStorageError("SQLite repository path is not configured.")
        if isinstance(candidate, Path):
            path = candidate
        else:
            path = Path(str(candidate))
        path = path.expanduser()
        if not path.exists():
            raise PromptStorageError(f"SQLite repository missing at {path}.")
        return path

    @property
    def collection(self) -> CollectionProtocol:
        """Expose the underlying Chroma collection."""
        if self._collection is None:
            raise PromptManagerError("Chroma collection is not initialised.")
        return self._collection

    @property
    def repository(self) -> PromptRepository:
        """Expose the SQLite repository."""
        return self._repository

    @property
    def db_path(self) -> Path | None:
        """Return the configured SQLite database path."""
        return self._db_path

    @property
    def chroma_path(self) -> Path:
        """Return the filesystem path backing the Chroma vector store."""
        return Path(self._chroma_path)

    @property
    def logs_path(self) -> Path:
        """Return the directory where application logs are written."""
        return self._logs_path

    @property
    def intent_classifier(self) -> IntentClassifier | None:
        """Expose the configured intent classifier for tooling hooks."""
        classifier: IntentClassifier | None = getattr(self, "_intent_classifier", None)
        return classifier

    @property
    def executor(self) -> CodexExecutor | None:
        """Return the configured prompt executor."""
        return self._executor

    def set_executor(self, executor: CodexExecutor | None) -> None:
        """Assign or replace the Codex executor at runtime."""
        self._executor = executor

    @property
    def history_tracker(self) -> HistoryTracker | None:
        """Expose the execution history tracker if configured."""
        return self._history_tracker

    def set_history_tracker(self, tracker: HistoryTracker | None) -> None:
        """Assign or replace the history tracker at runtime."""
        self._history_tracker = tracker

    @property
    def notification_center(self) -> NotificationCenter:
        """Expose the notification centre for UI and tooling integrations."""
        return self._notification_center

    @property
    def user_profile(self) -> UserProfile | None:
        """Return the stored single-user profile when available."""
        profile: UserProfile | None = getattr(self, "_user_profile", None)
        return profile

    def refresh_user_profile(self) -> UserProfile | None:
        """Reload the persisted profile from the repository."""
        try:
            profile = self._repository.get_user_profile()
        except RepositoryError:
            logger.warning("Unable to refresh user profile from storage", exc_info=True)
            cached: UserProfile | None = getattr(self, "_user_profile", None)
            return cached
        self._user_profile = profile
        return profile

    def set_intent_classifier(self, classifier: IntentClassifier | None) -> None:
        """Replace the runtime intent classifier implementation."""
        self._intent_classifier = classifier

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
            disconnect = getattr(pool, "disconnect", None) if pool is not None else None
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

    def __enter__(self) -> Self:
        """Support use of the manager as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close resources when exiting a context manager block."""
        self.close()
