"""Backend bootstrap mixin for Prompt Manager.

Updates:
  v0.1.1 - 2025-12-09 - Initialise Redis availability state.
  v0.1.0 - 2025-12-03 - Extract repository and backend wiring helpers from package init.
"""

from __future__ import annotations

import logging
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ..category_registry import CategoryRegistry
from ..embedding import EmbeddingProvider, EmbeddingSyncWorker
from ..exceptions import PromptStorageError
from ..notifications import NotificationCenter, notification_center as default_notification_center
from ..repository import RepositoryError
from .backends import (
    CollectionProtocol,
    NullEmbeddingWorker,
    RedisClientProtocol,
    build_chroma_client,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from chromadb.api import ClientAPI

    from models.category_model import PromptCategory

    from ..repository import PromptRepository

logger = logging.getLogger(__name__)

__all__ = ["BackendBootstrapMixin"]


class BackendBootstrapMixin:
    """Mixin encapsulating repository, Chroma, and embedding bootstrap logic."""

    _closed: bool
    _cache_ttl_seconds: int
    _redis_client: RedisClientProtocol | None
    _redis_unavailable_reason: str | None
    _repository: PromptRepository
    _category_registry: CategoryRegistry
    _collection: CollectionProtocol | None
    _collection_name: str
    _chroma_path: str
    _chroma_client: Any
    _db_path: Path | None
    _embedding_function: Any | None
    _embedding_provider: EmbeddingProvider
    _embedding_worker: EmbeddingSyncWorker | NullEmbeddingWorker | Any
    _notification_center: NotificationCenter
    _logs_path: Path
    _persist_embedding_from_worker: Any

    def _initialise_backends(
        self,
        *,
        chroma_path: str,
        db_path: str | Path | None,
        collection_name: str,
        cache_ttl_seconds: int,
        redis_client: RedisClientProtocol | None,
        chroma_client: ClientAPI | None,
        embedding_function: Any | None,
        repository: PromptRepository | None,
        category_definitions: Sequence[PromptCategory] | None,
        embedding_provider: EmbeddingProvider | None,
        embedding_worker: EmbeddingSyncWorker | NullEmbeddingWorker | None,
        enable_background_sync: bool,
        notification_center: NotificationCenter | None,
    ) -> None:
        """Wire repositories, Chroma collection, and embedding background worker."""
        self._closed = False
        self._cache_ttl_seconds = cache_ttl_seconds
        self._redis_client = redis_client
        self._redis_unavailable_reason = None

        resolved_db_path = Path(db_path).expanduser() if db_path is not None else None
        resolved_chroma_path = Path(chroma_path).expanduser()
        self._db_path = resolved_db_path
        self._chroma_path = str(resolved_chroma_path)
        self._collection_name = collection_name
        self._embedding_function = embedding_function
        self._logs_path = Path("data") / "logs"

        if repository is None:
            manager_module = import_module("core.prompt_manager")
            repo_cls = cast("type[PromptRepository]", manager_module.PromptRepository)
            try:
                repo_instance = repo_cls(str(resolved_db_path))
            except RepositoryError as exc:
                raise PromptStorageError("Unable to initialise SQLite repository") from exc
        else:
            repo_instance = repository

        self._repository = repo_instance

        self._category_registry = CategoryRegistry(self._repository, category_definitions)

        client, collection = build_chroma_client(
            self._chroma_path,
            self._collection_name,
            embedding_function,
            chroma_client=chroma_client,
        )
        self._chroma_client = client
        self._collection = collection

        self._notification_center = notification_center or default_notification_center

        self._embedding_provider = embedding_provider or EmbeddingProvider(embedding_function)
        if enable_background_sync:
            worker_logger = logger.getChild("embedding_sync")
            self._embedding_worker = embedding_worker or EmbeddingSyncWorker(
                provider=self._embedding_provider,
                fetch_prompt=self._repository.get,
                persist_callback=self._persist_embedding_from_worker,
                notification_center=self._notification_center,
                logger=worker_logger,
            )
        else:
            self._embedding_worker = embedding_worker or NullEmbeddingWorker()
