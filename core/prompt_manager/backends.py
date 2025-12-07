"""Backend bootstrap and protocol helpers for Prompt Manager.

Updates:
  v0.1.0 - 2025-12-03 - Extract Redis/Chroma bootstrap helpers from package init.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import chromadb
from chromadb.errors import ChromaError

from ..exceptions import PromptStorageError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping
    from uuid import UUID

    from chromadb.api import ClientAPI

logger = logging.getLogger(__name__)

os.environ.setdefault("CHROMA_TELEMETRY_IMPLEMENTATION", "none")
os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "0")

RedisValue = str | bytes | memoryview

__all__ = [
    "CollectionProtocol",
    "RedisClientProtocol",
    "RedisConnectionPoolProtocol",
    "RedisValue",
    "NullEmbeddingWorker",
    "build_chroma_client",
    "mute_posthog_capture",
]


class RedisConnectionPoolProtocol(Protocol):
    """Subset of redis-py connection pool used for diagnostics."""

    connection_kwargs: Mapping[str, Any]

    def disconnect(self) -> None:
        """Close all pooled connections."""


class RedisClientProtocol(Protocol):
    """Subset of redis-py client behaviour used within the manager."""

    connection_pool: RedisConnectionPoolProtocol | None

    def ping(self) -> bool:
        """Return True if the Redis server responds."""
        ...

    def dbsize(self) -> int:
        """Return the number of keys stored in Redis."""
        ...

    def info(self) -> Mapping[str, Any]:
        """Return diagnostic information from Redis."""
        ...

    def get(self, name: str) -> RedisValue | None:
        """Return a cached value when present."""
        ...

    def setex(self, name: str, time: int, value: RedisValue) -> bool:
        """Store a value with the specified TTL."""
        ...

    def delete(self, *names: str) -> int:
        """Remove one or more cache entries."""
        ...

    def close(self) -> None:
        """Release client resources."""
        ...


class CollectionProtocol(Protocol):
    """Minimal Chroma collection surface consumed by the manager."""

    def count(self) -> int:
        """Return the number of stored embeddings."""
        ...

    def delete(self, **kwargs: Any) -> Any:
        """Remove embeddings matching the supplied filters."""
        ...

    def upsert(self, **kwargs: Any) -> Any:
        """Insert or update embeddings."""
        ...

    def query(self, **kwargs: Any) -> Mapping[str, Any]:
        """Run a similarity query against the collection."""
        ...

    def peek(self, **kwargs: Any) -> Any:
        """Inspect raw collection entries."""
        ...


class NullEmbeddingWorker:
    """Embedding worker placeholder used when background sync is disabled."""

    def schedule(self, _: UUID) -> None:  # pragma: no cover - trivial noop
        """Ignore schedule requests when sync is disabled."""
        return

    def stop(self) -> None:  # pragma: no cover - trivial noop
        """No-op stop hook for interface parity."""
        return


def mute_posthog_capture() -> None:
    """Disable PostHog telemetry to avoid noisy logs in local environments."""
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


def build_chroma_client(
    chroma_path: str | Path,
    collection_name: str,
    embedding_function: Any | None = None,
    *,
    chroma_client: ClientAPI | None = None,
) -> tuple[Any, CollectionProtocol]:
    """Return a Chroma client and initialised collection located at chroma_path."""
    resolved_path = Path(chroma_path).expanduser()
    resolved_path.mkdir(parents=True, exist_ok=True)

    if chroma_client is None:
        chroma_settings: Any | None = None
        try:
            from chromadb.config import Settings as ChromaSettings

            chroma_settings = ChromaSettings(
                anonymized_telemetry=False,
                is_persistent=True,
                persist_directory=str(resolved_path),
            )
        except Exception:  # pragma: no cover - optional dependency guard
            chroma_settings = None

        persistent_kwargs: dict[str, Any] = {"path": str(resolved_path)}
        if chroma_settings is not None:
            persistent_kwargs["settings"] = chroma_settings

        try:
            try:
                client = chromadb.PersistentClient(**persistent_kwargs)
            except TypeError as exc:
                if "unexpected keyword argument 'settings'" not in str(exc):
                    raise
                persistent_kwargs.pop("settings", None)
                client = chromadb.PersistentClient(**persistent_kwargs)
        except ChromaError as exc:
            logger.warning(
                "Chroma persistent client unavailable, using in-memory fallback: %s",
                exc,
            )
            client = chromadb.EphemeralClient()
    else:
        client = chroma_client

    try:
        collection = cast(
            "CollectionProtocol",
            client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=embedding_function,
            ),
        )
    except ChromaError as exc:  # pragma: no cover - exercised via higher-level tests
        raise PromptStorageError("Unable to initialise ChromaDB collection") from exc

    return client, collection
