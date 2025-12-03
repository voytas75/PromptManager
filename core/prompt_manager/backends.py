"""Backend bootstrap and protocol helpers for Prompt Manager.

Updates:
  v0.1.0 - 2025-12-03 - Extract Redis/Chroma bootstrap helpers from package init.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Protocol, cast
from uuid import UUID

import chromadb
from chromadb.errors import ChromaError

from ..exceptions import PromptStorageError

if TYPE_CHECKING:  # pragma: no cover - typing only
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

    def ping(self) -> bool: ...

    def dbsize(self) -> int: ...

    def info(self) -> Mapping[str, Any]: ...

    def get(self, name: str) -> RedisValue | None: ...

    def setex(self, name: str, time: int, value: RedisValue) -> bool: ...

    def delete(self, *names: str) -> int: ...

    def close(self) -> None: ...


class CollectionProtocol(Protocol):
    """Minimal Chroma collection surface consumed by the manager."""
    def count(self) -> int: ...

    def delete(self, **kwargs: Any) -> Any: ...

    def upsert(self, **kwargs: Any) -> Any: ...

    def query(self, **kwargs: Any) -> Mapping[str, Any]: ...

    def peek(self, **kwargs: Any) -> Any: ...


class NullEmbeddingWorker:
    """Embedding worker placeholder used when background sync is disabled."""
    def schedule(self, _: UUID) -> None:  # pragma: no cover - trivial noop
        return

    def stop(self) -> None:  # pragma: no cover - trivial noop
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
