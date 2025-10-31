"""Embedding provider and background synchronisation helpers for Prompt Manager.

Updates: v0.1.0 - 2025-11-05 - Introduce embedding provider with retry logic and sync worker.
"""

from __future__ import annotations

import hashlib
import logging
import queue
import threading
import time
import uuid
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from models.prompt_model import Prompt


EmbeddingFunction = Callable[[Sequence[str]], Sequence[Sequence[float]]]


class EmbeddingProviderError(Exception):
    """Base exception for embedding provider failures."""


class EmbeddingGenerationError(EmbeddingProviderError):
    """Raised when generating embeddings fails after retries."""


class DefaultEmbeddingFunction:
    """Deterministic, lightweight embedding fallback.

    The implementation hashes the input text and produces a 32-dimensional
    vector in the range [0.0, 1.0]. It serves as a placeholder until the
    application is wired to a real embedding backend.
    """

    _VECTOR_LENGTH = 32

    def __call__(self, texts: Sequence[str]) -> List[List[float]]:
        results: List[List[float]] = []
        for text in texts:
            digest = hashlib.blake2b(text.encode("utf-8"), digest_size=self._VECTOR_LENGTH).digest()
            vector = [byte / 255.0 for byte in digest]
            results.append(vector)
        return results


class EmbeddingProvider:
    """Generate embeddings with retry logic and pluggable backends."""

    def __init__(
        self,
        embedding_function: Optional[EmbeddingFunction] = None,
        *,
        max_retries: int = 3,
        retry_delay_seconds: float = 0.3,
    ) -> None:
        self._embedding_function: EmbeddingFunction = embedding_function or DefaultEmbeddingFunction()
        self._max_retries = max(1, max_retries)
        self._retry_delay_seconds = max(0.0, retry_delay_seconds)

    def embed(self, text: str) -> List[float]:
        """Return an embedding vector for the supplied text."""

        last_error: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                vectors = self._embedding_function([text])
                if not vectors or not isinstance(vectors[0], (list, tuple)):
                    message = "Embedding function returned invalid payload"
                    raise EmbeddingGenerationError(message)
                return [float(value) for value in vectors[0]]
            except Exception as exc:  # noqa: BLE001 - propagate after retries
                last_error = exc
                if attempt < self._max_retries:
                    time.sleep(self._retry_delay_seconds * attempt)
        raise EmbeddingGenerationError("Unable to generate embedding") from last_error


class EmbeddingSyncWorker:
    """Background worker that keeps prompt embeddings in sync."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        fetch_prompt: Callable[[uuid.UUID], Prompt],
        persist_callback: Callable[[Prompt, Sequence[float]], None],
        *,
        logger: Optional[logging.Logger] = None,
        max_attempts: int = 3,
        retry_delay_seconds: float = 0.5,
    ) -> None:
        self._provider = provider
        self._fetch_prompt = fetch_prompt
        self._persist_callback = persist_callback
        self._logger = logger or logging.getLogger("prompt_manager.embedding_worker")
        self._max_attempts = max(1, max_attempts)
        self._retry_delay_seconds = max(0.0, retry_delay_seconds)
        self._queue: "queue.Queue[Tuple[uuid.UUID, int]]" = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="prompt-embedding-sync", daemon=True)
        self._thread.start()

    def schedule(self, prompt_id: uuid.UUID) -> None:
        """Queue a prompt identifier for embedding synchronisation."""

        self._queue.put((prompt_id, 0))

    def stop(self) -> None:
        """Signal the background thread to stop and drain the queue."""

        self._stop_event.set()
        self._queue.put_nowait((uuid.uuid4(), self._max_attempts))  # sentinel to unblock
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                prompt_id, attempts = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if self._stop_event.is_set():
                    return
                self._process(prompt_id, attempts)
            finally:
                self._queue.task_done()

    def _process(self, prompt_id: uuid.UUID, attempts: int) -> None:
        try:
            prompt = self._fetch_prompt(prompt_id)
        except Exception as exc:  # noqa: BLE001 - repository surface handles logging upstream
            self._logger.debug("Unable to fetch prompt for embedding sync", exc_info=exc, extra={"prompt_id": str(prompt_id)})
            self._maybe_reschedule(prompt_id, attempts)
            return

        try:
            vector = self._provider.embed(prompt.document)
        except EmbeddingGenerationError as exc:
            self._logger.warning(
                "Embedding generation failed during background sync",
                extra={"prompt_id": str(prompt_id), "attempt": attempts + 1},
            )
            self._maybe_reschedule(prompt_id, attempts, backoff=True)
            return

        try:
            self._persist_callback(prompt, vector)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Persisting embedding failed",
                exc_info=exc,
                extra={"prompt_id": str(prompt_id), "attempt": attempts + 1},
            )
            self._maybe_reschedule(prompt_id, attempts, backoff=True)

    def _maybe_reschedule(self, prompt_id: uuid.UUID, attempts: int, *, backoff: bool = False) -> None:
        next_attempt = attempts + 1
        if next_attempt >= self._max_attempts:
            self._logger.error(
                "Exceeded embedding sync attempts",
                extra={"prompt_id": str(prompt_id), "attempts": next_attempt},
            )
            return
        if backoff and self._retry_delay_seconds:
            time.sleep(self._retry_delay_seconds * next_attempt)
        self._queue.put((prompt_id, next_attempt))


__all__ = [
    "EmbeddingProvider",
    "EmbeddingProviderError",
    "EmbeddingGenerationError",
    "EmbeddingSyncWorker",
]
