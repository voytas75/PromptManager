"""Embedding provider and background synchronisation helpers for Prompt Manager.

Updates: v0.6.0 - 2025-11-07 - Add configurable LiteLLM and sentence-transformer backends.
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

from .litellm_adapter import get_embedding

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


class LiteLLMEmbeddingFunction:
    """Call LiteLLM embedding endpoint for semantic vectors."""

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str],
        api_base: Optional[str],
        timeout_seconds: float = 10.0,
    ) -> None:
        if not model:
            raise ValueError("LiteLLM embedding backend requires a model name.")
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._timeout_seconds = timeout_seconds

    def __call__(self, texts: Sequence[str]) -> List[List[float]]:
        embedding_fn, LiteLLMException = get_embedding()
        inputs = list(texts)
        if not inputs:
            return []
        request = {
            "model": self._model,
            "input": inputs,
            "timeout": self._timeout_seconds,
        }
        if self._api_key:
            request["api_key"] = self._api_key
        if self._api_base:
            request["api_base"] = self._api_base
        try:
            response = embedding_fn(**request)  # type: ignore[arg-type]
        except LiteLLMException as exc:  # type: ignore[misc]
            raise EmbeddingGenerationError(f"LiteLLM embedding request failed: {exc}") from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise EmbeddingGenerationError("LiteLLM embedding request failed unexpectedly") from exc
        try:
            data = response["data"]  # type: ignore[index]
        except (KeyError, TypeError) as exc:  # pragma: no cover - defensive
            raise EmbeddingGenerationError("LiteLLM returned an unexpected embedding payload") from exc
        vectors: List[List[float]] = []
        for index, item in enumerate(data):
            raw_vector = getattr(item, "get", lambda key, default=None: None)("embedding")
            if raw_vector is None:
                raise EmbeddingGenerationError(f"LiteLLM response missing embedding at index {index}")
            if not isinstance(raw_vector, (list, tuple)):
                raise EmbeddingGenerationError("LiteLLM embedding payload is not a vector.")
            vectors.append([float(value) for value in raw_vector])
        if len(vectors) == 1 and len(inputs) > 1:
            # Some providers return a single vector even for batched input.
            vectors = vectors * len(inputs)
        if len(vectors) != len(inputs):
            raise EmbeddingGenerationError("LiteLLM embedding response length mismatch.")
        return vectors


class SentenceTransformersEmbeddingFunction:
    """Use sentence-transformers models for local embeddings."""

    def __init__(self, model: str, *, device: Optional[str] = None) -> None:
        if not model:
            raise ValueError("sentence-transformers backend requires a model name.")
        self._model_name = model
        self._device = device
        self._model = self._load_model()

    def _load_model(self):
        try:  # pragma: no cover - runtime dependency
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is not installed") from exc
        return SentenceTransformer(self._model_name, device=self._device)

    def __call__(self, texts: Sequence[str]) -> List[List[float]]:
        inputs = list(texts)
        if not inputs:
            return []
        embeddings = self._model.encode(
            inputs,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()
        return [[float(value) for value in vector] for vector in embeddings]


def create_embedding_function(
    backend: str,
    *,
    model: Optional[str],
    api_key: Optional[str],
    api_base: Optional[str],
    device: Optional[str] = None,
    timeout_seconds: float = 10.0,
) -> Optional[EmbeddingFunction]:
    """Return an embedding function for the configured backend or None for the default."""

    backend_normalised = (backend or "deterministic").strip().lower()
    if backend_normalised in {"", "deterministic", "default"}:
        return None
    if backend_normalised in {"litellm", "openai"}:
        return LiteLLMEmbeddingFunction(
            model=model or "",
            api_key=api_key,
            api_base=api_base,
            timeout_seconds=timeout_seconds,
        )
    if backend_normalised in {"sentence-transformers", "sentence_transformers", "st"}:
        return SentenceTransformersEmbeddingFunction(model or "", device=device)
    raise ValueError(f"Unsupported embedding backend: {backend}")


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
    "DefaultEmbeddingFunction",
    "LiteLLMEmbeddingFunction",
    "SentenceTransformersEmbeddingFunction",
    "create_embedding_function",
]
