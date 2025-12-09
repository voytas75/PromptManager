"""Embedding provider and background synchronisation helpers for Prompt Manager.

Updates:
  v0.7.5 - 2025-12-09 - Expose LiteLLM embedding function name for Chroma telemetry.
  v0.7.4 - 2025-11-30 - Document embedding helpers and fix docstring spacing for lint compliance.
  v0.7.3 - 2025-11-29 - Guard Prompt import for typing and wrap logging/reporting lines.
  v0.7.2 - 2025-12-06 - Align embedding function signatures with updated Chroma client expectations.
  v0.7.1 - 2025-11-05 - Stop forwarding LiteLLM embedding timeouts by default.
  v0.7.0 - 2025-11-11 - Publish notifications for background embedding sync progress.
  v0.6.0 - 2025-11-07 - Add configurable LiteLLM and sentence-transformer backends.
  v0.1.0 - 2025-11-05 - Introduce embedding provider with retry logic and sync worker.
"""

from __future__ import annotations

import hashlib
import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

from .litellm_adapter import get_embedding
from .notifications import (
    NotificationCenter,
    NotificationLevel,
    notification_center as default_notification_center,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
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

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        """Return deterministic embedding vectors for each supplied input string."""
        results: list[list[float]] = []
        for text in input:
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
        api_key: str | None,
        api_base: str | None,
        timeout_seconds: float | None = None,
    ) -> None:
        """Store LiteLLM credentials and request options for embedding calls."""
        if not model:
            raise ValueError("LiteLLM embedding backend requires a model name.")
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._timeout_seconds = timeout_seconds

    @property
    def name(self) -> str:
        """Identifier surfaced to Chroma for telemetry/diagnostics."""
        return f"litellm:{self._model}"

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        """Return embedding vectors by invoking the configured LiteLLM backend."""
        embedding_fn, LiteLLMException = get_embedding()
        inputs = list(input)
        if not inputs:
            return []
        request: dict[str, Any] = {
            "model": self._model,
            "input": inputs,
        }
        if self._timeout_seconds is not None:
            request["timeout"] = self._timeout_seconds
        if self._api_key:
            request["api_key"] = self._api_key
        if self._api_base:
            request["api_base"] = self._api_base
        try:
            response = embedding_fn(**request)  # type: ignore[arg-type]
            payload = self._extract_payload(response)
            data = self._extract_data_array(payload)
        except LiteLLMException as exc:  # type: ignore[misc]
            raise EmbeddingGenerationError(f"LiteLLM embedding request failed: {exc}") from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise EmbeddingGenerationError("LiteLLM embedding request failed unexpectedly") from exc
        vectors: list[list[float]] = []
        for index, item in enumerate(data):
            raw_vector = self._extract_embedding_vector(item, index)
            try:
                vectors.append([float(value) for value in raw_vector])
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise EmbeddingGenerationError(
                    "LiteLLM embedding payload contains non-numeric values"
                ) from exc
        if len(vectors) == 1 and len(inputs) > 1:
            # Some providers return a single vector even for batched input.
            vectors = vectors * len(inputs)
        if len(vectors) != len(inputs):
            raise EmbeddingGenerationError("LiteLLM embedding response length mismatch.")
        return vectors

    @staticmethod
    def _extract_payload(response: Any) -> Any:
        """Return a JSON-like payload from LiteLLM responses."""
        for attr in ("model_dump", "dict", "json"):
            if hasattr(response, attr):
                attr_func = getattr(response, attr)
                if callable(attr_func):
                    try:
                        return attr_func()
                    except Exception:  # noqa: BLE001 - fall back to original payload
                        continue
        return response

    @staticmethod
    def _extract_data_array(payload: Any) -> Sequence[Any]:
        """Extract the embedding data array from LiteLLM responses."""
        data_obj: Any
        if isinstance(payload, Mapping) and "data" in payload:
            payload_map = cast("Mapping[str, Any]", payload)
            data_obj = payload_map.get("data")
        else:
            attr_source = cast("Any", payload)
            data_obj = attr_source
            if hasattr(attr_source, "data"):
                data_obj = attr_source.data
        if not isinstance(data_obj, Sequence) or isinstance(data_obj, (str, bytes)):
            raise EmbeddingGenerationError("LiteLLM embedding response missing data array.")
        return cast("Sequence[Any]", data_obj)

    @staticmethod
    def _extract_embedding_vector(item: Any, index: int) -> Sequence[Any]:
        """Return the embedding vector sequence from a data entry."""
        candidate: Any
        if isinstance(item, Mapping):
            mapping_item = cast("Mapping[str, Any]", item)
            candidate = mapping_item.get("embedding")
        elif hasattr(item, "embedding"):
            candidate = item.embedding
        elif hasattr(item, "model_dump"):
            try:
                dumped = item.model_dump()
            except Exception:  # noqa: BLE001 - fallback when model_dump fails
                dumped = None
            candidate = (
                cast("Mapping[str, Any]", dumped).get("embedding")
                if isinstance(dumped, Mapping)
                else None
            )
        else:
            candidate = item

        if candidate is None:
            raise EmbeddingGenerationError(f"LiteLLM response missing embedding at index {index}")
        if isinstance(candidate, Mapping):
            candidate = cast("Mapping[str, Any]", candidate).get("embedding")
        if not isinstance(candidate, Sequence) or isinstance(candidate, (str, bytes)):
            raise EmbeddingGenerationError("LiteLLM embedding payload is not a vector.")
        return cast("Sequence[Any]", candidate)


class SentenceTransformersEmbeddingFunction:
    """Use sentence-transformers models for local embeddings."""

    def __init__(self, model: str, *, device: str | None = None) -> None:
        """Load a sentence-transformers model for local embedding generation."""
        if not model:
            raise ValueError("sentence-transformers backend requires a model name.")
        self._model_name = model
        self._device = device
        self._model: Any = self._load_model()

    def _load_model(self) -> Any:
        try:  # pragma: no cover - runtime dependency
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is not installed") from exc
        sentence_transformer = cast("Any", SentenceTransformer)
        model: Any = sentence_transformer(self._model_name, device=self._device)
        return model

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        """Return embeddings for the supplied text batch using the loaded model."""
        inputs = list(input)
        if not inputs:
            return []
        embeddings: Any = self._model.encode(
            inputs,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()
        if not isinstance(embeddings, Sequence):
            raise EmbeddingGenerationError("sentence-transformers returned invalid embeddings")
        embedding_rows = cast("Sequence[Sequence[Any]]", embeddings)
        return [[float(value) for value in vector] for vector in embedding_rows]


def create_embedding_function(
    backend: str,
    *,
    model: str | None,
    api_key: str | None,
    api_base: str | None,
    device: str | None = None,
    timeout_seconds: float | None = None,
) -> EmbeddingFunction | None:
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
        embedding_function: EmbeddingFunction | None = None,
        *,
        max_retries: int = 3,
        retry_delay_seconds: float = 0.3,
    ) -> None:
        """Initialise the provider with the callable backend and retry strategy."""
        self._embedding_function: EmbeddingFunction = (
            embedding_function or DefaultEmbeddingFunction()
        )
        self._max_retries = max(1, max_retries)
        self._retry_delay_seconds = max(0.0, retry_delay_seconds)

    def embed(self, text: str) -> list[float]:
        """Return an embedding vector for the supplied text."""
        last_error: Exception | None = None
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
        notification_center: NotificationCenter | None = None,
        logger: logging.Logger | None = None,
        max_attempts: int = 3,
        retry_delay_seconds: float = 0.5,
    ) -> None:
        """Run a background thread that keeps prompt embeddings up to date."""
        self._provider = provider
        self._fetch_prompt = fetch_prompt
        self._persist_callback = persist_callback
        self._notification_center = notification_center or default_notification_center
        self._logger = logger or logging.getLogger("prompt_manager.embedding_worker")
        self._max_attempts = max(1, max_attempts)
        self._retry_delay_seconds = max(0.0, retry_delay_seconds)
        self._queue: queue.Queue[tuple[uuid.UUID, int]] = queue.Queue()
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
            self._logger.debug(
                "Unable to fetch prompt for embedding sync",
                exc_info=exc,
                extra={"prompt_id": str(prompt_id)},
            )
            self._maybe_reschedule(prompt_id, attempts)
            return

        label = prompt.name or str(prompt.id)
        metadata = {"prompt_id": str(prompt.id), "attempt": attempts + 1}
        task_id = f"embedding-sync:{prompt.id}:{attempts + 1}"

        try:
            with self._notification_center.track_task(
                title="Embedding synchronisation",
                task_id=task_id,
                start_message=f"Updating embedding for '{label}'…",
                success_message=f"Embedding updated for '{label}'.",
                failure_message=f"Embedding sync failed for '{label}'",
                metadata=metadata,
                level=NotificationLevel.INFO,
                failure_level=NotificationLevel.WARNING,
            ):
                vector = self._provider.embed(prompt.document)
                try:
                    self._persist_callback(prompt, vector)
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError("Persisting embedding failed") from exc
        except EmbeddingGenerationError:
            self._logger.warning(
                "Embedding generation failed during background sync",
                extra={"prompt_id": str(prompt_id), "attempt": attempts + 1},
            )
            self._maybe_reschedule(prompt_id, attempts, backoff=True)
        except RuntimeError as exc:
            self._logger.error(
                "Persisting embedding failed",
                exc_info=exc.__cause__ if exc.__cause__ is not None else exc,
                extra={"prompt_id": str(prompt_id), "attempt": attempts + 1},
            )
            self._maybe_reschedule(prompt_id, attempts, backoff=True)
        except Exception as exc:  # noqa: BLE001 - defensive catch-all
            self._logger.error(
                "Unexpected embedding sync failure",
                exc_info=exc,
                extra={"prompt_id": str(prompt_id), "attempt": attempts + 1},
            )
            self._maybe_reschedule(prompt_id, attempts, backoff=True)

    def _maybe_reschedule(
        self,
        prompt_id: uuid.UUID,
        attempts: int,
        *,
        backoff: bool = False,
    ) -> None:
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
