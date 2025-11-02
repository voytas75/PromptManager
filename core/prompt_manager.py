"""High-level CRUD manager for prompt records backed by SQLite, ChromaDB, and Redis.

Updates: v0.13.2 - 2025-11-16 - Add task template CRUD APIs and apply workflow.
Updates: v0.12.0 - 2025-11-15 - Add LiteLLM-backed prompt engineering workflow.
Updates: v0.11.0 - 2025-11-12 - Add multi-turn chat execution with conversation history logging.
Updates: v0.10.0 - 2025-11-11 - Emit notifications for managed LLM and embedding tasks.
Updates: v0.9.0 - 2025-11-11 - Track single-user preferences and personalise intent suggestions.
Updates: v0.8.0 - 2025-11-09 - Capture prompt ratings from executions and update quality scores.
Updates: v0.7.0 - 2025-11-08 - Integrate prompt execution pipeline with history logging.
Updates: v0.6.0 - 2025-11-06 - Add intent-aware search suggestions and ranking helpers.
Updates: v0.5.0 - 2025-11-05 - Integrate LiteLLM name generator and catalogue seeding.
Updates: v0.4.1 - 2025-11-05 - Add lifecycle shutdown hooks and mute Chroma telemetry.
Updates: v0.4.0 - 2025-11-05 - Add embedding provider with background sync and retry handling.
Updates: v0.3.0 - 2025-11-03 - Require explicit DB path; accept resolved settings inputs.
Updates: v0.2.0 - 2025-10-31 - Add SQLite repository integration with ChromaDB/Redis sync.
Updates: v0.1.0 - 2025-10-30 - Initial PromptManager with CRUD and search support.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union, cast

import json
import logging
import os
import uuid

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


def _normalise_conversation(
    messages: Optional[Sequence[Mapping[str, str]]],
) -> List[Dict[str, str]]:
    """Return a sanitised copy of conversation messages for execution and logging."""

    normalised: List[Dict[str, str]] = []
    if not messages:
        return normalised
    for index, message in enumerate(messages):
        role = str(message.get("role", "")).strip()
        if not role:
            raise PromptExecutionError(f"Conversation entry {index} is missing a role.")
        content = message.get("content")
        if content is None:
            raise PromptExecutionError(f"Conversation entry {index} is missing content.")
        normalised.append({"role": role, "content": str(content)})
    return normalised

try:
    import redis
    from redis.exceptions import RedisError
except ImportError:  # pragma: no cover - redis optional during development
    redis = None  # type: ignore
    RedisError = Exception  # type: ignore[misc,assignment]

from models.prompt_model import Prompt, PromptExecution, TaskTemplate, UserProfile

from .embedding import EmbeddingGenerationError, EmbeddingProvider, EmbeddingSyncWorker
from .execution import CodexExecutionResult, CodexExecutor, ExecutionError
from .history_tracker import HistoryTracker, HistoryTrackerError
from .intent_classifier import IntentClassifier, IntentPrediction, rank_by_hints
from .name_generation import (
    LiteLLMDescriptionGenerator,
    LiteLLMNameGenerator,
    DescriptionGenerationError,
    NameGenerationError,
)
from .prompt_engineering import PromptEngineer, PromptEngineeringError, PromptRefinement
from .repository import PromptRepository, RepositoryError, RepositoryNotFoundError
from .notifications import (
    NotificationCenter,
    NotificationLevel,
    notification_center as default_notification_center,
)

logger = logging.getLogger(__name__)


class PromptManagerError(Exception):
    """Base exception for PromptManager failures."""


class PromptNotFoundError(PromptManagerError):
    """Raised when a prompt cannot be located in the backing store."""


class PromptExecutionUnavailable(PromptManagerError):
    """Raised when prompt execution is not configured for the manager."""


class PromptExecutionError(PromptManagerError):
    """Raised when executing a prompt via LiteLLM fails."""


class PromptHistoryError(PromptManagerError):
    """Raised when manual history operations fail."""


class PromptStorageError(PromptManagerError):
    """Raised when interactions with persistent backends fail."""


class PromptCacheError(PromptManagerError):
    """Raised when Redis cache lookups or writes fail."""


class PromptEngineeringUnavailable(PromptManagerError):
    """Raised when prompt refinement is requested without an engineer configured."""


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
        name_generator: Optional[LiteLLMNameGenerator] = None,
        description_generator: Optional[LiteLLMDescriptionGenerator] = None,
        prompt_engineer: Optional[PromptEngineer] = None,
        intent_classifier: Optional[IntentClassifier] = None,
        notification_center: Optional[NotificationCenter] = None,
        executor: Optional[CodexExecutor] = None,
        user_profile: Optional[UserProfile] = None,
        history_tracker: Optional[HistoryTracker] = None,
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
            notification_center: Optional NotificationCenter override for task events.
            executor: Optional CodexExecutor responsible for running prompts.
            history_tracker: Optional execution history tracker for persistence.
            user_profile: Optional profile to seed single-user personalisation state.
            prompt_engineer: Optional prompt refinement helper using LiteLLM.
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
            self._embedding_worker = embedding_worker or _NullEmbeddingWorker()
        self._name_generator = name_generator
        self._description_generator = description_generator
        self._prompt_engineer = prompt_engineer
        self._intent_classifier = intent_classifier
        self._executor = executor
        self._history_tracker = history_tracker
        if user_profile is not None:
            self._user_profile: Optional[UserProfile] = user_profile
        else:
            try:
                self._user_profile = self._repository.get_user_profile()
            except RepositoryError:
                logger.warning("Unable to load persisted user profile", exc_info=True)
                self._user_profile = None

    @dataclass(slots=True)
    class IntentSuggestions:
        """Intent-aware search recommendations returned to callers."""

        prediction: IntentPrediction
        prompts: List[Prompt]
        fallback_used: bool = False

    @dataclass(slots=True)
    class ExecutionOutcome:
        """Aggregate data returned after executing a prompt."""

        result: CodexExecutionResult
        history_entry: Optional[PromptExecution]
        conversation: List[Dict[str, str]]

    @dataclass(slots=True)
    class TemplateApplication:
        """Result returned when applying a task template."""

        template: TaskTemplate
        prompts: List[Prompt]

    @property
    def collection(self) -> Collection:
        """Expose the underlying Chroma collection."""
        return cast(Collection, self._collection)

    @property
    def repository(self) -> PromptRepository:
        """Expose the SQLite repository."""
        return self._repository

    @property
    def intent_classifier(self) -> Optional[IntentClassifier]:
        """Expose the configured intent classifier for tooling hooks."""

        return self._intent_classifier

    @property
    def executor(self) -> Optional[CodexExecutor]:
        """Return the configured prompt executor."""

        return self._executor

    def set_executor(self, executor: Optional[CodexExecutor]) -> None:
        """Assign or replace the Codex executor at runtime."""

        self._executor = executor

    @property
    def history_tracker(self) -> Optional[HistoryTracker]:
        """Expose the execution history tracker if configured."""

        return self._history_tracker

    def set_history_tracker(self, tracker: Optional[HistoryTracker]) -> None:
        """Assign or replace the history tracker at runtime."""

        self._history_tracker = tracker

    @property
    def notification_center(self) -> NotificationCenter:
        """Expose the notification centre for UI and tooling integrations."""

        return self._notification_center

    @property
    def user_profile(self) -> Optional[UserProfile]:
        """Return the stored single-user profile when available."""

        return self._user_profile

    @property
    def prompt_engineer(self) -> Optional[PromptEngineer]:
        """Return the configured prompt engineering helper, if any."""

        return self._prompt_engineer

    def refresh_user_profile(self) -> Optional[UserProfile]:
        """Reload the persisted profile from the repository."""

        try:
            self._user_profile = self._repository.get_user_profile()
        except RepositoryError:
            logger.warning("Unable to refresh user profile from storage", exc_info=True)
        return self._user_profile

    def generate_prompt_name(self, context: str) -> str:
        """Return a prompt name using the configured LiteLLM generator."""
        if self._name_generator is None:
            raise NameGenerationError(
                "LiteLLM name generator is not configured. Set PROMPT_MANAGER_LITELLM_MODEL."
            )
        task_id = f"name-gen:{uuid.uuid4()}"
        metadata = {"context_length": len(context or "")}
        with self._notification_center.track_task(
            title="Prompt name generation",
            task_id=task_id,
            start_message="Generating prompt name via LiteLLM…",
            success_message="Prompt name generated.",
            failure_message="Prompt name generation failed",
            metadata=metadata,
            level=NotificationLevel.INFO,
        ):
            try:
                suggestion = self._name_generator.generate(context)
            except Exception as exc:
                raise NameGenerationError(str(exc)) from exc
        return suggestion

    def generate_prompt_description(self, context: str) -> str:
        """Return a prompt description using the configured LiteLLM generator."""

        if self._description_generator is None:
            raise DescriptionGenerationError(
                "LiteLLM description generator is not configured. Set PROMPT_MANAGER_LITELLM_MODEL."
            )
        task_id = f"description-gen:{uuid.uuid4()}"
        metadata = {"context_length": len(context or "")}
        with self._notification_center.track_task(
            title="Prompt description generation",
            task_id=task_id,
            start_message="Generating prompt description via LiteLLM…",
            success_message="Prompt description generated.",
            failure_message="Prompt description generation failed",
            metadata=metadata,
            level=NotificationLevel.INFO,
        ):
            try:
                summary = self._description_generator.generate(context)
            except Exception as exc:
                raise DescriptionGenerationError(str(exc)) from exc
        return summary

    def refine_prompt_text(
        self,
        prompt_text: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        negative_constraints: Optional[Sequence[str]] = None,
    ) -> PromptRefinement:
        """Improve a prompt using the configured prompt engineer."""

        if not prompt_text.strip():
            raise PromptEngineeringError("Prompt refinement requires non-empty prompt text.")
        if self._prompt_engineer is None:
            raise PromptEngineeringUnavailable(
                "Prompt engineering is not configured. Set PROMPT_MANAGER_LITELLM_MODEL to enable refinement."
            )
        task_id = f"prompt-refine:{uuid.uuid4()}"
        metadata = {
            "prompt_length": len(prompt_text or ""),
            "has_name": bool(name),
            "tag_count": len(tags or []),
        }
        with self._notification_center.track_task(
            title="Prompt refinement",
            task_id=task_id,
            start_message="Analysing prompt via LiteLLM…",
            success_message="Prompt refined.",
            failure_message="Prompt refinement failed",
            metadata=metadata,
            level=NotificationLevel.INFO,
        ):
            try:
                return self._prompt_engineer.refine(
                    prompt_text,
                    name=name,
                    description=description,
                    category=category,
                    tags=tags,
                    negative_constraints=negative_constraints,
                )
            except PromptEngineeringError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                raise PromptEngineeringError("Prompt refinement failed unexpectedly.") from exc

    def execute_prompt(
        self,
        prompt_id: uuid.UUID,
        request_text: str,
        *,
        conversation: Optional[Sequence[Mapping[str, str]]] = None,
    ) -> "PromptManager.ExecutionOutcome":
        """Execute a prompt via LiteLLM and persist the outcome when configured."""

        if not request_text.strip():
            raise PromptExecutionError("Prompt execution requires non-empty input text.")
        if self._executor is None:
            raise PromptExecutionUnavailable(
                "Prompt execution is not configured. Provide LiteLLM credentials and model."
            )

        conversation_history = _normalise_conversation(conversation)
        prompt = self.get_prompt(prompt_id)
        task_id = f"prompt-exec:{prompt.id}:{uuid.uuid4()}"
        metadata = {
            "prompt_id": str(prompt.id),
            "prompt_name": prompt.name,
            "request_length": len(request_text or ""),
        }
        with self._notification_center.track_task(
            title="Prompt execution",
            task_id=task_id,
            start_message=f"Running '{prompt.name}' via LiteLLM…",
            success_message=f"Completed '{prompt.name}'.",
            failure_message=f"Prompt execution failed for '{prompt.name}'",
            metadata=metadata,
            level=NotificationLevel.INFO,
        ):
            try:
                result = self._executor.execute(
                    prompt,
                    request_text,
                    conversation=conversation_history,
                )
            except ExecutionError as exc:
                failed_messages = list(conversation_history)
                failed_messages.append({"role": "user", "content": request_text.strip()})
                self._log_execution_failure(
                    prompt.id,
                    request_text,
                    str(exc),
                    conversation=failed_messages,
                )
                raise PromptExecutionError(str(exc)) from exc

            augmented_conversation = list(conversation_history)
            augmented_conversation.append({"role": "user", "content": request_text.strip()})
            if result.response_text:
                augmented_conversation.append(
                    {"role": "assistant", "content": result.response_text}
                )
            history_entry = self._log_execution_success(
                prompt.id,
                request_text,
                result,
                conversation=augmented_conversation,
            )
            try:
                self.increment_usage(prompt.id)
            except PromptManagerError:
                logger.debug(
                    "Prompt executed but usage counter update failed",
                    extra={"prompt_id": str(prompt.id)},
                    exc_info=True,
                )

        return PromptManager.ExecutionOutcome(
            result=result,
            history_entry=history_entry,
            conversation=augmented_conversation,
        )

    def save_execution_result(
        self,
        prompt_id: uuid.UUID,
        request_text: str,
        response_text: str,
        *,
        duration_ms: Optional[int] = None,
        usage: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        rating: Optional[float] = None,
    ) -> PromptExecution:
        """Persist a manual prompt execution entry (e.g., from GUI Save Result)."""

        tracker = self._history_tracker
        if tracker is None:
            raise PromptExecutionUnavailable(
                "Execution history is not configured; cannot save results manually."
            )
        payload_metadata: Dict[str, Any] = {"manual": True}
        if metadata:
            payload_metadata.update(dict(metadata))
        if usage:
            payload_metadata.setdefault("usage", dict(usage))
        if rating is not None:
            payload_metadata["rating"] = rating
        try:
            execution = tracker.record_success(
                prompt_id=prompt_id,
                request_text=request_text,
                response_text=response_text,
                duration_ms=duration_ms,
                metadata=payload_metadata,
                rating=rating,
            )
        except HistoryTrackerError as exc:
            raise PromptHistoryError(str(exc)) from exc
        if rating is not None:
            self._apply_rating(prompt_id, rating)
        return execution

    def update_execution_note(self, execution_id: uuid.UUID, note: Optional[str]) -> PromptExecution:
        """Update the note metadata for a history entry."""

        tracker = self._history_tracker
        if tracker is None:
            raise PromptExecutionUnavailable(
                "Execution history is not configured; cannot update saved results."
            )
        try:
            return tracker.update_note(execution_id, note)
        except HistoryTrackerError as exc:
            raise PromptHistoryError(str(exc)) from exc

    def list_recent_executions(self, *, limit: int = 20) -> List[PromptExecution]:
        """Return recently logged executions if history tracking is enabled."""

        tracker = self._history_tracker
        if tracker is None:
            return []
        try:
            return tracker.list_recent(limit=limit)
        except HistoryTrackerError:
            logger.warning("Unable to list execution history", exc_info=True)
            return []

    def list_executions_for_prompt(
        self,
        prompt_id: uuid.UUID,
        *,
        limit: int = 20,
    ) -> List[PromptExecution]:
        """Return execution history for a specific prompt."""

        tracker = self._history_tracker
        if tracker is None:
            return []
        try:
            return tracker.list_for_prompt(prompt_id, limit=limit)
        except HistoryTrackerError:
            logger.warning(
                "Unable to list execution history for prompt",
                extra={"prompt_id": str(prompt_id)},
                exc_info=True,
            )
            return []

    def query_executions(
        self,
        *,
        status: Optional[str] = None,
        prompt_id: Optional[uuid.UUID] = None,
        search: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[PromptExecution]:
        """Return executions filtered by status, prompt, and search term."""

        tracker = self._history_tracker
        if tracker is None:
            return []
        try:
            return tracker.query_executions(
                status=status,
                prompt_id=prompt_id,
                search=search,
                limit=limit,
            )
        except HistoryTrackerError:
            logger.warning("Unable to query execution history", exc_info=True)
            return []

    def set_name_generator(
        self,
        model: Optional[str],
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str],
    ) -> None:
        """Configure the LiteLLM name generator at runtime."""
        if not model:
            self._name_generator = None
            self._description_generator = None
            self._prompt_engineer = None
            return
        try:
            self._name_generator = LiteLLMNameGenerator(
                model=model,
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
            )
            self._description_generator = LiteLLMDescriptionGenerator(
                model=model,
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
            )
            self._prompt_engineer = PromptEngineer(
                model=model,
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
            )
        except RuntimeError as exc:
            raise NameGenerationError(str(exc)) from exc
        if self._intent_classifier is not None and model:
            # Name generation and intent classification may share LiteLLM routing;
            # future integrations can configure richer classification when desired.
            logger.debug("LiteLLM powered features enabled for intent classifier")

    def _log_execution_success(
        self,
        prompt_id: uuid.UUID,
        request_text: str,
        result: CodexExecutionResult,
        *,
        conversation: Optional[Sequence[Mapping[str, str]]] = None,
    ) -> Optional[PromptExecution]:
        """Persist a successful execution outcome when the tracker is available."""

        tracker = self._history_tracker
        if tracker is None:
            return None
        usage_metadata: Dict[str, Any] = {}
        try:
            usage_metadata = dict(result.usage)
        except Exception:  # pragma: no cover - defensive
            usage_metadata = {}
        metadata: Dict[str, Any] = {"usage": usage_metadata}
        if conversation:
            metadata["conversation"] = list(conversation)
        try:
            return tracker.record_success(
                prompt_id=prompt_id,
                request_text=request_text,
                response_text=result.response_text,
                duration_ms=result.duration_ms,
                metadata=metadata,
            )
        except HistoryTrackerError:
            logger.warning(
                "Prompt executed but history logging failed",
                extra={"prompt_id": str(prompt_id)},
                exc_info=True,
            )
            return None

    def _log_execution_failure(
        self,
        prompt_id: uuid.UUID,
        request_text: str,
        error_message: str,
        *,
        conversation: Optional[Sequence[Mapping[str, str]]] = None,
    ) -> Optional[PromptExecution]:
        """Persist a failed execution attempt when history tracking is enabled."""

        tracker = self._history_tracker
        if tracker is None:
            return None
        metadata: Optional[Mapping[str, Any]] = None
        if conversation:
            metadata = {"conversation": list(conversation)}
        try:
            return tracker.record_failure(
                prompt_id=prompt_id,
                request_text=request_text,
                error_message=error_message,
                metadata=metadata,
            )
        except HistoryTrackerError:
            logger.warning(
                "Prompt execution failed and could not be logged",
                extra={"prompt_id": str(prompt_id)},
                exc_info=True,
            )
            return None

    def set_intent_classifier(self, classifier: Optional[IntentClassifier]) -> None:
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

    # Template management ------------------------------------------------ #

    def list_templates(self, *, include_inactive: bool = False) -> List[TaskTemplate]:
        """Return available task templates."""

        try:
            return self._repository.list_templates(include_inactive=include_inactive)
        except RepositoryError as exc:
            raise PromptStorageError("Unable to list task templates") from exc

    def get_template(self, template_id: uuid.UUID) -> TaskTemplate:
        """Return a single task template."""

        try:
            return self._repository.get_template(template_id)
        except RepositoryNotFoundError as exc:
            raise PromptNotFoundError(str(exc)) from exc
        except RepositoryError as exc:
            raise PromptStorageError(f"Unable to load template {template_id}") from exc

    def create_template(self, template: TaskTemplate) -> TaskTemplate:
        """Persist a new task template."""

        template.touch()
        try:
            stored = self._repository.add_template(template)
        except RepositoryError as exc:
            raise PromptStorageError(f"Failed to persist template {template.id}") from exc
        return stored

    def update_template(self, template: TaskTemplate) -> TaskTemplate:
        """Persist updates to an existing task template."""

        template.touch()
        try:
            updated = self._repository.update_template(template)
        except RepositoryNotFoundError as exc:
            raise PromptNotFoundError(str(exc)) from exc
        except RepositoryError as exc:
            raise PromptStorageError(f"Failed to update template {template.id}") from exc
        return updated

    def delete_template(self, template_id: uuid.UUID) -> None:
        """Delete a task template."""

        try:
            self._repository.delete_template(template_id)
        except RepositoryNotFoundError as exc:
            raise PromptNotFoundError(str(exc)) from exc
        except RepositoryError as exc:
            raise PromptStorageError(f"Failed to delete template {template_id}") from exc

    def apply_template(self, template_id: uuid.UUID) -> "PromptManager.TemplateApplication":
        """Return template metadata and associated prompts for task workflows."""

        template = self.get_template(template_id)
        try:
            prompts = self._repository.get_prompts_for_ids(template.prompt_ids)
        except RepositoryError as exc:
            raise PromptStorageError("Unable to load prompts for template") from exc
        missing = len(template.prompt_ids) - len(prompts)
        if missing > 0:
            logger.warning(
                "Template references missing prompts",
                extra={
                    "template_id": str(template.id),
                    "expected": len(template.prompt_ids),
                    "resolved": len(prompts),
                },
            )
        return PromptManager.TemplateApplication(template=template, prompts=prompts)

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

    def suggest_prompts(self, query_text: str, *, limit: int = 5) -> "PromptManager.IntentSuggestions":
        """Return intent-ranked prompt recommendations for the supplied query."""

        if limit <= 0:
            raise ValueError("limit must be a positive integer")

        stripped = query_text.strip()
        if not stripped:
            try:
                baseline = self.repository.list(limit=limit)
            except RepositoryError as exc:
                raise PromptStorageError("Unable to load prompts for suggestions") from exc
            personalised = self._personalize_ranked_prompts(baseline)
            return PromptManager.IntentSuggestions(
                IntentPrediction.general(), personalised[:limit], fallback_used=True
            )

        prediction = (
            self._intent_classifier.classify(stripped)
            if self._intent_classifier is not None
            else IntentPrediction.general()
        )

        augmented_query_parts = [stripped]
        if prediction.category_hints:
            augmented_query_parts.append(
                "Intent categories: " + ", ".join(prediction.category_hints)
            )
        if prediction.tag_hints:
            augmented_query_parts.append("Intent tags: " + ", ".join(prediction.tag_hints))
        augmented_query = "\n".join(augmented_query_parts)

        suggestions: List[Prompt] = []
        fallback_used = False
        try:
            raw_results = self.search_prompts(augmented_query, limit=max(limit * 2, 10))
        except PromptManagerError:
            raw_results = []

        ranked = rank_by_hints(
            raw_results,
            category_hints=prediction.category_hints,
            tag_hints=prediction.tag_hints,
        )
        ranked = self._personalize_ranked_prompts(ranked)

        seen_ids: set[uuid.UUID] = set()
        for prompt in ranked:
            if prompt.id in seen_ids:
                continue
            suggestions.append(prompt)
            seen_ids.add(prompt.id)
            if len(suggestions) >= limit:
                break

        if len(suggestions) < limit:
            fallback_used = True
            try:
                fallback_results = self.search_prompts(stripped, limit=max(limit * 2, 10))
            except PromptManagerError:
                fallback_results = []
            else:
                fallback_results = self._personalize_ranked_prompts(fallback_results)
            for prompt in fallback_results:
                if prompt.id in seen_ids:
                    continue
                suggestions.append(prompt)
                seen_ids.add(prompt.id)
                if len(suggestions) >= limit:
                    break

        if not suggestions:
            fallback_used = True
            try:
                suggestions = self.repository.list(limit=limit)
            except RepositoryError as exc:
                raise PromptStorageError("Unable to load prompts for suggestions") from exc

        personalised = self._personalize_ranked_prompts(suggestions)
        return PromptManager.IntentSuggestions(
            prediction=prediction,
            prompts=personalised[:limit],
            fallback_used=fallback_used,
        )

    def _personalize_ranked_prompts(self, prompts: Sequence[Prompt]) -> List[Prompt]:
        """Bias prompt order using stored user preferences while preserving stability."""

        if not prompts:
            return []
        profile = self._user_profile
        if profile is None:
            return list(prompts)

        favorite_categories = profile.favorite_categories(limit=5)
        favorite_tags = profile.favorite_tags(limit=8)
        if not favorite_categories and not favorite_tags:
            return list(prompts)

        category_weights = {name: (len(favorite_categories) - idx) * 2 for idx, name in enumerate(favorite_categories)}
        tag_weights = {name: len(favorite_tags) - idx for idx, name in enumerate(favorite_tags)}

        scored: List[tuple[float, int, Prompt]] = []
        for index, prompt in enumerate(prompts):
            score = 0.0
            category = (prompt.category or "").strip()
            if category in category_weights:
                score += float(category_weights[category])
            for tag in prompt.tags or []:
                weight = tag_weights.get(tag)
                if weight:
                    score += float(weight)
            scored.append((score, index, prompt))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [prompt for _, _, prompt in scored]

    def increment_usage(self, prompt_id: uuid.UUID) -> None:
        """Increment usage counter for a prompt."""
        prompt = self.get_prompt(prompt_id)
        prompt.usage_count += 1
        self.update_prompt(prompt)
        self._record_prompt_usage(prompt)

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

    def _apply_rating(self, prompt_id: uuid.UUID, rating: float) -> None:
        """Update prompt aggregates from a new rating."""
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
        prompt.last_modified = datetime.now(timezone.utc)

        try:
            self.update_prompt(prompt)
        except PromptManagerError:
            logger.warning(
                "Unable to persist rating update for prompt",
                extra={"prompt_id": str(prompt_id)},
                exc_info=True,
            )

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
    "PromptExecutionUnavailable",
    "PromptExecutionError",
    "PromptEngineeringUnavailable",
]
