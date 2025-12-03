"""Prompt Manager package façade and orchestration layer.

Updates:
  v0.14.11 - 2025-12-03 - Extract versioning, diff, and fork APIs into mixin module.
  v0.14.10 - 2025-12-03 - Move search and suggestion APIs into dedicated mixin.
  v0.14.9 - 2025-12-03 - Move execution and benchmarking APIs into mixin module.
  v0.14.8 - 2025-12-02 - Move maintenance utilities into dedicated mixin module.
  v0.14.7 - 2025-12-02 - Extract category, response style, and note APIs into mixins.
  v0.14.6 - 2025-11-30 - Ensure Chroma directories exist and stabilise Chroma client bootstrap.
  v0.14.5 - 2025-11-29 - Reformat header/imports and wrap CLI analytics summaries.
  v0.14.4 - 2025-02-14 - Add embedding diagnostics helper for CLI integration.
  v0.14.3 - 2025-11-28 - Add data snapshot helper for maintenance workflows.
  v0.14.2 - 2025-11-28 - Add benchmarks, scenario refresh APIs, and category health stats.
  v0.14.1 - 2025-11-24 - Add LiteLLM category suggestion workflow with classifier fallback.
  pre-v0.14.1 - 2025-11-30 - Consolidated history covering releases v0.1.0–v0.14.0.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3 as _sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import chromadb
from chromadb.errors import ChromaError

# ---------------------------------------------------------------------------
# NOTE: Transitional refactor
# ---------------------------
# This module is in the process of being split into a dedicated package
# `core.prompt_manager` as part of the ongoing modularisation effort
# (see AGENTS.md guidelines – KISS, DRY, maintainability).  Common exception
# classes have been migrated to ``core.exceptions`` so that they can be shared
# across sub‑modules once the split is complete.
# ---------------------------------------------------------------------------
from config import LITELLM_ROUTED_WORKFLOWS
from models.prompt_model import Prompt, PromptVersion, UserProfile
from prompt_templates import DEFAULT_PROMPT_TEMPLATES, PROMPT_TEMPLATE_KEYS

from ..category_registry import CategoryRegistry
from ..embedding import EmbeddingGenerationError, EmbeddingProvider, EmbeddingSyncWorker
from ..exceptions import (
    CategoryError,
    CategoryNotFoundError,
    CategoryStorageError,
    CategorySuggestionError,
    DescriptionGenerationError,
    PromptCacheError,
    PromptEngineeringUnavailable,
    PromptExecutionError,
    PromptExecutionUnavailable,
    PromptHistoryError,
    PromptManagerError,
    PromptNoteError,
    PromptNoteNotFoundError,
    PromptNoteStorageError,
    PromptNotFoundError,
    PromptStorageError,
    PromptVersionError,
    PromptVersionNotFoundError,
    ResponseStyleError,
    ResponseStyleNotFoundError,
    ResponseStyleStorageError,
    ScenarioGenerationError,
)
from ..execution import CodexExecutor
from ..intent_classifier import IntentClassifier
from ..name_generation import (
    LiteLLMCategoryGenerator,
    LiteLLMDescriptionGenerator,
    LiteLLMNameGenerator,
    NameGenerationError,
)
from ..notifications import (
    NotificationCenter,
    NotificationLevel,
    notification_center as default_notification_center,
)
from ..prompt_engineering import (
    PromptEngineer,
    PromptEngineeringError,
    PromptRefinement,
)
from ..repository import (
    PromptRepository,
    RepositoryError,
    RepositoryNotFoundError,
)
from ..scenario_generation import LiteLLMScenarioGenerator
from .categories import CategorySupport
from .engineering import PromptEngineerFacade
from .execution import ExecutionResult, PromptExecutor
from .execution_history import (
    BenchmarkReport as _BenchmarkReport,
    BenchmarkRun as _BenchmarkRun,
    ExecutionHistoryMixin,
    ExecutionOutcome as _ExecutionOutcome,
)
from .generation import GenerationMixin
from .maintenance import MaintenanceMixin
from .prompt_notes import PromptNoteSupport
from .response_styles import ResponseStyleSupport
from .search import IntentSuggestions as _IntentSuggestions, PromptSearchMixin
from .storage import PromptStorage
from .versioning import (
    PromptVersionDiff as _PromptVersionDiff,
    PromptVersionMixin,
)

sqlite3 = _sqlite3

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Mapping, Sequence
    from types import TracebackType

    from chromadb.api import ClientAPI

    from models.category_model import PromptCategory

    from ..history_tracker import HistoryTracker
else:  # pragma: no cover - typing fallback
    ClientAPI = Any  # type: ignore[assignment]
    TracebackType = type(None)

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


def _parse_timestamp(value: Any) -> datetime | None:
    """Return a timezone-aware datetime when parsing succeeds."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


try:
    from redis.exceptions import RedisError as _RedisErrorType
except ImportError:  # pragma: no cover - redis optional during development
    _RedisErrorType = Exception

RedisError = _RedisErrorType


# ---------------------------------------------------------------------------
# Protocols for optional dependencies to keep type hints precise without
# importing heavy runtime modules when they are absent in development.
# ---------------------------------------------------------------------------
RedisValue = str | bytes | memoryview


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


logger = logging.getLogger(__name__)

# Public exports of this module (until full split is finished)
_REEXPORTED_EXCEPTIONS = (
    CategoryError,
    CategoryNotFoundError,
    CategoryStorageError,
    CategorySuggestionError,
    DescriptionGenerationError,
    NameGenerationError,
    PromptVersionError,
    PromptVersionNotFoundError,
    PromptManagerError,
    PromptNotFoundError,
    PromptExecutionUnavailable,
    PromptExecutionError,
    PromptHistoryError,
    PromptStorageError,
    PromptCacheError,
    PromptEngineeringUnavailable,
    ResponseStyleError,
    ResponseStyleNotFoundError,
    ResponseStyleStorageError,
    PromptNoteError,
    PromptNoteNotFoundError,
    PromptNoteStorageError,
    ScenarioGenerationError,
)

__all__ = [
    "CategoryError",
    "CategoryNotFoundError",
    "CategoryStorageError",
    "CategorySuggestionError",
    "DescriptionGenerationError",
    "NameGenerationError",
    "PromptManagerError",
    "PromptNotFoundError",
    "PromptExecutionUnavailable",
    "PromptExecutionError",
    "PromptHistoryError",
    "PromptStorageError",
    "PromptCacheError",
    "PromptEngineeringUnavailable",
    "ResponseStyleError",
    "ResponseStyleNotFoundError",
    "ResponseStyleStorageError",
    "PromptNoteError",
    "PromptNoteNotFoundError",
    "PromptNoteStorageError",
    "ScenarioGenerationError",
]


def _normalize_prompt_body(body: str | None) -> str:
    """Return canonical prompt body text for change detection comparisons."""
    if body is None:
        return ""
    return body.replace("\r\n", "\n")


class PromptManager(
    CategorySupport,
    ResponseStyleSupport,
    PromptNoteSupport,
    GenerationMixin,
    PromptSearchMixin,
    PromptVersionMixin,
    MaintenanceMixin,
    ExecutionHistoryMixin,
):
    """Manage prompt persistence, caching, and semantic search."""

    def __init__(
        self,
        chroma_path: str,
        db_path: str | Path | None = None,
        collection_name: str = "prompt_manager",
        cache_ttl_seconds: int = 300,
        redis_client: RedisClientProtocol | None = None,
        chroma_client: ClientAPI | None = None,
        embedding_function: Any | None = None,
        repository: PromptRepository | None = None,
        category_definitions: Sequence[PromptCategory] | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        embedding_worker: EmbeddingSyncWorker | None = None,
        enable_background_sync: bool = True,
        name_generator: LiteLLMNameGenerator | None = None,
        description_generator: LiteLLMDescriptionGenerator | None = None,
        scenario_generator: LiteLLMScenarioGenerator | None = None,
        category_generator: LiteLLMCategoryGenerator | None = None,
        prompt_engineer: PromptEngineer | None = None,
        structure_prompt_engineer: PromptEngineer | None = None,
        fast_model: str | None = None,
        inference_model: str | None = None,
        workflow_models: Mapping[str, str | None] | None = None,
        intent_classifier: IntentClassifier | None = None,
        notification_center: NotificationCenter | None = None,
        executor: CodexExecutor | None = None,
        user_profile: UserProfile | None = None,
        history_tracker: HistoryTracker | None = None,
        prompt_templates: Mapping[str, object] | None = None,
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
            embedding_provider: Optional provider wrapper used for ad-hoc embeddings.
            embedding_worker: Background worker handling embedding synchronisation.
            enable_background_sync: Toggle to start the embedding worker automatically.
            repository: Optional preconfigured repository instance (for example, in tests).
            category_definitions: Optional PromptCategory defaults for seeding the repository.
            name_generator: LiteLLM helper for generating prompt names.
            description_generator: LiteLLM helper for generating prompt descriptions.
            scenario_generator: LiteLLM helper for generating usage scenarios.
            fast_model: LiteLLM fast-tier model identifier configured for workflows.
            inference_model: LiteLLM inference-tier model identifier configured for workflows.
            workflow_models: Mapping of workflow identifiers to routing tiers.
            intent_classifier: Optional classifier instance for detecting prompt intent.
            notification_center: Optional NotificationCenter override for task events.
            executor: Optional CodexExecutor responsible for running prompts.
            history_tracker: Optional execution history tracker for persistence.
            user_profile: Optional profile to seed single-user personalisation state.
            prompt_engineer: Optional prompt refinement helper using LiteLLM.
            structure_prompt_engineer: Optional helper focusing on prompt structure tweaks.
            prompt_templates: Optional mapping of workflow identifiers to system prompt overrides.
            category_generator: Optional LiteLLM helper for suggesting prompt categories.
        """
        # Allow tests to supply repository directly; only require db_path when building one.
        if repository is None and db_path is None:
            raise ValueError("db_path must be provided when no repository is supplied")

        self._closed = False
        self._cache_ttl_seconds = cache_ttl_seconds
        self._redis_client: RedisClientProtocol | None = redis_client
        self._chroma_client: Any
        self._collection: CollectionProtocol | None = None
        resolved_db_path = Path(db_path).expanduser() if db_path is not None else None
        resolved_chroma_path = Path(chroma_path).expanduser()
        resolved_chroma_path.mkdir(parents=True, exist_ok=True)
        self._db_path = resolved_db_path
        self._chroma_path = str(resolved_chroma_path)
        self._collection_name = collection_name
        self._embedding_function = embedding_function
        self._logs_path = Path("data") / "logs"
        try:
            self._repository = repository or PromptRepository(str(resolved_db_path))
        except RepositoryError as exc:
            raise PromptStorageError("Unable to initialise SQLite repository") from exc
        self._category_registry = CategoryRegistry(self._repository, category_definitions)
        # Disable Chroma anonymized telemetry to avoid noisy PostHog errors in some environments
        # and respect privacy defaults. Use persistent client at the configured path.
        if chroma_client is None:
            chroma_settings: Any | None = None
            try:
                from chromadb.config import Settings as ChromaSettings

                chroma_settings = ChromaSettings(
                    anonymized_telemetry=False,
                    is_persistent=True,
                    persist_directory=str(resolved_chroma_path),
                )
            except Exception:  # pragma: no cover - defensive guard for optional dependency
                chroma_settings = None

            persistent_kwargs: dict[str, Any] = {"path": str(resolved_chroma_path)}
            if chroma_settings is not None:
                persistent_kwargs["settings"] = chroma_settings
            try:
                try:
                    persistent_client = chromadb.PersistentClient(**persistent_kwargs)
                except TypeError as exc:
                    if "unexpected keyword argument 'settings'" not in str(exc):
                        raise
                    persistent_kwargs.pop("settings", None)
                    persistent_client = chromadb.PersistentClient(**persistent_kwargs)
            except ChromaError as exc:
                logger.warning(
                    "Chroma persistent client unavailable, using in-memory fallback: %s",
                    exc,
                )
                self._chroma_client = cast("Any", chromadb.EphemeralClient())
            else:
                self._chroma_client = cast("Any", persistent_client)
        else:
            self._chroma_client = cast("Any", chroma_client)
        self._initialise_chroma_collection()

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
        self._scenario_generator = scenario_generator
        self._category_generator = category_generator
        self._prompt_engineer = prompt_engineer
        self._prompt_structure_engineer = structure_prompt_engineer or prompt_engineer
        self._litellm_fast_model: str | None = self._normalise_model_identifier(fast_model)
        if self._litellm_fast_model is None and getattr(name_generator, "model", None):
            self._litellm_fast_model = self._normalise_model_identifier(name_generator.model)
        self._litellm_inference_model: str | None = self._normalise_model_identifier(
            inference_model
        )
        self._litellm_workflow_models: dict[str, str] = {}
        if workflow_models:
            for key, value in workflow_models.items():
                key_str = str(key).strip()
                if key_str not in LITELLM_ROUTED_WORKFLOWS:
                    continue
                if value is None:
                    continue
                choice = str(value).strip().lower()
                if choice == "inference":
                    self._litellm_workflow_models[key_str] = "inference"
        self._litellm_stream: bool = False
        self._litellm_drop_params: Sequence[str] | None = None
        self._litellm_reasoning_effort: str | None = None
        for candidate in (
            name_generator,
            scenario_generator,
            prompt_engineer,
            executor,
            category_generator,
        ):
            if candidate is not None and getattr(candidate, "drop_params", None):
                raw_params = cast("Sequence[str]", candidate.drop_params)
                self._litellm_drop_params = tuple(str(param) for param in raw_params)
                break
        for candidate in (
            executor,
            name_generator,
            scenario_generator,
            prompt_engineer,
            category_generator,
        ):
            if candidate is not None and hasattr(candidate, "stream"):
                self._litellm_stream = bool(getattr(candidate, "stream", False))
                if self._litellm_stream:
                    break
        if executor is not None and getattr(executor, "reasoning_effort", None):
            effort = executor.reasoning_effort
            self._litellm_reasoning_effort = str(effort) if effort else None
        self._intent_classifier = intent_classifier or IntentClassifier()
        self._executor = executor
        if self._executor is not None:
            if self._litellm_drop_params:
                self._executor.drop_params = list(self._litellm_drop_params)
            if self._litellm_reasoning_effort:
                self._executor.reasoning_effort = self._litellm_reasoning_effort
            self._executor.stream = self._litellm_stream
        self._history_tracker = history_tracker
        if user_profile is not None:
            self._user_profile: UserProfile | None = user_profile
        else:
            try:
                self._user_profile = self._repository.get_user_profile()
            except RepositoryError:
                logger.warning("Unable to load persisted user profile", exc_info=True)
                self._user_profile = None
        self._prompt_templates: dict[str, str] = self._normalise_prompt_templates(prompt_templates)

    @staticmethod
    def _normalise_model_identifier(value: str | None) -> str | None:
        """Return a stripped model identifier when provided."""
        if value is None:
            return None
        text = str(value).strip()
        return text or None

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
        try:
            collection = cast(
                "CollectionProtocol",
                self._chroma_client.get_or_create_collection(
                    name=self._collection_name,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=self._embedding_function,
                ),
            )
            self._collection = collection
        except ChromaError as exc:
            raise PromptStorageError("Unable to initialise ChromaDB collection") from exc

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

    IntentSuggestions = _IntentSuggestions
    PromptVersionDiff = _PromptVersionDiff
    ExecutionOutcome = _ExecutionOutcome
    BenchmarkRun = _BenchmarkRun
    BenchmarkReport = _BenchmarkReport

    @dataclass(slots=True)
    class EmbeddingDimensionMismatch:
        """Stored prompt embedding vector that no longer matches the reference dimension."""

        prompt_id: uuid.UUID
        prompt_name: str
        stored_dimension: int

    @dataclass(slots=True)
    class MissingEmbedding:
        """Prompt record that is missing a persisted embedding vector."""

        prompt_id: uuid.UUID
        prompt_name: str

    @dataclass(slots=True)
    class EmbeddingDiagnostics:
        """Summary of embedding backend health and stored vector consistency."""

        backend_ok: bool
        backend_message: str
        backend_dimension: int | None
        inferred_dimension: int | None
        chroma_ok: bool
        chroma_message: str
        chroma_count: int | None
        repository_total: int
        prompts_with_embeddings: int
        missing_prompts: list[PromptManager.MissingEmbedding]
        mismatched_prompts: list[PromptManager.EmbeddingDimensionMismatch]
        consistent_counts: bool | None

    @dataclass(slots=True)
    class CategoryHealth:
        """Aggregated prompt and execution metrics for a category."""

        slug: str
        label: str
        total_prompts: int
        active_prompts: int
        success_rate: float | None
        last_executed_at: datetime | None

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
        return self._intent_classifier

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
        return self._user_profile

    @property
    def prompt_engineer(self) -> PromptEngineer | None:
        """Return the configured prompt engineering helper, if any."""
        return self._prompt_engineer

    @property
    def prompt_structure_engineer(self) -> PromptEngineer | None:
        """Return the configured structure-only prompt engineering helper."""
        return self._prompt_structure_engineer or self._prompt_engineer

    def get_category_health(self) -> list[PromptManager.CategoryHealth]:
        """Return prompt and execution health metrics for each category."""
        try:
            prompt_counts = self._repository.get_category_prompt_counts()
            execution_stats = self._repository.get_category_execution_statistics()
        except RepositoryError as exc:
            raise PromptStorageError("Unable to compute category health metrics") from exc

        categories = {
            category.slug: category
            for category in self._category_registry.all(include_archived=True)
        }
        slug_keys = set(prompt_counts.keys()) | set(execution_stats.keys()) | set(categories.keys())
        if not slug_keys:
            slug_keys.add("")

        results: list[PromptManager.CategoryHealth] = []
        for slug in sorted(slug_keys or {""}):
            category = categories.get(slug)
            label = (
                category.label
                if category
                else (slug.replace("-", " ").title() if slug else "Uncategorised")
            )
            counts = prompt_counts.get(slug, {"total_prompts": 0, "active_prompts": 0})
            stats = execution_stats.get(slug)
            success_rate: float | None = None
            last_executed_at: datetime | None = None
            if stats:
                total_runs = int(stats.get("total_runs", 0) or 0)
                success_runs = int(stats.get("success_runs", 0) or 0)
                success_rate = success_runs / total_runs if total_runs else None
                last_executed_at = _parse_timestamp(stats.get("last_executed_at"))
            results.append(
                PromptManager.CategoryHealth(
                    slug=slug or "",
                    label=label,
                    total_prompts=int(counts.get("total_prompts", 0) or 0),
                    active_prompts=int(counts.get("active_prompts", 0) or 0),
                    success_rate=success_rate,
                    last_executed_at=last_executed_at,
                )
            )

        return results

    def refresh_user_profile(self) -> UserProfile | None:
        """Reload the persisted profile from the repository."""
        try:
            self._user_profile = self._repository.get_user_profile()
        except RepositoryError:
            logger.warning("Unable to refresh user profile from storage", exc_info=True)
        return self._user_profile

    def diagnose_embeddings(
        self,
        *,
        sample_text: str = "Prompt Manager diagnostics probe",
    ) -> PromptManager.EmbeddingDiagnostics:
        """Return embedding backend health and stored vector consistency details."""
        provider = getattr(self, "_embedding_provider", None)
        if provider is None:
            raise PromptManagerError("Embedding provider is not configured.")

        backend_ok = True
        backend_message = "Embedding backend reachable."
        backend_dimension: int | None = None
        try:
            probe_vector = provider.embed(sample_text)
            backend_dimension = len(probe_vector)
            if backend_dimension == 0:
                backend_ok = False
                backend_message = "Embedding backend returned an empty vector."
        except EmbeddingGenerationError as exc:
            backend_ok = False
            backend_message = f"Unable to generate embeddings: {exc}"
        except Exception as exc:  # noqa: BLE001 - defensive diagnostics surface
            backend_ok = False
            backend_message = f"Unexpected embedding backend error: {exc}"

        chroma_ok = False
        chroma_message = "Chroma collection unavailable."
        chroma_count: int | None = None
        try:
            collection = self.collection
        except PromptManagerError as exc:
            chroma_message = str(exc)
        else:
            try:
                chroma_count = int(collection.count())
                chroma_ok = True
                chroma_message = "Chroma collection reachable."
            except Exception as exc:  # noqa: BLE001 - defensive diagnostics surface
                chroma_message = f"Unable to query Chroma collection: {exc}"

        try:
            prompts = self._repository.list()
        except RepositoryError as exc:
            raise PromptStorageError("Unable to load prompts for embedding diagnostics") from exc

        missing_prompts: list[PromptManager.MissingEmbedding] = []
        mismatched: list[PromptManager.EmbeddingDimensionMismatch] = []
        prompts_with_embeddings = 0
        inferred_dimension: int | None = None
        reference_dimension = backend_dimension if backend_dimension else None

        for prompt in prompts:
            vector = prompt.ext4
            if not vector:
                missing_prompts.append(
                    PromptManager.MissingEmbedding(
                        prompt_id=prompt.id,
                        prompt_name=prompt.name or "Unnamed prompt",
                    )
                )
                continue
            vector_values = list(vector)
            stored_dimension = len(vector_values)
            if stored_dimension == 0:
                missing_prompts.append(
                    PromptManager.MissingEmbedding(
                        prompt_id=prompt.id,
                        prompt_name=prompt.name or "Unnamed prompt",
                    )
                )
                continue
            prompts_with_embeddings += 1
            if reference_dimension is None:
                reference_dimension = stored_dimension
                inferred_dimension = stored_dimension
            if reference_dimension is not None and stored_dimension != reference_dimension:
                mismatched.append(
                    PromptManager.EmbeddingDimensionMismatch(
                        prompt_id=prompt.id,
                        prompt_name=prompt.name or "Unnamed prompt",
                        stored_dimension=stored_dimension,
                    )
                )

        consistent_counts: bool | None = None
        if chroma_count is not None:
            consistent_counts = chroma_count == prompts_with_embeddings

        return PromptManager.EmbeddingDiagnostics(
            backend_ok=backend_ok,
            backend_message=backend_message,
            backend_dimension=backend_dimension if backend_ok else None,
            inferred_dimension=inferred_dimension,
            chroma_ok=chroma_ok,
            chroma_message=chroma_message,
            chroma_count=chroma_count,
            repository_total=len(prompts),
            prompts_with_embeddings=prompts_with_embeddings,
            missing_prompts=missing_prompts,
            mismatched_prompts=mismatched,
            consistent_counts=consistent_counts,
        )

    def _build_execution_context_metadata(
        self,
        prompt: Prompt,
        *,
        stream_enabled: bool,
        executor_model: str | None,
        conversation_length: int,
        request_text: str,
        response_text: str,
        response_style: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return structured metadata describing the execution context."""
        prompt_metadata = {
            "id": str(prompt.id),
            "name": prompt.name,
            "category": prompt.category,
            "tags": list(prompt.tags),
            "version": prompt.version,
        }
        execution_metadata = {
            "model": executor_model,
            "stream_enabled": stream_enabled,
            "conversation_messages": conversation_length,
            "request_chars": len(request_text or ""),
            "response_chars": len(response_text or ""),
        }
        context: dict[str, Any] = {
            "prompt": prompt_metadata,
            "execution": execution_metadata,
        }
        if response_style:
            context["response_style"] = dict(response_style)
        return context

    def refine_prompt_text(
        self,
        prompt_text: str,
        *,
        name: str | None = None,
        description: str | None = None,
        category: str | None = None,
        tags: Sequence[str] | None = None,
        negative_constraints: Sequence[str] | None = None,
    ) -> PromptRefinement:
        """Improve a prompt using the configured prompt engineer."""
        if not prompt_text.strip():
            raise PromptEngineeringError("Prompt refinement requires non-empty prompt text.")
        engineer = self._prompt_structure_engineer or self._prompt_engineer
        if engineer is None:
            raise PromptEngineeringUnavailable(
                "Prompt engineering is not configured. Set PROMPT_MANAGER_LITELLM_MODEL "
                "to enable refinement."
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
                return engineer.refine(
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

    def refine_prompt_structure(
        self,
        prompt_text: str,
        *,
        name: str | None = None,
        description: str | None = None,
        category: str | None = None,
        tags: Sequence[str] | None = None,
    ) -> PromptRefinement:
        """Reformat a prompt to improve structure without changing intent."""
        if not prompt_text.strip():
            raise PromptEngineeringError("Prompt refinement requires non-empty prompt text.")
        engineer = self._prompt_structure_engineer or self._prompt_engineer
        if engineer is None:
            raise PromptEngineeringUnavailable(
                "Prompt engineering is not configured. Set PROMPT_MANAGER_LITELLM_MODEL "
                "to enable refinement."
            )
        task_id = f"prompt-structure-refine:{uuid.uuid4()}"
        metadata = {
            "prompt_length": len(prompt_text or ""),
            "mode": "structure",
        }
        with self._notification_center.track_task(
            title="Prompt structure refinement",
            task_id=task_id,
            start_message="Reformatting prompt via LiteLLM…",
            success_message="Prompt structure refined.",
            failure_message="Prompt structure refinement failed",
            metadata=metadata,
            level=NotificationLevel.INFO,
        ):
            try:
                return engineer.refine_structure(
                    prompt_text,
                    name=name,
                    description=description,
                    category=category,
                    tags=tags,
                )
            except PromptEngineeringError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                raise PromptEngineeringError("Prompt refinement failed unexpectedly.") from exc

    def set_name_generator(
        self,
        model: str | None,
        api_key: str | None,
        api_base: str | None,
        api_version: str | None,
        *,
        inference_model: str | None = None,
        workflow_models: Mapping[str, str | None] | None = None,
        drop_params: Sequence[str] | None = None,
        reasoning_effort: str | None = None,
        stream: bool | None = None,
        prompt_templates: Mapping[str, object] | None = None,
    ) -> None:
        """Configure LiteLLM-backed workflows at runtime."""
        self._litellm_fast_model = self._normalise_model_identifier(model)
        self._litellm_inference_model = self._normalise_model_identifier(inference_model)
        routing: dict[str, str] = {}
        if workflow_models:
            for key, value in workflow_models.items():
                workflow_key = str(key).strip()
                if workflow_key not in LITELLM_ROUTED_WORKFLOWS:
                    continue
                if value is None:
                    continue
                choice = str(value).strip().lower()
                if choice == "inference":
                    routing[workflow_key] = "inference"
        self._litellm_workflow_models = routing

        if drop_params is not None:
            cleaned_params = [str(item).strip() for item in drop_params if str(item).strip()]
            self._litellm_drop_params = tuple(cleaned_params) if cleaned_params else None
        else:
            self._litellm_drop_params = None

        self._litellm_reasoning_effort = reasoning_effort
        if stream is not None:
            self._litellm_stream = bool(stream)

        if prompt_templates is not None:
            self._prompt_templates = self._normalise_prompt_templates(prompt_templates)

        # Reset existing helpers before rebuilding.
        self._name_generator = None
        self._description_generator = None
        self._prompt_engineer = None
        self._prompt_structure_engineer = None
        self._scenario_generator = None
        self._category_generator = None
        self._executor = None

        if not (self._litellm_fast_model or self._litellm_inference_model):
            self._litellm_reasoning_effort = None
            self._litellm_stream = False
            return

        drop_params_payload = list(self._litellm_drop_params) if self._litellm_drop_params else None

        def _select_model(workflow: str) -> str | None:
            selection = self._litellm_workflow_models.get(workflow, "fast")
            if selection == "inference":
                return self._litellm_inference_model or self._litellm_fast_model
            return self._litellm_fast_model or self._litellm_inference_model

        def _construct(factory: Callable[..., Any], workflow: str, **extra: Any) -> Any | None:
            selected_model = _select_model(workflow)
            if not selected_model:
                return None
            return factory(
                model=selected_model,
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                drop_params=drop_params_payload,
                **extra,
            )

        template_overrides = dict(self._prompt_templates)
        try:
            self._name_generator = _construct(
                LiteLLMNameGenerator,
                "name_generation",
                system_prompt=template_overrides.get("name_generation"),
            )
            self._description_generator = _construct(
                LiteLLMDescriptionGenerator,
                "description_generation",
                system_prompt=template_overrides.get("description_generation"),
            )
            self._prompt_engineer = _construct(
                PromptEngineer,
                "prompt_engineering",
                system_prompt=template_overrides.get("prompt_engineering"),
            )
            structure_engineer = _construct(
                PromptEngineer,
                "prompt_structure_refinement",
                system_prompt=template_overrides.get("prompt_engineering"),
            )
            self._prompt_structure_engineer = structure_engineer or self._prompt_engineer
            self._scenario_generator = _construct(
                LiteLLMScenarioGenerator,
                "scenario_generation",
                system_prompt=template_overrides.get("scenario_generation"),
            )
            self._category_generator = _construct(
                LiteLLMCategoryGenerator,
                "category_generation",
                system_prompt=template_overrides.get("category_generation"),
            )
            self._executor = _construct(
                CodexExecutor,
                "prompt_execution",
                reasoning_effort=self._litellm_reasoning_effort,
                stream=self._litellm_stream,
            )
        except RuntimeError as exc:
            raise NameGenerationError(str(exc)) from exc

        if self._executor is not None:
            if self._litellm_drop_params:
                self._executor.drop_params = list(self._litellm_drop_params)
            if self._litellm_reasoning_effort:
                self._executor.reasoning_effort = self._litellm_reasoning_effort
            self._executor.stream = self._litellm_stream

        if self._intent_classifier is not None and (
            self._litellm_fast_model or self._litellm_inference_model
        ):
            logger.debug("LiteLLM powered features enabled for intent classifier")

    @staticmethod
    def _normalise_prompt_templates(overrides: Mapping[str, object] | None) -> dict[str, str]:
        """Return a cleaned mapping of workflow prompt overrides."""
        if not overrides:
            return {}
        cleaned: dict[str, str] = {}
        for key, text in overrides.items():
            if key not in PROMPT_TEMPLATE_KEYS:
                continue
            if not isinstance(text, str):
                continue
            stripped = text.strip()
            default_text = DEFAULT_PROMPT_TEMPLATES.get(key)
            if stripped and stripped != default_text:
                cleaned[key] = stripped
        return cleaned

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
            redis_close = cast(
                "Callable[[], Any] | None", getattr(self._redis_client, "close", None)
            )
            if callable(redis_close):
                try:
                    redis_close()
                except Exception:
                    logger.debug(
                        "Failed to close Redis client cleanly",
                        exc_info=True,
                    )
            pool = cast(
                "RedisConnectionPoolProtocol | None",
                getattr(self._redis_client, "connection_pool", None),
            )
            disconnect = cast("Callable[[], Any] | None", getattr(pool, "disconnect", None))
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

    def __enter__(self) -> PromptManager:
        """Support use of PromptManager as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close resources when exiting a context manager block."""
        self.close()

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

        body_changed = _normalize_prompt_body(previous_prompt.context) != _normalize_prompt_body(
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

    def delete_prompt(self, prompt_id: uuid.UUID) -> None:
        """Remove a prompt from all data stores."""
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
        prompt.last_modified = datetime.now(UTC)

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

    def _get_cached_prompt(self, prompt_id: uuid.UUID) -> Prompt | None:
        """Fetch prompt from Redis cache when available."""
        if self._redis_client is None:
            return None
        try:
            cached_value: RedisValue | None = self._redis_client.get(self._cache_key(prompt_id))
        except RedisError as exc:  # pragma: no cover - redis not in CI
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

    def _persist_embedding(
        self, prompt: Prompt, embedding: Sequence[float], *, is_new: bool
    ) -> None:
        """Persist embeddings to Chroma and refresh caches."""
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
    # New storage façade
    "PromptStorage",
    # Execution facade
    "PromptExecutor",
    "ExecutionResult",
    # Engineering facade
    "PromptEngineerFacade",
    "PromptEngineeringError",
    "PromptRefinement",
]
