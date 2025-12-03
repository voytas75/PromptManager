"""Prompt Manager package façade and orchestration layer.

Updates:
  v0.14.15 - 2025-12-03 - Move runtime lifecycle helpers into dedicated mixin module.
  v0.14.14 - 2025-12-03 - Move backend bootstrap helpers into dedicated module.
  v0.14.13 - 2025-12-03 - Extract LiteLLM workflows and prompt refinement helpers into mixins.
  v0.14.12 - 2025-12-03 - Extract prompt CRUD, caching, and embedding APIs into mixin.
  v0.14.11 - 2025-12-03 - Extract versioning, diff, and fork APIs into mixin module.
  v0.14.10 - 2025-12-03 - Move search and suggestion APIs into dedicated mixin.
  v0.14.9 - 2025-12-03 - Move execution and benchmarking APIs into mixin module.
  v0.14.8 - 2025-12-02 - Move maintenance utilities into dedicated mixin module.
  v0.14.7 - 2025-12-02 - Extract category, response style, and note APIs into mixins.
  pre-v0.14.7 - 2025-11-30 - Consolidated history covering releases v0.1.0–v0.14.6.
"""

from __future__ import annotations

import logging
import sqlite3 as _sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import chromadb as _chromadb
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
from models.prompt_model import UserProfile

from ..category_registry import CategoryRegistry
from ..embedding import EmbeddingProvider, EmbeddingSyncWorker
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
from ..execution import CodexExecutor  # noqa: TCH001
from ..intent_classifier import IntentClassifier
from ..name_generation import (
    LiteLLMCategoryGenerator,
    LiteLLMDescriptionGenerator,
    LiteLLMNameGenerator,
    NameGenerationError,
)
from ..notifications import NotificationCenter, notification_center as default_notification_center
from ..prompt_engineering import (
    PromptEngineer,
    PromptEngineeringError,
    PromptRefinement,
)
from ..repository import PromptRepository, RepositoryError, RepositoryNotFoundError
from ..scenario_generation import LiteLLMScenarioGenerator  # noqa: TCH001
from .analytics import (
    AnalyticsMixin,
    CategoryHealth as _CategoryHealth,
    EmbeddingDiagnostics as _EmbeddingDiagnostics,
    EmbeddingDimensionMismatch as _EmbeddingDimensionMismatch,
    MissingEmbedding as _MissingEmbedding,
)
from .backends import (
    CollectionProtocol,
    NullEmbeddingWorker,
    RedisClientProtocol,
    build_chroma_client,
    mute_posthog_capture,
)
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
from .lifecycle import PromptLifecycleMixin
from .maintenance import MaintenanceMixin
from .prompt_notes import PromptNoteSupport
from .refinement import PromptRefinementMixin
from .response_styles import ResponseStyleSupport
from .runtime import PromptRuntimeMixin
from .search import IntentSuggestions as _IntentSuggestions, PromptSearchMixin
from .storage import PromptStorage
from .versioning import (
    PromptVersionDiff as _PromptVersionDiff,
    PromptVersionMixin,
)
from .workflows import LiteLLMWorkflowMixin

sqlite3 = _sqlite3
chromadb = _chromadb

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence

    from chromadb.api import ClientAPI

    from models.category_model import PromptCategory
    from models.prompt_model import UserProfile

    from ..execution import CodexExecutor
    from ..history_tracker import HistoryTracker
    from ..scenario_generation import LiteLLMScenarioGenerator
else:  # pragma: no cover - typing fallback
    ClientAPI = Any  # type: ignore[assignment]

try:
    from redis.exceptions import RedisError as _RedisErrorType
except ImportError:  # pragma: no cover - redis optional during development
    _RedisErrorType = Exception

RedisError = _RedisErrorType

mute_posthog_capture()


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
    RepositoryNotFoundError,
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
    "RepositoryNotFoundError",
    "ChromaError",
    "chromadb",
]


class PromptManager(
    PromptRuntimeMixin,
    CategorySupport,
    ResponseStyleSupport,
    PromptNoteSupport,
    GenerationMixin,
    AnalyticsMixin,
    LiteLLMWorkflowMixin,
    PromptRefinementMixin,
    PromptLifecycleMixin,
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
        self._name_generator = name_generator
        self._description_generator = description_generator
        self._scenario_generator = scenario_generator
        self._category_generator = category_generator
        self._prompt_engineer = prompt_engineer
        self._prompt_structure_engineer = structure_prompt_engineer or prompt_engineer
        self._litellm_fast_model: str | None = self._normalise_model_identifier(fast_model)
        if self._litellm_fast_model is None and name_generator is not None:
            generator_model = getattr(name_generator, "model", None)
            if generator_model:
                self._litellm_fast_model = self._normalise_model_identifier(generator_model)
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

    IntentSuggestions = _IntentSuggestions
    PromptVersionDiff = _PromptVersionDiff
    ExecutionOutcome = _ExecutionOutcome
    BenchmarkRun = _BenchmarkRun
    BenchmarkReport = _BenchmarkReport
    CategoryHealth = _CategoryHealth
    EmbeddingDiagnostics = _EmbeddingDiagnostics
    EmbeddingDimensionMismatch = _EmbeddingDimensionMismatch
    MissingEmbedding = _MissingEmbedding


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
    "ChromaError",
    "chromadb",
]
