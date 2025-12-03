"""Prompt Manager package façade and orchestration layer.

Updates:
  v0.14.16 - 2025-12-03 - Extract backend bootstrap and LiteLLM wiring into mixins.
  v0.14.15 - 2025-12-03 - Move runtime lifecycle helpers into dedicated mixin module.
  v0.14.14 - 2025-12-03 - Move backend bootstrap helpers into dedicated module.
  v0.14.13 - 2025-12-03 - Extract LiteLLM workflows and prompt refinement helpers into mixins.
  v0.14.12 - 2025-12-03 - Extract prompt CRUD, caching, and embedding APIs into mixin.
  v0.14.11 - 2025-12-03 - Extract versioning, diff, and fork APIs into mixin module.
  v0.14.10 - 2025-12-03 - Move search and suggestion APIs into dedicated mixin.
  v0.14.9 - 2025-12-03 - Move execution and benchmarking APIs into mixin module.
  v0.14.8 - 2025-12-02 - Move maintenance utilities into dedicated mixin module.
  pre-v0.14.8 - 2025-11-30 - Consolidated history covering releases v0.1.0–v0.14.7.
"""

from __future__ import annotations

import logging
import sqlite3 as _sqlite3
from typing import TYPE_CHECKING, Any

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
from models.prompt_model import UserProfile

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
from ..name_generation import (
    LiteLLMCategoryGenerator,
    LiteLLMDescriptionGenerator,
    LiteLLMNameGenerator,
    NameGenerationError,
)
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
from .backends import RedisClientProtocol, mute_posthog_capture
from .bootstrap import BackendBootstrapMixin
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
from .litellm_helpers import LiteLLMWiringMixin
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
    from pathlib import Path

    from chromadb.api import ClientAPI

    from models.category_model import PromptCategory
    from models.prompt_model import UserProfile

    from ..embedding import EmbeddingProvider, EmbeddingSyncWorker
    from ..execution import CodexExecutor
    from ..history_tracker import HistoryTracker
    from ..intent_classifier import IntentClassifier
    from ..notifications import NotificationCenter
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
    BackendBootstrapMixin,
    LiteLLMWiringMixin,
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

        self._initialise_backends(
            chroma_path=chroma_path,
            db_path=db_path,
            collection_name=collection_name,
            cache_ttl_seconds=cache_ttl_seconds,
            redis_client=redis_client,
            chroma_client=chroma_client,
            embedding_function=embedding_function,
            repository=repository,
            category_definitions=category_definitions,
            embedding_provider=embedding_provider,
            embedding_worker=embedding_worker,
            enable_background_sync=enable_background_sync,
            notification_center=notification_center,
        )

        self._initialise_litellm_helpers(
            name_generator=name_generator,
            description_generator=description_generator,
            scenario_generator=scenario_generator,
            category_generator=category_generator,
            prompt_engineer=prompt_engineer,
            structure_prompt_engineer=structure_prompt_engineer,
            fast_model=fast_model,
            inference_model=inference_model,
            workflow_models=workflow_models,
            executor=executor,
            intent_classifier=intent_classifier,
            prompt_templates=prompt_templates,
        )
        self._history_tracker = history_tracker
        if user_profile is not None:
            self._user_profile: UserProfile | None = user_profile
        else:
            try:
                self._user_profile = self._repository.get_user_profile()
            except RepositoryError:
                logger.warning("Unable to load persisted user profile", exc_info=True)
                self._user_profile = None
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
