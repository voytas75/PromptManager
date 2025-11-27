"""Core service layer for Prompt Manager.

Updates: v0.9.0 - 2025-12-06 - Export PromptNote models and exception helpers.
Updates: v0.7.1 - 2025-11-30 - Restore catalogue import helpers for GUI workflows.
Updates: v0.6.0 - 2025-11-15 - Export prompt engineering helpers alongside manager API.
Updates: v0.5.0 - 2025-11-07 - Export embedding factory helper for external use.
Updates: v0.4.0 - 2025-11-06 - Export intent classifier utilities for GUI integration.
Updates: v0.3.0 - 2025-11-03 - Export build_prompt_manager factory for shared bootstrap.
Updates: v0.2.0 - 2025-10-31 - Surface PromptRepository alongside PromptManager.
Updates: v0.1.0 - 2025-10-30 - Expose PromptManager API.
"""

from models.category_model import PromptCategory
from models.prompt_note import PromptNote
from models.response_style import ResponseStyle

from .catalog_importer import (
    CatalogChangePlan,
    CatalogChangeType,
    CatalogDiff,
    CatalogDiffEntry,
    CatalogImportResult,
    diff_prompt_catalog,
    export_prompt_catalog,
    import_prompt_catalog,
    load_prompt_catalog,
)
from .category_registry import CategoryRegistry
from .embedding import create_embedding_function
from .execution import CodexExecutionResult, CodexExecutor, ExecutionError
from .factory import build_prompt_manager
from .history_tracker import (
    ExecutionAnalytics,
    HistoryTracker,
    HistoryTrackerError,
    PromptExecutionAnalytics,
)
from .intent_classifier import (
    IntentClassifier,
    IntentClassifierError,
    IntentLabel,
    IntentPrediction,
)
from .name_generation import (
    CategorySuggestionError,
    DescriptionGenerationError,
    LiteLLMCategoryGenerator,
    LiteLLMDescriptionGenerator,
    NameGenerationError,
)
from .notifications import (
    Notification,
    NotificationCenter,
    NotificationLevel,
    NotificationStatus,
    NotificationSubscription,
    notification_center,
)
from .prompt_engineering import PromptEngineer, PromptEngineeringError, PromptRefinement
from .prompt_manager import (
    CategoryError,
    CategoryNotFoundError,
    CategoryStorageError,
    PromptCacheError,
    PromptEngineeringUnavailable,
    PromptExecutionError,
    PromptExecutionUnavailable,
    PromptHistoryError,
    PromptManager,
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
)
from .repository import PromptRepository, RepositoryError, RepositoryNotFoundError
from .scenario_generation import LiteLLMScenarioGenerator, ScenarioGenerationError

__all__ = [
    "PromptManager",
    "PromptManagerError",
    "PromptNotFoundError",
    "PromptStorageError",
    "PromptCacheError",
    "PromptExecutionError",
    "PromptExecutionUnavailable",
    "PromptEngineeringUnavailable",
    "PromptHistoryError",
    "PromptVersionError",
    "PromptVersionNotFoundError",
    "ResponseStyleError",
    "ResponseStyleNotFoundError",
    "ResponseStyleStorageError",
    "PromptNoteError",
    "PromptNoteNotFoundError",
    "PromptNoteStorageError",
    "CategoryError",
    "CategoryNotFoundError",
    "CategoryStorageError",
    "PromptRepository",
    "RepositoryError",
    "RepositoryNotFoundError",
    "CategoryRegistry",
    "PromptCategory",
    "ResponseStyle",
    "PromptNote",
    "build_prompt_manager",
    "import_prompt_catalog",
    "load_prompt_catalog",
    "diff_prompt_catalog",
    "export_prompt_catalog",
    "CatalogImportResult",
    "CatalogDiff",
    "CatalogDiffEntry",
    "CatalogChangeType",
    "CatalogChangePlan",
    "NameGenerationError",
    "DescriptionGenerationError",
    "CategorySuggestionError",
    "IntentClassifier",
    "IntentClassifierError",
    "IntentPrediction",
    "IntentLabel",
    "create_embedding_function",
    "LiteLLMDescriptionGenerator",
    "LiteLLMCategoryGenerator",
    "LiteLLMScenarioGenerator",
    "CodexExecutor",
    "CodexExecutionResult",
    "ExecutionError",
    "PromptEngineer",
    "PromptEngineeringError",
    "PromptRefinement",
    "ScenarioGenerationError",
    "HistoryTracker",
    "HistoryTrackerError",
    "ExecutionAnalytics",
    "PromptExecutionAnalytics",
    "Notification",
    "NotificationCenter",
    "NotificationLevel",
    "NotificationStatus",
    "NotificationSubscription",
    "notification_center",
]
