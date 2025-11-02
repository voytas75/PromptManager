"""Core service layer for Prompt Manager.

Updates: v0.6.0 - 2025-11-15 - Export prompt engineering helpers alongside manager API.
Updates: v0.5.0 - 2025-11-07 - Export embedding factory helper for external use.
Updates: v0.4.0 - 2025-11-06 - Export intent classifier utilities for GUI integration.
Updates: v0.3.0 - 2025-11-03 - Export build_prompt_manager factory for shared bootstrap.
Updates: v0.2.0 - 2025-10-31 - Surface PromptRepository alongside PromptManager.
Updates: v0.1.0 - 2025-10-30 - Expose PromptManager API.
"""

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
from .factory import build_prompt_manager
from .intent_classifier import (
    IntentClassifier,
    IntentClassifierError,
    IntentLabel,
    IntentPrediction,
)
from .history_tracker import HistoryTracker, HistoryTrackerError
from .prompt_manager import (
    PromptCacheError,
    PromptExecutionError,
    PromptExecutionUnavailable,
    PromptEngineeringUnavailable,
    PromptHistoryError,
    PromptManager,
    PromptManagerError,
    PromptNotFoundError,
    PromptStorageError,
)
from .name_generation import NameGenerationError, DescriptionGenerationError, LiteLLMDescriptionGenerator
from .repository import PromptRepository, RepositoryError, RepositoryNotFoundError
from .embedding import create_embedding_function
from .execution import CodexExecutor, CodexExecutionResult, ExecutionError
from .notifications import (
    Notification,
    NotificationCenter,
    NotificationLevel,
    NotificationStatus,
    NotificationSubscription,
    notification_center,
)
from .prompt_engineering import PromptEngineer, PromptEngineeringError, PromptRefinement

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
    "PromptRepository",
    "RepositoryError",
    "RepositoryNotFoundError",
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
    "IntentClassifier",
    "IntentClassifierError",
    "IntentPrediction",
    "IntentLabel",
    "create_embedding_function",
    "LiteLLMDescriptionGenerator",
    "CodexExecutor",
    "CodexExecutionResult",
    "ExecutionError",
    "PromptEngineer",
    "PromptEngineeringError",
    "PromptRefinement",
    "HistoryTracker",
    "HistoryTrackerError",
    "Notification",
    "NotificationCenter",
    "NotificationLevel",
    "NotificationStatus",
    "NotificationSubscription",
    "notification_center",
]
