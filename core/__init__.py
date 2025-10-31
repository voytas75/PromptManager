"""Core service layer for Prompt Manager.

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
from .prompt_manager import (
    PromptCacheError,
    PromptManager,
    PromptManagerError,
    PromptNotFoundError,
    PromptStorageError,
)
from .name_generation import NameGenerationError
from .repository import PromptRepository, RepositoryError, RepositoryNotFoundError
from .embedding import create_embedding_function

__all__ = [
    "PromptManager",
    "PromptManagerError",
    "PromptNotFoundError",
    "PromptStorageError",
    "PromptCacheError",
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
    "IntentClassifier",
    "IntentClassifierError",
    "IntentPrediction",
    "IntentLabel",
    "create_embedding_function",
]
