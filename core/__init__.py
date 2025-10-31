"""Core service layer for Prompt Manager.

Updates: v0.3.0 - 2025-11-03 - Export build_prompt_manager factory for shared bootstrap.
Updates: v0.2.0 - 2025-10-31 - Surface PromptRepository alongside PromptManager.
Updates: v0.1.0 - 2025-10-30 - Expose PromptManager API.
"""

from .catalog_importer import CatalogImportResult, import_prompt_catalog, load_prompt_catalog
from .factory import build_prompt_manager
from .prompt_manager import (
    PromptCacheError,
    PromptManager,
    PromptManagerError,
    PromptNotFoundError,
    PromptStorageError,
)
from .repository import PromptRepository, RepositoryError, RepositoryNotFoundError

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
    "CatalogImportResult",
]
