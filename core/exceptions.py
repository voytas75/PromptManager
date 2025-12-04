"""Common exception classes for core package.

This module centralises shared exception definitions to reduce duplication
across the **core** package and to unblock incremental modularisation work.

The initial set has been extracted from the former monolithic
`core/prompt_manager.py`. Additional core-level exceptions should be added
here rather than redefining them in individual modules.

All exceptions ultimately inherit from :class:`PromptManagerError`, allowing
callers to catch a single base class for any manager-related failure while
still distinguishing individual error categories when needed.

Updates:
  v0.20.0 - 2025-12-04 - Add prompt chain exception hierarchy.
  v0.19.0 - 2025-12-02 - Centralise LiteLLM generation errors and category suggestion failures.
  v0.18.0 - 2025-11-22 - Add category exception hierarchy.
  v0.17.0 - 2025-11-22 - Add prompt versioning exception hierarchy.
  v0.16.0 - 2025-12-06 - Add PromptNote exception hierarchy.
  v0.15.0 - 2025-12-05 - Add ResponseStyle exception hierarchy.
  v0.14.0 - 2025-11-18 - Created module; migrated existing classes.
"""

from __future__ import annotations


class PromptManagerError(Exception):
    """Base exception for Prompt Manager failures."""


# ---------------------------------------------------------------------------
# Prompt‑specific errors (extracted from the former core.prompt_manager module)
# ---------------------------------------------------------------------------


class PromptNotFoundError(PromptManagerError):
    """Raised when a prompt cannot be located in the backing store."""


class PromptExecutionUnavailable(PromptManagerError):
    """Raised when prompt execution is not configured for the manager."""


class PromptExecutionError(PromptManagerError):
    """Raised when executing a prompt via an LLM fails."""


class PromptHistoryError(PromptManagerError):
    """Raised when manual history operations fail."""


class PromptStorageError(PromptManagerError):
    """Raised when interactions with persistent backends fail."""


class PromptCacheError(PromptManagerError):
    """Raised when Redis cache lookups or writes fail."""


class PromptEngineeringUnavailable(PromptManagerError):
    """Raised when prompt refinement is requested without an engineer configured."""


class CategoryError(PromptManagerError):
    """Base class for prompt category management failures."""


class CategoryNotFoundError(CategoryError):
    """Raised when a requested category does not exist."""


class CategoryStorageError(CategoryError):
    """Raised when persisting or loading categories fails."""


class CategorySuggestionError(CategoryError):
    """Raised when LiteLLM category suggestions fail or return invalid data."""


class PromptGenerationError(PromptManagerError):
    """Base class for LiteLLM prompt metadata generation failures."""


class NameGenerationError(PromptGenerationError):
    """Raised when a prompt name cannot be generated."""


class DescriptionGenerationError(PromptGenerationError):
    """Raised when a prompt description cannot be generated."""


class ScenarioGenerationError(PromptGenerationError):
    """Raised when LiteLLM usage scenarios cannot be generated."""


class ResponseStyleError(PromptManagerError):
    """Base class for response style workflow failures."""


class ResponseStyleNotFoundError(ResponseStyleError):
    """Raised when a response style cannot be located."""


class ResponseStyleStorageError(ResponseStyleError):
    """Raised when response style persistence fails."""


class PromptNoteError(PromptManagerError):
    """Base class for prompt note workflow failures."""


class PromptNoteNotFoundError(PromptNoteError):
    """Raised when a prompt note cannot be found."""


class PromptNoteStorageError(PromptNoteError):
    """Raised when persistence for prompt notes fails."""


class PromptVersionError(PromptManagerError):
    """Base class for prompt versioning workflow failures."""


class PromptVersionNotFoundError(PromptVersionError):
    """Raised when the requested prompt version or fork lineage entry is missing."""


class PromptShareError(PromptManagerError):
    """Base class for prompt sharing workflow failures."""


class ShareProviderError(PromptShareError):
    """Raised when a remote sharing provider rejects or fails a request."""


class PromptChainError(PromptManagerError):
    """Base class for prompt chain workflow failures."""


class PromptChainNotFoundError(PromptChainError):
    """Raised when a prompt chain cannot be located."""


class PromptChainStorageError(PromptChainError):
    """Raised when prompt chain persistence fails."""


class PromptChainExecutionError(PromptChainError):
    """Raised when executing a prompt chain fails."""
