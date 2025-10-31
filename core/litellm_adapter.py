"""Shared LiteLLM adapters for Prompt Manager.

Updates: v0.6.0 - 2025-11-07 - Provide shared completion/embedding import helpers.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Type

_completion: Optional[Callable[..., object]] = None
_embedding: Optional[Callable[..., object]] = None
_LiteLLMException: Type[Exception] = Exception


def _ensure_loaded() -> None:
    """Import LiteLLM helpers lazily to keep the dependency optional."""

    global _completion, _embedding, _LiteLLMException
    if _completion is not None and _embedding is not None:
        return
    try:  # pragma: no cover - runtime import path
        from litellm import completion as litellm_completion, embedding as litellm_embedding
        try:
            from litellm.exceptions import LiteLLMException as litellm_exception  # type: ignore[attr-defined]
        except ImportError:
            litellm_exception = Exception  # type: ignore[assignment]
    except ImportError as exc:
        raise RuntimeError("litellm is not installed") from exc
    _completion = litellm_completion
    _embedding = litellm_embedding
    _LiteLLMException = litellm_exception  # type: ignore[misc]


def get_completion() -> Tuple[Callable[..., object], Type[Exception]]:
    """Return the LiteLLM completion callable and exception type."""

    _ensure_loaded()
    assert _completion is not None  # pragma: no cover - defensive
    return _completion, _LiteLLMException


def get_embedding() -> Tuple[Callable[..., object], Type[Exception]]:
    """Return the LiteLLM embedding callable and exception type."""

    _ensure_loaded()
    assert _embedding is not None  # pragma: no cover - defensive
    return _embedding, _LiteLLMException


__all__ = ["get_completion", "get_embedding"]
