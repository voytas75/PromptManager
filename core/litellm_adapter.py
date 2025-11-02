"""Shared LiteLLM adapters for Prompt Manager.

Updates: v0.6.2 - 2025-11-14 - Raise actionable error when LiteLLM is missing.
Updates: v0.6.1 - 2025-11-07 - Tolerate LiteLLM builds without embedding support.
Updates: v0.6.0 - 2025-11-07 - Provide shared completion/embedding import helpers.
"""

from __future__ import annotations

import importlib
from typing import Callable, Optional, Tuple, Type


class LiteLLMNotInstalledError(RuntimeError):
    """Raised when LiteLLM is not available in the current environment."""

_completion: Optional[Callable[..., object]] = None
_embedding: Optional[Callable[..., object]] = None
_LiteLLMException: Type[Exception] = Exception


def _ensure_loaded() -> None:
    """Import LiteLLM helpers lazily to keep the dependency optional."""

    global _completion, _embedding, _LiteLLMException
    if _completion is not None:
        return
    try:  # pragma: no cover - runtime import path
        litellm = importlib.import_module("litellm")
    except ImportError as exc:
        raise LiteLLMNotInstalledError(
            "LiteLLM integration requires the optional dependency 'litellm'. "
            "Install it with `pip install litellm` or `pip install .[llm]`."
        ) from exc

    completion = getattr(litellm, "completion", None)
    if completion is None:
        raise RuntimeError("litellm completion API is unavailable in the installed version.")
    embedding = getattr(litellm, "embedding", None)

    try:
        exceptions_module = importlib.import_module("litellm.exceptions")
        LiteLLMException = getattr(exceptions_module, "LiteLLMException", Exception)
    except ImportError:
        LiteLLMException = Exception  # type: ignore[assignment]

    _completion = completion
    _embedding = embedding
    _LiteLLMException = LiteLLMException  # type: ignore[misc]


def get_completion() -> Tuple[Callable[..., object], Type[Exception]]:
    """Return the LiteLLM completion callable and exception type."""

    _ensure_loaded()
    assert _completion is not None  # pragma: no cover - defensive
    return _completion, _LiteLLMException


def get_embedding() -> Tuple[Callable[..., object], Type[Exception]]:
    """Return the LiteLLM embedding callable and exception type."""

    _ensure_loaded()
    if _embedding is None:
        raise RuntimeError(
            "litellm embedding API is unavailable. Install a version that exposes `litellm.embedding` "
            "or configure a different embedding backend."
        )
    return _embedding, _LiteLLMException


__all__ = ["get_completion", "get_embedding", "LiteLLMNotInstalledError"]
