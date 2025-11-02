"""Shared LiteLLM adapters for Prompt Manager.

Updates: v0.7.0 - 2025-11-02 - Strip configured and retry drop parameters instead of forwarding them to providers.
Updates: v0.6.3 - 2025-11-17 - Retry completion calls without unsupported parameters when models reject them.
Updates: v0.6.2 - 2025-11-14 - Raise actionable error when LiteLLM is missing.
Updates: v0.6.1 - 2025-11-07 - Tolerate LiteLLM builds without embedding support.
Updates: v0.6.0 - 2025-11-07 - Provide shared completion/embedding import helpers.
"""

from __future__ import annotations

import importlib
import logging
from typing import Callable, Dict, Iterable, Optional, Sequence, Set, Tuple, Type


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


def call_completion_with_fallback(
    request: Dict[str, object],
    completion: Callable[..., object],
    lite_llm_exception: Type[Exception],
    *,
    drop_candidates: Optional[Iterable[str]] = None,
    pre_dropped: Optional[Iterable[str]] = None,
) -> object:
    """Invoke LiteLLM completion and retry without unsupported params if necessary."""

    try:
        return completion(**request)  # type: ignore[arg-type]
    except lite_llm_exception as exc:  # type: ignore[arg-type]
        existing_drop: Set[str] = set()
        if pre_dropped:
            existing_drop.update(
                str(item).strip() for item in pre_dropped if str(item).strip()
            )
        unsupported = _detect_unsupported_parameters(
            str(exc), request.keys(), drop_candidates
        )
        if not unsupported:
            raise
        trimmed_request = {
            key: value for key, value in request.items() if key not in unsupported
        }
        merged_drop = sorted(existing_drop.union(unsupported))
        logging.getLogger("prompt_manager.litellm").info(
            "LiteLLM model rejected parameters %s; retrying request without them.",
            ", ".join(merged_drop),
        )
        return completion(**trimmed_request)  # type: ignore[arg-type]


def apply_configured_drop_params(
    request: Dict[str, object],
    drop_params: Optional[Sequence[str]],
) -> Tuple[str, ...]:
    """Remove configured parameters from request dict and return the dropped set."""

    if not drop_params:
        return tuple()

    dropped: list[str] = []
    for raw_key in drop_params:
        key = str(raw_key).strip()
        if not key:
            continue
        if key in request:
            request.pop(key, None)
            dropped.append(key)
    seen: Set[str] = set()
    ordered_unique = [item for item in dropped if not (item in seen or seen.add(item))]
    return tuple(ordered_unique)


def _detect_unsupported_parameters(
    message: str,
    parameters: Iterable[str],
    drop_candidates: Optional[Iterable[str]] = None,
) -> Set[str]:
    lowered = message.lower()
    indicators = (
        "not support",
        "unsupported",
        "not allowed",
        "additional property",
        "additional properties",
        "unexpected",
        "unknown",
    )
    if not any(token in lowered for token in indicators):
        return set()

    candidates = set(drop_candidates or {"max_tokens", "max_output_tokens", "temperature", "timeout"})
    unsupported: Set[str] = set()
    for key in parameters:
        if key not in candidates:
            continue
        key_forms = (key, key.replace("_", " "), key.replace("_", "-"))
        if any(form in lowered for form in key_forms):
            unsupported.add(key)
    return unsupported


__all__ = [
    "get_completion",
    "get_embedding",
    "LiteLLMNotInstalledError",
    "call_completion_with_fallback",
    "apply_configured_drop_params",
]
