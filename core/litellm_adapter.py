"""Shared LiteLLM adapters for Prompt Manager.

Updates:
  v0.7.2 - 2025-11-29 - Gate typing-only collection imports for Ruff TC003 compliance.
  v0.7.1 - 2025-11-29 - Reformat docstring and wrap long retry diagnostics strings.
  v0.7.0 - 2025-11-02 - Strip drop parameters before LiteLLM retries to match provider support.
  v0.6.3 - 2025-11-17 - Retry completion calls without unsupported parameters rejected by models.
  v0.6.2 - 2025-11-14 - Raise actionable error when LiteLLM is missing.
  v0.6.1 - 2025-11-07 - Tolerate LiteLLM builds without embedding support.
  v0.6.0 - 2025-11-07 - Provide shared completion/embedding import helpers.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence


class LiteLLMNotInstalledError(RuntimeError):
    """Raised when LiteLLM is not available in the current environment."""

_completion: Callable[..., object] | None = None
_embedding: Callable[..., object] | None = None
_LiteLLMException: type[Exception] = Exception


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


def get_completion() -> tuple[Callable[..., object], type[Exception]]:
    """Return the LiteLLM completion callable and exception type."""

    _ensure_loaded()
    assert _completion is not None  # pragma: no cover - defensive
    return _completion, _LiteLLMException


def get_embedding() -> tuple[Callable[..., object], type[Exception]]:
    """Return the LiteLLM embedding callable and exception type."""

    _ensure_loaded()
    if _embedding is None:
        raise RuntimeError(
            "litellm embedding API is unavailable. Install a version that exposes "
            "`litellm.embedding` or configure a different embedding backend."
        )
    return _embedding, _LiteLLMException


def call_completion_with_fallback(
    request: dict[str, object],
    completion: Callable[..., object],
    lite_llm_exception: type[Exception],
    *,
    drop_candidates: Iterable[str] | None = None,
    pre_dropped: Iterable[str] | None = None,
) -> object:
    """Invoke LiteLLM completion and retry without unsupported params if necessary."""

    try:
        return completion(**request)  # type: ignore[arg-type]
    except lite_llm_exception as exc:  # type: ignore[arg-type]
        existing_drop: set[str] = set()
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
    request: dict[str, object],
    drop_params: Sequence[str] | None,
) -> tuple[str, ...]:
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
    seen: set[str] = set()
    ordered_unique = [item for item in dropped if not (item in seen or seen.add(item))]
    return tuple(ordered_unique)


def _detect_unsupported_parameters(
    message: str,
    parameters: Iterable[str],
    drop_candidates: Iterable[str] | None = None,
) -> set[str]:
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

    candidates = set(
        drop_candidates or {"max_tokens", "max_output_tokens", "temperature", "timeout"}
    )
    unsupported: set[str] = set()
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
