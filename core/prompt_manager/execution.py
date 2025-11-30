"""Execution façade for Prompt Manager.

This module provides lightweight wrappers around the existing low-level
executor classes (e.g. :class:`core.execution.CodexExecutor`). The goal is to
decouple the high-level PromptManager API from direct dependency on a specific
implementation so that we may later introduce alternative execution back-ends
(streaming, batch, mock for testing) without touching calling code.

Updates:
  v0.14.3 - 2025-11-30 - Document initializer and tighten docstring spacing.
  v0.14.2 - 2025-11-29 - Move typing-only imports behind TYPE_CHECKING for Ruff.
  v0.14.1 - 2025-11-27 - Align façade with CodexExecutor prompt-centric API.
  v0.14.0 - 2025-11-18 - Initial scaffold with proxy implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..execution import CodexExecutionResult, CodexExecutor, ExecutionError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from models.prompt_model import Prompt

__all__ = [
    "ExecutionError",
    "ExecutionResult",
    "PromptExecutor",
]

# Public alias to keep call‑sites descriptive
ExecutionResult = CodexExecutionResult


class PromptExecutor:
    """Facade over :class:`core.execution.CodexExecutor`."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        api_base: str | None = None,
        api_version: str | None = None,
        timeout_seconds: float | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
        drop_params: Sequence[str] | None = None,
        reasoning_effort: str | None = None,
        stream: bool = False,
    ) -> None:
        """Initialise executor façade with optional LiteLLM tuning parameters."""
        self._executor = CodexExecutor(
            model=model,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            drop_params=drop_params,
            reasoning_effort=reasoning_effort,
            stream=stream,
        )

    def execute(
        self,
        prompt: Prompt,
        request_text: str,
        *,
        conversation: Sequence[Mapping[str, str]] | None = None,
        stream: bool | None = None,
        on_stream: Callable[[str], None] | None = None,
    ) -> ExecutionResult:
        """Run ``request_text`` against ``prompt`` via LiteLLM."""
        return self._executor.execute(
            prompt,
            request_text,
            conversation=conversation,
            stream=stream,
            on_stream=on_stream,
        )

    def save_transcript(self, conversation: Iterable[str], path: str | Path) -> None:
        """Persist a plain-text transcript for debugging sessions."""
        resolved_path = Path(path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [str(line) for line in conversation]
        resolved_path.write_text("\n".join(lines), encoding="utf-8")
