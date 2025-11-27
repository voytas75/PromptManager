"""Execution façade for Prompt Manager.

This module provides lightweight wrappers around the existing low‑level
executor classes (e.g. :class:`core.execution.CodexExecutor`).  The goal is to
decouple the high‑level PromptManager API from direct dependency on a specific
implementation so that we may later introduce alternative execution back‑ends
(streaming, batch, mock for testing) without touching calling code.

Updates: v0.14.1 – 2025‑11‑27 – Align façade with CodexExecutor prompt-centric API.
Updates: v0.14.0 – 2025‑11‑18 – Initial scaffold with proxy implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence

from models.prompt_model import Prompt

from ..execution import CodexExecutionResult, CodexExecutor, ExecutionError

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
        api_key: Optional[str] = None,
        *,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
        drop_params: Optional[Sequence[str]] = None,
        reasoning_effort: Optional[str] = None,
        stream: bool = False,
    ) -> None:
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
        conversation: Optional[Sequence[Mapping[str, str]]] = None,
        stream: Optional[bool] = None,
        on_stream: Optional[Callable[[str], None]] = None,
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
