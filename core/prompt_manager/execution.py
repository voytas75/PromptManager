"""Execution façade for Prompt Manager.

This module provides lightweight wrappers around the existing low‑level
executor classes (e.g. :class:`core.execution.CodexExecutor`).  The goal is to
decouple the high‑level PromptManager API from direct dependency on a specific
implementation so that we may later introduce alternative execution back‑ends
(streaming, batch, mock for testing) without touching calling code.

Updates: v0.14.0 – 2025‑11‑18 – Initial scaffold with proxy implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
        model_name: str,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self._executor = CodexExecutor(
            model_name,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )

    # ------------------------------------------------------------------
    # Delegations – we purposefully expose only minimal subset. Additional
    # methods can be added as they are consumed by PromptManager.
    # ------------------------------------------------------------------

    def execute(
        self,
        messages: Sequence[Mapping[str, str]] | None,
        *,
        stream: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExecutionResult:
        """Run the messages through the underlying executor."""

        return self._executor.execute(messages, stream=stream, metadata=metadata)

    async def aexecute(
        self,
        messages: Sequence[Mapping[str, str]] | None,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExecutionResult:  # noqa: D401
        """Async variant of :meth:`execute`."""

        return await self._executor.aexecute(messages, metadata=metadata)

    # Helper: save run transcript to disk – useful in debugging.
    def save_transcript(self, conversation: Iterable[str], path: str | Path) -> None:  # noqa: D401,E501
        self._executor.save_transcript(conversation, path)
