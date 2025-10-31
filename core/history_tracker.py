"""Prompt execution history tracking utilities.

Updates: v0.1.0 - 2025-11-08 - Introduce HistoryTracker for execution logs.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Mapping, Optional

from models.prompt_model import ExecutionStatus, PromptExecution

from .repository import PromptRepository, RepositoryError, RepositoryNotFoundError

_DEFAULT_MAX_REQUEST_LENGTH = 16_000
_DEFAULT_MAX_RESPONSE_LENGTH = 24_000


class HistoryTrackerError(Exception):
    """Raised when execution history cannot be persisted."""


def _clip(text: Optional[str], max_length: int) -> str:
    """Return a whitespace-trimmed string clipped to max_length."""
    if not text:
        return ""
    trimmed = text.strip()
    if len(trimmed) > max_length:
        return trimmed[: max_length - 3] + "..."
    return trimmed


@dataclass(slots=True)
class HistoryTracker:
    """Manage prompt execution history using the SQLite repository."""

    repository: PromptRepository
    max_request_chars: int = _DEFAULT_MAX_REQUEST_LENGTH
    max_response_chars: int = _DEFAULT_MAX_RESPONSE_LENGTH

    def record_success(
        self,
        prompt_id: uuid.UUID,
        request_text: str,
        response_text: str,
        *,
        duration_ms: Optional[int] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> PromptExecution:
        """Persist a successful execution."""
        execution = self._build_execution(
            prompt_id=prompt_id,
            request_text=request_text,
            response_text=response_text,
            status=ExecutionStatus.SUCCESS,
            duration_ms=duration_ms,
            metadata=metadata,
        )
        return self._store(execution)

    def record_failure(
        self,
        prompt_id: uuid.UUID,
        request_text: str,
        error_message: str,
        *,
        duration_ms: Optional[int] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> PromptExecution:
        """Persist a failed execution attempt."""
        execution = self._build_execution(
            prompt_id=prompt_id,
            request_text=request_text,
            response_text="",
            status=ExecutionStatus.FAILED,
            error_message=error_message,
            duration_ms=duration_ms,
            metadata=metadata,
        )
        return self._store(execution)

    def get(self, execution_id: uuid.UUID) -> PromptExecution:
        """Return a single execution entry."""
        try:
            return self.repository.get_execution(execution_id)
        except RepositoryNotFoundError as exc:
            raise HistoryTrackerError(str(exc)) from exc
        except RepositoryError as exc:
            raise HistoryTrackerError(f"Unable to fetch execution {execution_id}: {exc}") from exc

    def list_recent(self, *, limit: int = 20) -> List[PromptExecution]:
        """Return recent executions."""
        try:
            return self.repository.list_executions(limit=limit)
        except RepositoryError as exc:
            raise HistoryTrackerError(f"Unable to list executions: {exc}") from exc

    def list_for_prompt(
        self,
        prompt_id: uuid.UUID,
        *,
        limit: int = 20,
    ) -> List[PromptExecution]:
        """Return recent executions for a specific prompt."""
        try:
            return self.repository.list_executions_for_prompt(prompt_id, limit=limit)
        except RepositoryError as exc:
            raise HistoryTrackerError(
                f"Unable to list executions for prompt {prompt_id}: {exc}"
            ) from exc

    def _build_execution(
        self,
        *,
        prompt_id: uuid.UUID,
        request_text: str,
        response_text: str,
        status: ExecutionStatus,
        duration_ms: Optional[int],
        metadata: Optional[Mapping[str, object]],
        error_message: Optional[str] = None,
        executed_at: Optional[datetime] = None,
    ) -> PromptExecution:
        """Create a PromptExecution dataclass instance with sanitised payloads."""

        execution_metadata = dict(metadata) if metadata else None
        return PromptExecution(
            id=uuid.uuid4(),
            prompt_id=prompt_id,
            request_text=_clip(request_text, self.max_request_chars),
            response_text=_clip(response_text, self.max_response_chars) or None,
            status=status,
            error_message=error_message.strip() if error_message else None,
            duration_ms=duration_ms,
            executed_at=executed_at or datetime.now(timezone.utc),
            metadata=execution_metadata,
        )

    def _store(self, execution: PromptExecution) -> PromptExecution:
        """Persist the execution entry via the repository."""
        try:
            return self.repository.add_execution(execution)
        except RepositoryError as exc:
            raise HistoryTrackerError(f"Unable to persist execution {execution.id}: {exc}") from exc


__all__ = ["HistoryTracker", "HistoryTrackerError"]
