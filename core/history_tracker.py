"""Prompt execution history tracking utilities.

Updates: v0.3.1 - 2025-11-28 - Add per-prompt analytics helper for benchmarks and maintenance surfaces.
Updates: v0.3.0 - 2025-11-24 - Persist execution context metadata (response styles, runtime config).
Updates: v0.2.0 - 2025-11-09 - Capture optional ratings alongside execution logs.
Updates: v0.1.0 - 2025-11-08 - Introduce HistoryTracker for execution logs.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Mapping, Optional

from models.prompt_model import ExecutionStatus, PromptExecution

from .repository import PromptRepository, RepositoryError, RepositoryNotFoundError

_DEFAULT_MAX_REQUEST_LENGTH = 16_000
_DEFAULT_MAX_RESPONSE_LENGTH = 24_000

logger = logging.getLogger("prompt_manager.history_tracker")


class HistoryTrackerError(Exception):
    """Raised when execution history cannot be persisted."""


@dataclass(slots=True)
class PromptExecutionAnalytics:
    """Per-prompt aggregates for execution history."""

    prompt_id: uuid.UUID
    name: str
    total_runs: int
    success_rate: float
    average_duration_ms: Optional[float]
    average_rating: Optional[float]
    rating_trend: Optional[float]
    last_executed_at: Optional[datetime]


@dataclass(slots=True)
class ExecutionAnalytics:
    """Aggregated execution statistics across the catalogue."""

    total_runs: int
    success_rate: float
    average_duration_ms: Optional[float]
    average_rating: Optional[float]
    prompt_breakdown: List[PromptExecutionAnalytics]
    window_start: Optional[datetime] = None


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
        rating: Optional[float] = None,
        context_metadata: Optional[Mapping[str, object]] = None,
    ) -> PromptExecution:
        """Persist a successful execution."""
        metadata_payload: dict[str, object] = dict(metadata) if metadata else {}
        if context_metadata:
            metadata_payload["context"] = dict(context_metadata)
        execution = self._build_execution(
            prompt_id=prompt_id,
            request_text=request_text,
            response_text=response_text,
            status=ExecutionStatus.SUCCESS,
            duration_ms=duration_ms,
            metadata=metadata_payload or None,
            rating=rating,
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
        context_metadata: Optional[Mapping[str, object]] = None,
    ) -> PromptExecution:
        """Persist a failed execution attempt, including optional context metadata."""

        metadata_payload: dict[str, object] = dict(metadata) if metadata else {}
        if context_metadata:
            metadata_payload["context"] = dict(context_metadata)
        execution = self._build_execution(
            prompt_id=prompt_id,
            request_text=request_text,
            response_text="",
            status=ExecutionStatus.FAILED,
            error_message=error_message,
            duration_ms=duration_ms,
            metadata=metadata_payload or None,
            rating=None,
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
    def query_executions(
        self,
        *,
        status: Optional[ExecutionStatus] = None,
        prompt_id: Optional[uuid.UUID] = None,
        search: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[PromptExecution]:
        """Return executions filtered by the provided parameters."""

        status_value = status.value if isinstance(status, ExecutionStatus) else status
        search_term = search.strip() if search else None
        try:
            return self.repository.list_executions_filtered(
                status=status_value,
                prompt_id=prompt_id,
                search=search_term,
                limit=limit,
            )
        except RepositoryError as exc:
            raise HistoryTrackerError(f"Unable to query execution history: {exc}") from exc

    def summarize(
        self,
        *,
        window_days: Optional[int] = 30,
        prompt_limit: int = 5,
        trend_window: int = 5,
    ) -> ExecutionAnalytics:
        """Return aggregated execution metrics for the stored history."""

        since: Optional[datetime] = None
        if window_days is not None:
            since = datetime.now(timezone.utc) - timedelta(days=max(window_days, 0))
        try:
            summary_row, prompt_rows = self.repository.get_execution_analytics(
                since=since,
                limit=prompt_limit,
            )
        except RepositoryError as exc:
            raise HistoryTrackerError(f"Unable to compute execution analytics: {exc}") from exc

        total_runs = int(summary_row.get("total_runs", 0) or 0)
        success_runs = int(summary_row.get("success_runs", 0) or 0)
        average_duration = _coerce_float(summary_row.get("avg_duration_ms"))
        average_rating = _coerce_float(summary_row.get("avg_rating"))
        success_rate = success_runs / total_runs if total_runs else 0.0

        prompt_stats: List[PromptExecutionAnalytics] = []
        for row in prompt_rows:
            prompt_identifier = row.get("prompt_id")
            if not prompt_identifier:
                continue
            try:
                prompt_id = uuid.UUID(str(prompt_identifier))
            except (TypeError, ValueError):
                continue
            total = int(row.get("total_runs", 0) or 0)
            success_total = int(row.get("success_runs", 0) or 0)
            prompt_success_rate = success_total / total if total else 0.0
            prompt_average_duration = _coerce_float(row.get("avg_duration_ms"))
            prompt_average_rating = _coerce_float(row.get("avg_rating"))
            last_executed = _parse_datetime(row.get("last_executed_at"))
            rating_trend = self._compute_rating_trend(
                prompt_id,
                prompt_average_rating,
                trend_window,
            )
            prompt_stats.append(
                PromptExecutionAnalytics(
                    prompt_id=prompt_id,
                    name=str(row.get("prompt_name") or "Unknown Prompt"),
                    total_runs=total,
                    success_rate=prompt_success_rate,
                    average_duration_ms=prompt_average_duration,
                    average_rating=prompt_average_rating,
                    rating_trend=rating_trend,
                    last_executed_at=last_executed,
                )
            )

        return ExecutionAnalytics(
            total_runs=total_runs,
            success_rate=success_rate,
            average_duration_ms=average_duration,
            average_rating=average_rating,
            prompt_breakdown=prompt_stats,
            window_start=since,
        )

    def summarize_prompt(
        self,
        prompt_id: uuid.UUID,
        *,
        window_days: Optional[int] = None,
        trend_window: int = 5,
    ) -> Optional[PromptExecutionAnalytics]:
        """Return aggregate execution metrics for a specific prompt."""

        since: Optional[datetime] = None
        if window_days is not None:
            since = datetime.now(timezone.utc) - timedelta(days=max(window_days, 0))
        try:
            stats = self.repository.get_prompt_execution_statistics(prompt_id, since=since)
        except RepositoryError as exc:
            raise HistoryTrackerError(
                f"Unable to compute execution analytics for prompt {prompt_id}: {exc}"
            ) from exc

        total_runs = int(stats.get("total_runs", 0) or 0)
        if total_runs == 0:
            return None

        success_runs = int(stats.get("success_runs", 0) or 0)
        average_duration = _coerce_float(stats.get("avg_duration_ms"))
        average_rating = _coerce_float(stats.get("avg_rating"))
        last_executed = _parse_datetime(stats.get("last_executed_at"))
        rating_trend = self._compute_rating_trend(
            prompt_id,
            average_rating,
            trend_window,
        )

        return PromptExecutionAnalytics(
            prompt_id=prompt_id,
            name=str(stats.get("prompt_name") or ""),
            total_runs=total_runs,
            success_rate=success_runs / total_runs if total_runs else 0.0,
            average_duration_ms=average_duration,
            average_rating=average_rating,
            rating_trend=rating_trend,
            last_executed_at=last_executed,
        )

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
        rating: Optional[float] = None,
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
            rating=rating,
        )

    def _store(self, execution: PromptExecution) -> PromptExecution:
        """Persist the execution entry via the repository."""
        try:
            return self.repository.add_execution(execution)
        except RepositoryError as exc:
            raise HistoryTrackerError(f"Unable to persist execution {execution.id}: {exc}") from exc

    def update_note(self, execution_id: uuid.UUID, note: Optional[str]) -> PromptExecution:
        """Update or clear the note metadata for an execution."""

        try:
            existing = self.repository.get_execution(execution_id)
        except RepositoryNotFoundError as exc:
            raise HistoryTrackerError(str(exc)) from exc
        except RepositoryError as exc:
            raise HistoryTrackerError(f"Unable to load execution {execution_id}: {exc}") from exc

        metadata = dict(existing.metadata or {})
        if note:
            metadata["note"] = note
        else:
            metadata.pop("note", None)
        if not metadata:
            metadata = {}
        existing.metadata = metadata or None
        try:
            return self.repository.update_execution(existing)
        except RepositoryNotFoundError as exc:
            raise HistoryTrackerError(str(exc)) from exc
        except RepositoryError as exc:
            raise HistoryTrackerError(f"Unable to update execution {execution_id}: {exc}") from exc

    def _compute_rating_trend(
        self,
        prompt_id: uuid.UUID,
        baseline_average: Optional[float],
        trend_window: int,
    ) -> Optional[float]:
        if trend_window <= 0:
            return None
        try:
            recent = self.repository.list_executions_for_prompt(prompt_id, limit=trend_window)
        except RepositoryError as exc:
            logger.debug(
                "Unable to compute rating trend",
                exc_info=exc,
                extra={"prompt_id": str(prompt_id)},
            )
            return None
        ratings = [entry.rating for entry in recent if entry.rating is not None]
        if not ratings:
            return None
        recent_average = sum(ratings) / len(ratings)
        if baseline_average is None:
            return recent_average
        return recent_average - baseline_average


def _coerce_float(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: Optional[object]) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


__all__ = [
    "ExecutionAnalytics",
    "HistoryTracker",
    "HistoryTrackerError",
    "PromptExecutionAnalytics",
]
