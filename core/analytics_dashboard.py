"""High-level analytics aggregation helpers for dashboards and diagnostics.

Updates:
  v0.1.3 - 2025-12-05 - Treat window filtering as day-based so bucket boundaries match dashboard expectations.
  v0.1.2 - 2025-11-30 - Move typing-only imports into TYPE_CHECKING for lint compliance.
  v0.1.1 - 2025-11-30 - Fix function docstring spacing for lint compliance.
  v0.1.0 - 2025-11-28 - Introduce analytics snapshot builder for CLI and GUI surfaces.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.prompt_manager import PromptHistoryError, PromptManager, PromptManagerError

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from collections.abc import Mapping
    from uuid import UUID

    from core.history_tracker import ExecutionAnalytics

logger = logging.getLogger("prompt_manager.analytics")


@dataclass(slots=True)
class UsageFrequencyEntry:
    """Summarise prompt usage volume for ranking charts."""

    prompt_id: UUID
    name: str
    usage_count: int
    success_rate: float | None
    last_executed_at: datetime | None


@dataclass(slots=True)
class ModelCostEntry:
    """Aggregated token usage per model."""

    model: str
    run_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(slots=True)
class BenchmarkStatsEntry:
    """Aggregated benchmark execution metrics."""

    model: str
    run_count: int
    success_rate: float
    average_duration_ms: float | None
    total_tokens: int


@dataclass(slots=True)
class IntentSuccessPoint:
    """Success ratio for intent executions bucketed by day."""

    bucket: datetime
    success_rate: float
    success: int
    total: int


@dataclass(slots=True)
class AnalyticsSnapshot:
    """Container for dashboard datasets."""

    execution: ExecutionAnalytics | None
    usage_frequency: list[UsageFrequencyEntry]
    model_costs: list[ModelCostEntry]
    benchmark_stats: list[BenchmarkStatsEntry]
    intent_success: list[IntentSuccessPoint]
    embedding: PromptManager.EmbeddingDiagnostics | None


def build_analytics_snapshot(
    manager: PromptManager,
    *,
    window_days: int = 30,
    prompt_limit: int = 5,
    usage_log_path: Path | None = None,
) -> AnalyticsSnapshot:
    """Collect aggregated metrics for dashboards and diagnostics outputs."""
    since: datetime | None
    if window_days <= 0:
        since = None
    else:
        since = datetime.now(UTC) - timedelta(days=window_days)

    usage_path = (
        Path(usage_log_path)
        if usage_log_path is not None
        else Path("data") / "logs" / "intent_usage.jsonl"
    )

    execution_summary: ExecutionAnalytics | None = None
    try:
        execution_summary = manager.get_execution_analytics(
            window_days=window_days if window_days > 0 else None,
            prompt_limit=prompt_limit,
            trend_window=5,
        )
    except PromptHistoryError as exc:
        logger.debug("Unable to load execution analytics", exc_info=exc)

    usage_frequency = _collect_usage_frequency(manager, limit=prompt_limit, since=since)
    model_costs = _collect_model_costs(manager, since=since)
    benchmark_stats = _collect_benchmark_stats(manager, since=since)
    intent_success = _collect_intent_success_trend(usage_path, since=since)

    embedding_summary: PromptManager.EmbeddingDiagnostics | None = None
    try:
        embedding_summary = manager.diagnose_embeddings()
    except PromptManagerError as exc:
        logger.debug("Embedding diagnostics unavailable", exc_info=exc)

    return AnalyticsSnapshot(
        execution=execution_summary,
        usage_frequency=usage_frequency,
        model_costs=model_costs,
        benchmark_stats=benchmark_stats,
        intent_success=intent_success,
        embedding=embedding_summary,
    )


def snapshot_dataset_rows(snapshot: AnalyticsSnapshot, dataset: str) -> list[dict[str, object]]:
    """Return dataset rows for CSV export based on the requested key."""
    key = dataset.lower()
    if key == "usage":
        return [
            {
                "prompt_id": str(entry.prompt_id),
                "prompt_name": entry.name,
                "usage_count": entry.usage_count,
                "success_rate": (
                    round(entry.success_rate * 100, 2) if entry.success_rate is not None else None
                ),
                "last_executed_at": (
                    entry.last_executed_at.isoformat() if entry.last_executed_at else None
                ),
            }
            for entry in snapshot.usage_frequency
        ]
    if key == "model_costs":
        return [
            {
                "model": entry.model,
                "run_count": entry.run_count,
                "prompt_tokens": entry.prompt_tokens,
                "completion_tokens": entry.completion_tokens,
                "total_tokens": entry.total_tokens,
            }
            for entry in snapshot.model_costs
        ]
    if key == "benchmark":
        return [
            {
                "model": entry.model,
                "run_count": entry.run_count,
                "success_rate": round(entry.success_rate * 100, 2),
                "average_duration_ms": entry.average_duration_ms,
                "total_tokens": entry.total_tokens,
            }
            for entry in snapshot.benchmark_stats
        ]
    if key == "intent":
        return [
            {
                "bucket": point.bucket.date().isoformat(),
                "success_rate": round(point.success_rate * 100, 2),
                "success": point.success,
                "total": point.total,
            }
            for point in snapshot.intent_success
        ]
    if key == "embedding":
        report = snapshot.embedding
        if report is None:
            return []
        return [
            {
                "backend_ok": report.backend_ok,
                "backend_message": report.backend_message,
                "backend_dimension": report.backend_dimension,
                "inferred_dimension": report.inferred_dimension,
                "chroma_ok": report.chroma_ok,
                "chroma_message": report.chroma_message,
                "chroma_count": report.chroma_count,
                "repository_total": report.repository_total,
                "prompts_with_embeddings": report.prompts_with_embeddings,
                "missing_prompts": len(report.missing_prompts),
                "mismatched_prompts": len(report.mismatched_prompts),
                "consistent_counts": report.consistent_counts,
            }
        ]
    raise ValueError(f"Unknown analytics dataset: {dataset}")


def _collect_usage_frequency(
    manager: PromptManager,
    *,
    limit: int,
    since: datetime | None,
) -> list[UsageFrequencyEntry]:
    try:
        prompts = manager.repository.list()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Unable to list prompts for usage analytics", exc_info=exc)
        return []

    if not prompts:
        return []

    ranked = sorted(prompts, key=lambda prompt: prompt.usage_count, reverse=True)
    top_prompts = ranked[: max(1, limit)]
    entries: list[UsageFrequencyEntry] = []
    for prompt in top_prompts:
        stats: Mapping[str, Any]
        try:
            stats = manager.repository.get_prompt_execution_statistics(prompt.id, since=since)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Unable to load usage stats for prompt", exc_info=exc)
            stats = {}
        total_runs = _coerce_int(stats.get("total_runs"), default=0)
        success_runs = _coerce_int(stats.get("success_runs"), default=0)
        success_rate: float | None = success_runs / total_runs if total_runs else None
        last_executed = stats.get("last_executed_at")
        timestamp = _coerce_datetime(last_executed)
        entries.append(
            UsageFrequencyEntry(
                prompt_id=prompt.id,
                name=prompt.name,
                usage_count=prompt.usage_count,
                success_rate=success_rate,
                last_executed_at=timestamp,
            )
        )
    return entries


def _collect_model_costs(
    manager: PromptManager,
    *,
    since: datetime | None,
) -> list[ModelCostEntry]:
    try:
        rows = manager.repository.get_model_usage_breakdown(since=since)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Unable to compute model usage breakdown", exc_info=exc)
        return []
    entries: list[ModelCostEntry] = []
    for row in rows:
        model = str(row.get("model") or "unknown").strip() or "unknown"
        entries.append(
            ModelCostEntry(
                model=model,
                run_count=_coerce_int(row.get("run_count"), default=0),
                prompt_tokens=_coerce_int(row.get("prompt_tokens"), default=0),
                completion_tokens=_coerce_int(row.get("completion_tokens"), default=0),
                total_tokens=_coerce_int(row.get("total_tokens"), default=0),
            )
        )
    return entries


def _collect_benchmark_stats(
    manager: PromptManager,
    *,
    since: datetime | None,
) -> list[BenchmarkStatsEntry]:
    try:
        rows = manager.repository.get_benchmark_execution_stats(since=since)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Unable to compute benchmark analytics", exc_info=exc)
        return []
    entries: list[BenchmarkStatsEntry] = []
    for row in rows:
        model = str(row.get("model") or "unknown").strip() or "unknown"
        total_runs = int(row.get("total_runs", 0) or 0)
        success_runs = int(row.get("success_runs", 0) or 0)
        success_rate = success_runs / total_runs if total_runs else 0.0
        avg_duration = row.get("avg_duration_ms")
        try:
            avg_duration_value: float | None = (
                float(avg_duration) if avg_duration is not None else None
            )
        except (TypeError, ValueError):  # pragma: no cover - defensive
            avg_duration_value = None
        entries.append(
            BenchmarkStatsEntry(
                model=model,
                run_count=total_runs,
                success_rate=success_rate,
                average_duration_ms=avg_duration_value,
                total_tokens=_coerce_int(row.get("total_tokens"), default=0),
            )
        )
    return entries


def _collect_intent_success_trend(
    usage_path: Path,
    *,
    since: datetime | None,
) -> list[IntentSuccessPoint]:
    if not usage_path.exists():
        return []
    try:
        lines = usage_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:  # pragma: no cover - defensive
        logger.debug("Unable to read usage log", exc_info=exc)
        return []

    day_threshold: datetime | None = None
    if since is not None:
        tzinfo = since.tzinfo or UTC
        day_threshold = datetime.combine(since.date(), datetime.min.time(), tzinfo=tzinfo)

    buckets: dict[date, dict[str, int]] = {}
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event") != "execute":
            continue
        timestamp = _coerce_datetime(payload.get("timestamp"))
        if timestamp is None:
            continue
        if day_threshold is not None and timestamp < day_threshold:
            continue
        bucket_day = timestamp.date()
        bucket = buckets.setdefault(bucket_day, {"success": 0, "total": 0})
        bucket["total"] += 1
        if payload.get("success") is True:
            bucket["success"] += 1
    points: list[IntentSuccessPoint] = []
    for bucket_day in sorted(buckets.keys()):
        data = buckets[bucket_day]
        total = data["total"]
        success = data["success"]
        success_rate = success / total if total else 0.0
        bucket_dt = datetime.combine(bucket_day, datetime.min.time(), tzinfo=UTC)
        points.append(
            IntentSuccessPoint(
                bucket=bucket_dt,
                success_rate=success_rate,
                success=success,
                total=total,
            )
        )
    return points


def _coerce_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time(), tzinfo=UTC)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed_dt = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed_dt.tzinfo is None:
            parsed_dt = parsed_dt.replace(tzinfo=UTC)
        return parsed_dt
    return None


def _coerce_int(value: object | None, *, default: int = 0) -> int:
    if value is None:
        return default
    coerced_value: Any = value
    try:
        return int(coerced_value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "AnalyticsSnapshot",
    "BenchmarkStatsEntry",
    "IntentSuccessPoint",
    "ModelCostEntry",
    "UsageFrequencyEntry",
    "build_analytics_snapshot",
    "snapshot_dataset_rows",
]
