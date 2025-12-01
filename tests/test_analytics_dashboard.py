"""Unit tests for analytics snapshot helpers."""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from core import (
    AnalyticsSnapshot,
    PromptManager,
    PromptRepository,
    build_analytics_snapshot,
    snapshot_dataset_rows,
)
from core.history_tracker import ExecutionAnalytics, PromptExecutionAnalytics
from models.prompt_model import ExecutionStatus, Prompt, PromptExecution

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path


class _StubAnalyticsManager:
    def __init__(
        self,
        repository: PromptRepository,
        execution: ExecutionAnalytics,
        embedding_report: PromptManager.EmbeddingDiagnostics,
    ) -> None:
        self.repository = repository
        self._execution = execution
        self._embedding = embedding_report

    def get_execution_analytics(
        self,
        *,
        window_days: int | None = None,
        prompt_limit: int = 5,
        trend_window: int = 5,
    ) -> ExecutionAnalytics:
        return self._execution

    def diagnose_embeddings(self) -> PromptManager.EmbeddingDiagnostics:
        return self._embedding


def _make_prompt(name: str, usage_count: int) -> Prompt:
    return Prompt(
        id=uuid.uuid4(),
        name=name,
        description=f"{name} description",
        category="tests",
        usage_count=usage_count,
    )


def _add_execution(
    repository: PromptRepository,
    *,
    prompt_id: uuid.UUID,
    status: ExecutionStatus = ExecutionStatus.SUCCESS,
    metadata: dict[str, Any] | None = None,
) -> None:
    execution = PromptExecution(
        id=uuid.uuid4(),
        prompt_id=prompt_id,
        request_text="demo",
        response_text="ok",
        status=status,
        duration_ms=120,
        executed_at=datetime.now(UTC),
        metadata=metadata,
    )
    repository.add_execution(execution)


def test_build_analytics_snapshot_aggregates_repository_data(tmp_path: Path) -> None:
    db_path = tmp_path / "analytics.db"
    repo = PromptRepository(str(db_path))
    prompt_a = _make_prompt("Alpha", 10)
    prompt_b = _make_prompt("Beta", 6)
    repo.add(prompt_a)
    repo.add(prompt_b)

    _add_execution(
        repo,
        prompt_id=prompt_a.id,
        metadata={
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "context": {"execution": {"model": "gpt-fast"}},
        },
    )
    _add_execution(
        repo,
        prompt_id=prompt_a.id,
        metadata={
            "usage": {"prompt_tokens": 5, "completion_tokens": 15, "total_tokens": 20},
            "context": {"execution": {"model": "gpt-fast"}},
        },
    )
    _add_execution(
        repo,
        prompt_id=prompt_b.id,
        metadata={
            "benchmark": True,
            "model": "gpt-benchmark",
            "usage": {"total_tokens": 40},
        },
    )

    execution_stats = ExecutionAnalytics(
        total_runs=3,
        success_rate=0.75,
        average_duration_ms=150.0,
        average_rating=4.4,
        prompt_breakdown=[
            PromptExecutionAnalytics(
                prompt_id=prompt_a.id,
                name=prompt_a.name,
                total_runs=2,
                success_rate=1.0,
                average_duration_ms=140.0,
                average_rating=4.8,
                rating_trend=0.1,
                last_executed_at=datetime.now(UTC),
            )
        ],
        window_start=datetime.now(UTC),
    )

    embedding_report = PromptManager.EmbeddingDiagnostics(
        backend_ok=True,
        backend_message="ok",
        backend_dimension=32,
        inferred_dimension=None,
        chroma_ok=True,
        chroma_message="ok",
        chroma_count=2,
        repository_total=2,
        prompts_with_embeddings=2,
        missing_prompts=[],
        mismatched_prompts=[],
        consistent_counts=True,
    )

    manager = _StubAnalyticsManager(repo, execution_stats, embedding_report)

    usage_log = tmp_path / "intent_usage.jsonl"
    usage_log.write_text(
        "\n".join(
            [
                '{"event": "execute", "timestamp": "2025-11-28T10:00:00+00:00", "success": true}',
                '{"event": "execute", "timestamp": "2025-11-28T11:00:00+00:00", "success": false}',
            ]
        ),
        encoding="utf-8",
    )

    snapshot = build_analytics_snapshot(
        manager,
        window_days=7,
        prompt_limit=2,
        usage_log_path=usage_log,
    )

    assert snapshot.execution is execution_stats
    assert snapshot.embedding is embedding_report
    assert snapshot.model_costs and snapshot.model_costs[0].total_tokens == 50
    assert snapshot.benchmark_stats and snapshot.benchmark_stats[0].model == "gpt-benchmark"
    assert snapshot.usage_frequency and snapshot.usage_frequency[0].name == "Alpha"
    assert snapshot.intent_success and snapshot.intent_success[0].success == 1
    assert snapshot.intent_success[0].total == 2


def test_snapshot_dataset_rows_serialises_embedding_report() -> None:
    snapshot = AnalyticsSnapshot(
        execution=None,
        usage_frequency=[],
        model_costs=[],
        benchmark_stats=[],
        intent_success=[],
        embedding=PromptManager.EmbeddingDiagnostics(
            backend_ok=False,
            backend_message="error",
            backend_dimension=None,
            inferred_dimension=64,
            chroma_ok=False,
            chroma_message="down",
            chroma_count=None,
            repository_total=3,
            prompts_with_embeddings=1,
            missing_prompts=[],
            mismatched_prompts=[],
            consistent_counts=False,
        ),
    )

    rows = snapshot_dataset_rows(snapshot, "embedding")

    assert rows
    assert rows[0]["backend_ok"] is False
    assert rows[0]["repository_total"] == 3
