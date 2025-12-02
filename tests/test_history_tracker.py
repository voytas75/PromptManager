"""Unit tests for HistoryTracker execution logging."""

from __future__ import annotations

import uuid

import pytest

from core.history_tracker import ExecutionAnalytics, HistoryTracker
from core.repository import PromptRepository
from models.prompt_model import Prompt


def _make_prompt(name: str = "Execution Test") -> Prompt:
    return Prompt(
        id=uuid.uuid4(),
        name=name,
        description="History tracker test prompt",
        category="tests",
        context="Review the provided Python snippet for issues.",
    )


def test_history_tracker_records_success_and_failure(tmp_path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    prompt = _make_prompt()
    repo.add(prompt)
    tracker = HistoryTracker(repo, max_request_chars=20, max_response_chars=20)

    request = "print('hello world that should be clipped')"
    success = tracker.record_success(
        prompt_id=prompt.id,
        request_text=request,
        response_text="All clear!",
        duration_ms=123,
        metadata={"usage": {"prompt_tokens": 5}},
        rating=7,
        context_metadata={
            "prompt": {"id": str(prompt.id), "category": prompt.category},
            "execution": {"model": "gpt-4o-mini"},
        },
    )
    assert success.response_text == "All clear!"
    assert success.request_text.endswith("...")
    assert success.duration_ms == 123
    assert success.rating == 7
    assert success.metadata is not None
    assert success.metadata.get("context", {}).get("prompt", {}).get("id") == str(prompt.id)

    failure = tracker.record_failure(
        prompt_id=prompt.id,
        request_text="raise ValueError('boom')",
        error_message="Timeout",
        context_metadata={
            "execution": {"model": "gpt-4o-mini", "stream_enabled": False},
            "prompt": {"id": str(prompt.id)},
        },
    )
    assert failure.status.value == "failed"
    assert "Timeout" == failure.error_message
    assert failure.metadata is not None
    assert failure.metadata.get("context", {}).get("execution", {}).get("model") == "gpt-4o-mini"

    recent = tracker.list_recent(limit=5)
    assert {entry.id for entry in recent} == {success.id, failure.id}

    by_prompt = tracker.list_for_prompt(prompt.id, limit=10)
    assert {entry.id for entry in by_prompt} == {success.id, failure.id}


def test_history_tracker_updates_note(tmp_path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    prompt = _make_prompt()
    repo.add(prompt)
    tracker = HistoryTracker(repo)

    execution = tracker.record_success(
        prompt_id=prompt.id,
        request_text="demo",
        response_text="result",
    )
    updated = tracker.update_note(execution.id, "Needs follow-up")
    assert updated.metadata and updated.metadata.get("note") == "Needs follow-up"

    cleared = tracker.update_note(execution.id, "")
    assert not (cleared.metadata or {}).get("note")


def test_history_tracker_summarize_returns_metrics(tmp_path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    prompt = _make_prompt()
    repo.add(prompt)
    tracker = HistoryTracker(repo)

    tracker.record_success(
        prompt_id=prompt.id,
        request_text="demo",
        response_text="ok",
        duration_ms=100,
        rating=4.0,
    )
    tracker.record_failure(
        prompt_id=prompt.id,
        request_text="demo",
        error_message="boom",
    )
    tracker.record_success(
        prompt_id=prompt.id,
        request_text="demo",
        response_text="better",
        duration_ms=50,
        rating=5.0,
    )

    analytics = tracker.summarize(window_days=None, prompt_limit=3, trend_window=2)
    assert isinstance(analytics, ExecutionAnalytics)
    assert analytics.total_runs == 3
    assert pytest.approx(analytics.success_rate, rel=1e-3) == 2 / 3
    assert analytics.prompt_breakdown, "Expected per-prompt breakdown"
    stats = analytics.prompt_breakdown[0]
    assert stats.total_runs == 3
    assert pytest.approx(stats.success_rate, rel=1e-3) == 2 / 3
    assert stats.rating_trend is not None
    assert stats.average_duration_ms and stats.average_duration_ms >= 50


def test_history_tracker_summarize_handles_empty_history(tmp_path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    tracker = HistoryTracker(repo)

    analytics = tracker.summarize(window_days=None)

    assert analytics.total_runs == 0
    assert analytics.prompt_breakdown == []


def test_history_tracker_summarize_prompt_returns_metrics(tmp_path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    prompt = _make_prompt()
    repo.add(prompt)
    tracker = HistoryTracker(repo)

    tracker.record_success(
        prompt_id=prompt.id,
        request_text="demo",
        response_text="ok",
        duration_ms=80,
        rating=4.5,
    )
    tracker.record_failure(
        prompt_id=prompt.id,
        request_text="demo",
        error_message="boom",
    )

    stats = tracker.summarize_prompt(prompt.id, window_days=None, trend_window=2)
    assert stats is not None
    assert stats.prompt_id == prompt.id
    assert stats.total_runs == 2
    assert stats.success_rate == pytest.approx(0.5)
    assert stats.average_duration_ms == pytest.approx(80.0)
    assert stats.average_rating == pytest.approx(4.5)
    assert stats.rating_trend is not None

    missing = tracker.summarize_prompt(uuid.uuid4(), window_days=None)
    assert missing is None
