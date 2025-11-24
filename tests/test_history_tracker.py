"""Unit tests for HistoryTracker execution logging."""

from __future__ import annotations

import uuid

from core.history_tracker import HistoryTracker
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
    )
    assert failure.status.value == "failed"
    assert "Timeout" == failure.error_message

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
