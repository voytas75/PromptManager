"""Tests for PromptManager prompt execution workflow."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from core import (
    CodexExecutionResult,
    ExecutionError,
    HistoryTracker,
    PromptExecutionError,
    PromptManager,
    PromptRepository,
)
from models.prompt_model import Prompt


class _StubEmbeddingProvider:
    def embed(self, _: str) -> list[float]:
        return [0.0, 0.1, 0.2]


class _StubChromaCollection:
    def __init__(self) -> None:
        self.calls: Dict[str, list[str]] = {}

    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[Dict[str, Any]],
        embeddings: Optional[list[list[float]]] = None,
    ) -> None:
        self.calls["add"] = ids

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[Dict[str, Any]],
        embeddings: Optional[list[list[float]]] = None,
    ) -> None:
        self.calls["upsert"] = ids


class _StubChromaClient:
    def __init__(self) -> None:
        self.collection = _StubChromaCollection()

    def get_or_create_collection(
        self,
        name: str,
        metadata: Dict[str, Any],
        embedding_function: Optional[Any] = None,
    ) -> _StubChromaCollection:
        return self.collection

    def close(self) -> None:
        return


class _StubExecutor:
    def __init__(self) -> None:
        self.called_with: Optional[str] = None
        self.conversation_length: int = 0

    def execute(  # type: ignore[override]
        self,
        prompt: Prompt,
        request_text: str,
        *,
        conversation=None,
    ) -> CodexExecutionResult:
        self.called_with = request_text
        if conversation is not None:
            self.conversation_length = len(list(conversation))
        return CodexExecutionResult(
            prompt_id=prompt.id,
            request_text=request_text,
            response_text="Execution complete.",
            duration_ms=42,
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            raw_response={"choices": [{"message": {"content": "Execution complete."}}]},
        )


class _FailingExecutor:
    def execute(  # type: ignore[override]
        self,
        prompt: Prompt,
        request_text: str,
        *,
        conversation=None,
    ) -> CodexExecutionResult:
        raise ExecutionError("model timeout")


def _make_prompt(name: str = "Executable Prompt") -> Prompt:
    return Prompt(
        id=uuid.uuid4(),
        name=name,
        description="Prompt for execution tests",
        category="tests",
        context="Review the provided Python snippet for issues.",
    )


def _manager_with_dependencies(
    tmp_path: Path,
    executor: Any,
    *,
    history: bool = True,
) -> tuple[PromptManager, Prompt, HistoryTracker | None]:
    db_path = tmp_path / "prompt_manager.db"
    chroma_path = tmp_path / "chroma"
    repository = PromptRepository(str(db_path))
    prompt = _make_prompt()
    repository.add(prompt)
    tracker = HistoryTracker(repository) if history else None
    manager = PromptManager(
        chroma_path=str(chroma_path),
        db_path=str(db_path),
        cache_ttl_seconds=60,
        chroma_client=_StubChromaClient(),
        embedding_function=None,
        repository=repository,
        embedding_provider=_StubEmbeddingProvider(),
        enable_background_sync=False,
        executor=executor,
        history_tracker=tracker,
    )
    return manager, prompt, tracker


def test_execute_prompt_returns_outcome_and_logs_history(tmp_path: Path) -> None:
    executor = _StubExecutor()
    manager, prompt, tracker = _manager_with_dependencies(tmp_path, executor)

    outcome = manager.execute_prompt(prompt.id, "print('hello')")

    assert executor.called_with == "print('hello')"
    assert outcome.result.response_text == "Execution complete."
    assert outcome.history_entry is not None
    assert outcome.history_entry.status.value == "success"
    assert outcome.conversation == [
        {"role": "user", "content": "print('hello')"},
        {"role": "assistant", "content": "Execution complete."},
    ]
    assert outcome.history_entry.metadata is not None
    assert outcome.history_entry.metadata["conversation"] == outcome.conversation

    recent = manager.list_recent_executions()
    assert recent and recent[0].prompt_id == prompt.id

    history_for_prompt = manager.list_executions_for_prompt(prompt.id)
    assert history_for_prompt and history_for_prompt[0].prompt_id == prompt.id

    updated_prompt = manager.repository.get(prompt.id)
    assert updated_prompt.usage_count >= 1
    manager.close()


def test_execute_prompt_logs_failure(tmp_path: Path) -> None:
    manager, prompt, tracker = _manager_with_dependencies(tmp_path, _FailingExecutor())

    with pytest.raises(PromptExecutionError):
        manager.execute_prompt(prompt.id, "raise Exception")

    history = tracker.list_for_prompt(prompt.id)
    assert history and history[0].status.value == "failed"
    assert history[0].metadata is not None
    assert history[0].metadata["conversation"] == [
        {"role": "user", "content": "raise Exception"}
    ]
    manager.close()


def test_execute_prompt_supports_conversation(tmp_path: Path) -> None:
    executor = _StubExecutor()
    manager, prompt, tracker = _manager_with_dependencies(tmp_path, executor)

    first = manager.execute_prompt(prompt.id, "print('hello')")
    assert first.conversation[-1]["role"] == "assistant"

    follow_up = manager.execute_prompt(
        prompt.id,
        "Thanks!",
        conversation=first.conversation,
    )

    assert executor.conversation_length == len(first.conversation)
    assert follow_up.conversation == [
        {"role": "user", "content": "print('hello')"},
        {"role": "assistant", "content": "Execution complete."},
        {"role": "user", "content": "Thanks!"},
        {"role": "assistant", "content": "Execution complete."},
    ]
    assert follow_up.history_entry is not None
    assert follow_up.history_entry.metadata is not None
    assert follow_up.history_entry.metadata["conversation"] == follow_up.conversation
    manager.close()


def test_save_execution_result_records_manual_entry(tmp_path: Path) -> None:
    executor = _StubExecutor()
    manager, prompt, tracker = _manager_with_dependencies(tmp_path, executor)

    outcome = manager.execute_prompt(prompt.id, "print('hello')")
    manual = manager.save_execution_result(
        prompt.id,
        outcome.result.request_text,
        outcome.result.response_text,
        duration_ms=outcome.result.duration_ms,
        usage=outcome.result.usage,
        metadata={"note": "Reviewed manually"},
        rating=8,
    )
    assert manual.metadata and manual.metadata.get("note") == "Reviewed manually"
    assert manual.metadata.get("manual") is True
    assert manual.rating == 8
    entries = manager.list_recent_executions()
    assert any(entry.id == manual.id for entry in entries)
    refreshed_prompt = manager.get_prompt(prompt.id)
    assert refreshed_prompt.rating_count == 1
    assert refreshed_prompt.rating_sum == pytest.approx(8.0)
    assert refreshed_prompt.quality_score == pytest.approx(8.0)
    manager.close()


def test_update_execution_note(tmp_path: Path) -> None:
    executor = _StubExecutor()
    manager, prompt, tracker = _manager_with_dependencies(tmp_path, executor)

    outcome = manager.execute_prompt(prompt.id, "print('hi')")
    saved = manager.save_execution_result(
        prompt.id,
        outcome.result.request_text,
        outcome.result.response_text,
    )
    updated = manager.update_execution_note(saved.id, "Updated note")
    assert updated.metadata and updated.metadata.get("note") == "Updated note"
    manager.close()


def test_history_methods_without_tracker(tmp_path: Path) -> None:
    manager, prompt, tracker = _manager_with_dependencies(tmp_path, _StubExecutor(), history=False)
    assert tracker is None
    assert manager.list_recent_executions() == []
    assert manager.list_executions_for_prompt(prompt.id) == []
    manager.close()
