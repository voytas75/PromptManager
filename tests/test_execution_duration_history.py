"""Controlled-clock execution duration and real SQLite history regressions."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from core import PromptExecutionError, PromptRepository, execution
from core.execution import CodexExecutor, ExecutionError
from models.prompt_model import ExecutionStatus
from tests.test_prompt_manager_execution import (
    _make_prompt,  # pyright: ignore[reportPrivateUsage]
    _manager_with_dependencies,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path


class _LiteLLMError(Exception):
    """Provider-like exception without a provider dependency."""


@dataclass
class _Clock:
    milliseconds: int = 10_000

    def perf_counter(self) -> float:
        return self.milliseconds / 1000

    def advance(self, milliseconds: int) -> None:
        self.milliseconds += milliseconds


@dataclass
class _Completion:
    clock: _Clock
    interruption: type[Exception] | None = None
    calls: int = 0
    events: list[str] = field(default_factory=lambda: list[str]())

    def __call__(self, **request: Any) -> object:
        self.calls += 1
        self.events.append("request")
        self.clock.advance(250)
        if request.get("stream"):
            # Return a lazy iterator: no chunk work happens during the request.
            return self.stream()
        return _Response(self.clock)

    def stream(self) -> Iterator[dict[str, Any]]:
        self.events.append("first")
        self.clock.advance(500)
        yield {"choices": [{"delta": {"content": "Hello"}}]}
        self.clock.advance(750)
        if self.interruption is not None:
            self.events.append("interrupted")
            raise self.interruption("synthetic interruption")
        self.events.append("second")
        yield {"choices": [{"delta": {"content": " world "}}]}
        self.events.append("usage")
        self.clock.advance(250)
        yield {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 6}}
        self.events.append("exhausted")
        self.clock.advance(250)

    def get_completion(self) -> tuple[Callable[..., Any], type[Exception]]:
        return self, _LiteLLMError


@dataclass
class _Response:
    clock: _Clock

    def model_dump(self) -> dict[str, Any]:
        # Local non-stream processing must remain outside the request duration.
        self.clock.advance(500)
        return {
            "choices": [{"message": {"content": "Hello world "}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 6},
        }


@pytest.fixture
def completion(monkeypatch: pytest.MonkeyPatch) -> _Completion:
    clock = _Clock()
    fake = _Completion(clock)
    # Replace this module's time reference, not the global time.perf_counter.
    monkeypatch.setattr(execution, "time", SimpleNamespace(perf_counter=clock.perf_counter))
    monkeypatch.setattr(execution, "get_completion", fake.get_completion)
    return fake


_USAGE = {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}


@pytest.mark.parametrize("callback_mode", ["none", "collect", "raise"])
def test_stream_duration_covers_request_chunks_usage_and_exhaustion(
    completion: _Completion, callback_mode: str
) -> None:
    chunks: list[str] = []

    def on_stream(delta: str) -> None:
        chunks.append(delta)
        completion.clock.advance(125)
        if callback_mode == "raise":
            raise RuntimeError("synthetic callback failure")

    result = CodexExecutor(model="gpt-test").execute(
        _make_prompt(),
        "Say hello",
        stream=True,
        on_stream=None if callback_mode == "none" else on_stream,
    )

    assert completion.calls == 1
    assert completion.events == ["request", "first", "second", "usage", "exhausted"]
    assert chunks == ([] if callback_mode == "none" else ["Hello", " world "])
    assert result.response_text == "Hello world"
    assert result.usage == _USAGE
    assert result.raw_response["usage"] == _USAGE
    assert result.raw_response["streamed"] is True
    assert len(cast("list[object]", result.raw_response["chunks"])) == 3
    assert result.duration_ms == (2000 if callback_mode == "none" else 2250)


def test_non_stream_duration_excludes_local_response_processing(completion: _Completion) -> None:
    result = CodexExecutor(model="gpt-test", stream=True).execute(
        _make_prompt(), "Say hello", stream=False
    )

    assert completion.calls == 1
    assert completion.events == ["request"]
    assert completion.clock.milliseconds == 10_750
    assert result.response_text == "Hello world"
    assert result.usage == _USAGE
    assert result.duration_ms == 250


@pytest.mark.parametrize(
    ("interruption", "message"),
    [
        (_LiteLLMError, "Streaming interrupted: synthetic interruption"),
        (ValueError, "Unexpected error while streaming LiteLLM response"),
    ],
)
def test_interrupted_stream_raises_without_success_result(
    completion: _Completion, interruption: type[Exception], message: str
) -> None:
    completion.interruption = interruption
    chunks: list[str] = []
    with pytest.raises(ExecutionError, match=f"^{message}$") as error:
        CodexExecutor(model="gpt-test").execute(
            _make_prompt(), "Say hello", stream=True, on_stream=chunks.append
        )

    assert isinstance(error.value.__cause__, interruption)
    assert completion.calls == 1
    assert chunks == ["Hello"]
    assert completion.events == ["request", "first", "interrupted"]


@pytest.mark.parametrize("stream", [True, False])
def test_public_manager_returns_full_duration(
    tmp_path: Path, completion: _Completion, stream: bool
) -> None:
    manager, prompt, _ = _manager_with_dependencies(
        tmp_path, CodexExecutor(model="gpt-test", stream=stream)
    )
    try:
        outcome = manager.execute_prompt(prompt.id, "Say hello")
        assert outcome.result.response_text == "Hello world"
        assert outcome.result.usage == _USAGE
        assert outcome.history_entry is not None
        assert outcome.history_entry.status is ExecutionStatus.SUCCESS
        assert outcome.result.duration_ms == (2000 if stream else 250)
        assert outcome.history_entry.duration_ms == outcome.result.duration_ms
    finally:
        manager.close()


@pytest.mark.parametrize("stream", [True, False])
def test_public_manager_persists_full_duration_in_real_sqlite(
    tmp_path: Path, completion: _Completion, stream: bool
) -> None:
    manager, prompt, _ = _manager_with_dependencies(
        tmp_path, CodexExecutor(model="gpt-test", stream=stream)
    )
    try:
        outcome = manager.execute_prompt(prompt.id, "Say hello")
        assert outcome.history_entry is not None
        execution_id = outcome.history_entry.id
    finally:
        manager.close()

    db_path = tmp_path / "prompt_manager.db"
    # A separate connection verifies the committed row, not the returned object.
    with closing(sqlite3.connect(db_path)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT * FROM prompt_executions").fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == str(execution_id)
    assert row["prompt_id"] == str(prompt.id)
    assert row["status"] == "success"
    assert row["response_text"] == "Hello world"
    assert json.loads(row["metadata"])["usage"] == _USAGE
    assert row["duration_ms"] == (2000 if stream else 250)
    assert (
        PromptRepository(str(db_path)).get_execution(execution_id).duration_ms == row["duration_ms"]
    )


@pytest.mark.parametrize("interruption", [_LiteLLMError, ValueError])
def test_public_manager_persists_interruption_without_success_or_partial_tokens(
    tmp_path: Path, completion: _Completion, interruption: type[Exception]
) -> None:
    completion.interruption = interruption
    manager, prompt, _ = _manager_with_dependencies(tmp_path, CodexExecutor(model="gpt-test"))
    chunks: list[str] = []
    try:
        with pytest.raises(PromptExecutionError) as error:
            manager.execute_prompt(prompt.id, "Say hello", stream=True, on_stream=chunks.append)
        assert isinstance(error.value.__cause__, ExecutionError)
        assert isinstance(error.value.__cause__.__cause__, interruption)
        assert manager.repository.get(prompt.id).usage_count == 0
        entries = manager.list_executions_for_prompt(prompt.id)
        assert len(entries) == 1
        assert entries[0].status is ExecutionStatus.FAILED
        assert entries[0].duration_ms is None
        totals = manager.get_token_usage_totals()
        assert (totals.prompt_tokens, totals.completion_tokens, totals.total_tokens) == (0, 0, 0)
    finally:
        manager.close()

    assert completion.calls == 1
    assert chunks == ["Hello"]
    with closing(sqlite3.connect(tmp_path / "prompt_manager.db")) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT * FROM prompt_executions").fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "failed"
    assert row["duration_ms"] is None
    assert not row["response_text"]
    metadata = json.loads(row["metadata"])
    assert "usage" not in metadata
    assert metadata["conversation"] == [{"role": "user", "content": "Say hello"}]
