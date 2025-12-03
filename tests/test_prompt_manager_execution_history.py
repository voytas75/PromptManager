"""Unit tests for execution history mixin."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from core.execution import CodexExecutionResult, CodexExecutor
from core.notifications import NotificationCenter, NotificationLevel
from core.prompt_manager.execution_history import ExecutionHistoryMixin
from models.prompt_model import ExecutionStatus, Prompt, PromptExecution

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from core.history_tracker import HistoryTracker
else:  # pragma: no cover - type fallback
    Callable = Any
    Mapping = Any
    Sequence = Any
    HistoryTracker = Any


def _make_prompt() -> Prompt:
    return Prompt(
        id=uuid.uuid4(),
        name="Test Prompt",
        description="Prompt for execution history tests",
        category="tests",
        context="Reply with a summary of the request.",
    )


class _NotificationStub(NotificationCenter):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    @contextmanager
    def track_task(
        self,
        *,
        title: str,
        task_id: str | None = None,
        start_message: str,
        success_message: str,
        failure_message: str | None = None,
        metadata: dict[str, Any] | None = None,
        level: NotificationLevel = NotificationLevel.INFO,
        failure_level: NotificationLevel = NotificationLevel.ERROR,
    ) -> Any:
        self.calls.append(
            {
                "title": title,
                "task_id": task_id,
                "start": start_message,
                "success": success_message,
                "failure": failure_message,
                "metadata": metadata,
                "level": level,
                "failure_level": failure_level,
            }
        )
        yield


class _ExecutorStub(CodexExecutor):
    def __init__(self) -> None:
        super().__init__(model="stub-model")
        self.calls: list[str] = []

    def execute(  # type: ignore[override]
        self,
        prompt: Prompt,
        request_text: str,
        *,
        conversation: Sequence[Mapping[str, str]] | None = None,
        stream: bool | None = None,
        on_stream: Callable[[str], None] | None = None,
    ) -> CodexExecutionResult:
        self.calls.append(request_text)
        return CodexExecutionResult(
            prompt_id=prompt.id,
            request_text=request_text,
            response_text="Execution complete.",
            duration_ms=25,
            usage={"prompt_tokens": 5, "completion_tokens": 7},
            raw_response={"id": "stub"},
        )


@dataclass
class _HistoryStub:
    successes: int = 0
    manual_successes: int = 0

    def record_success(
        self,
        *,
        prompt_id: uuid.UUID,
        request_text: str,
        response_text: str | None,
        duration_ms: int | None,
        metadata: dict[str, Any] | None,
        context_metadata: dict[str, Any] | None,
        rating: float | None = None,
    ) -> PromptExecution:
        self.successes += 1
        if metadata and metadata.get("manual"):
            self.manual_successes += 1
        return PromptExecution(
            id=uuid.uuid4(),
            prompt_id=prompt_id,
            request_text=request_text,
            response_text=response_text,
            status=ExecutionStatus.SUCCESS,
            duration_ms=duration_ms,
            metadata=metadata,
        )

    def record_failure(self, **_: Any) -> None:
        return None

    def update_note(self, execution_id: uuid.UUID, note: str | None) -> PromptExecution:
        return PromptExecution(
            id=execution_id,
            prompt_id=uuid.uuid4(),
            request_text="",
            response_text=None,
            status=ExecutionStatus.SUCCESS,
        )

    def list_recent(self, *, limit: int = 20) -> list[PromptExecution]:
        return []

    def list_for_prompt(self, prompt_id: uuid.UUID, *, limit: int = 20) -> list[PromptExecution]:
        return []

    def query_executions(
        self,
        *,
        status: ExecutionStatus | None,
        prompt_id: uuid.UUID | None,
        search: str | None,
        limit: int | None,
    ) -> list[PromptExecution]:
        return []

    def summarize(self, *, window_days: int | None, prompt_limit: int, trend_window: int):
        return None

    def summarize_prompt(
        self,
        prompt_id: uuid.UUID,
        *,
        window_days: int | None,
        trend_window: int,
    ) -> None:
        return None


class _ExecutionHarness(ExecutionHistoryMixin):
    def __init__(self, prompt: Prompt) -> None:
        self._notification_center = _NotificationStub()
        self._executor = _ExecutorStub()
        self._history_tracker = cast("HistoryTracker", _HistoryStub())
        self._litellm_fast_model = None
        self._litellm_inference_model = None
        self._prompts = {prompt.id: prompt}
        self.usage_calls: list[uuid.UUID] = []
        self.rating_calls: list[tuple[uuid.UUID, float]] = []

    def get_prompt(self, prompt_id: uuid.UUID) -> Prompt:
        return self._prompts[prompt_id]

    def increment_usage(self, prompt_id: uuid.UUID) -> None:
        self.usage_calls.append(prompt_id)

    def _apply_rating(self, prompt_id: uuid.UUID, rating: float) -> None:
        self.rating_calls.append((prompt_id, rating))

    @property
    def executor(self) -> _ExecutorStub:
        return cast("_ExecutorStub", self._executor)

    @property
    def tracker(self) -> _HistoryStub:
        return cast("_HistoryStub", self._history_tracker)

    @property
    def notifications(self) -> _NotificationStub:
        return cast("_NotificationStub", self._notification_center)


def test_execute_prompt_records_history_and_usage() -> None:
    prompt = _make_prompt()
    harness = _ExecutionHarness(prompt)

    outcome = harness.execute_prompt(prompt.id, "Summarise this code snippet.")

    assert outcome.history_entry is not None
    assert harness.executor.calls == ["Summarise this code snippet."]
    assert harness.usage_calls == [prompt.id]
    assert harness.notifications.calls, "Notification task should be recorded"
    assert harness.tracker.successes == 1


def test_save_execution_result_persists_manual_entry_and_rating() -> None:
    prompt = _make_prompt()
    harness = _ExecutionHarness(prompt)

    entry = harness.save_execution_result(
        prompt.id,
        "Manual input",
        "Manual response",
        rating=0.75,
        usage={"prompt_tokens": 3},
        metadata={"source": "test"},
    )

    assert entry.response_text == "Manual response"
    assert harness.rating_calls == [(prompt.id, 0.75)]
    assert harness.tracker.manual_successes == 1
