"""Execution workflow and history mixin for Prompt Manager.

Updates:
  v0.1.0 - 2025-12-03 - Extract execution and history APIs into dedicated mixin.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from models.prompt_model import ExecutionStatus, Prompt, PromptExecution

from ..exceptions import (
    PromptExecutionError,
    PromptExecutionUnavailable,
    PromptHistoryError,
    PromptManagerError,
)
from ..execution import CodexExecutionResult, CodexExecutor, ExecutionError
from ..history_tracker import (
    ExecutionAnalytics,
    HistoryTrackerError,
    PromptExecutionAnalytics,
)
from ..notifications import NotificationCenter, NotificationLevel

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import Protocol

    from ..history_tracker import HistoryTracker

    class _PromptAccessor(Protocol):
        def get_prompt(self, prompt_id: uuid.UUID) -> Prompt: ...

        def increment_usage(self, prompt_id: uuid.UUID) -> None: ...

        def _apply_rating(self, prompt_id: uuid.UUID, rating: float) -> None: ...

else:  # pragma: no cover - typing fallback
    HistoryTracker = Any

logger = logging.getLogger(__name__)

__all__ = [
    "ExecutionHistoryMixin",
    "ExecutionOutcome",
    "BenchmarkRun",
    "BenchmarkReport",
]


def _normalise_conversation(
    messages: Sequence[Mapping[str, str]] | None,
) -> list[dict[str, str]]:
    """Return a sanitised copy of conversation messages for execution and logging."""
    normalised: list[dict[str, str]] = []
    if not messages:
        return normalised
    for index, message in enumerate(messages):
        role = str(message.get("role", "")).strip()
        if not role:
            raise PromptExecutionError(f"Conversation entry {index} is missing a role.")
        content = message.get("content")
        if content is None:
            raise PromptExecutionError(f"Conversation entry {index} is missing content.")
        normalised.append({"role": role, "content": str(content)})
    return normalised


@dataclass(slots=True)
class ExecutionOutcome:
    """Aggregate data returned after executing a prompt."""

    result: CodexExecutionResult
    history_entry: PromptExecution | None
    conversation: list[dict[str, str]]


@dataclass(slots=True)
class BenchmarkRun:
    """Single benchmark result for a prompt/model pair."""

    prompt_id: uuid.UUID
    prompt_name: str
    model: str
    duration_ms: int | None
    usage: Mapping[str, Any]
    response_preview: str
    error: str | None
    history: PromptExecutionAnalytics | None


@dataclass(slots=True)
class BenchmarkReport:
    """Structured response returned by benchmark_prompts."""

    runs: list[BenchmarkRun]


class ExecutionHistoryMixin:
    """Shared execution workflows, benchmarking, and history logging helpers."""

    _notification_center: NotificationCenter
    _executor: CodexExecutor | None
    _history_tracker: HistoryTracker | None
    _litellm_fast_model: str | None
    _litellm_inference_model: str | None

    def _build_execution_context_metadata(
        self,
        prompt: Prompt,
        *,
        stream_enabled: bool,
        executor_model: str | None,
        conversation_length: int,
        request_text: str,
        response_text: str,
        response_style: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return structured metadata describing the execution context."""
        prompt_metadata = {
            "id": str(prompt.id),
            "name": prompt.name,
            "category": prompt.category,
            "tags": list(prompt.tags),
            "version": prompt.version,
        }
        execution_metadata = {
            "model": executor_model,
            "stream_enabled": stream_enabled,
            "conversation_messages": conversation_length,
            "request_chars": len(request_text or ""),
            "response_chars": len(response_text or ""),
        }
        context: dict[str, Any] = {
            "prompt": prompt_metadata,
            "execution": execution_metadata,
        }
        if response_style:
            context["response_style"] = dict(response_style)
        return context

    def execute_prompt(
        self,
        prompt_id: uuid.UUID,
        request_text: str,
        *,
        conversation: Sequence[Mapping[str, str]] | None = None,
        stream: bool | None = None,
        on_stream: Callable[[str], None] | None = None,
    ) -> ExecutionOutcome:
        """Execute a prompt via LiteLLM and persist the outcome when configured."""
        if not request_text.strip():
            raise PromptExecutionError("Prompt execution requires non-empty input text.")
        if self._executor is None:
            raise PromptExecutionUnavailable(
                "Prompt execution is not configured. Provide LiteLLM credentials and model."
            )

        conversation_history = _normalise_conversation(conversation)
        deps = cast("_PromptAccessor", self)
        prompt = deps.get_prompt(prompt_id)
        stream_enabled = self._executor.stream if stream is None else bool(stream)
        task_id = f"prompt-exec:{prompt.id}:{uuid.uuid4()}"
        metadata = {
            "prompt_id": str(prompt.id),
            "prompt_name": prompt.name,
            "request_length": len(request_text or ""),
        }
        with self._notification_center.track_task(
            title="Prompt execution",
            task_id=task_id,
            start_message=f"Running '{prompt.name}' via LiteLLM…",
            success_message=f"Completed '{prompt.name}'.",
            failure_message=f"Prompt execution failed for '{prompt.name}'",
            metadata=metadata,
            level=NotificationLevel.INFO,
        ):
            try:
                result = self._executor.execute(
                    prompt,
                    request_text,
                    conversation=conversation_history,
                    stream=stream,
                    on_stream=on_stream,
                )
            except ExecutionError as exc:
                failed_messages = list(conversation_history)
                failed_messages.append({"role": "user", "content": request_text.strip()})
                failure_context = self._build_execution_context_metadata(
                    prompt,
                    stream_enabled=stream_enabled,
                    executor_model=getattr(self._executor, "model", None),
                    conversation_length=len(failed_messages),
                    request_text=request_text,
                    response_text="",
                )
                self._log_execution_failure(
                    prompt.id,
                    request_text,
                    str(exc),
                    conversation=failed_messages,
                    context_metadata=failure_context,
                )
                raise PromptExecutionError(str(exc)) from exc

            augmented_conversation = list(conversation_history)
            augmented_conversation.append({"role": "user", "content": request_text.strip()})
            if result.response_text:
                augmented_conversation.append(
                    {"role": "assistant", "content": result.response_text}
                )
            context_metadata = self._build_execution_context_metadata(
                prompt,
                stream_enabled=stream_enabled,
                executor_model=getattr(self._executor, "model", None),
                conversation_length=len(augmented_conversation),
                request_text=request_text,
                response_text=result.response_text,
            )
            history_entry = self._log_execution_success(
                prompt.id,
                request_text,
                result,
                conversation=augmented_conversation,
                context_metadata=context_metadata,
            )
            try:
                deps.increment_usage(prompt.id)
            except PromptManagerError:
                logger.debug(
                    "Prompt executed but usage counter update failed",
                    extra={"prompt_id": str(prompt.id)},
                    exc_info=True,
                )

        return ExecutionOutcome(
            result=result,
            history_entry=history_entry,
            conversation=augmented_conversation,
        )

    def benchmark_prompts(
        self,
        prompt_ids: Sequence[uuid.UUID],
        request_text: str,
        *,
        models: Sequence[str] | None = None,
        persist_history: bool = False,
        history_window_days: int | None = 30,
        trend_window: int = 5,
    ) -> BenchmarkReport:
        """Execute prompts across one or more models and return benchmark data."""
        if not prompt_ids:
            raise PromptExecutionError("At least one prompt must be provided for benchmarking.")
        text = (request_text or "").strip()
        if not text:
            raise PromptExecutionError("Benchmark execution requires non-empty input text.")
        if self._executor is None:
            raise PromptExecutionUnavailable(
                "Prompt execution is not configured. Provide LiteLLM credentials and model."
            )

        unique_prompt_ids: list[uuid.UUID] = []
        seen_prompts: set[uuid.UUID] = set()
        for prompt_id in prompt_ids:
            if prompt_id in seen_prompts:
                continue
            seen_prompts.add(prompt_id)
            unique_prompt_ids.append(prompt_id)

        deps = cast("_PromptAccessor", self)
        prompts: list[Prompt] = [deps.get_prompt(prompt_id) for prompt_id in unique_prompt_ids]

        model_candidates: list[str] = []
        if models:
            for candidate in models:
                value = str(candidate or "").strip()
                if value and value not in model_candidates:
                    model_candidates.append(value)
        if not model_candidates:
            for candidate in (
                getattr(self._executor, "model", None),
                self._litellm_fast_model,
                self._litellm_inference_model,
            ):
                if candidate and candidate not in model_candidates:
                    model_candidates.append(str(candidate))
        if not model_candidates:
            raise PromptExecutionUnavailable("No LiteLLM models are configured for benchmarking.")

        base_executor = self._executor

        def _executor_for_model(model_name: str) -> CodexExecutor:
            if base_executor.model == model_name:
                return base_executor
            if not isinstance(base_executor, CodexExecutor):
                raise PromptExecutionUnavailable(
                    "Only configured LiteLLM executors can benchmark against multiple models."
                )
            drop_params = list(base_executor.drop_params) if base_executor.drop_params else None
            return CodexExecutor(
                model=model_name,
                api_key=base_executor.api_key,
                api_base=base_executor.api_base,
                api_version=base_executor.api_version,
                timeout_seconds=base_executor.timeout_seconds,
                max_output_tokens=base_executor.max_output_tokens,
                temperature=base_executor.temperature,
                drop_params=drop_params,
                reasoning_effort=base_executor.reasoning_effort,
                stream=base_executor.stream,
            )

        tracker = self._history_tracker
        prompt_analytics: dict[uuid.UUID, PromptExecutionAnalytics | None] = {}
        if tracker is not None:
            for prompt in prompts:
                try:
                    prompt_analytics[prompt.id] = tracker.summarize_prompt(
                        prompt.id,
                        window_days=history_window_days,
                        trend_window=trend_window,
                    )
                except HistoryTrackerError:
                    prompt_analytics[prompt.id] = None
        else:
            prompt_analytics = {prompt.id: None for prompt in prompts}

        runs: list[BenchmarkRun] = []
        for prompt in prompts:
            history = prompt_analytics.get(prompt.id)
            for model_name in model_candidates:
                executor_for_model = _executor_for_model(model_name)
                try:
                    result = executor_for_model.execute(
                        prompt,
                        text,
                        conversation=None,
                        stream=False,
                        on_stream=None,
                    )
                except ExecutionError as exc:
                    runs.append(
                        BenchmarkRun(
                            prompt_id=prompt.id,
                            prompt_name=prompt.name,
                            model=model_name,
                            duration_ms=None,
                            usage={},
                            response_preview="",
                            error=str(exc),
                            history=history,
                        )
                    )
                    if persist_history:
                        self._log_execution_failure(
                            prompt.id,
                            text,
                            str(exc),
                            conversation=[{"role": "user", "content": text}],
                            context_metadata=None,
                            extra_metadata={"benchmark": True, "model": model_name},
                        )
                    continue

                try:
                    usage_data = dict(result.usage)
                except Exception:  # pragma: no cover - defensive
                    usage_data = {}

                preview = (result.response_text or "").strip()
                if len(preview) > 400:
                    preview = preview[:397].rstrip() + "…"

                runs.append(
                    BenchmarkRun(
                        prompt_id=prompt.id,
                        prompt_name=prompt.name,
                        model=model_name,
                        duration_ms=result.duration_ms,
                        usage=usage_data,
                        response_preview=preview,
                        error=None,
                        history=history,
                    )
                )

                if persist_history:
                    conversation_payload = [
                        {"role": "user", "content": text},
                        {"role": "assistant", "content": result.response_text},
                    ]
                    context_metadata = self._build_execution_context_metadata(
                        prompt,
                        stream_enabled=bool(getattr(executor_for_model, "stream", False)),
                        executor_model=model_name,
                        conversation_length=len(conversation_payload),
                        request_text=text,
                        response_text=result.response_text,
                    )
                    self._log_execution_success(
                        prompt.id,
                        text,
                        result,
                        conversation=conversation_payload,
                        context_metadata=context_metadata,
                        extra_metadata={"benchmark": True, "model": model_name},
                    )

        return BenchmarkReport(runs=runs)

    def save_execution_result(
        self,
        prompt_id: uuid.UUID,
        request_text: str,
        response_text: str,
        *,
        duration_ms: int | None = None,
        usage: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        rating: float | None = None,
        context_metadata: Mapping[str, Any] | None = None,
    ) -> PromptExecution:
        """Persist a manual prompt execution entry (e.g., from GUI Save Result)."""
        tracker = self._history_tracker
        if tracker is None:
            raise PromptExecutionUnavailable(
                "Execution history is not configured; cannot save results manually."
            )
        payload_metadata: dict[str, Any] = {"manual": True}
        if metadata:
            payload_metadata.update(dict(metadata))
        if usage:
            payload_metadata.setdefault("usage", dict(usage))
        if rating is not None:
            payload_metadata["rating"] = rating
        try:
            execution = tracker.record_success(
                prompt_id=prompt_id,
                request_text=request_text,
                response_text=response_text,
                duration_ms=duration_ms,
                metadata=payload_metadata,
                rating=rating,
                context_metadata=context_metadata,
            )
        except HistoryTrackerError as exc:
            raise PromptHistoryError(str(exc)) from exc
        if rating is not None:
            cast("Any", self)._apply_rating(prompt_id, rating)
        return execution

    def update_execution_note(self, execution_id: uuid.UUID, note: str | None) -> PromptExecution:
        """Update the note metadata for a history entry."""
        tracker = self._history_tracker
        if tracker is None:
            raise PromptExecutionUnavailable(
                "Execution history is not configured; cannot update saved results."
            )
        try:
            return tracker.update_note(execution_id, note)
        except HistoryTrackerError as exc:
            raise PromptHistoryError(str(exc)) from exc

    def list_recent_executions(self, *, limit: int = 20) -> list[PromptExecution]:
        """Return recently logged executions if history tracking is enabled."""
        tracker = self._history_tracker
        if tracker is None:
            return []
        try:
            return tracker.list_recent(limit=limit)
        except HistoryTrackerError:
            logger.warning("Unable to list execution history", exc_info=True)
            return []

    def list_executions_for_prompt(
        self,
        prompt_id: uuid.UUID,
        *,
        limit: int = 20,
    ) -> list[PromptExecution]:
        """Return execution history for a specific prompt."""
        tracker = self._history_tracker
        if tracker is None:
            return []
        try:
            return tracker.list_for_prompt(prompt_id, limit=limit)
        except HistoryTrackerError:
            logger.warning(
                "Unable to list execution history for prompt",
                extra={"prompt_id": str(prompt_id)},
                exc_info=True,
            )
            return []

    def query_executions(
        self,
        *,
        status: ExecutionStatus | str | None = None,
        prompt_id: uuid.UUID | None = None,
        search: str | None = None,
        limit: int | None = None,
    ) -> list[PromptExecution]:
        """Return executions filtered by status, prompt, and search term."""
        tracker = self._history_tracker
        if tracker is None:
            return []
        status_filter: ExecutionStatus | None = None
        if isinstance(status, ExecutionStatus):
            status_filter = status
        elif isinstance(status, str):
            try:
                status_filter = ExecutionStatus(status)
            except ValueError:
                logger.debug("Ignoring unknown execution status filter", extra={"status": status})
        try:
            return tracker.query_executions(
                status=status_filter,
                prompt_id=prompt_id,
                search=search,
                limit=limit,
            )
        except HistoryTrackerError:
            logger.warning("Unable to query execution history", exc_info=True)
            return []

    def get_execution_analytics(
        self,
        *,
        window_days: int | None = 30,
        prompt_limit: int = 5,
        trend_window: int = 5,
    ) -> ExecutionAnalytics | None:
        """Return aggregated execution analytics for downstream consumers."""
        tracker = self._history_tracker
        if tracker is None:
            return None
        try:
            return tracker.summarize(
                window_days=window_days,
                prompt_limit=prompt_limit,
                trend_window=trend_window,
            )
        except HistoryTrackerError as exc:
            raise PromptHistoryError(str(exc)) from exc

    def _log_execution_success(
        self,
        prompt_id: uuid.UUID,
        request_text: str,
        result: CodexExecutionResult,
        *,
        conversation: Sequence[Mapping[str, str]] | None = None,
        context_metadata: Mapping[str, Any] | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> PromptExecution | None:
        """Persist a successful execution outcome when the tracker is available."""
        tracker = self._history_tracker
        if tracker is None:
            return None
        usage_metadata: dict[str, Any] = {}
        try:
            usage_metadata = dict(result.usage)
        except Exception:  # pragma: no cover - defensive
            usage_metadata = {}
        metadata: dict[str, Any] = {"usage": usage_metadata}
        if conversation:
            metadata["conversation"] = list(conversation)
        if extra_metadata:
            for key, value in extra_metadata.items():
                metadata[key] = value
        try:
            return tracker.record_success(
                prompt_id=prompt_id,
                request_text=request_text,
                response_text=result.response_text,
                duration_ms=result.duration_ms,
                metadata=metadata,
                context_metadata=context_metadata,
            )
        except HistoryTrackerError:
            logger.warning(
                "Prompt executed but history logging failed",
                extra={"prompt_id": str(prompt_id)},
                exc_info=True,
            )
            return None

    def _log_execution_failure(
        self,
        prompt_id: uuid.UUID,
        request_text: str,
        error_message: str,
        *,
        conversation: Sequence[Mapping[str, str]] | None = None,
        context_metadata: Mapping[str, Any] | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> PromptExecution | None:
        """Persist a failed execution attempt when history tracking is enabled."""
        tracker = self._history_tracker
        if tracker is None:
            return None
        metadata_payload: dict[str, Any] = {}
        if conversation:
            metadata_payload["conversation"] = list(conversation)
        if extra_metadata:
            for key, value in extra_metadata.items():
                metadata_payload[key] = value
        try:
            return tracker.record_failure(
                prompt_id=prompt_id,
                request_text=request_text,
                error_message=error_message,
                metadata=metadata_payload or None,
                context_metadata=context_metadata,
            )
        except HistoryTrackerError:
            logger.warning(
                "Prompt execution failed and could not be logged",
                extra={"prompt_id": str(prompt_id)},
                exc_info=True,
            )
            return None
