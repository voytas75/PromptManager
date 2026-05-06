"""Prompt chain orchestration helpers for Prompt Manager.

Updates:
  v0.2.2 - 2025-12-09 - Surface offline-friendly messaging when execution is disabled.
  v0.2.1 - 2025-12-07 - Wrap web search context with shared formatting helpers.
  v0.2.0 - 2025-12-06 - Switch chains to plain-text inputs and linear step piping.
  v0.1.5 - 2025-12-06 - Honor prompt template overrides for chain summaries.
  v0.1.4 - 2025-12-06 - Summarize final chain output via LiteLLM when chains opt in.
  v0.1.3 - 2025-12-05 - Add optional web search enrichment ahead of each chain step.
  v0.1.2 - 2025-12-05 - Summarize the last step response when chains request it.
  v0.1.1 - 2025-12-05 - Surface streaming callbacks during prompt chain execution.
  v0.1.0 - 2025-12-04 - Introduce prompt chain CRUD and execution mixin.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from prompt_templates import get_default_prompt

from ..exceptions import (
    PromptChainExecutionError,
    PromptChainNotFoundError,
    PromptChainStorageError,
    PromptExecutionError,
    PromptManagerError,
    WebSearchError,
)
from ..execution import ExecutionError
from ..litellm_adapter import (
    LiteLLMNotInstalledError,
    apply_configured_drop_params,
    call_completion_with_fallback,
    get_completion,
    serialise_litellm_response,
)
from ..repository import RepositoryError, RepositoryNotFoundError
from ..web_search.context_formatting import (
    build_numbered_search_results,
    wrap_search_results_block,
)
from .execution_history import ExecutionOutcome

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import Protocol

    from models.prompt_chain_model import PromptChain, PromptChainStep
    from models.prompt_model import Prompt, PromptExecution

    from ..execution import CodexExecutionResult
    from ..notifications import NotificationCenter
    from ..repository import PromptRepository

    class _PromptChainHost(Protocol):
        def get_prompt(self, prompt_id: uuid.UUID) -> Prompt: ...

        def build_execution_context_metadata(
            self,
            prompt: Prompt,
            *,
            stream_enabled: bool,
            executor_model: str | None,
            conversation_length: int,
            request_text: str,
            response_text: str,
            response_style: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]: ...

        def log_execution_failure(
            self,
            prompt_id: uuid.UUID,
            request_text: str,
            error_message: str,
            *,
            conversation: Sequence[Mapping[str, str]] | None = None,
            context_metadata: Mapping[str, Any] | None = None,
            extra_metadata: Mapping[str, Any] | None = None,
        ) -> PromptExecution | None: ...

        def log_execution_success(
            self,
            prompt_id: uuid.UUID,
            request_text: str,
            result: CodexExecutionResult,
            *,
            conversation: Sequence[Mapping[str, str]] | None = None,
            context_metadata: Mapping[str, Any] | None = None,
            extra_metadata: Mapping[str, Any] | None = None,
        ) -> PromptExecution | None: ...

        def llm_status_message(self, capability: str) -> str: ...

        def increment_usage(self, prompt_id: uuid.UUID) -> None: ...


logger = logging.getLogger(__name__)

__all__ = [
    "PromptChainMixin",
    "PromptChainRunResult",
    "PromptChainStepRun",
]

_CHAIN_WEB_SEARCH_RESULT_LIMIT = 10
_CHAIN_SUMMARY_MAX_TOKENS = 220
_CHAIN_SUMMARY_SYSTEM_PROMPT = get_default_prompt("chain_summary") or (
    "You summarise the outcome of automated multi-step workflows for prompt engineers. "
    "Capture the key result, blockers, or next actions in two concise sentences. "
    "Do not invent information, include markdown headings, or reference step numbers."
)


@dataclass(slots=True)
class PromptChainStepRun:
    """Outcome metadata for a single chain step."""

    step: PromptChainStep
    status: str
    outcome: ExecutionOutcome | None
    error: str | None = None
    prompt_name: str | None = None
    request_text: str | None = None
    response_text: str | None = None
    duration_ms: int | None = None
    web_search_requested: bool = False
    web_search_applied: bool = False
    skip_reason: str | None = None
    step_label: str | None = None
    step_output_key: str | None = None


@dataclass(slots=True)
class PromptChainRunResult:
    """Aggregate response returned after executing a prompt chain."""

    chain: PromptChain
    chain_input: str
    step_outputs: dict[str, str]
    steps: list[PromptChainStepRun]
    final_output_text: str | None = None
    final_summary_text: str | None = None
    step_aliases: dict[str, str] | None = None
    final_step_id: uuid.UUID | None = None
    final_step_output_key: str | None = None
    final_step_label: str | None = None


class PromptChainMixin:
    """Expose prompt chain CRUD plus execution helpers."""

    _repository: PromptRepository
    _notification_center: NotificationCenter
    _chain_run_history_limit = 20

    def list_prompt_chains(self, include_inactive: bool = False) -> list[PromptChain]:
        """Return stored prompt chain definitions."""
        try:
            return self._repository.list_chains(include_inactive=include_inactive)
        except RepositoryError as exc:
            raise PromptChainStorageError("Unable to list prompt chains") from exc

    def get_prompt_chain(self, chain_id: uuid.UUID) -> PromptChain:
        """Return a single prompt chain."""
        try:
            return self._repository.get_chain(chain_id)
        except RepositoryNotFoundError as exc:
            raise PromptChainNotFoundError(str(exc)) from exc
        except RepositoryError as exc:
            raise PromptChainStorageError(f"Unable to load prompt chain {chain_id}") from exc

    def save_prompt_chain(self, chain: PromptChain) -> PromptChain:
        """Create or update a prompt chain depending on whether it exists."""
        try:
            exists = self._repository.chain_exists(chain.id)
        except RepositoryError as exc:
            raise PromptChainStorageError("Unable to resolve prompt chain state") from exc
        try:
            if exists:
                return self._repository.update_chain(chain)
            return self._repository.add_chain(chain)
        except RepositoryNotFoundError as exc:
            raise PromptChainNotFoundError(str(exc)) from exc
        except RepositoryError as exc:
            raise PromptChainStorageError("Unable to persist prompt chain") from exc

    def delete_prompt_chain(self, chain_id: uuid.UUID) -> None:
        """Remove a prompt chain definition."""
        try:
            self._repository.delete_chain(chain_id)
        except RepositoryNotFoundError as exc:
            raise PromptChainNotFoundError(str(exc)) from exc
        except RepositoryError as exc:
            raise PromptChainStorageError(f"Unable to delete prompt chain {chain_id}") from exc

    def run_prompt_chain(
        self,
        chain_id: uuid.UUID,
        *,
        chain_input: str,
        stream_callback: Callable[[PromptChainStep, str, bool], None] | None = None,
        use_web_search: bool | None = None,
        web_search_limit: int = _CHAIN_WEB_SEARCH_RESULT_LIMIT,
    ) -> PromptChainRunResult:
        """Execute the specified prompt chain sequentially."""
        host = cast("_PromptChainHost", self)
        chain = self.get_prompt_chain(chain_id)
        if not chain.is_active:
            raise PromptChainExecutionError(f"Prompt chain '{chain.name}' is inactive.")
        if not chain.steps:
            raise PromptChainExecutionError(f"Prompt chain '{chain.name}' has no steps configured.")
        if getattr(self, "_executor", None) is None:
            raise PromptChainExecutionError(
                "Prompt execution is not configured. Provide LiteLLM credentials before "
                "running prompt chains."
            )
        raw_input = chain_input or ""
        if not raw_input.strip():
            raise PromptChainExecutionError("Prompt chain input must not be empty.")

        previous_response = raw_input
        outputs: dict[str, str] = {}
        step_aliases: dict[str, str] = {}
        step_runs: list[PromptChainStepRun] = []
        task_id = f"prompt-chain:{chain.id}:{uuid.uuid4()}"
        web_search_enabled = bool(use_web_search)
        safe_search_limit = max(1, int(web_search_limit or _CHAIN_WEB_SEARCH_RESULT_LIMIT))
        with self._notification_center.track_task(
            title="Prompt chain",
            task_id=task_id,
            start_message=f"Running chain '{chain.name}'…",
            success_message=f"Chain '{chain.name}' completed.",
            failure_message=f"Chain '{chain.name}' failed.",
        ):
            for step in chain.steps:
                status = "pending"
                outcome: ExecutionOutcome | None = None
                error: str | None = None
                request_text = previous_response
                if not request_text.strip() and step.order_index == 1:
                    raise PromptChainExecutionError(
                        f"Step {step.order_index} in '{chain.name}' received empty input."
                    )
                prompt = host.get_prompt(step.prompt_id)
                step_output_key = f"step_{step.order_index}"
                step_label = step.output_variable or step_output_key
                request_text = self._maybe_enrich_with_web_search(
                    prompt,
                    request_text,
                    use_web_search=web_search_enabled,
                    web_search_limit=safe_search_limit,
                )
                web_search_applied = request_text != previous_response
                try:
                    outcome = self._execute_chain_step(
                        chain,
                        step,
                        prompt,
                        request_text,
                        stream_callback=stream_callback,
                    )
                    status = "success"
                    response_text = outcome.result.response_text or ""
                    previous_response = response_text
                    outputs[step_output_key] = response_text
                    if step.output_variable:
                        step_aliases[step.output_variable] = step_output_key
                except PromptExecutionError as exc:
                    status = "failed"
                    error = str(exc)
                    step_runs.append(
                        PromptChainStepRun(
                            step=step,
                            status=status,
                            outcome=None,
                            error=error,
                            prompt_name=prompt.name,
                            request_text=request_text,
                            response_text=None,
                            duration_ms=None,
                            web_search_requested=web_search_enabled,
                            web_search_applied=web_search_applied,
                            skip_reason=None,
                            step_label=step_label,
                            step_output_key=step_output_key,
                        )
                    )
                    if step.stop_on_failure:
                        message = (
                            f"Prompt chain '{chain.name}' failed at step "
                            f"{step.order_index}: {error}"
                        )
                        raise PromptChainExecutionError(message) from exc
                    continue
                step_runs.append(
                    PromptChainStepRun(
                        step=step,
                        status=status,
                        outcome=outcome,
                        prompt_name=getattr(prompt, "name", None),
                        request_text=request_text,
                        response_text=response_text,
                        duration_ms=outcome.result.duration_ms,
                        web_search_requested=web_search_enabled,
                        web_search_applied=web_search_applied,
                        skip_reason=None,
                        step_label=step_label,
                        step_output_key=step_output_key,
                    )
                )
        summary_text = (
            self._build_chain_summary(step_runs) if chain.summarize_last_response else None
        )
        final_output_text = None
        final_step_id = None
        final_step_output_key = None
        final_step_label = None
        for index in range(len(step_runs) - 1, -1, -1):
            candidate = step_runs[index]
            if candidate.status != "success" or candidate.outcome is None:
                continue
            final_output_text = (candidate.outcome.result.response_text or "").strip() or None
            final_step_id = candidate.step.id
            final_step_output_key = (
                candidate.step_output_key or f"step_{candidate.step.order_index}"
            )
            final_step_label = (
                candidate.step_label or candidate.step.output_variable or final_step_output_key
            )
            break
        result = PromptChainRunResult(
            chain=chain,
            chain_input=raw_input,
            step_outputs=outputs,
            steps=step_runs,
            final_output_text=final_output_text,
            final_summary_text=summary_text,
            step_aliases=step_aliases or None,
            final_step_id=final_step_id,
            final_step_output_key=final_step_output_key,
            final_step_label=final_step_label,
        )
        self._record_chain_run_history(result)
        return result

    def _build_chain_history_record(
        self,
        result: PromptChainRunResult,
    ) -> dict[str, str | None]:
        final_status = "success"
        if any(step_run.status == "failed" for step_run in result.steps):
            final_status = "failed"
        elif any(step_run.status == "skipped" for step_run in result.steps):
            final_status = "skipped"

        def _preview(text: str | None, limit: int = 160) -> str:
            value = (text or "").strip()
            if len(value) <= limit:
                return value
            return value[: limit - 1].rstrip() + "…"

        record: dict[str, str | None] = {
            "chain_id": str(result.chain.id),
            "chain_name": result.chain.name,
            "run_timestamp": datetime.now(UTC).isoformat(),
            "status": final_status,
            "input_preview": _preview(result.chain_input),
            "final_output_preview": _preview(result.final_output_text),
            "final_step_output_key": result.final_step_output_key,
        }
        if result.final_step_id is not None:
            record["final_step_id"] = str(result.final_step_id)
        if result.final_step_label is not None:
            record["final_step_label"] = result.final_step_label
        return record

    def _record_chain_run_history(self, result: PromptChainRunResult) -> None:
        records = getattr(self, "_chain_run_history_records", None)
        if records is None:
            self._chain_run_history_records = []
            records = self._chain_run_history_records
        records.insert(0, self._build_chain_history_record(result))
        del records[self._chain_run_history_limit :]

    def list_recent_prompt_chain_runs(self, *, limit: int = 20) -> list[dict[str, str | None]]:
        """Return bounded recent prompt-chain run records in newest-first order."""
        records = getattr(self, "_chain_run_history_records", [])
        safe_limit = max(1, int(limit or self._chain_run_history_limit))
        return list(records[:safe_limit])

    # Internal helpers ------------------------------------------------- #

    def _maybe_enrich_with_web_search(
        self,
        prompt: Prompt,
        request_text: str,
        *,
        use_web_search: bool,
        web_search_limit: int,
    ) -> str:
        """Prepend live web context to *request_text* when enabled."""
        if not use_web_search:
            return request_text
        trimmed = request_text.strip()
        if not trimmed:
            return request_text
        service = getattr(self, "web_search_service", None)
        is_available = bool(getattr(service, "is_available", lambda: False)())
        if service is None or not is_available:
            return request_text
        query = self._build_web_search_query(prompt, trimmed)
        if not query:
            return request_text
        try:
            result = asyncio.run(service.search(query, limit=web_search_limit))
        except RuntimeError:
            logger.debug("Event loop already running; skipping web search enrichment")
            return request_text
        except WebSearchError:
            logger.debug(
                "Web search provider failed during prompt chain run",
                exc_info=True,
                extra={"prompt_id": str(prompt.id)},
            )
            return request_text
        except Exception:  # pragma: no cover - defensive
            logger.debug("Unexpected error running web search", exc_info=True)
            return request_text
        documents = getattr(result, "documents", [])
        lines = self._collect_web_context_lines(documents)
        if not lines:
            return request_text
        provider_label = (result.provider or "Web search").strip() or "Web search"
        numbered_context = build_numbered_search_results(lines)
        if not numbered_context:
            return request_text
        formatted_block = wrap_search_results_block(numbered_context)
        if not formatted_block:
            return request_text
        return f"{provider_label} findings:\n{formatted_block}\n\nUser request:\n{request_text}"

    @staticmethod
    def _normalise_conversation(
        messages: Sequence[Mapping[str, str]] | None,
    ) -> list[dict[str, str]]:
        """Return a sanitised copy of conversation messages for chain execution."""
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

    @staticmethod
    def _prompt_tags(prompt: Prompt) -> list[str]:
        raw_tags = getattr(prompt, "tags", None)
        if not isinstance(raw_tags, list):
            return []
        return cast("list[str]", raw_tags)

    @staticmethod
    def _build_web_search_query(prompt: Prompt, request_text: str) -> str:
        parts: list[str] = []
        if prompt.name:
            parts.append(prompt.name.strip())
        if prompt.category:
            parts.append(prompt.category.strip())
        tags = PromptChainMixin._prompt_tags(prompt)
        if tags:
            parts.append(", ".join(tag.strip() for tag in tags[:3] if tag.strip()))
        description = (getattr(prompt, "description", "") or "").strip()
        if description:
            parts.append(description[:160])
        context = (getattr(prompt, "context", "") or "").strip()
        if context:
            parts.append(context[:160])
        text = request_text.strip()
        if text:
            parts.append(text[:200])
        query = " ".join(part for part in parts if part).strip()
        return query[:512]

    @classmethod
    def _collect_web_context_lines(
        cls,
        documents: Sequence[object],
    ) -> list[str]:
        lines: list[str] = []
        for document in documents:
            if document is None:
                continue
            url = str(getattr(document, "url", "") or "").strip()
            if not url:
                continue
            snippet = cls._extract_document_snippet(document)
            if not snippet:
                continue
            title = str(getattr(document, "title", "") or "").strip()
            prefix = f"{title}: " if title else ""
            lines.append(f"{prefix}{snippet} (Source: {url})")
        return lines

    @staticmethod
    def _extract_document_snippet(document: object) -> str:
        parts: list[str] = []
        summary = getattr(document, "summary", None)
        if summary:
            summary_text = str(summary).strip()
            if summary_text:
                parts.append(summary_text)
        highlights = getattr(document, "highlights", None)
        if isinstance(highlights, list):
            entries = cast("list[object]", highlights)
            for entry in entries:
                text = str(entry or "").strip()
                if text:
                    parts.append(text)
        return " ".join(parts).strip()

    def _execute_chain_step(
        self,
        chain: PromptChain,
        step: PromptChainStep,
        prompt: Prompt,
        request_text: str,
        *,
        stream_callback: Callable[[PromptChainStep, str, bool], None] | None = None,
    ) -> ExecutionOutcome:
        """Execute a single prompt without emitting GUI toasts per step."""
        host = cast("_PromptChainHost", self)
        executor = getattr(self, "_executor", None)
        if executor is None:
            message = (
                host.llm_status_message("Prompt execution")
                if hasattr(host, "llm_status_message")
                else "Prompt execution is not configured for this manager instance."
            )
            raise PromptChainExecutionError(message)
        conversation_history = self._normalise_conversation(None)
        stream_enabled = executor.stream

        def _handle_stream(chunk: str) -> None:
            if stream_callback is None:
                return
            if not chunk:
                return
            try:
                stream_callback(step, chunk, False)
            except Exception:  # pragma: no cover - callback failures should not bubble
                logger.debug("Prompt chain stream callback failed", exc_info=True)

        try:
            result = executor.execute(
                prompt,
                request_text,
                conversation=conversation_history,
                on_stream=_handle_stream if stream_callback else None,
            )
            if stream_callback is not None and result.response_text:
                try:
                    stream_callback(step, result.response_text, True)
                except Exception:  # pragma: no cover - defensive
                    logger.debug("Prompt chain final stream callback failed", exc_info=True)
        except ExecutionError as exc:
            failed_messages = list(conversation_history)
            failed_messages.append({"role": "user", "content": request_text.strip()})
            failure_context = host.build_execution_context_metadata(
                prompt,
                stream_enabled=stream_enabled,
                executor_model=getattr(executor, "model", None),
                conversation_length=len(failed_messages),
                request_text=request_text,
                response_text="",
            )
            host.log_execution_failure(
                prompt.id,
                request_text,
                str(exc),
                conversation=failed_messages,
                context_metadata=failure_context,
                extra_metadata={
                    "chain": {
                        "chain_id": str(chain.id),
                        "chain_name": chain.name,
                        "step_id": str(step.id),
                        "step_order": step.order_index,
                        "prompt_id": str(prompt.id),
                    }
                },
            )
            raise PromptExecutionError(str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise PromptExecutionError(str(exc)) from exc

        augmented_conversation = list(conversation_history)
        augmented_conversation.append({"role": "user", "content": request_text.strip()})
        if result.response_text:
            augmented_conversation.append({"role": "assistant", "content": result.response_text})
        context_metadata = host.build_execution_context_metadata(
            prompt,
            stream_enabled=stream_enabled,
            executor_model=getattr(executor, "model", None),
            conversation_length=len(augmented_conversation),
            request_text=request_text,
            response_text=result.response_text,
        )
        history_entry = host.log_execution_success(
            prompt.id,
            request_text,
            result,
            conversation=augmented_conversation,
            context_metadata=context_metadata,
            extra_metadata={
                "chain": {
                    "chain_id": str(chain.id),
                    "chain_name": chain.name,
                    "step_id": str(step.id),
                    "step_order": step.order_index,
                    "prompt_id": str(prompt.id),
                    "prompt_name": prompt.name,
                }
            },
        )
        try:
            host.increment_usage(prompt.id)
        except PromptManagerError:
            logger.debug(
                "Failed to increment usage after chain step",
                extra={"prompt_id": str(prompt.id)},
                exc_info=True,
            )
        return ExecutionOutcome(
            result=result,
            history_entry=history_entry,
            conversation=augmented_conversation,
        )

    def _build_chain_summary(self, step_runs: list[PromptChainStepRun]) -> str | None:
        """Return a compact summary sourced from the final successful step output."""
        response_text = self._last_successful_response(step_runs)
        if not response_text:
            return None
        summary = self._summarize_response_with_litellm(response_text)
        if summary:
            return summary
        return _summarize_response_text(response_text)

    @staticmethod
    def _last_successful_response(step_runs: list[PromptChainStepRun]) -> str | None:
        """Return the response text from the final successful step in *step_runs*."""
        for step_run in reversed(step_runs):
            outcome = step_run.outcome
            if outcome is None or step_run.status != "success":
                continue
            response_text = (outcome.result.response_text or "").strip()
            if response_text:
                return response_text
        return None

    def _summarize_response_with_litellm(self, response_text: str) -> str | None:
        """Generate a condensed summary via LiteLLM when models are configured."""
        trimmed = response_text.strip()
        if not trimmed:
            return None
        prompt_templates = getattr(self, "_prompt_templates", None)
        prompt_overrides = (
            cast("dict[str, str]", prompt_templates) if isinstance(prompt_templates, dict) else {}
        )
        system_prompt = prompt_overrides.get("chain_summary") or _CHAIN_SUMMARY_SYSTEM_PROMPT
        executor = getattr(self, "_executor", None)
        model = (
            getattr(self, "_litellm_fast_model", None)
            or getattr(self, "_litellm_inference_model", None)
            or getattr(executor, "model", None)
        )
        if not model:
            return None
        try:
            completion, LiteLLMException = get_completion()
        except LiteLLMNotInstalledError:
            logger.debug("LiteLLM unavailable; falling back to deterministic chain summary")
            return None

        request: dict[str, object] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": trimmed,
                },
            ],
            "temperature": 0.2,
            "max_tokens": _CHAIN_SUMMARY_MAX_TOKENS,
        }
        timeout_seconds = getattr(executor, "timeout_seconds", None)
        if timeout_seconds is not None:
            request["timeout"] = timeout_seconds
        for attr_name in ("api_key", "api_base", "api_version"):
            value = getattr(executor, attr_name, None)
            if value:
                request[attr_name] = value
        drop_params = getattr(self, "_litellm_drop_params", None)
        if drop_params is None and executor is not None:
            drop_params = getattr(executor, "drop_params", None)
        dropped = apply_configured_drop_params(request, drop_params)
        try:
            response = call_completion_with_fallback(
                request,
                completion,
                LiteLLMException,
                drop_candidates={
                    "max_tokens",
                    "max_output_tokens",
                    "temperature",
                    "timeout",
                },
                pre_dropped=dropped,
            )
        except LiteLLMException:
            logger.warning("LiteLLM chain summary generation failed", exc_info=True)
            return None
        except Exception:
            logger.warning("Unexpected error while summarizing chain output", exc_info=True)
            return None
        payload = serialise_litellm_response(response)
        if payload is None:
            return None
        try:
            message = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None
        if not isinstance(message, str):
            return None
        summary = message.strip()
        return summary or None


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _summarize_response_text(text: str, *, max_length: int = 360) -> str:
    """Return a deterministic summary suitable for UI display."""
    collapsed = " ".join(text.split())
    if not collapsed:
        return ""
    if len(collapsed) <= max_length:
        return collapsed
    sentences = [
        segment.strip() for segment in _SENTENCE_BOUNDARY.split(collapsed) if segment.strip()
    ]
    summary_parts: list[str] = []
    for sentence in sentences:
        summary_parts.append(sentence)
        joined = " ".join(summary_parts)
        if len(joined) >= max_length * 0.7:
            break
    summary = " ".join(summary_parts).strip()
    if len(summary) > max_length:
        summary = summary[:max_length].rstrip()
        last_space = summary.rfind(" ")
        if last_space > max_length * 0.4:
            summary = summary[:last_space]
    if len(summary) < len(collapsed):
        summary = summary.rstrip(".") + "…"
    return summary
