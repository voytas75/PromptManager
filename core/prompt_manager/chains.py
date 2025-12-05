"""Prompt chain orchestration helpers for Prompt Manager.

Updates:
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
from typing import TYPE_CHECKING, Any, Sequence

from jsonschema import Draft202012Validator, exceptions as jsonschema_exceptions

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
from ..templating import TemplateRenderer
from .execution_history import ExecutionOutcome, _normalise_conversation

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from collections.abc import Callable, Mapping

    from models.prompt_chain_model import PromptChain, PromptChainStep
    from models.prompt_model import Prompt

    from ..notifications import NotificationCenter
    from ..repository import PromptRepository

logger = logging.getLogger(__name__)

__all__ = [
    "PromptChainMixin",
    "PromptChainRunResult",
    "PromptChainStepRun",
]

_CHAIN_WEB_SEARCH_RESULT_LIMIT = 10
_CHAIN_SUMMARY_MAX_TOKENS = 220
_CHAIN_SUMMARY_SYSTEM_PROMPT = (
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


@dataclass(slots=True)
class PromptChainRunResult:
    """Aggregate response returned after executing a prompt chain."""

    chain: PromptChain
    variables: dict[str, Any]
    outputs: dict[str, Any]
    steps: list[PromptChainStepRun]
    summary: str | None = None


class PromptChainMixin:
    """Expose prompt chain CRUD plus execution helpers."""

    _repository: PromptRepository
    _notification_center: NotificationCenter

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
        variables: Mapping[str, Any] | None = None,
        stream_callback: Callable[[PromptChainStep, str, bool], None] | None = None,
        use_web_search: bool | None = None,
        web_search_limit: int = _CHAIN_WEB_SEARCH_RESULT_LIMIT,
    ) -> PromptChainRunResult:
        """Execute the specified prompt chain sequentially."""
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
        context: dict[str, Any] = dict(variables or {})
        validator = self._build_validator(chain)
        if validator is not None:
            try:
                validator.validate(context)
            except jsonschema_exceptions.ValidationError as exc:
                raise PromptChainExecutionError(
                    f"Prompt chain variables failed validation: {exc.message}"
                ) from exc

        renderer = TemplateRenderer()
        outputs: dict[str, Any] = {}
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
                if step.condition:
                    should_run = self._evaluate_condition(renderer, step.condition, context)
                    if not should_run:
                        status = "skipped"
                        step_runs.append(PromptChainStepRun(step=step, status=status, outcome=None))
                        continue
                request_text = self._render_template(renderer, step.input_template, context)
                if not request_text.strip():
                    raise PromptChainExecutionError(
                        f"Step {step.order_index} in '{chain.name}' produced an empty request."
                    )
                prompt = self.get_prompt(step.prompt_id)
                request_text = self._maybe_enrich_with_web_search(
                    prompt,
                    request_text,
                    use_web_search=web_search_enabled,
                    web_search_limit=safe_search_limit,
                )
                try:
                    outcome = self._execute_chain_step(
                        chain,
                        step,
                        prompt,
                        request_text,
                        stream_callback=stream_callback,
                    )
                    status = "success"
                    response_text = outcome.result.response_text
                    context[step.output_variable] = response_text
                    outputs[step.output_variable] = response_text
                    context["_last_response"] = response_text
                except PromptExecutionError as exc:
                    status = "failed"
                    error = str(exc)
                    step_runs.append(
                        PromptChainStepRun(step=step, status=status, outcome=None, error=error)
                    )
                    if step.stop_on_failure:
                        message = (
                            f"Prompt chain '{chain.name}' failed at step "
                            f"{step.order_index}: {error}"
                        )
                        raise PromptChainExecutionError(message) from exc
                    continue
                step_runs.append(PromptChainStepRun(step=step, status=status, outcome=outcome))
        summary_text = (
            self._build_chain_summary(step_runs)
            if chain.summarize_last_response
            else None
        )
        return PromptChainRunResult(
            chain=chain,
            variables=context,
            outputs=outputs,
            steps=step_runs,
            summary=summary_text,
        )

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
        context_block = "\n".join(lines)
        return (
            f"{provider_label} findings:\n"
            f"{context_block}\n\n"
            "User request:\n"
            f"{request_text}"
        )

    @staticmethod
    def _build_web_search_query(prompt: Prompt, request_text: str) -> str:
        parts: list[str] = []
        if prompt.name:
            parts.append(prompt.name.strip())
        if prompt.category:
            parts.append(prompt.category.strip())
        tags = getattr(prompt, "tags", None) or []
        if tags:
            parts.append(
                ", ".join(tag.strip() for tag in tags[:3] if isinstance(tag, str) and tag.strip())
            )
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
            lines.append(f"- {prefix}{snippet} (Source: {url})")
        return lines

    @staticmethod
    def _extract_document_snippet(document: object) -> str:
        parts: list[str] = []
        summary = getattr(document, "summary", None)
        if summary:
            summary_text = str(summary).strip()
            if summary_text:
                parts.append(summary_text)
        highlights = getattr(document, "highlights", None) or []
        if isinstance(highlights, Sequence) and not isinstance(highlights, (str, bytes, bytearray)):
            for entry in highlights:
                text = str(entry or "").strip()
                if text:
                    parts.append(text)
        return " ".join(parts).strip()

    def _build_validator(self, chain: PromptChain) -> Draft202012Validator | None:
        schema = chain.variables_schema
        if not schema:
            return None
        try:
            return Draft202012Validator(schema)
        except jsonschema_exceptions.SchemaError:
            logger.warning(
                "Ignoring invalid variables_schema for chain",
                extra={"chain_id": str(chain.id), "chain_name": chain.name},
                exc_info=True,
            )
            return None

    def _render_template(
        self,
        renderer: TemplateRenderer,
        template_text: str,
        variables: Mapping[str, Any],
    ) -> str:
        result = renderer.render(template_text, variables)
        if result.errors:
            raise PromptChainExecutionError("; ".join(result.errors))
        return result.rendered_text

    def _evaluate_condition(
        self,
        renderer: TemplateRenderer,
        template_text: str,
        variables: Mapping[str, Any],
    ) -> bool:
        rendered = self._render_template(renderer, template_text, variables)
        normalised = rendered.strip().lower()
        return normalised not in {"", "0", "false", "no"}

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
        executor = getattr(self, "_executor", None)
        if executor is None:
            raise PromptChainExecutionError(
                "Prompt execution is not configured for this manager instance."
            )
        conversation_history = _normalise_conversation(None)
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
            failure_context = self._build_execution_context_metadata(
                prompt,
                stream_enabled=stream_enabled,
                executor_model=getattr(executor, "model", None),
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
        context_metadata = self._build_execution_context_metadata(
            prompt,
            stream_enabled=stream_enabled,
            executor_model=getattr(executor, "model", None),
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
            self.increment_usage(prompt.id)
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
                {"role": "system", "content": _CHAIN_SUMMARY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Summarize the following response from a prompt chain so collaborators "
                        "understand the outcome in two concise sentences. Focus on the "
                        "result and any follow-up actions, avoid markdown, and never invent "
                        f"details.\n\n{trimmed}"
                    ),
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
        segment.strip()
        for segment in _SENTENCE_BOUNDARY.split(collapsed)
        if segment.strip()
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
