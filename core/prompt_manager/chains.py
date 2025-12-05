"""Prompt chain orchestration helpers for Prompt Manager.

Updates:
  v0.1.1 - 2025-12-05 - Surface streaming callbacks during prompt chain execution.
  v0.1.0 - 2025-12-04 - Introduce prompt chain CRUD and execution mixin.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from jsonschema import Draft202012Validator, exceptions as jsonschema_exceptions

from ..exceptions import (
    PromptChainExecutionError,
    PromptChainNotFoundError,
    PromptChainStorageError,
    PromptExecutionError,
    PromptManagerError,
)
from ..execution import ExecutionError
from ..repository import RepositoryError, RepositoryNotFoundError
from ..templating import TemplateRenderer
from .execution_history import ExecutionOutcome, _normalise_conversation

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from collections.abc import Mapping

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
        return PromptChainRunResult(
            chain=chain,
            variables=context,
            outputs=outputs,
            steps=step_runs,
        )

    # Internal helpers ------------------------------------------------- #

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
