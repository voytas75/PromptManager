"""LiteLLM-backed prompt execution helpers.

Updates: v0.3.1 - 2025-11-05 - Rely on provider defaults instead of forcing LiteLLM timeouts.
Updates: v0.3.0 - 2025-11-02 - Support reasoning-effort configuration and max_output_tokens for new OpenAI model families.
Updates: v0.2.1 - 2025-11-14 - Surface missing LiteLLM dependency as ExecutionError.
Updates: v0.2.0 - 2025-11-12 - Support multi-turn conversation payloads for LiteLLM execution.
Updates: v0.1.0 - 2025-11-08 - Introduce CodexExecutor for running prompts via LiteLLM.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from models.prompt_model import Prompt

from .litellm_adapter import apply_configured_drop_params, call_completion_with_fallback, get_completion

logger = logging.getLogger("prompt_manager.execution")


class ExecutionError(Exception):
    """Raised when LiteLLM prompt execution fails."""


def _supports_reasoning(model: str) -> bool:
    """Return True when the target model supports OpenAI reasoning payloads."""

    lowered = model.lower()
    reasoning_markers = ("o1", "o3", "o4", "gpt-4.1", "gpt-5")
    return any(marker in lowered for marker in reasoning_markers)


@dataclass(slots=True)
class CodexExecutionResult:
    """Container for prompt execution responses."""

    prompt_id: uuid.UUID
    request_text: str
    response_text: str
    duration_ms: int
    usage: Mapping[str, Any]
    raw_response: Mapping[str, Any]


@dataclass(slots=True)
class CodexExecutor:
    """Execute prompts against GPT-style models via LiteLLM."""

    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    timeout_seconds: Optional[float] = None
    max_output_tokens: int = 1024
    temperature: float = 0.2
    drop_params: Optional[Sequence[str]] = None
    reasoning_effort: Optional[str] = None

    def execute(
        self,
        prompt: Prompt,
        request_text: str,
        *,
        conversation: Optional[Sequence[Mapping[str, str]]] = None,
    ) -> CodexExecutionResult:
        """Run the supplied request through LiteLLM and return the response."""

        try:
            completion, LiteLLMException = get_completion()
        except RuntimeError as exc:
            raise ExecutionError(str(exc)) from exc
        instructions = prompt.context or prompt.description
        if not instructions:
            raise ExecutionError(f"Prompt {prompt.id} is missing context to execute.")

        payload_messages: list[Mapping[str, str]] = [
            {"role": "system", "content": instructions},
        ]
        normalised_conversation: list[Mapping[str, str]] = []
        if conversation:
            for index, message in enumerate(conversation):
                role = str(message.get("role", "")).strip()
                if not role:
                    raise ExecutionError(
                        f"Conversation message at index {index} is missing a role."
                    )
                content = message.get("content")
                if content is None:
                    raise ExecutionError(
                        f"Conversation message '{role}' is missing content."
                    )
                normalised_conversation.append({"role": role, "content": str(content)})
        payload_messages.extend(normalised_conversation)
        payload_messages.append({"role": "user", "content": request_text.strip()})

        request: Dict[str, Any] = {
            "model": self.model,
            "messages": payload_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "max_output_tokens": self.max_output_tokens,
        }
        if self.timeout_seconds is not None:
            request["timeout"] = self.timeout_seconds
        if self.api_key:
            request["api_key"] = self.api_key
        if self.api_base:
            request["api_base"] = self.api_base
        if self.api_version:
            request["api_version"] = self.api_version
        reasoning_payload: Optional[Dict[str, str]] = None
        if self.reasoning_effort and _supports_reasoning(self.model):
            reasoning_payload = {"effort": self.reasoning_effort}
            request["reasoning"] = reasoning_payload
        dropped_params = apply_configured_drop_params(request, self.drop_params)
        if dropped_params:
            logger.debug(
                "Dropping LiteLLM parameters before execution",
                extra={
                    "prompt_id": str(prompt.id),
                    "dropped_params": list(dropped_params),
                },
            )

        logger.debug(
            "Executing prompt via LiteLLM",
            extra={
                "prompt_id": str(prompt.id),
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_output_tokens,
            },
        )
        started = time.perf_counter()
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
                    "reasoning",
                },
                pre_dropped=dropped_params,
            )
        except LiteLLMException as exc:  # type: ignore[arg-type]
            raise ExecutionError(f"LiteLLM execution failed: {exc}") from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise ExecutionError("Unexpected error while calling LiteLLM") from exc
        duration_ms = int((time.perf_counter() - started) * 1000)

        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:  # pragma: no cover - defensive
            raise ExecutionError("LiteLLM returned an unexpected payload") from exc

        usage: Mapping[str, Any]
        if isinstance(response, Mapping):
            usage = response.get("usage") or {}
        else:  # pragma: no cover - defensive
            usage = {}

        result = CodexExecutionResult(
            prompt_id=prompt.id,
            request_text=request_text,
            response_text=str(content).strip(),
            duration_ms=duration_ms,
            usage=usage,
            raw_response=response,
        )
        logger.debug(
            "Prompt executed",
            extra={
                "prompt_id": str(prompt.id),
                "duration_ms": duration_ms,
                "tokens_prompt": usage.get("prompt_tokens"),
                "tokens_completion": usage.get("completion_tokens"),
            },
        )
        return result


__all__ = ["CodexExecutor", "CodexExecutionResult", "ExecutionError"]
