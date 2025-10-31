"""LiteLLM-backed prompt execution helpers.

Updates: v0.1.0 - 2025-11-08 - Introduce CodexExecutor for running prompts via LiteLLM.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from models.prompt_model import Prompt

from .litellm_adapter import get_completion

logger = logging.getLogger("prompt_manager.execution")


class ExecutionError(Exception):
    """Raised when LiteLLM prompt execution fails."""


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
    timeout_seconds: float = 30.0
    max_output_tokens: int = 1024
    temperature: float = 0.2

    def execute(
        self,
        prompt: Prompt,
        request_text: str,
        *,
        extra_messages: Optional[Sequence[Mapping[str, str]]] = None,
    ) -> CodexExecutionResult:
        """Run the supplied request through LiteLLM and return the response."""

        completion, LiteLLMException = get_completion()
        instructions = prompt.context or prompt.description
        if not instructions:
            raise ExecutionError(f"Prompt {prompt.id} is missing context to execute.")

        payload_messages: list[Mapping[str, str]] = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": request_text.strip()},
        ]
        if extra_messages:
            payload_messages.extend(extra_messages)

        request: Dict[str, Any] = {
            "model": self.model,
            "messages": payload_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "timeout": self.timeout_seconds,
        }
        if self.api_key:
            request["api_key"] = self.api_key
        if self.api_base:
            request["api_base"] = self.api_base
        if self.api_version:
            request["api_version"] = self.api_version

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
            response = completion(**request)  # type: ignore[arg-type]
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
