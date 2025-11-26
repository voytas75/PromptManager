"""LiteLLM-backed prompt execution helpers.

Updates: v0.4.0 - 2025-11-26 - Add streaming support for LiteLLM prompt execution.
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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

from models.prompt_model import Prompt

from .litellm_adapter import (
    apply_configured_drop_params,
    call_completion_with_fallback,
    get_completion,
)

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
    stream: bool = False

    def execute(
        self,
        prompt: Prompt,
        request_text: str,
        *,
        conversation: Optional[Sequence[Mapping[str, str]]] = None,
        stream: Optional[bool] = None,
        on_stream: Optional[Callable[[str], None]] = None,
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

        stream_enabled = self.stream if stream is None else stream

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
        if stream_enabled:
            request["stream"] = True
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
                    "stream",
                },
                pre_dropped=dropped_params,
            )
        except LiteLLMException as exc:  # type: ignore[arg-type]
            raise ExecutionError(f"LiteLLM execution failed: {exc}") from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise ExecutionError("Unexpected error while calling LiteLLM") from exc
        duration_ms = int((time.perf_counter() - started) * 1000)

        usage: Mapping[str, Any]
        response_text: str
        raw_payload: Mapping[str, Any]

        if stream_enabled:
            if not isinstance(response, Iterable):  # pragma: no cover - defensive
                raise ExecutionError("LiteLLM streaming response is not iterable")
            response_text, usage, raw_payload = _consume_streaming_response(
                response,
                on_stream,
            )
        else:
            try:
                content = response["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:  # pragma: no cover - defensive
                raise ExecutionError("LiteLLM returned an unexpected payload") from exc

            if isinstance(response, Mapping):
                usage = response.get("usage") or {}
                raw_payload = response
            else:  # pragma: no cover - defensive
                usage = {}
                raw_payload = {"raw": response}
            response_text = str(content).strip()

        if stream_enabled:
            response_text = response_text.strip()

        if not isinstance(usage, Mapping):  # pragma: no cover - defensive
            usage = {}
        else:
            usage = dict(usage)

        if isinstance(raw_payload, Mapping):
            raw_payload = dict(raw_payload)
        else:  # pragma: no cover - defensive
            raw_payload = {"raw": raw_payload}

        result = CodexExecutionResult(
            prompt_id=prompt.id,
            request_text=request_text,
            response_text=response_text,
            duration_ms=duration_ms,
            usage=usage,
            raw_response=raw_payload,
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


def _consume_streaming_response(
    stream: Iterable[Any],
    on_stream: Optional[Callable[[str], None]] = None,
) -> tuple[str, Mapping[str, Any], Mapping[str, Any]]:
    """Aggregate LiteLLM streaming chunks into final response text and metadata."""

    accumulated: list[str] = []
    usage: Mapping[str, Any] = {}
    serialised_chunks: list[Any] = []

    for chunk in stream:
        serialised = _serialise_chunk(chunk)
        serialised_chunks.append(serialised)
        text_delta = _extract_stream_text(serialised)
        if text_delta:
            accumulated.append(text_delta)
            if on_stream is not None:
                try:
                    on_stream(text_delta)
                except Exception:  # pragma: no cover - callback failures should not bubble
                    logger.warning("Streaming callback raised an exception", exc_info=True)
        chunk_usage = _extract_stream_usage(serialised)
        if chunk_usage:
            usage = chunk_usage

    final_text = "".join(accumulated)
    raw_payload: Dict[str, Any] = {
        "streamed": True,
        "chunks": serialised_chunks,
    }
    if final_text:
        raw_payload["choices"] = [
            {"message": {"role": "assistant", "content": final_text}}
        ]
    if usage:
        raw_payload["usage"] = usage
    return final_text, usage, raw_payload


def _serialise_chunk(chunk: Any) -> Any:
    """Best-effort conversion of LiteLLM streaming chunks into serialisable objects."""

    if isinstance(chunk, Mapping):
        return dict(chunk)
    model_dump = getattr(chunk, "model_dump", None)
    if callable(model_dump):  # pragma: no branch - pydantic v2
        try:
            return model_dump()
        except Exception:
            logger.debug("Unable to serialise streaming chunk via model_dump", exc_info=True)
    as_dict = getattr(chunk, "dict", None)
    if callable(as_dict):
        try:
            return as_dict()
        except Exception:
            logger.debug("Unable to serialise streaming chunk via dict()", exc_info=True)
    return chunk


def _extract_stream_text(payload: Any) -> str:
    """Return textual delta from a LiteLLM streaming payload when available."""

    if not isinstance(payload, Mapping):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, Mapping):
        return ""
    delta = first.get("delta")
    if isinstance(delta, Mapping):
        content = delta.get("content")
        if content:
            return str(content)
    message = first.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
        if content:
            return str(content)
    text = first.get("text")
    if isinstance(text, str):
        return text
    return ""


def _extract_stream_usage(payload: Any) -> Mapping[str, Any]:
    """Return usage metadata from a streaming payload if present."""

    if not isinstance(payload, Mapping):
        return {}
    usage = payload.get("usage")
    if isinstance(usage, Mapping):
        return usage
    return {}
