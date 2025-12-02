"""LiteLLM-backed prompt scenario generation utilities.

Updates:
  v0.1.5 - 2025-12-01 - Accept LiteLLM ModelResponse payloads during scenario refresh.
  v0.1.4 - 2025-11-30 - Align LiteLLM request prompt text with CLI contract.
  v0.1.3 - 2025-11-27 - Strip Markdown code fences before parsing scenarios.
  v0.1.2 - 2025-11-23 - Support configurable system prompt overrides.
  v0.1.1 - 2025-11-05 - Remove explicit LiteLLM timeout to rely on provider defaults.
  v0.1.0 - 2025-11-19 - Introduce scenario generator for prompt usage guidance.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from prompt_templates import SCENARIO_GENERATION_PROMPT

from .litellm_adapter import (
    apply_configured_drop_params,
    call_completion_with_fallback,
    get_completion,
    serialise_litellm_response,
)


class ScenarioGenerationError(Exception):
    """Raised when usage scenarios cannot be generated."""


def _normalise_scenarios(candidates: Sequence[str], limit: int) -> list[str]:
    """Return up to *limit* distinct, trimmed scenario strings."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = candidate.strip()
        if not text:
            continue
        if text.startswith(("-", "*")):
            text = text[1:].strip()
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def _strip_code_fences(response_text: str) -> str:
    """Remove leading/trailing Markdown code fences to aid parsing."""
    stripped = response_text.lstrip()
    if not stripped.startswith("```"):
        return response_text
    lines = stripped.splitlines()
    if not lines:
        return ""
    lines = lines[1:]
    while lines and lines[-1].strip().startswith("```"):
        lines.pop()
    return "\n".join(lines).strip()


def _extract_candidates(response_text: str) -> list[str]:
    """Parse LiteLLM output into a list of candidate scenario strings."""
    response_text = _strip_code_fences(response_text)
    stripped = response_text.strip()
    if not stripped:
        return []
    try:
        payload: object = json.loads(stripped)
    except json.JSONDecodeError:
        cleaned: list[str] = []
        for line in stripped.splitlines():
            text = line.strip()
            if not text or text in {"[", "]"}:
                continue
            if text.endswith(",]"):
                text = text[:-2].strip()
            elif text.endswith(","):
                text = text[:-1].strip()
            if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'", "`"}:
                text = text[1:-1].strip()
            if text.startswith("```") or text.endswith("```"):
                continue
            if text:
                cleaned.append(text)
        return cleaned
    if isinstance(payload, list):
        items = cast("Sequence[object]", payload)
        return [str(item) for item in items]
    if isinstance(payload, str):
        return [payload]
    if isinstance(payload, dict):
        mapping_payload = cast("Mapping[str, object]", payload)
        return [str(value) for value in mapping_payload.values()]
    return [str(payload)]


@dataclass(slots=True)
class LiteLLMScenarioGenerator:
    """Generate prompt usage scenarios via LiteLLM chat completions."""

    model: str
    api_key: str | None = None
    api_base: str | None = None
    timeout_seconds: float | None = None
    api_version: str | None = None
    drop_params: Sequence[str] | None = None
    default_max_scenarios: int = 3
    system_prompt: str | None = None

    def generate(self, context: str, *, max_scenarios: int | None = None) -> list[str]:
        """Return a ranked list of usage scenarios for the supplied prompt body."""
        if not context.strip():
            raise ScenarioGenerationError("Prompt context is required to generate scenarios.")
        completion, LiteLLMException = get_completion()

        limit = max_scenarios if max_scenarios is not None else self.default_max_scenarios
        limit = max(1, min(int(limit), 5))

        request: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_prompt_text()},
                {
                    "role": "user",
                    # NOTE: Using an f-string avoids ``str.format`` parsing braces that may
                    # legitimately appear inside *context*.  ``str.format`` would treat any
                    # ``{`` / ``}`` pairs in the prompt body as replacement fields and raise
                    # a ``ValueError`` like "expected ':' after conversion specifier".  An
                    # f-string is evaluated prior to concatenation, so braces inside
                    # *context* remain intact.
                    "content": (
                        f"Return up to {limit} scenarios for this prompt.\n\n"
                        f"Prompt:\n{context.strip()}"
                    ),
                },
            ],
            "temperature": 0.4,
            "max_tokens": 256,
        }
        if self.timeout_seconds is not None:
            request["timeout"] = self.timeout_seconds
        if self.api_key:
            request["api_key"] = self.api_key
        if self.api_base:
            request["api_base"] = self.api_base
        if self.api_version:
            request["api_version"] = self.api_version

        dropped_params = apply_configured_drop_params(request, self.drop_params)
        if dropped_params:
            logger.debug(
                "Dropping LiteLLM parameters for scenario generation",
                extra={"dropped_params": list(dropped_params)},
            )

        try:
            response = call_completion_with_fallback(
                request,
                completion,
                LiteLLMException,
                drop_candidates={"max_tokens", "max_output_tokens", "temperature", "timeout"},
                pre_dropped=dropped_params,
            )
        except LiteLLMException as exc:  # type: ignore[arg-type]
            raise ScenarioGenerationError(str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise ScenarioGenerationError("Unexpected error while calling LiteLLM") from exc

        response_mapping = serialise_litellm_response(response)
        if response_mapping is None:
            raise ScenarioGenerationError("LiteLLM returned an unexpected payload")
        choices_value = response_mapping.get("choices")
        if not isinstance(choices_value, Sequence) or not choices_value:
            raise ScenarioGenerationError("LiteLLM returned an unexpected payload")
        choices_seq = cast("Sequence[Any]", choices_value)
        first_choice = choices_seq[0]
        if not isinstance(first_choice, Mapping):
            raise ScenarioGenerationError("LiteLLM returned an unexpected payload")
        first_mapping = cast("Mapping[str, Any]", first_choice)
        message_value = first_mapping.get("message")
        if not isinstance(message_value, Mapping):
            raise ScenarioGenerationError("LiteLLM returned an unexpected payload")
        message_mapping = cast("Mapping[str, Any]", message_value)
        content_value = message_mapping.get("content")
        if content_value is None:
            raise ScenarioGenerationError("LiteLLM response is missing content")
        message = content_value

        candidates = _extract_candidates(str(message))
        scenarios = _normalise_scenarios(candidates, limit)
        if not scenarios:
            raise ScenarioGenerationError("LiteLLM returned no scenarios to surface.")
        return scenarios

    def _system_prompt_text(self) -> str:
        """Return the configured or default scenario template."""
        return (self.system_prompt or SCENARIO_GENERATION_PROMPT).strip()


__all__ = ["LiteLLMScenarioGenerator", "ScenarioGenerationError"]
logger = logging.getLogger(__name__)
