"""LiteLLM-backed prompt engineering utilities for refining prompts.

Updates:
  v0.1.5 - 2025-11-29 - Reformat payload builders and guard long literals.
  v0.1.4 - 2025-11-23 - Allow overriding the system prompt via settings.
  v0.1.3 - 2025-11-22 - Add structure-only refinement mode with targeted instructions.
  v0.1.2 - 2025-11-05 - Stop forwarding LiteLLM timeout parameter to avoid premature failures.
  v0.1.1 - 2025-11-02 - Drop configured LiteLLM parameters locally before refinement calls.
  v0.1.0 - 2025-11-15 - Introduce prompt refinement helper using meta-prompt rules.
"""


from __future__ import annotations

import json
import logging
from collections.abc import (
    Iterable,
    Iterable as IterableABC,
    Mapping,
    Sequence,
    Sequence as SequenceABC,
)
from dataclasses import dataclass
from typing import Any, cast

from prompt_templates import PROMPT_ENGINEERING_PROMPT

from .litellm_adapter import (
    apply_configured_drop_params,
    call_completion_with_fallback,
    get_completion,
)

logger = logging.getLogger(__name__)


class PromptEngineeringError(Exception):
    """Raised when prompt refinement fails."""


@dataclass(slots=True)
class PromptRefinement:
    """Structured result returned by the prompt engineering workflow."""

    improved_prompt: str
    analysis: str
    checklist: Sequence[str]
    warnings: Sequence[str]
    confidence: float
    raw_response: dict[str, Any] | None = None


def _format_list(items: Iterable[Any] | None) -> list[str]:
    """Normalise iterable inputs into a list of trimmed strings."""

    if not items:
        return []
    formatted: list[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            formatted.append(text)
    return formatted


def _strip_code_fence(text: str) -> str:
    """Remove Markdown code fences from LiteLLM responses."""

    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


@dataclass(slots=True)
class PromptEngineer:
    """Refine prompts using the meta-prompt ruleset via LiteLLM."""

    model: str
    api_key: str | None = None
    api_base: str | None = None
    api_version: str | None = None
    temperature: float = 0.25
    top_p: float = 0.9
    timeout_seconds: float | None = None
    max_tokens: int = 1200
    drop_params: Sequence[str] | None = None
    system_prompt: str | None = None

    def refine(
        self,
        prompt_text: str,
        *,
        name: str | None = None,
        description: str | None = None,
        category: str | None = None,
        tags: Sequence[str] | None = None,
        negative_constraints: Sequence[str] | None = None,
        structure_only: bool = False,
    ) -> PromptRefinement:
        """Return a refined prompt and supporting analysis via LiteLLM."""

        base_prompt = (prompt_text or "").strip()
        if not base_prompt:
            raise PromptEngineeringError("Prompt text is required for refinement.")

        completion, LiteLLMException = get_completion()
        user_payload = self._build_user_payload(
            base_prompt,
            name=name,
            description=description,
            category=category,
            tags=tags,
            negative_constraints=negative_constraints,
            structure_only=structure_only,
        )
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_prompt_text()},
                {"role": "user", "content": user_payload},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
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
                "Dropping LiteLLM parameters for prompt engineering",
                extra={
                    "model": self.model,
                    "dropped_params": list(dropped_params),
                },
            )

        logger.debug(
            "Refining prompt via LiteLLM",
            extra={
                "model": self.model,
                "prompt_length": len(base_prompt),
                "has_name": bool(name),
                "tag_count": len(tags or ()),
            },
        )
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
                    "top_p",
                },
                pre_dropped=dropped_params,
            )
        except LiteLLMException as exc:  # type: ignore[arg-type]
            raise PromptEngineeringError(f"LiteLLM refinement failed: {exc}") from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise PromptEngineeringError("Unexpected error while calling LiteLLM") from exc

        message_text = self._extract_message(response)
        try:
            parsed = _parse_refinement_payload(_strip_code_fence(message_text))
        except ValueError as exc:
            raise PromptEngineeringError("Prompt refinement returned invalid JSON") from exc

        improved = str(parsed.get("improved_prompt", "")).strip()
        if not improved:
            raise PromptEngineeringError("Prompt refinement did not return an improved prompt.")

        analysis = str(parsed.get("analysis", "")).strip()
        checklist = _format_list(parsed.get("checklist"))
        warnings = _format_list(parsed.get("warnings"))
        confidence_raw = parsed.get("confidence", 0.0)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(confidence, 1.0))

        raw_response: dict[str, Any] | None = None
        if isinstance(response, Mapping):
            response_mapping = cast("Mapping[str, Any]", response)
            raw_response = {str(key): value for key, value in response_mapping.items()}

        logger.debug(
            "Prompt refinement completed",
            extra={
                "delta": len(improved) - len(base_prompt),
                "checklist_items": len(checklist),
                "warnings": len(warnings),
            },
        )

        return PromptRefinement(
            improved_prompt=improved,
            analysis=analysis,
            checklist=checklist,
            warnings=warnings,
            confidence=confidence,
            raw_response=raw_response,
        )

    def refine_structure(
        self,
        prompt_text: str,
        *,
        name: str | None = None,
        description: str | None = None,
        category: str | None = None,
        tags: Sequence[str] | None = None,
        negative_constraints: Sequence[str] | None = None,
    ) -> PromptRefinement:
        """Return a refinement focused on formatting and structural improvements."""

        constraints = list(negative_constraints or [])
        constraints.append(
            "Do not introduce new instructions or change the prompt's semantic intent; "
            "reorganise sections and formatting only."
        )
        return self.refine(
            prompt_text,
            name=name,
            description=description,
            category=category,
            tags=tags,
            negative_constraints=constraints,
            structure_only=True,
        )

    def _build_user_payload(
        self,
        prompt_text: str,
        *,
        name: str | None,
        description: str | None,
        category: str | None,
        tags: Sequence[str] | None,
        negative_constraints: Sequence[str] | None,
        structure_only: bool,
    ) -> str:
        """Compose the user instruction message for LiteLLM."""

        sections: list[str] = ["### Prompt Engineering Request"]
        if name:
            sections.append(f"Prompt Name: {name.strip()}")
        if description:
            sections.append(f"Description: {description.strip()}")
        if category:
            sections.append(f"Category: {category.strip()}")
        if tags:
            formatted_tags = ", ".join(tag.strip() for tag in tags if tag.strip())
            if formatted_tags:
                sections.append(f"Tags: {formatted_tags}")
        if negative_constraints:
            blocked = ", ".join(
                constraint.strip()
                for constraint in negative_constraints
                if constraint and str(constraint).strip()
            )
            if blocked:
                sections.append(f"Negative Constraints: {blocked}")

        sections.append("\n### Current Prompt\n" + prompt_text.strip())
        if structure_only:
            sections.append(
                "\n### Tasks\n"
                "1. Reorganise the prompt into clearly labelled sections (context, instructions, "
                "constraints, and output format) using consistent headings or delimiters.\n"
                "2. Keep the original wording and requirements intact; only adjust ordering and "
                "grouping for clarity.\n"
                "3. Highlight missing structural components (for example, the output contract) in "
                "the warnings list."
            )
            sections.append(
                "Do not add new requirements or examples beyond light reformatting. "
                "If information is missing, call it out in warnings rather than inventing details."
            )
        else:
            sections.append(
                "\n### Tasks\n"
                "1. Identify gaps or violations of the meta-prompt rules.\n"
                "2. Produce an improved prompt that retains the author's intent and integrates "
                "explicit instructions, context, guardrails, plus the output contract.\n"
                "3. Note residual risks, missing information, or follow-up questions in the "
                "warnings list."
            )
            sections.append(
                "Ensure the improved prompt is production-ready, uses deterministic language, and "
                "includes explicit delimiters or sections where appropriate."
            )
        return "\n".join(sections)

    @staticmethod
    def _extract_message(response: Any) -> str:
        """Extract the primary message content from the LiteLLM response payload."""

        def _normalise_content(value: Any) -> str | None:
            if value is None:
                return None
            if isinstance(value, str):
                text = value.strip()
                return text or None
            mapping_value = _as_mapping(value)
            if mapping_value is not None:
                for key in ("content", "text", "value"):
                    if key in mapping_value:
                        normalised = _normalise_content(mapping_value[key])
                        if normalised:
                            return normalised
                return None
            if isinstance(value, IterableABC) and not isinstance(value, (bytes, bytearray, str)):
                parts: list[str] = []
                iterable_value = cast("Iterable[Any]", value)
                for item in iterable_value:
                    normalised = _normalise_content(item)
                    if normalised:
                        parts.append(normalised)
                if parts:
                    return "\n".join(parts)
                return None
            for attr in ("content", "text", "value"):
                if hasattr(value, attr):
                    normalised = _normalise_content(getattr(value, attr))
                    if normalised:
                        return normalised
            try:
                text = str(value).strip()
            except Exception:  # pragma: no cover - defensive
                return None
            return text or None

        def _extract_from_choice(choice: Any) -> str | None:
            mapping_choice = _as_mapping(choice)
            if mapping_choice is not None:
                message = mapping_choice.get("message")
                normalised = _normalise_content(message)
                if normalised:
                    return normalised
                text = mapping_choice.get("text")
                if text is not None:
                    return _normalise_content(text)
            else:
                if hasattr(choice, "message"):
                    normalised = _normalise_content(choice.message)
                    if normalised:
                        return normalised
                if hasattr(choice, "text"):
                    normalised = _normalise_content(choice.text)
                    if normalised:
                        return normalised
            return None

        choices_value: Any = None
        response_mapping = _as_mapping(response)
        if response_mapping is not None:
            choices_value = response_mapping.get("choices")
        elif hasattr(response, "choices"):
            choices_value = response.choices
        elif isinstance(response, SequenceABC) and not isinstance(
            response,
            (bytes, bytearray, str),
        ):
            choices_value = cast("Sequence[Any]", response)

        choices_sequence: Sequence[Any] = ()
        if isinstance(choices_value, Mapping):
            choices_sequence = [choices_value]
        elif isinstance(choices_value, SequenceABC) and not isinstance(
            choices_value, (bytes, bytearray, str)
        ):
            choices_sequence = cast("Sequence[Any]", choices_value)

        for choice in choices_sequence:
            content = _extract_from_choice(choice)
            if content:
                return content

        if isinstance(response, str):
            return response

        snippet = _summarise_response_for_error(response)
        message = "LiteLLM response did not include message content."
        if snippet:
            message += f" Payload: {snippet}"
        raise PromptEngineeringError(message)

    def _system_prompt_text(self) -> str:
        """Return the configured system prompt or the default meta-prompt."""

        return (self.system_prompt or PROMPT_ENGINEERING_PROMPT).strip()


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    """Return a mapping view for dynamic LiteLLM payload objects when possible."""

    if isinstance(value, Mapping):
        return cast("Mapping[str, Any]", value)
    for attr in ("model_dump", "dict"):
        candidate = getattr(value, attr, None)
        if callable(candidate):
            try:
                result = candidate()
            except Exception:  # pragma: no cover - best effort normalisation
                continue
            if isinstance(result, Mapping):
                return cast("Mapping[str, Any]", result)
    return None


def _summarise_response_for_error(response: Any) -> str:
    """Return a short string describing an unexpected LiteLLM payload."""

    try:
        if isinstance(response, (dict, list)):
            text = json.dumps(response)
        else:
            text = str(response)
    except Exception:  # pragma: no cover - defensive
        return ""
    snippet = text.strip()
    if len(snippet) > 120:
        snippet = snippet[:117].rstrip() + "..."
    return snippet


def _parse_refinement_payload(text: str) -> dict[str, Any]:
    """Parse the refined prompt payload, tolerating leading/trailing text."""

    stripped = text.strip()
    if not stripped:
        raise ValueError("empty payload")
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, Mapping):
            parsed_mapping = cast("Mapping[str, Any]", parsed)
            return {str(key): value for key, value in parsed_mapping.items()}
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            parsed_obj, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed_obj, Mapping):
            parsed_mapping = cast("Mapping[str, Any]", parsed_obj)
            return {str(key): value for key, value in parsed_mapping.items()}
    raise ValueError("unable to parse prompt engineering payload")


__all__ = ["PromptEngineer", "PromptEngineeringError", "PromptRefinement"]
