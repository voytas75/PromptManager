"""LiteLLM-backed prompt engineering utilities for refining prompts.

Updates: v0.1.2 - 2025-11-05 - Stop forwarding LiteLLM timeout parameter to avoid premature failures.
Updates: v0.1.1 - 2025-11-02 - Drop configured LiteLLM parameters locally before refinement calls.
Updates: v0.1.0 - 2025-11-15 - Introduce prompt refinement helper using meta-prompt rules.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

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
    raw_response: Optional[dict[str, Any]] = None


def _format_list(items: Optional[Iterable[Any]]) -> list[str]:
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
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    temperature: float = 0.25
    top_p: float = 0.9
    timeout_seconds: Optional[float] = None
    max_tokens: int = 1200
    drop_params: Optional[Sequence[str]] = None

    _SYSTEM_PROMPT = (
        "You are a meticulous prompt engineering assistant."
        " Analyse and refine prompts using the following ordered rules."
        "\n\n"
        "### Rules for LLM Prompt Query Language\n"
        "1. Clarity and Specificity: define the task, detail level, tone, and explicitly state"
        " what is out of scope.\n"
        "2. Context Completeness: include relevant background, constraints, and summarise"
        " large context within token budgets.\n"
        "3. Output Structure Control: specify the desired output structure, format, and how"
        " to handle unknowns or uncertainty.\n"
        "4. Role/Goal Separation: optionally assign a role that is limited to the task and"
        " restate the goal without implying sentience.\n"
        "5. Deterministic Language: use imperative phrasing and explicit delimiters to"
        " separate instructions from data.\n"
        "6. Anthropomorphism Control: default to non-anthropomorphic framing unless the"
        " use-case demands a persona.\n"
        "7. Prevent Agency Misinterpretation: avoid implying memory, intent, or autonomy"
        " beyond the prompt context.\n"
        "8. Psychological Transparency: request reasoning steps and confidence when"
        " helpful, balancing brevity and clarity.\n"
        "9. Example-Driven Prompting: include examples (and counter-examples when"
        " relevant) to anchor expectations.\n"
        "10. Iterative Refinement: document refinements, versioning, and measurement"
        " cues for future iterations.\n"
        "11. Hybrid Natural-Formal Syntax: combine natural language with structured"
        " schemas (e.g., JSON sections) when precision matters.\n"
        "12. Guardrails and Safety: surface security, privacy, or policy constraints and"
        " reference handling instructions.\n"
        "13. Token and Resource Efficiency: stay concise, highlight required vs optional"
        " context, and note truncation strategies.\n"
        "14. Role-Play for Engagement: only when the use-case benefits from it and make"
        " it explicit that the persona is simulated.\n"
        "15. Conversational Polishing: optional—politeness is secondary to clarity.\n"
        "Always preserve the author's intent while fixing omissions or ambiguities."
        " Return precise, security conscious prompts with actionable structure."
        " Produce deterministic guidance that can run unchanged in production systems."
        "\n\nReturn a JSON object with the keys: analysis (string), improved_prompt (string),"
        " checklist (array of strings), warnings (array of strings), and confidence (number 0-1)."
        " If no changes are warranted set improved_prompt equal to the original but justify"
        " the decision in analysis."
    )

    def refine(
        self,
        prompt_text: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        negative_constraints: Optional[Sequence[str]] = None,
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
        )
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._SYSTEM_PROMPT},
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
                drop_candidates={"max_tokens", "max_output_tokens", "temperature", "timeout", "top_p"},
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

        raw_response: Optional[dict[str, Any]] = None
        if isinstance(response, dict):
            raw_response = response

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

    def _build_user_payload(
        self,
        prompt_text: str,
        *,
        name: Optional[str],
        description: Optional[str],
        category: Optional[str],
        tags: Optional[Sequence[str]],
        negative_constraints: Optional[Sequence[str]],
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
                constraint.strip() for constraint in negative_constraints if constraint and str(constraint).strip()
            )
            if blocked:
                sections.append(f"Negative Constraints: {blocked}")

        sections.append("\n### Current Prompt\n" + prompt_text.strip())
        sections.append(
            "\n### Tasks\n"
            "1. Identify gaps or violations of the meta-prompt rules.\n"
            "2. Produce an improved prompt that retains the author's intent, integrates explicit"
            " instructions, context, and guardrails, and specifies the output contract.\n"
            "3. Note any residual risks, missing information, or follow-up questions in the warnings list."
        )
        sections.append(
            "Ensure the improved prompt is ready for production use, uses deterministic language,"
            " and includes explicit delimiters or sections where appropriate."
        )
        return "\n".join(sections)

    @staticmethod
    def _extract_message(response: Any) -> str:
        """Extract the primary message content from the LiteLLM response payload."""

        def _normalise_content(value: Any) -> Optional[str]:
            if value is None:
                return None
            if isinstance(value, str):
                text = value.strip()
                return text or None
            if isinstance(value, dict):
                for key in ("content", "text", "value"):
                    if key in value:
                        normalised = _normalise_content(value[key])
                        if normalised:
                            return normalised
                return None
            if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, str)):
                parts: list[str] = []
                for item in value:
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

        def _extract_from_choice(choice: Any) -> Optional[str]:
            if choice is None:
                return None
            if isinstance(choice, dict):
                message = choice.get("message")
                normalised = _normalise_content(message)
                if normalised:
                    return normalised
                text = choice.get("text")
                if text:
                    return _normalise_content(text)
            else:
                if hasattr(choice, "message"):
                    normalised = _normalise_content(getattr(choice, "message"))
                    if normalised:
                        return normalised
                if hasattr(choice, "text"):
                    normalised = _normalise_content(getattr(choice, "text"))
                    if normalised:
                        return normalised
            return None

        choices: Optional[Any] = None
        if isinstance(response, dict):
            choices = response.get("choices")
        elif hasattr(response, "choices"):
            choices = getattr(response, "choices")
        elif isinstance(response, list):
            choices = response

        if isinstance(choices, dict):
            choices = [choices]

        if isinstance(choices, Iterable) and not isinstance(choices, (bytes, bytearray, str)):
            for choice in choices:
                content = _extract_from_choice(choice)
                if content:
                    return content

        if isinstance(response, str):
            return response

        snippet = _summarise_response_for_error(response)
        raise PromptEngineeringError(
            "LiteLLM response did not include message content." + (f" Payload: {snippet}" if snippet else "")
        )


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
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            parsed, end = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("unable to parse prompt engineering payload")


__all__ = ["PromptEngineer", "PromptEngineeringError", "PromptRefinement"]
