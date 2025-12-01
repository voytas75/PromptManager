"""LiteLLM-backed prompt metadata generation utilities.

Updates:
  v0.7.9 - 2025-12-01 - Normalise LiteLLM ModelResponse payloads before parsing text.
  v0.7.8 - 2025-11-29 - Move Sequence import behind TYPE_CHECKING per Ruff TC003.
  v0.7.7 - 2025-11-29 - Guard PromptCategory import for typing and wrap long literals.
  v0.7.6 - 2025-11-24 - Add category suggestion helper leveraging LiteLLM.
  v0.7.5 - 2025-11-23 - Allow custom system prompt overrides supplied via settings.
  v0.7.4 - 2025-11-05 - Remove explicit LiteLLM timeout to avoid premature cancellation errors.
  v0.7.3 - 2025-11-02 - Strip configured drop parameters before calling LiteLLM.
  v0.7.2 - 2025-11-17 - Retry without unsupported LiteLLM parameters when models reject them.
  v0.7.1 - 2025-11-11 - Summarise LiteLLM errors for friendlier GUI fallbacks.
  v0.7.0 - 2025-11-07 - Add description generator alongside name helper.
  v0.6.0 - 2025-11-07 - Share LiteLLM import helper with embedding adapters.
  v0.5.0 - 2025-11-05 - Introduce LiteLLM name generator with graceful fallbacks.
"""


from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from prompt_templates import (
    CATEGORY_GENERATION_PROMPT,
    DESCRIPTION_GENERATION_PROMPT,
    NAME_GENERATION_PROMPT,
)

from .litellm_adapter import (
    apply_configured_drop_params,
    call_completion_with_fallback,
    get_completion,
    serialise_litellm_response,
)

if TYPE_CHECKING:  # pragma: no cover - imported for annotations only
    from collections.abc import Sequence

    from models.category_model import PromptCategory

logger = logging.getLogger(__name__)


class NameGenerationError(Exception):
    """Raised when a prompt name cannot be generated."""


class DescriptionGenerationError(Exception):
    """Raised when a prompt description cannot be generated."""


class CategorySuggestionError(Exception):
    """Raised when a prompt category cannot be suggested."""


@dataclass(slots=True)
class LiteLLMNameGenerator:
    """Generate prompt names via LiteLLM chat completion API."""

    model: str
    api_key: str | None = None
    api_base: str | None = None
    timeout_seconds: float | None = None
    api_version: str | None = None
    drop_params: Sequence[str] | None = None
    system_prompt: str | None = None

    def generate(self, context: str) -> str:
        """Return an LLM-generated prompt name from contextual text."""
        completion, LiteLLMException = get_completion()
        if not context.strip():
            raise NameGenerationError("Prompt context is required to generate a name.")

        request: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_prompt_text()},
                {
                    "role": "user",
                    "content": (
                        "Suggest a concise prompt name for the following content:\n\n"
                        f"{context.strip()}"
                    ),
                },
            ],
            "temperature": 0.2,
            "max_tokens": 16,
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
                "Dropping LiteLLM parameters for name generation",
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
            raise NameGenerationError(_summarise_litellm_error(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise NameGenerationError("Unexpected error while calling LiteLLM") from exc

        payload = serialise_litellm_response(response)
        if payload is None:
            raise NameGenerationError("LiteLLM returned an unexpected payload")

        try:
            message = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:  # pragma: no cover - defensive
            raise NameGenerationError("LiteLLM returned an unexpected payload") from exc

        suggestion = message.strip()
        if not suggestion:
            raise NameGenerationError("LiteLLM returned an empty suggestion.")
        return suggestion

    def _system_prompt_text(self) -> str:
        """Return the configured or default system prompt."""
        return (self.system_prompt or NAME_GENERATION_PROMPT).strip()


@dataclass(slots=True)
class LiteLLMDescriptionGenerator:
    """Generate succinct prompt descriptions via LiteLLM."""

    model: str
    api_key: str | None = None
    api_base: str | None = None
    timeout_seconds: float | None = None
    api_version: str | None = None
    drop_params: Sequence[str] | None = None
    system_prompt: str | None = None

    def generate(self, context: str) -> str:
        """Return a succinct description for the prompt *context* text."""
        completion, LiteLLMException = get_completion()
        if not context.strip():
            raise DescriptionGenerationError(
                "Prompt context is required to generate a description."
            )
        request: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_prompt_text()},
                {
                    "role": "user",
                    "content": (
                        "Write a short description for this prompt that explains when to use it "
                        "and the value it delivers:\n\n"
                        f"{context.strip()}"
                    ),
                },
            ],
            "temperature": 0.3,
            "max_tokens": 120,
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
                "Dropping LiteLLM parameters for description generation",
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
            raise DescriptionGenerationError(_summarise_litellm_error(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise DescriptionGenerationError("Unexpected error while calling LiteLLM") from exc

        # Defensive extraction of the assistant message returned by LiteLLM.  The
        # payload structure **should** be
        # ``{"choices": [{"message": {"content": "..."}}]}`` but we have seen
        # third‑party gateways occasionally return ``null`` or other non‑string
        # values for *content*.  Instead of leaking an AttributeError (which
        # bubbles up as ``'NoneType' object has no attribute 'strip'`` and gives
        # users no actionable clue), convert any structural irregularity into a
        # predictable ``DescriptionGenerationError`` with a concise message.

        payload = serialise_litellm_response(response)
        if payload is None:
            raise DescriptionGenerationError("LiteLLM returned an unexpected payload structure")

        try:
            message = payload["choices"][0]["message"].get("content")  # type: ignore[index]
        except (
            KeyError,
            IndexError,
            TypeError,
            AttributeError,
        ) as exc:  # pragma: no cover - defensive
            raise DescriptionGenerationError(
                "LiteLLM returned an unexpected payload structure"
            ) from exc

        if not isinstance(message, str):
            raise DescriptionGenerationError("LiteLLM returned a non‑text description.")

        summary = message.strip()
        if not summary:
            raise DescriptionGenerationError("LiteLLM returned an empty description.")

        return summary

    def _system_prompt_text(self) -> str:
        """Return the configured or default description system prompt."""
        return (self.system_prompt or DESCRIPTION_GENERATION_PROMPT).strip()


@dataclass(slots=True)
class LiteLLMCategoryGenerator:
    """Suggest prompt categories from the configured catalogue via LiteLLM."""

    model: str
    api_key: str | None = None
    api_base: str | None = None
    timeout_seconds: float | None = None
    api_version: str | None = None
    drop_params: Sequence[str] | None = None
    system_prompt: str | None = None
    max_categories: int = 24

    def generate(self, context: str, *, categories: Sequence[PromptCategory]) -> str:
        """Return the label of the best-fit category for the supplied prompt."""
        completion, LiteLLMException = get_completion()
        prompt_text = context.strip()
        if not prompt_text:
            raise CategorySuggestionError("Prompt context is required to suggest a category.")
        if not categories:
            raise CategorySuggestionError(
                "At least one category is required to suggest a category."
            )

        formatted_categories = self._format_categories(categories)
        request: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_prompt_text()},
                {
                    "role": "user",
                    "content": (
                        "Select the single best category label from the list below "
                        "that matches the prompt.\n"
                        "Respond with the exact label text only, without numbering "
                        "or explanations.\n\n"
                        f"Categories:\n{formatted_categories}\n\nPrompt:\n{prompt_text}"
                    ),
                },
            ],
            "temperature": 0.1,
            "max_tokens": 12,
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
                "Dropping LiteLLM parameters for category suggestion",
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
            raise CategorySuggestionError(_summarise_litellm_error(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise CategorySuggestionError("Unexpected error while calling LiteLLM") from exc

        payload = serialise_litellm_response(response)
        if payload is None:
            raise CategorySuggestionError("LiteLLM returned an unexpected payload")

        try:
            message = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:  # pragma: no cover - defensive
            raise CategorySuggestionError("LiteLLM returned an unexpected payload") from exc

        suggestion = str(message).strip()
        if not suggestion:
            raise CategorySuggestionError("LiteLLM returned an empty category suggestion.")
        return suggestion

    def _format_categories(self, categories: Sequence[PromptCategory]) -> str:
        """Return a newline separated bullet list of categories with descriptions."""
        entries: list[str] = []
        limit = max(1, self.max_categories)
        for category in list(categories)[:limit]:
            label = (category.label or category.slug or "Uncategorised").strip()
            description = (category.description or "").strip()
            entry = f"- {label}"
            if description:
                entry += f": {description}"
            entries.append(entry)
        return "\n".join(entries)

    def _system_prompt_text(self) -> str:
        """Return the configured or default category system prompt."""
        return (self.system_prompt or CATEGORY_GENERATION_PROMPT).strip()


def _summarise_litellm_error(exc: Exception) -> str:
    """Return a concise, user-friendly message for LiteLLM failures."""
    text = str(exc).strip()
    if not text:
        return "LiteLLM request failed."

    lowered = text.lower()
    if "content_filter" in lowered or "responsibleaipolicyviolation" in lowered:
        return (
            "Azure OpenAI blocked this request by content policy. "
            "Adjust the prompt text and try again."
        )
    if "timeout" in lowered or "timed out" in lowered:
        return "LiteLLM request timed out. Please retry or check network connectivity."
    return f"LiteLLM request failed: {text}" if not text.startswith("LiteLLM") else text


__all__ = [
    "LiteLLMNameGenerator",
    "LiteLLMDescriptionGenerator",
    "LiteLLMCategoryGenerator",
    "NameGenerationError",
    "DescriptionGenerationError",
    "CategorySuggestionError",
]
