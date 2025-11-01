"""LiteLLM-backed prompt metadata generation utilities.

Updates: v0.7.1 - 2025-11-11 - Summarise LiteLLM errors for friendlier GUI fallbacks.
Updates: v0.7.0 - 2025-11-07 - Add description generator alongside name helper.
Updates: v0.6.0 - 2025-11-07 - Share LiteLLM import helper with embedding adapters.
Updates: v0.5.0 - 2025-11-05 - Introduce LiteLLM name generator with graceful fallbacks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

from .litellm_adapter import get_completion

class NameGenerationError(Exception):
    """Raised when a prompt name cannot be generated."""


class DescriptionGenerationError(Exception):
    """Raised when a prompt description cannot be generated."""


@dataclass(slots=True)
class LiteLLMNameGenerator:
    """Generate prompt names via LiteLLM chat completion API."""

    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    timeout_seconds: float = 10.0
    api_version: Optional[str] = None

    _SYSTEM_PROMPT = (
        "You generate concise, descriptive prompt names for a prompt catalogue. "
        "Return a title of at most 5 words. Avoid punctuation except spaces."
    )

    def generate(self, context: str) -> str:
        """Return an LLM-generated prompt name from contextual text."""
        completion, LiteLLMException = get_completion()
        if not context.strip():
            raise NameGenerationError("Prompt context is required to generate a name.")

        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Suggest a concise prompt name for the following content:\n\n{context.strip()}",
                },
            ],
            "temperature": 0.2,
            "max_tokens": 16,
            "timeout": self.timeout_seconds,
        }
        if self.api_key:
            request["api_key"] = self.api_key
        if self.api_base:
            request["api_base"] = self.api_base
        if self.api_version:
            request["api_version"] = self.api_version

        try:
            response = completion(**request)  # type: ignore[arg-type]
        except LiteLLMException as exc:  # type: ignore[arg-type]
            raise NameGenerationError(_summarise_litellm_error(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise NameGenerationError("Unexpected error while calling LiteLLM") from exc

        try:
            message = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:  # pragma: no cover - defensive
            raise NameGenerationError("LiteLLM returned an unexpected payload") from exc

        suggestion = message.strip()
        if not suggestion:
            raise NameGenerationError("LiteLLM returned an empty suggestion.")
        return suggestion


@dataclass(slots=True)
class LiteLLMDescriptionGenerator:
    """Generate succinct prompt descriptions via LiteLLM."""

    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    timeout_seconds: float = 12.0
    api_version: Optional[str] = None

    _SYSTEM_PROMPT = (
        "You write concise catalogue descriptions for reusable AI prompts. "
        "Summarise the intent, inputs, and expected outcomes in 2 sentences. "
        "Avoid bullet lists and marketing fluff."
    )

    def generate(self, context: str) -> str:
        completion, LiteLLMException = get_completion()
        if not context.strip():
            raise DescriptionGenerationError("Prompt context is required to generate a description.")
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._SYSTEM_PROMPT},
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
            "timeout": self.timeout_seconds,
        }
        if self.api_key:
            request["api_key"] = self.api_key
        if self.api_base:
            request["api_base"] = self.api_base
        if self.api_version:
            request["api_version"] = self.api_version

        try:
            response = completion(**request)  # type: ignore[arg-type]
        except LiteLLMException as exc:  # type: ignore[arg-type]
            raise DescriptionGenerationError(_summarise_litellm_error(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise DescriptionGenerationError("Unexpected error while calling LiteLLM") from exc

        try:
            message = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:  # pragma: no cover - defensive
            raise DescriptionGenerationError("LiteLLM returned an unexpected payload") from exc

        summary = message.strip()
        if not summary:
            raise DescriptionGenerationError("LiteLLM returned an empty description.")
        return summary


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
    "NameGenerationError",
    "DescriptionGenerationError",
]
