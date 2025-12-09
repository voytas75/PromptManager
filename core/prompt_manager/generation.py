"""LiteLLM generation and category insight helpers for Prompt Manager.

Updates:
  v0.1.1 - 2025-12-09 - Provide offline-friendly errors when LiteLLM is missing.
  v0.1.0 - 2025-12-02 - Extract prompt generation and category insight APIs into mixin.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping, MutableMapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar, cast

from models.category_model import PromptCategory, slugify_category

from ..exceptions import (
    CategorySuggestionError,
    DescriptionGenerationError,
    NameGenerationError,
    PromptManagerError,
    PromptStorageError,
    ScenarioGenerationError,
)
from ..intent_classifier import IntentLabel
from ..notifications import NotificationLevel

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from models.prompt_model import Prompt

    from . import PromptManager as _PromptManager
else:  # pragma: no cover - runtime fallback for typing name
    Prompt = Any
    _PromptManager = Any

logger = logging.getLogger(__name__)

__all__ = ["GenerationMixin"]

_PromptManagerErrorT = TypeVar("_PromptManagerErrorT", bound=PromptManagerError)


_CATEGORY_DEFAULT_MAP: dict[IntentLabel, str] = {
    IntentLabel.ANALYSIS: "Code Analysis",
    IntentLabel.DEBUG: "Reasoning / Debugging",
    IntentLabel.REFACTOR: "Refactoring",
    IntentLabel.ENHANCEMENT: "Enhancement",
    IntentLabel.DOCUMENTATION: "Documentation",
    IntentLabel.REPORTING: "Reporting",
    IntentLabel.GENERAL: "General",
}

_CATEGORY_KEYWORD_HINTS: Sequence[tuple[str, str]] = (
    ("bug", "Reasoning / Debugging"),
    ("error", "Reasoning / Debugging"),
    ("refactor", "Refactoring"),
    ("optimize", "Enhancement"),
    ("optimiz", "Enhancement"),
    ("document", "Documentation"),
    ("summary", "Reporting"),
    ("report", "Reporting"),
)

_CATEGORY_INSIGHT_KEY = "category_insight"


def _match_category_label(
    candidate: str | None,
    categories: Sequence[PromptCategory],
) -> str | None:
    """Return the stored category label that best matches *candidate*."""
    if not candidate:
        return None
    text = str(candidate).strip()
    if not text:
        return None
    lowered = text.lower()
    slug = slugify_category(text)
    for category in categories:
        if category.label.lower() == lowered:
            return category.label
        if slug and slug == category.slug:
            return category.label
    return None


class GenerationMixin:
    """Prompt generation helpers shared across the manager and GUI flows."""

    def _as_prompt_manager(self) -> _PromptManager:
        """Return *self* with PromptManager-specific typing for helper access."""
        return cast("_PromptManager", self)

    def _raise_llm_unavailable(
        self,
        error_type: type[_PromptManagerErrorT],
        capability: str,
    ) -> NoReturn:
        """Raise a capability-specific error with the manager's offline message."""
        manager = self._as_prompt_manager()
        message = manager.llm_status_message(capability)
        raise error_type(message)

    def generate_prompt_name(self, context: str) -> str:
        """Return a prompt name using the configured LiteLLM generator."""
        manager = self._as_prompt_manager()
        if manager._name_generator is None:
            self._raise_llm_unavailable(
                NameGenerationError,
                "Prompt name generation",
            )
        task_id = f"name-gen:{uuid.uuid4()}"
        metadata = {"context_length": len(context or "")}
        with manager._notification_center.track_task(
            title="Prompt name generation",
            task_id=task_id,
            start_message="Generating prompt name via LiteLLM…",
            success_message="Prompt name generated.",
            failure_message="Prompt name generation failed",
            metadata=metadata,
            level=NotificationLevel.INFO,
        ):
            try:
                suggestion = manager._name_generator.generate(context)
            except Exception as exc:  # pragma: no cover - backend determined
                raise NameGenerationError(str(exc)) from exc
        return suggestion

    def generate_prompt_description(
        self,
        context: str,
        *,
        allow_fallback: bool = True,
        prompt: Prompt | None = None,
    ) -> str:
        """Return a prompt description using LiteLLM with an optional deterministic fallback."""
        manager = self._as_prompt_manager()
        text = (context or "").strip()
        if not text:
            raise DescriptionGenerationError(
                "Prompt context is required to generate a description."
            )
        if manager._description_generator is None:
            if allow_fallback:
                logger.debug("Description generator missing; using fallback summary")
                return manager._build_description_fallback(text, prompt=prompt)
            self._raise_llm_unavailable(
                DescriptionGenerationError,
                "Prompt description generation",
            )
        task_id = f"description-gen:{uuid.uuid4()}"
        metadata = {"context_length": len(text)}
        with manager._notification_center.track_task(
            title="Prompt description generation",
            task_id=task_id,
            start_message="Generating prompt description via LiteLLM…",
            success_message="Prompt description generated.",
            failure_message="Prompt description generation failed",
            metadata=metadata,
            level=NotificationLevel.INFO,
        ):
            try:
                summary = manager._description_generator.generate(text)
            except Exception as exc:  # pragma: no cover - backend determined
                if allow_fallback:
                    logger.warning(
                        "LiteLLM description generation failed; falling back",
                        exc_info=True,
                    )
                    return manager._build_description_fallback(text, prompt=prompt)
                raise DescriptionGenerationError(str(exc)) from exc
        return summary

    def generate_prompt_scenarios(
        self,
        context: str,
        *,
        max_scenarios: int = 3,
    ) -> list[str]:
        """Return usage scenarios for a prompt via the configured LiteLLM helper."""
        manager = self._as_prompt_manager()
        if manager._scenario_generator is None:
            self._raise_llm_unavailable(
                ScenarioGenerationError,
                "Prompt scenario generation",
            )
        task_id = f"scenario-gen:{uuid.uuid4()}"
        metadata = {
            "context_length": len(context or ""),
            "max_scenarios": max(0, int(max_scenarios)),
        }
        with manager._notification_center.track_task(
            title="Prompt scenario generation",
            task_id=task_id,
            start_message="Generating scenarios via LiteLLM…",
            success_message="Prompt scenarios generated.",
            failure_message="Prompt scenario generation failed",
            metadata=metadata,
            level=NotificationLevel.INFO,
        ):
            try:
                scenarios = manager._scenario_generator.generate(
                    context,
                    max_scenarios=max_scenarios,
                )
            except Exception as exc:  # pragma: no cover - backend determined
                raise ScenarioGenerationError(str(exc)) from exc
        return scenarios

    def refresh_prompt_scenarios(
        self,
        prompt_id: uuid.UUID,
        *,
        max_scenarios: int = 3,
    ) -> Prompt:
        """Regenerate and persist scenarios for the specified prompt."""
        manager = self._as_prompt_manager()
        prompt = manager.get_prompt(prompt_id)
        context_source = prompt.context or prompt.description
        if not context_source:
            raise ScenarioGenerationError(
                "Prompt is missing context or description; unable to generate scenarios."
            )
        scenarios = manager.generate_prompt_scenarios(context_source, max_scenarios=max_scenarios)
        prompt.scenarios = list(scenarios)
        ext5_payload: MutableMapping[str, Any] | None
        if isinstance(prompt.ext5, MutableMapping):
            ext5_payload = prompt.ext5
        elif isinstance(prompt.ext5, Mapping):
            ext5_payload = dict(prompt.ext5)
        else:
            ext5_payload = None
        if scenarios:
            if ext5_payload is None:
                ext5_payload = {}
            ext5_payload["scenarios"] = list(scenarios)
        elif ext5_payload is not None:
            ext5_payload.pop("scenarios", None)
            if not ext5_payload:
                ext5_payload = None
        prompt.ext5 = ext5_payload
        prompt.last_modified = datetime.now(UTC)
        try:
            return manager.update_prompt(prompt)
        except PromptManagerError as exc:
            raise PromptStorageError("Failed to persist refreshed scenarios") from exc

    def generate_prompt_category(self, context: str) -> str:
        """Suggest a prompt category using LiteLLM with classifier-based fallback."""
        manager = self._as_prompt_manager()
        text = (context or "").strip()
        if not text:
            return ""
        categories = manager.list_categories()
        if not categories:
            return ""

        if manager._category_generator is not None:
            suggestion = manager._run_category_generator(text, categories)
            if suggestion:
                matched = _match_category_label(suggestion, categories)
                if matched:
                    return matched

        return manager._fallback_category_from_context(text, categories)

    def _run_category_generator(
        self,
        context: str,
        categories: Sequence[PromptCategory],
    ) -> str:
        """Return category suggestion from LiteLLM, logging failures for fallbacks."""
        manager = self._as_prompt_manager()
        if manager._category_generator is None:
            return ""
        task_id = f"category-suggest:{uuid.uuid4()}"
        metadata = {
            "context_length": len(context or ""),
            "category_count": len(categories),
        }
        with manager._notification_center.track_task(
            title="Prompt category suggestion",
            task_id=task_id,
            start_message="Suggesting prompt category via LiteLLM…",
            success_message="Prompt category suggested.",
            failure_message="Prompt category suggestion failed",
            metadata=metadata,
            level=NotificationLevel.INFO,
        ):
            try:
                return manager._category_generator.generate(context, categories=categories)
            except CategorySuggestionError as exc:
                logger.debug(
                    "LiteLLM category suggestion failed",
                    extra={"error": str(exc)},
                    exc_info=True,
                )
            except Exception:  # pragma: no cover - defensive
                logger.warning("LiteLLM category suggestion failed unexpectedly", exc_info=True)
        return ""

    def _fallback_category_from_context(
        self,
        context: str,
        categories: Sequence[PromptCategory],
    ) -> str:
        """Return a category suggestion using heuristics and classifier hints."""
        manager = self._as_prompt_manager()
        classifier = getattr(manager, "_intent_classifier", None)
        if classifier is not None:
            prediction = classifier.classify(context)
            if prediction.category_hints:
                for hint in prediction.category_hints:
                    matched = _match_category_label(hint, categories)
                    if matched:
                        return matched
                return prediction.category_hints[0]
            fallback = _CATEGORY_DEFAULT_MAP.get(prediction.label)
            if fallback:
                resolved = _match_category_label(fallback, categories)
                if resolved:
                    return resolved
                return fallback

        lowered = context.lower()
        for keyword, category in _CATEGORY_KEYWORD_HINTS:
            if keyword in lowered:
                resolved = _match_category_label(category, categories)
                return resolved or category

        default_label = _match_category_label("General", categories)
        if default_label:
            return default_label
        return categories[0].label if categories else "General"

    def _update_category_insight(
        self,
        prompt: Prompt,
        *,
        previous_prompt: Prompt | None,
    ) -> None:
        """Capture LiteLLM-backed category drift metadata on the prompt."""
        manager = self._as_prompt_manager()
        if manager._category_generator is None:
            manager._set_category_insight_metadata(prompt, None)
            return

        context_text = manager._category_context_text(prompt)
        if not context_text:
            manager._set_category_insight_metadata(prompt, None)
            return

        categories = manager.list_categories()
        if not categories:
            manager._set_category_insight_metadata(prompt, None)
            return

        try:
            suggestion_raw = manager._category_generator.generate(
                context_text, categories=categories
            )
        except CategorySuggestionError as exc:
            logger.debug(
                "Category drift suggestion failed",
                extra={"prompt_id": str(prompt.id), "error": str(exc)},
                exc_info=True,
            )
            manager._set_category_insight_metadata(prompt, None)
            return
        except Exception:  # pragma: no cover - defensive
            logger.warning(
                "Category drift suggestion failed unexpectedly",
                extra={"prompt_id": str(prompt.id)},
                exc_info=True,
            )
            manager._set_category_insight_metadata(prompt, None)
            return

        suggestion_label = _match_category_label(suggestion_raw, categories) or suggestion_raw
        timestamp = datetime.now(UTC).isoformat()
        current_label = (prompt.category or "").strip()
        labels_match = bool(current_label) and current_label.lower() == suggestion_label.lower()
        previous_insight = manager._extract_category_insight(previous_prompt)
        if labels_match:
            adoption_payload = manager._build_category_adoption_insight(
                previous_prompt,
                previous_insight,
                suggestion_label,
                suggestion_raw,
                current_label,
                timestamp,
            )
            if adoption_payload:
                manager._set_category_insight_metadata(prompt, adoption_payload)
            else:
                manager._set_category_insight_metadata(prompt, None)
            return

        insight: dict[str, Any] = {
            "status": "suggested",
            "suggested_label": suggestion_label,
            "suggestion_raw": suggestion_raw,
            "current_label": current_label,
            "updated_at": timestamp,
        }
        if previous_prompt is not None:
            insight["previous_category"] = previous_prompt.category
        manager._set_category_insight_metadata(prompt, insight)

    @staticmethod
    def _category_context_text(prompt: Prompt) -> str:
        """Return the most descriptive text for category inference."""
        for candidate in (prompt.context, prompt.description, prompt.document):
            text = (candidate or "").strip()
            if text:
                return text
        return ""

    @staticmethod
    def _extract_category_insight(prompt: Prompt | None) -> dict[str, Any] | None:
        """Return the stored category insight mapping, if present."""
        if prompt is None:
            return None
        ext2 = prompt.ext2 if isinstance(prompt.ext2, Mapping) else None
        if not ext2:
            return None
        payload = ext2.get(_CATEGORY_INSIGHT_KEY)
        if isinstance(payload, Mapping):
            return {str(key): value for key, value in payload.items()}
        return None

    def _build_category_adoption_insight(
        self,
        previous_prompt: Prompt | None,
        previous_insight: Mapping[str, Any] | None,
        suggestion_label: str,
        suggestion_raw: str,
        current_label: str,
        timestamp: str,
    ) -> dict[str, Any] | None:
        """Return metadata describing an accepted LiteLLM category suggestion."""
        if previous_prompt is None or previous_insight is None:
            return None
        status = str(previous_insight.get("status") or "").strip().lower()
        if status != "suggested":
            return None
        previous_suggested = str(previous_insight.get("suggested_label") or "").strip()
        if not previous_suggested or previous_suggested.lower() != suggestion_label.lower():
            return None
        previous_label = (previous_prompt.category or "").strip()
        if not previous_label or previous_label.lower() == current_label.lower():
            return None
        return {
            "status": "adopted",
            "suggested_label": suggestion_label,
            "suggestion_raw": suggestion_raw,
            "previous_label": previous_label,
            "current_label": current_label,
            "adopted_at": timestamp,
        }

    @staticmethod
    def _set_category_insight_metadata(
        prompt: Prompt,
        insight: Mapping[str, Any] | None,
    ) -> None:
        """Persist or clear category insight metadata on the prompt record."""
        if isinstance(prompt.ext2, MutableMapping):
            metadata: dict[str, Any] = dict(prompt.ext2)
        elif isinstance(prompt.ext2, Mapping):
            metadata = dict(prompt.ext2)
        else:
            metadata = {}

        if insight is None:
            if metadata.pop(_CATEGORY_INSIGHT_KEY, None) is not None:
                prompt.ext2 = metadata or None
            return

        metadata[_CATEGORY_INSIGHT_KEY] = dict(insight)
        prompt.ext2 = metadata

    def _build_description_fallback(self, context: str, prompt: Prompt | None) -> str:
        """Return a deterministic summary derived from prompt metadata and context."""
        segments: list[str] = []
        if prompt is not None:
            name = (prompt.name or "").strip()
            category = (prompt.category or "").strip() or "General"
            if name:
                segments.append(f"{name} focuses on {category.lower()} workflows.")
            tags = ", ".join(tag.strip() for tag in prompt.tags if tag and str(tag).strip())
            if tags:
                segments.append(f"Common tags: {tags}.")
            scenario_text = ""
            for scenario in getattr(prompt, "scenarios", []) or []:
                candidate = str(scenario or "").strip()
                if candidate:
                    scenario_text = candidate
                    break
            if scenario_text:
                trimmed = scenario_text.rstrip(".")
                segments.append(f"Example use: {trimmed}.")
        snippet = context.strip()
        if snippet:
            segments.append(f"Overview: {snippet[:200]}")
        if not segments:
            return "No description available."
        return " ".join(segments)
