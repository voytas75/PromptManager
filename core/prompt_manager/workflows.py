"""LiteLLM workflow configuration helpers for Prompt Manager.

Updates:
  v0.1.0 - 2025-12-03 - Extract runtime LiteLLM configuration API from PromptManager.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence  # noqa: TCH003
from importlib import import_module
from typing import TYPE_CHECKING, Any

from config import LITELLM_ROUTED_WORKFLOWS
from prompt_templates import DEFAULT_PROMPT_TEMPLATES, PROMPT_TEMPLATE_KEYS

from ..exceptions import NameGenerationError

if TYPE_CHECKING:
    from ..intent_classifier import IntentClassifier
    from . import PromptManager as _PromptManager
else:
    _PromptManager = Any

__all__ = ["LiteLLMWorkflowMixin"]

logger = logging.getLogger(__name__)


def _resolve_factory(name: str) -> Callable[..., Any]:
    """Return a constructor from the core.prompt_manager module."""
    module = import_module("core.prompt_manager")
    factory = getattr(module, name)
    if not callable(factory):  # pragma: no cover - defensive
        raise NameGenerationError(f"{name} is not callable.")
    return factory


class LiteLLMWorkflowMixin:
    """Mixin that encapsulates LiteLLM-backed workflow configuration."""

    @staticmethod
    def _normalise_model_identifier(value: str | None) -> str | None:
        """Return a stripped model identifier when provided."""
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _normalise_prompt_templates(
        overrides: Mapping[str, object] | None,
    ) -> dict[str, str]:
        """Return a cleaned mapping of workflow prompt overrides."""
        if not overrides:
            return {}
        cleaned: dict[str, str] = {}
        for key, text in overrides.items():
            if key not in PROMPT_TEMPLATE_KEYS:
                continue
            if not isinstance(text, str):
                continue
            stripped = text.strip()
            default_text = DEFAULT_PROMPT_TEMPLATES.get(key)
            if stripped and stripped != default_text:
                cleaned[key] = stripped
        return cleaned

    def set_name_generator(
        self,
        model: str | None,
        api_key: str | None,
        api_base: str | None,
        api_version: str | None,
        *,
        inference_model: str | None = None,
        workflow_models: Mapping[str, str | None] | None = None,
        drop_params: Sequence[str] | None = None,
        reasoning_effort: str | None = None,
        stream: bool | None = None,
        prompt_templates: Mapping[str, object] | None = None,
    ) -> None:
        """Configure LiteLLM-backed workflows at runtime."""
        self._litellm_fast_model = self._normalise_model_identifier(model)
        self._litellm_inference_model = self._normalise_model_identifier(inference_model)
        routing: dict[str, str] = {}
        if workflow_models:
            for key, value in workflow_models.items():
                workflow_key = str(key).strip()
                if workflow_key not in LITELLM_ROUTED_WORKFLOWS:
                    continue
                if value is None:
                    continue
                choice = str(value).strip().lower()
                if choice == "inference":
                    routing[workflow_key] = "inference"
        self._litellm_workflow_models = routing

        if drop_params is not None:
            cleaned_params = [str(item).strip() for item in drop_params if str(item).strip()]
            self._litellm_drop_params = tuple(cleaned_params) if cleaned_params else None
        else:
            self._litellm_drop_params = None

        self._litellm_reasoning_effort = reasoning_effort
        if stream is not None:
            self._litellm_stream = bool(stream)

        if prompt_templates is not None:
            self._prompt_templates = self._normalise_prompt_templates(prompt_templates)

        # Reset existing helpers before rebuilding.
        self._name_generator = None
        self._description_generator = None
        self._prompt_engineer = None
        self._prompt_structure_engineer = None
        self._scenario_generator = None
        self._category_generator = None
        self._executor = None

        if not (self._litellm_fast_model or self._litellm_inference_model):
            self._litellm_reasoning_effort = None
            self._litellm_stream = False
            return

        drop_params_payload = list(self._litellm_drop_params) if self._litellm_drop_params else None

        def _select_model(workflow: str) -> str | None:
            selection = self._litellm_workflow_models.get(workflow, "fast")
            if selection == "inference":
                return self._litellm_inference_model or self._litellm_fast_model
            return self._litellm_fast_model or self._litellm_inference_model

        def _construct(factory_name: str, workflow: str, **extra: Any) -> Any | None:
            selected_model = _select_model(workflow)
            if not selected_model:
                return None
            factory = _resolve_factory(factory_name)
            return factory(
                model=selected_model,
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                drop_params=drop_params_payload,
                **extra,
            )

        template_overrides = dict(getattr(self, "_prompt_templates", {}))
        try:
            self._name_generator = _construct(
                "LiteLLMNameGenerator",
                "name_generation",
                system_prompt=template_overrides.get("name_generation"),
            )
            self._description_generator = _construct(
                "LiteLLMDescriptionGenerator",
                "description_generation",
                system_prompt=template_overrides.get("description_generation"),
            )
            self._prompt_engineer = _construct(
                "PromptEngineer",
                "prompt_engineering",
                system_prompt=template_overrides.get("prompt_engineering"),
            )
            structure_engineer = _construct(
                "PromptEngineer",
                "prompt_structure_refinement",
                system_prompt=template_overrides.get("prompt_engineering"),
            )
            self._prompt_structure_engineer = structure_engineer or self._prompt_engineer
            self._scenario_generator = _construct(
                "LiteLLMScenarioGenerator",
                "scenario_generation",
                system_prompt=template_overrides.get("scenario_generation"),
            )
            self._category_generator = _construct(
                "LiteLLMCategoryGenerator",
                "category_generation",
                system_prompt=template_overrides.get("category_generation"),
            )
            self._executor = _construct(
                "CodexExecutor",
                "prompt_execution",
                reasoning_effort=self._litellm_reasoning_effort,
                stream=self._litellm_stream,
            )
        except RuntimeError as exc:
            raise NameGenerationError(str(exc)) from exc

        if self._executor is not None:
            if self._litellm_drop_params:
                self._executor.drop_params = list(self._litellm_drop_params)
            if self._litellm_reasoning_effort:
                self._executor.reasoning_effort = self._litellm_reasoning_effort
            self._executor.stream = self._litellm_stream

        intent_classifier: IntentClassifier | None = getattr(self, "_intent_classifier", None)
        if intent_classifier is not None and (
            self._litellm_fast_model or self._litellm_inference_model
        ):
            logger.debug("LiteLLM powered features enabled for intent classifier")
