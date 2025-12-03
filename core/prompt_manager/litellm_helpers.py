"""LiteLLM helper wiring mixin for Prompt Manager.

Updates:
  v0.1.0 - 2025-12-03 - Extract LiteLLM helper initialisation from package init.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from config import LITELLM_ROUTED_WORKFLOWS

from ..intent_classifier import IntentClassifier
from .workflows import LiteLLMWorkflowMixin

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence

    from ..execution import CodexExecutor
    from ..name_generation import (
        LiteLLMCategoryGenerator,
        LiteLLMDescriptionGenerator,
        LiteLLMNameGenerator,
    )
    from ..prompt_engineering import PromptEngineer
    from ..scenario_generation import LiteLLMScenarioGenerator

__all__ = ["LiteLLMWiringMixin"]


class LiteLLMWiringMixin(LiteLLMWorkflowMixin):
    """Mixin responsible for initial LiteLLM helper and executor wiring."""

    _name_generator: LiteLLMNameGenerator | None
    _description_generator: LiteLLMDescriptionGenerator | None
    _scenario_generator: LiteLLMScenarioGenerator | None
    _category_generator: LiteLLMCategoryGenerator | None
    _prompt_engineer: PromptEngineer | None
    _prompt_structure_engineer: PromptEngineer | None
    _litellm_fast_model: str | None
    _litellm_inference_model: str | None
    _litellm_workflow_models: dict[str, str]
    _litellm_stream: bool
    _litellm_drop_params: Sequence[str] | None
    _litellm_reasoning_effort: str | None
    _intent_classifier: IntentClassifier | None
    _executor: CodexExecutor | None
    _prompt_templates: dict[str, str]

    def _initialise_litellm_helpers(
        self,
        *,
        name_generator: LiteLLMNameGenerator | None,
        description_generator: LiteLLMDescriptionGenerator | None,
        scenario_generator: LiteLLMScenarioGenerator | None,
        category_generator: LiteLLMCategoryGenerator | None,
        prompt_engineer: PromptEngineer | None,
        structure_prompt_engineer: PromptEngineer | None,
        fast_model: str | None,
        inference_model: str | None,
        workflow_models: Mapping[str, str | None] | None,
        executor: CodexExecutor | None,
        intent_classifier: IntentClassifier | None,
        prompt_templates: Mapping[str, object] | None,
    ) -> None:
        """Configure LiteLLM helpers, routing, and executor policies."""
        self._name_generator = name_generator
        self._description_generator = description_generator
        self._scenario_generator = scenario_generator
        self._category_generator = category_generator
        self._prompt_engineer = prompt_engineer
        self._prompt_structure_engineer = structure_prompt_engineer or prompt_engineer

        self._litellm_fast_model = self._normalise_model_identifier(fast_model)
        if self._litellm_fast_model is None and name_generator is not None:
            generator_model = getattr(name_generator, "model", None)
            if generator_model:
                self._litellm_fast_model = self._normalise_model_identifier(generator_model)

        self._litellm_inference_model = self._normalise_model_identifier(inference_model)

        routing: dict[str, str] = {}
        if workflow_models:
            for key, value in workflow_models.items():
                key_str = str(key).strip()
                if key_str not in LITELLM_ROUTED_WORKFLOWS:
                    continue
                if value is None:
                    continue
                choice = str(value).strip().lower()
                if choice == "inference":
                    routing[key_str] = "inference"
        self._litellm_workflow_models = routing

        self._litellm_stream = False
        self._litellm_drop_params = None
        self._litellm_reasoning_effort = None

        for candidate in (
            name_generator,
            scenario_generator,
            prompt_engineer,
            executor,
            category_generator,
        ):
            if candidate is not None and getattr(candidate, "drop_params", None):
                raw_params = cast("Sequence[str]", candidate.drop_params)
                self._litellm_drop_params = tuple(str(param) for param in raw_params)
                break

        for candidate in (
            executor,
            name_generator,
            scenario_generator,
            prompt_engineer,
            category_generator,
        ):
            if candidate is not None and hasattr(candidate, "stream"):
                self._litellm_stream = bool(getattr(candidate, "stream", False))
                if self._litellm_stream:
                    break

        if executor is not None and getattr(executor, "reasoning_effort", None):
            effort = executor.reasoning_effort
            self._litellm_reasoning_effort = str(effort) if effort else None

        self._intent_classifier = intent_classifier or IntentClassifier()

        self._executor = executor
        if self._executor is not None:
            if self._litellm_drop_params:
                self._executor.drop_params = list(self._litellm_drop_params)
            if self._litellm_reasoning_effort:
                self._executor.reasoning_effort = self._litellm_reasoning_effort
            self._executor.stream = self._litellm_stream

        self._prompt_templates = self._normalise_prompt_templates(prompt_templates)
