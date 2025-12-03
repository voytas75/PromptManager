"""Prompt creation and editing helpers extracted from :mod:`gui.main_window`.

Updates:
  v0.15.81 - 2025-12-01 - Provide reusable prompt generation + editor factory service.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from core import IntentLabel, PromptManager, PromptManagerError

from .catalog_workflow_controller import CatalogWorkflowController
from .language_tools import detect_language
from .prompt_editor_flow import PromptDialogFactory, PromptEditorFlow

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Callable, Sequence
    from uuid import UUID

    from PySide6.QtWidgets import QWidget

    from core.prompt_engineering import PromptRefinement
    from models.prompt_model import Prompt
else:  # pragma: no cover - runtime placeholders for type-only names
    from typing import Any as _Any

    Callable = _Any
    Sequence = _Any
    Prompt = _Any
    PromptRefinement = _Any
    QWidget = object


@dataclass(slots=True)
class PromptGenerationComponents:
    """Artifacts produced when building prompt editor workflows."""

    dialog_factory: PromptDialogFactory
    editor_flow: PromptEditorFlow
    catalog_controller: CatalogWorkflowController


class PromptGenerationService:
    """Centralize prompt generation utilities and dialog wiring."""

    def __init__(
        self,
        *,
        manager: PromptManager,
        execute_context_handler: Callable[[Prompt, QWidget | None, str | None], None],
        load_prompts: Callable[[str], None],
        current_search_text: Callable[[], str],
        select_prompt: Callable[[UUID], None],
        delete_prompt: Callable[[Prompt], None],
        status_callback: Callable[[str, int], None],
        error_callback: Callable[[str, str], None],
        current_prompt_supplier: Callable[[], Prompt | None],
        open_version_history_dialog: Callable[[Prompt | None], None],
    ) -> None:
        """Store collaborators used for dialog factories and generators."""
        self._manager = manager
        self._execute_context_handler = execute_context_handler
        self._load_prompts = load_prompts
        self._current_search_text = current_search_text
        self._select_prompt = select_prompt
        self._delete_prompt = delete_prompt
        self._status_callback = status_callback
        self._error_callback = error_callback
        self._current_prompt_supplier = current_prompt_supplier
        self._open_version_history_dialog = open_version_history_dialog

    def rebuild_components(self, parent: QWidget) -> PromptGenerationComponents:
        """Create dialog factories, flows, and controllers for prompt editing."""
        prompt_engineer = self._manager.prompt_engineer and self.refine_prompt_body or None
        structure_refiner = (
            self.refine_prompt_structure if self._manager.prompt_structure_engineer else None
        )

        def _execute_context_from_dialog(
            prompt_obj: Prompt,
            context_text: str,
            dialog: QWidget | None,
        ) -> None:
            self._execute_context_handler(
                prompt_obj,
                parent=dialog,
                context_override=context_text,
            )

        dialog_factory = PromptDialogFactory(
            manager=self._manager,
            name_generator=self.generate_prompt_name,
            description_generator=self.generate_prompt_description,
            category_generator=self.generate_prompt_category,
            tags_generator=self.generate_prompt_tags,
            scenario_generator=self.generate_prompt_scenarios,
            prompt_engineer=prompt_engineer,
            structure_refiner=structure_refiner,
            version_history_handler=self._open_version_history_dialog,
            execute_context_handler=_execute_context_from_dialog,
        )
        editor_flow = PromptEditorFlow(
            parent=parent,
            manager=self._manager,
            dialog_factory=dialog_factory,
            load_prompts=self._load_prompts,
            current_search_text=self._current_search_text,
            select_prompt=self._select_prompt,
            delete_prompt=self._delete_prompt,
            status_callback=self._status_callback,
            error_callback=self._error_callback,
        )
        catalog_controller = CatalogWorkflowController(
            parent,
            self._manager,
            load_prompts=self._load_prompts,
            current_search_text=self._current_search_text,
            select_prompt=self._select_prompt,
            current_prompt=self._current_prompt_supplier,
            show_status=self._status_callback,
            generate_category=self.generate_prompt_category,
            generate_tags=self.generate_prompt_tags,
        )
        return PromptGenerationComponents(
            dialog_factory=dialog_factory,
            editor_flow=editor_flow,
            catalog_controller=catalog_controller,
        )

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------
    def generate_prompt_name(self, context: str) -> str:
        """Delegate name generation to :class:`core.PromptManager`."""
        if not context.strip():
            return ""
        return self._manager.generate_prompt_name(context)

    def generate_prompt_description(self, context: str) -> str:
        """Delegate description generation to :class:`core.PromptManager`."""
        if not context.strip():
            return ""
        return self._manager.generate_prompt_description(context)

    def generate_prompt_scenarios(self, context: str) -> list[str]:
        """Produce scenario suggestions for *context*."""
        text = (context or "").strip()
        if not text:
            return []
        return list(self._manager.generate_prompt_scenarios(text))

    def generate_prompt_category(self, context: str) -> str:
        """Suggest the best category for *context*."""
        text = (context or "").strip()
        if not text:
            return ""
        try:
            return self._manager.generate_prompt_category(text)
        except PromptManagerError:
            return ""

    def generate_prompt_tags(self, context: str) -> list[str]:
        """Infer tags from the current classifier + language heuristics."""
        text = (context or "").strip()
        if not text:
            return []
        tags: list[str] = []
        classifier = self._manager.intent_classifier
        if classifier is not None:
            prediction = classifier.classify(text)
            tags.extend(prediction.tag_hints)
            tags.extend(prediction.language_hints)
            default_tag_map = {
                IntentLabel.ANALYSIS: "analysis",
                IntentLabel.DEBUG: "debugging",
                IntentLabel.REFACTOR: "refactor",
                IntentLabel.ENHANCEMENT: "enhancement",
                IntentLabel.DOCUMENTATION: "documentation",
                IntentLabel.REPORTING: "reporting",
                IntentLabel.GENERAL: "general",
            }
            default_tag = default_tag_map.get(prediction.label)
            if default_tag:
                tags.append(default_tag)
        detected = detect_language(text)
        if detected.code and detected.code not in {"", "plain"}:
            tags.append(detected.code)
        lowered = text.lower()
        keyword_tags = {
            "security": "security",
            "performance": "performance",
            "optimize": "optimization",
            "refactor": "refactor",
            "document": "documentation",
            "explain": "explanation",
            "bug": "bugfix",
        }
        for keyword, tag in keyword_tags.items():
            if keyword in lowered:
                tags.append(tag)
        unique: list[str] = []
        seen: set[str] = set()
        for tag in tags:
            normalized = tag.strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(normalized)
        return unique[:8]

    # ------------------------------------------------------------------
    # Refinement helpers
    # ------------------------------------------------------------------
    def refine_prompt_body(
        self,
        prompt_text: str,
        *,
        name: str | None = None,
        description: str | None = None,
        category: str | None = None,
        tags: Sequence[str] | None = None,
    ) -> PromptRefinement:
        """Delegate prompt refinement to :class:`core.PromptManager`."""
        return self._manager.refine_prompt_text(
            prompt_text,
            name=name,
            description=description,
            category=category,
            tags=tags,
        )

    def refine_prompt_structure(
        self,
        prompt_text: str,
        *,
        name: str | None = None,
        description: str | None = None,
        category: str | None = None,
        tags: Sequence[str] | None = None,
    ) -> PromptRefinement:
        """Delegate structure-only prompt refinement to :class:`core.PromptManager`."""
        return self._manager.refine_prompt_structure(
            prompt_text,
            name=name,
            description=description,
            category=category,
            tags=tags,
        )


__all__ = ["PromptGenerationComponents", "PromptGenerationService"]
