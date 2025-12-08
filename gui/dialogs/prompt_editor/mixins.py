"""Helper mixins shared by the prompt editor dialog.

Updates:
  v0.1.1 - 2025-12-08 - Added QWidget-aware helpers and type hints for Pyright.
  v0.1.0 - 2025-12-04 - Extracted category, generation, and refinement helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast, runtime_checkable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QWidget

from core import (
    DescriptionGenerationError,
    NameGenerationError,
    PromptEngineeringUnavailable,
    ScenarioGenerationError,
)
from core.prompt_engineering import PromptEngineeringError, PromptRefinement
from models.category_model import PromptCategory, slugify_category

from ..base import (
    fallback_generate_description,
    fallback_generate_scenarios,
    fallback_suggest_prompt_name,
    logger,
)
from .refined_dialog import PromptRefinedDialog

try:
    from ...processing_indicator import ProcessingIndicator
except ImportError:  # pragma: no cover - fallback when loaded outside package
    from gui.processing_indicator import ProcessingIndicator

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from PySide6.QtWidgets import QComboBox, QLineEdit, QPlainTextEdit

    _NameGenerator = Callable[[str], str]
    _DescriptionGenerator = Callable[[str], str]
    _CategoryProvider = Callable[[], Sequence[PromptCategory]]
    _CategoryGenerator = Callable[[str], str]
    _TagsGenerator = Callable[[str], Sequence[str]]
    _ScenarioGenerator = Callable[[str], Sequence[str]]
else:  # pragma: no cover - runtime fallbacks for postponed annotations
    from collections.abc import Callable, Sequence  # type: ignore[assignment]

    from PySide6.QtWidgets import QComboBox, QLineEdit, QPlainTextEdit  # type: ignore

    _NameGenerator = Callable  # type: ignore[assignment]
    _DescriptionGenerator = Callable  # type: ignore[assignment]
    _CategoryProvider = Callable  # type: ignore[assignment]
    _CategoryGenerator = Callable  # type: ignore[assignment]
    _TagsGenerator = Callable  # type: ignore[assignment]
    _ScenarioGenerator = Callable  # type: ignore[assignment]

_TaskResult = TypeVar("_TaskResult")


@runtime_checkable
class _CategoryAware(Protocol):
    """Protocol describing mixins that expose category helpers."""

    def _current_category_value(self) -> str:  # pragma: no cover - typing helper
        """Return the canonical category value."""
        ...


class _PromptDialogAssistMixin:
    """Provide shared helper methods for prompt dialog subclasses."""

    _name_generator: _NameGenerator | None
    _description_generator: _DescriptionGenerator | None
    _context_input: QPlainTextEdit
    _name_input: QLineEdit
    _description_input: QPlainTextEdit

    def _widget_parent(self) -> QWidget:
        """Return the QWidget instance to use as parent for dialogs."""
        if isinstance(self, QWidget):
            return self
        parent = getattr(self, "parentWidget", None)
        if callable(parent):
            widget = parent()
            if isinstance(widget, QWidget):
                return widget
        msg = "Prompt dialog mixins must be used with QWidget subclasses"
        raise RuntimeError(msg)

    def _run_with_indicator(
        self,
        message: str,
        func: Callable[..., _TaskResult],
        *args: Any,
        **kwargs: Any,
    ) -> _TaskResult:
        """Execute ``func`` within a background task while showing a busy UI."""
        indicator = cast("Any", ProcessingIndicator(self._widget_parent(), message))
        result = indicator.run(func, *args, **kwargs)
        return cast("_TaskResult", result)

    def _on_generate_name_clicked(self) -> None:
        """Generate a prompt name from the context field."""
        try:
            suggestion = self._run_with_indicator(
                "Generating prompt name…",
                self._generate_name,
                self._context_input.toPlainText(),
            )
        except NameGenerationError as exc:
            QMessageBox.warning(self._widget_parent(), "Name generation failed", str(exc))
            return
        if suggestion:
            self._name_input.setText(suggestion)

    def _on_context_changed(self) -> None:
        """Auto-suggest a prompt name when none has been supplied."""
        if getattr(self, "_source_prompt", None) is not None:
            return
        if self._name_generator is None:
            return
        current_name = self._name_input.text().strip()
        if current_name:
            return
        try:
            suggestion = self._generate_name(self._context_input.toPlainText())
        except NameGenerationError:
            return
        if suggestion:
            self._name_input.setText(suggestion)

    def _generate_name(self, context: str) -> str:
        """Generate a prompt name using LiteLLM when configured."""
        context = context.strip()
        if not context:
            return ""
        if self._name_generator is None:
            logger.info("LiteLLM disabled (model not configured); using fallback name suggestion")
            return fallback_suggest_prompt_name(context)
        try:
            return self._name_generator(context)
        except NameGenerationError as exc:
            message = str(exc).strip() or "unknown reason"
            if "not configured" in message.lower():
                logger.info(
                    "LiteLLM disabled (%s); using fallback name suggestion",
                    message,
                )
            else:
                logger.warning(
                    "Name generation failed (%s); using fallback suggestion",
                    message,
                    exc_info=exc,
                )
            return fallback_suggest_prompt_name(context)

    def _generate_description(self, context: str) -> str:
        """Generate a description using LiteLLM when configured."""
        context = context.strip()
        if not context:
            return ""
        if self._description_generator is None:
            logger.info(
                "LiteLLM disabled (model not configured); using fallback description summary"
            )
            return fallback_generate_description(context)
        try:
            return self._description_generator(context)
        except DescriptionGenerationError as exc:
            message = str(exc).strip() or "unknown reason"
            if "not configured" in message.lower():
                logger.info(
                    "LiteLLM disabled (%s); using fallback description summary",
                    message,
                )
            else:
                logger.warning(
                    "Description generation failed (%s); using fallback summary",
                    message,
                    exc_info=exc,
                )
            return fallback_generate_description(context)


class _PromptDialogScenarioMixin(_PromptDialogAssistMixin):
    """Manage scenario collection and generation helpers."""

    _scenario_generator: _ScenarioGenerator | None
    _scenarios_input: QPlainTextEdit

    def _collect_scenarios(self) -> list[str]:
        """Return the unique scenarios listed in the dialog."""
        scenarios: list[str] = []
        seen: set[str] = set()
        for line in self._scenarios_input.toPlainText().splitlines():
            text = line.strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            scenarios.append(text)
        return scenarios

    def _set_scenarios(self, scenarios: Sequence[str]) -> None:
        """Populate the scenarios editor with sanitized entries."""
        sanitized = [str(item).strip() for item in scenarios if str(item).strip()]
        unique: list[str] = []
        seen: set[str] = set()
        for scenario in sanitized:
            key = scenario.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(scenario)
        self._scenarios_input.setPlainText("\n".join(unique))

    def _generate_scenarios(self, context: str) -> list[str]:
        """Generate scenarios using configured helpers with heuristic fallback."""
        context_text = context.strip()
        if not context_text:
            return []

        if self._scenario_generator is not None:
            try:
                generated = self._scenario_generator(context_text) or []
            except ScenarioGenerationError as exc:
                logger.warning("Scenario generation failed: %s", exc, exc_info=True)
                fallback = fallback_generate_scenarios(context_text)
                if fallback:
                    return fallback
                raise
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Scenario generator raised unexpected error: %s", exc, exc_info=True)
                generated = []
            else:
                scenarios = [str(item).strip() for item in generated if str(item).strip()]
                if scenarios:
                    return scenarios
        return fallback_generate_scenarios(context_text)

    def _on_generate_scenarios_clicked(self) -> None:
        """Populate the scenarios field using analysis of the prompt body."""
        context = self._context_input.toPlainText()
        if not context.strip():
            QMessageBox.information(
                self._widget_parent(),
                "Prompt required",
                "Provide a prompt body before generating scenarios.",
            )
            return
        try:
            scenarios = self._run_with_indicator(
                "Generating example scenarios…",
                self._generate_scenarios,
                context,
            )
        except ScenarioGenerationError as exc:
            QMessageBox.warning(self._widget_parent(), "Scenario generation failed", str(exc))
            return
        if not scenarios:
            QMessageBox.information(
                self._widget_parent(),
                "No scenarios available",
                "The assistant could not derive example scenarios for this prompt.",
            )
            return
        self._set_scenarios(scenarios)


class _PromptDialogCategoryMixin(_PromptDialogAssistMixin):
    """Handle category registry interactions and metadata suggestions."""

    _category_provider: _CategoryProvider | None
    _category_generator: _CategoryGenerator | None
    _tags_generator: _TagsGenerator | None
    _category_input: QComboBox
    _tags_input: QLineEdit
    _categories: list[PromptCategory]

    def _populate_category_options(self) -> None:
        """Populate the category selector from the registry provider."""
        categories = self._load_categories()
        current_text = self._category_input.currentText().strip()
        self._category_input.blockSignals(True)
        self._category_input.clear()
        for category in categories:
            self._category_input.addItem(category.label)
        self._category_input.blockSignals(False)
        if current_text:
            self._set_category_value(current_text)

    def _load_categories(self) -> list[PromptCategory]:
        """Return available categories, refreshing from the provider when possible."""
        if self._category_provider is None:
            return self._categories
        try:
            categories = list(self._category_provider())
        except Exception as exc:  # pragma: no cover - provider errors surface in GUI
            logger.warning("Unable to load categories for prompt dialog: %s", exc, exc_info=True)
            return self._categories
        categories.sort(key=lambda category: category.label.lower())
        self._categories = categories
        return self._categories

    def _set_category_value(self, value: str | None) -> None:
        """Set the category selector text while aligning with registry labels."""
        resolved = self._resolve_category_label(value)
        if not resolved:
            self._category_input.setEditText("")
            self._category_input.setCurrentIndex(-1)
            return
        index = self._category_input.findText(resolved, Qt.MatchFlag.MatchFixedString)
        if index >= 0:
            self._category_input.setCurrentIndex(index)
            return
        self._category_input.setEditText(resolved)

    def _current_category_value(self) -> str:
        """Return the canonical category text from the selector."""
        text = self._category_input.currentText().strip()
        if not text:
            return ""
        resolved = self._resolve_category_label(text)
        if resolved != text:
            self._set_category_value(resolved)
        return resolved

    def _resolve_category_label(self, value: str | None) -> str:
        """Return the stored category label when the value matches an entry."""
        text = (value or "").strip()
        if not text:
            return ""
        slug = slugify_category(text)
        lowered = text.lower()
        for category in self._categories:
            if category.label.lower() == lowered:
                return category.label
            if slug and category.slug == slug:
                return category.label
        return text

    def _on_generate_category_clicked(self) -> None:
        """Generate a category suggestion based on the prompt body."""
        if self._category_generator is None:
            return
        context = self._context_input.toPlainText()
        try:
            suggestion = (
                self._run_with_indicator(
                    "Generating category suggestion…",
                    self._category_generator,
                    context,
                )
                or ""
            ).strip()
        except Exception as exc:  # noqa: BLE001 - surface generator failures to the user
            QMessageBox.warning(self._widget_parent(), "Category suggestion failed", str(exc))
            return
        if suggestion:
            self._set_category_value(suggestion)
        else:
            QMessageBox.information(
                self._widget_parent(),
                "No suggestion available",
                "The assistant could not determine a suitable category.",
            )

    def _on_generate_tags_clicked(self) -> None:
        """Generate tag suggestions based on the prompt body."""
        if self._tags_generator is None:
            return
        context = self._context_input.toPlainText()
        try:
            suggestions = (
                self._run_with_indicator(
                    "Generating tags…",
                    self._tags_generator,
                    context,
                )
                or []
            )
        except Exception as exc:  # noqa: BLE001 - surface generator failures to the user
            QMessageBox.warning(self._widget_parent(), "Tag suggestion failed", str(exc))
            return
        tags = [tag.strip() for tag in suggestions if str(tag).strip()]
        if tags:
            self._tags_input.setText(", ".join(tags))
        else:
            QMessageBox.information(
                self._widget_parent(),
                "No suggestion available",
                "The assistant could not determine relevant tags.",
            )


class _PromptDialogRefinementMixin(_PromptDialogAssistMixin):
    """Handle general/structure-specific prompt refinement workflows."""

    _prompt_engineer: Callable[..., PromptRefinement] | None
    _structure_refiner: Callable[..., PromptRefinement] | None
    _tags_input: QLineEdit
    _category_input: QComboBox

    def _on_refine_clicked(self) -> None:
        """Invoke the general prompt refinement workflow."""
        self._run_refinement(
            self._prompt_engineer,
            unavailable_title="Prompt refinement unavailable",
            unavailable_message="Configure LiteLLM in Settings to enable prompt engineering.",
            result_title="Prompt refined",
            indicator_message="Refining prompt…",
        )

    def _on_refine_structure_clicked(self) -> None:
        """Invoke the structure-only refinement workflow."""
        self._run_refinement(
            self._structure_refiner,
            unavailable_title="Prompt refinement unavailable",
            unavailable_message="Configure LiteLLM in Settings to enable prompt engineering.",
            result_title="Prompt structure refined",
            indicator_message="Improving prompt structure…",
        )

    def _run_refinement(
        self,
        handler: Callable[..., PromptRefinement] | None,
        *,
        unavailable_title: str,
        unavailable_message: str,
        result_title: str,
        indicator_message: str,
    ) -> None:
        """Execute a refinement handler and surface the summary to the user."""
        if handler is None:
            QMessageBox.information(self._widget_parent(), unavailable_title, unavailable_message)
            return

        prompt_text = self._context_input.toPlainText()
        if not prompt_text.strip():
            QMessageBox.information(
                self._widget_parent(),
                "Prompt required",
                "Enter prompt text before running refinement.",
            )
            return

        name = self._name_input.text().strip() or None
        description = self._description_input.toPlainText().strip() or None
        category = self._current_category_value() if isinstance(self, _CategoryAware) else None
        tags: list[str] = []
        if hasattr(self, "_tags_input"):
            tags = [tag.strip() for tag in self._tags_input.text().split(",") if tag.strip()]

        try:
            result = self._run_with_indicator(
                indicator_message,
                handler,
                prompt_text,
                name=name,
                description=description,
                category=category or None,
                tags=tags,
            )
        except PromptEngineeringError as exc:
            QMessageBox.warning(self._widget_parent(), "Prompt refinement failed", str(exc))
            return
        except PromptEngineeringUnavailable as exc:
            QMessageBox.warning(self._widget_parent(), "Prompt refinement unavailable", str(exc))
            return
        except Exception as exc:  # pragma: no cover - defensive
            QMessageBox.warning(
                self._widget_parent(),
                "Prompt refinement failed",
                f"Unexpected error: {exc}",
            )
            return

        self._context_input.setPlainText(result.improved_prompt)
        summary = self._format_refinement_summary(result)
        result_dialog = PromptRefinedDialog(summary, self._widget_parent(), title=result_title)
        result_dialog.exec()

    @staticmethod
    def _format_refinement_summary(result: PromptRefinement) -> str:
        """Compose a human-readable summary of the refinement output."""
        summary_parts: list[str] = []
        if result.analysis:
            summary_parts.append(result.analysis)
        if result.checklist:
            checklist = "\n".join(f"• {item}" for item in result.checklist)
            summary_parts.append(f"Checklist:\n{checklist}")
        if result.warnings:
            warnings = "\n".join(f"• {item}" for item in result.warnings)
            summary_parts.append(f"Warnings:\n{warnings}")
        if summary_parts:
            return "\n\n".join(summary_parts)
        return "The prompt has been updated with the refined version."


__all__ = [
    "_PromptDialogAssistMixin",
    "_PromptDialogCategoryMixin",
    "_PromptDialogRefinementMixin",
    "_PromptDialogScenarioMixin",
]
