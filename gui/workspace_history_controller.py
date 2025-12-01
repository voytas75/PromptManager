"""Coordinate prompt selection, lineage, and template preview updates.

Updates:
  v0.15.82 - 2025-12-01 - Extract selection + lineage handling from gui.main_window.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from core import PromptManager, PromptVersionError

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Callable
    from uuid import UUID

    from PySide6.QtWidgets import QListView

    from models.prompt_model import Prompt

    from .controllers.execution_controller import ExecutionController
    from .prompt_list_model import PromptListModel
    from .template_preview_controller import TemplatePreviewController
    from .widgets import PromptDetailWidget
else:  # pragma: no cover - runtime placeholders for type-only imports
    from typing import Any as _Any

    Callable = _Any
    QListView = _Any
    Prompt = _Any
    ExecutionController = _Any
    PromptListModel = _Any
    TemplatePreviewController = _Any
    PromptDetailWidget = _Any


class WorkspaceHistoryController:
    """Encapsulate prompt selection, lineage, and template preview tasks."""

    def __init__(
        self,
        *,
        manager: PromptManager,
        model: PromptListModel,
        detail_widget: PromptDetailWidget,
        list_view: QListView,
        current_prompt_supplier: Callable[[], Prompt | None],
        template_detail_widget_supplier: Callable[[], PromptDetailWidget | None],
        template_preview_controller_supplier: Callable[[], TemplatePreviewController | None],
        execution_controller_supplier: Callable[[], ExecutionController | None],
    ) -> None:
        """Store collaborators required to synchronize selection + lineage state."""
        self._manager = manager
        self._model = model
        self._detail_widget = detail_widget
        self._list_view = list_view
        self._current_prompt_supplier = current_prompt_supplier
        self._template_detail_widget_supplier = template_detail_widget_supplier
        self._template_preview_controller_supplier = template_preview_controller_supplier
        self._execution_controller_supplier = execution_controller_supplier

    def handle_selection_changed(self) -> None:
        """Update detail + template panels when the current prompt changes."""
        prompt = self._current_prompt_supplier()
        if prompt is None:
            self._detail_widget.clear()
            execution_controller = self._execution_controller_supplier()
            if execution_controller is not None:
                execution_controller.handle_prompt_selection_change(None)
            self._update_template_preview(None)
            template_detail = self._template_detail_widget_supplier()
            if template_detail is not None:
                template_detail.clear()
            return

        execution_controller = self._execution_controller_supplier()
        if execution_controller is not None:
            execution_controller.handle_prompt_selection_change(prompt.id)

        self._detail_widget.display_prompt(prompt)
        template_detail = self._template_detail_widget_supplier()
        if template_detail is not None:
            template_detail.display_prompt(prompt)
        self._update_prompt_lineage_summary(prompt)
        self._update_template_preview(prompt)

    def select_prompt(self, prompt_id: UUID) -> None:
        """Highlight *prompt_id* in the list view when present."""
        for row, prompt in enumerate(self._model.prompts()):
            if prompt.id == prompt_id:
                index = self._model.index(row, 0)
                self._list_view.setCurrentIndex(index)
                break

    def _update_prompt_lineage_summary(self, prompt: Prompt) -> None:
        summary_parts: list[str] = []
        try:
            parent_link = self._manager.get_prompt_parent_fork(prompt.id)
        except PromptVersionError:
            parent_link = None
        if parent_link is not None:
            summary_parts.append(f"Forked from {parent_link.source_prompt_id}")

        try:
            children = self._manager.list_prompt_forks(prompt.id)
        except PromptVersionError:
            children = []
        if children:
            child_label = "fork" if len(children) == 1 else "forks"
            summary_parts.append(f"{len(children)} {child_label}")
        summary_text = " | ".join(summary_parts) if summary_parts else "No lineage data yet."
        self._detail_widget.update_lineage_summary(summary_text)
        template_detail = self._template_detail_widget_supplier()
        if template_detail is not None:
            template_detail.update_lineage_summary(summary_text)

    def _update_template_preview(self, prompt: Prompt | None) -> None:
        controller = self._template_preview_controller_supplier()
        if controller is None:
            return
        controller.update_preview(prompt)


__all__ = ["WorkspaceHistoryController"]
