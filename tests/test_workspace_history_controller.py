"""Focused tests for workspace lineage summary updates.

Updates:
  v0.1.0 - 2026-04-10 - Cover human-readable parent lineage summaries for forks.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, cast

from gui.workspace_history_controller import WorkspaceHistoryController
from models.prompt_model import Prompt, PromptForkLink

if TYPE_CHECKING:
    from collections.abc import Callable

    from PySide6.QtWidgets import QListView

    from core import PromptManager
    from gui.controllers.execution_controller import ExecutionController
    from gui.prompt_list_model import PromptListModel
    from gui.template_preview_controller import TemplatePreviewController
    from gui.widgets import PromptDetailWidget
else:  # pragma: no cover - runtime placeholders for typing-only imports
    Callable = Any
    ExecutionController = Any
    PromptDetailWidget = Any
    PromptListModel = Any
    PromptManager = Any
    QListView = Any
    TemplatePreviewController = Any


class _ManagerStub:
    def __init__(self, *, parent_prompt: Prompt, parent_link: PromptForkLink) -> None:
        self._parent_prompt = parent_prompt
        self._parent_link = parent_link

    def get_prompt_parent_fork(self, prompt_id: uuid.UUID) -> PromptForkLink | None:  # noqa: ARG002
        return self._parent_link

    def list_prompt_forks(self, prompt_id: uuid.UUID) -> list[PromptForkLink]:  # noqa: ARG002
        return []

    def get_prompt(self, prompt_id: uuid.UUID) -> Prompt:
        assert prompt_id == self._parent_prompt.id
        return self._parent_prompt


class _PromptListModelStub:
    def prompts(self) -> list[Prompt]:
        return []

    def index(self, row: int, column: int) -> tuple[int, int]:
        return (row, column)


class _PromptDetailWidgetStub:
    def __init__(self) -> None:
        self.lineage_summary: str | None = None

    def clear(self) -> None:
        self.lineage_summary = None

    def display_prompt(self, prompt: Prompt) -> None:  # noqa: ARG002
        return

    def update_lineage_summary(self, text: str | None) -> None:
        self.lineage_summary = text


class _ListViewStub:
    def setCurrentIndex(self, index: Any) -> None:  # noqa: N802
        self.index = index


def _as_prompt_manager(manager: _ManagerStub) -> PromptManager:
    return cast("PromptManager", manager)


def _as_prompt_list_model(model: _PromptListModelStub) -> PromptListModel:
    return cast("PromptListModel", model)


def _as_prompt_detail_widget(widget: _PromptDetailWidgetStub) -> PromptDetailWidget:
    return cast("PromptDetailWidget", widget)


def _as_list_view(list_view: _ListViewStub) -> QListView:
    return cast("QListView", list_view)


def _template_detail_supplier(
    widget: _PromptDetailWidgetStub,
) -> Callable[[], PromptDetailWidget | None]:
    return cast("Callable[[], PromptDetailWidget | None]", lambda: widget)


def _template_preview_supplier() -> Callable[[], TemplatePreviewController | None]:
    return cast("Callable[[], TemplatePreviewController | None]", lambda: None)


def _execution_controller_supplier() -> Callable[[], ExecutionController | None]:
    return cast("Callable[[], ExecutionController | None]", lambda: None)


def test_workspace_history_controller_uses_parent_prompt_name_in_lineage_summary() -> None:
    """Lineage summary should prefer a readable parent prompt name over the raw UUID."""
    parent_prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000201"),
        name="Source prompt",
        description="Parent description",
        category="General",
    )
    fork_prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000202"),
        name="Fork prompt",
        description="Child description",
        category="General",
    )
    parent_link = PromptForkLink(
        id=1,
        source_prompt_id=parent_prompt.id,
        child_prompt_id=fork_prompt.id,
        created_at=fork_prompt.created_at,
    )
    detail_widget = _PromptDetailWidgetStub()
    template_detail_widget = _PromptDetailWidgetStub()
    manager = _ManagerStub(parent_prompt=parent_prompt, parent_link=parent_link)
    controller = WorkspaceHistoryController(
        manager=_as_prompt_manager(manager),
        model=_as_prompt_list_model(_PromptListModelStub()),
        detail_widget=_as_prompt_detail_widget(detail_widget),
        list_view=_as_list_view(_ListViewStub()),
        current_prompt_supplier=lambda: fork_prompt,
        template_detail_widget_supplier=_template_detail_supplier(template_detail_widget),
        template_preview_controller_supplier=_template_preview_supplier(),
        execution_controller_supplier=_execution_controller_supplier(),
    )

    controller.handle_selection_changed()

    assert detail_widget.lineage_summary == "Forked from Source prompt"
    assert template_detail_widget.lineage_summary == "Forked from Source prompt"
