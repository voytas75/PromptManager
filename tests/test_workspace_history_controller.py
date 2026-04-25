"""Focused tests for workspace lineage summary updates.

Updates:
  v0.2.0 - 2026-04-10 - Cover bounded changed-from-parent lineage cues for forks.
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


class _ExecutionEntryStub:
    def __init__(
        self,
        *,
        status: str = "success",
        model: str = "gpt-4o-mini",
        duration_ms: int | None = 120,
        prompt_version: int = 1,
        conversation_messages: int = 3,
        rating: float | None = None,
    ) -> None:
        self.status = type("_Status", (), {"value": status})()
        self.duration_ms = duration_ms
        self.executed_at = None
        self.rating = rating
        self.metadata = {
            "context": {
                "execution": {"model": model},
                "run": {
                    "kind": "prompt_execution",
                    "prompt_version": prompt_version,
                    "conversation_messages": conversation_messages,
                },
            }
        }


class _ManagerStub:
    def __init__(
        self,
        *,
        parent_prompt: Prompt | None = None,
        parent_link: PromptForkLink | None = None,
        execution_entries: dict[uuid.UUID, list[_ExecutionEntryStub]] | None = None,
    ) -> None:
        self._parent_prompt = parent_prompt
        self._parent_link = parent_link
        self._execution_entries = execution_entries or {}

    def get_prompt_parent_fork(self, prompt_id: uuid.UUID) -> PromptForkLink | None:  # noqa: ARG002
        return self._parent_link

    def list_prompt_forks(self, prompt_id: uuid.UUID) -> list[PromptForkLink]:  # noqa: ARG002
        return []

    def get_prompt(self, prompt_id: uuid.UUID) -> Prompt:
        assert self._parent_prompt is not None
        assert prompt_id == self._parent_prompt.id
        return self._parent_prompt

    def list_execution_history(
        self,
        prompt_id: uuid.UUID,
        *,
        limit: int = 20,
    ) -> list[_ExecutionEntryStub]:
        return list(self._execution_entries.get(prompt_id, [])[:limit])


class _PromptListModelStub:
    def prompts(self) -> list[Prompt]:
        return []

    def index(self, row: int, column: int) -> tuple[int, int]:
        return (row, column)


class _PromptDetailWidgetStub:
    def __init__(self) -> None:
        self.lineage_summary: str | None = None
        self.decision_summary: str | None = None
        self.next_action_summary: str | None = None
        self.run_summary: str | None = None

    def clear(self) -> None:
        self.lineage_summary = None
        self.decision_summary = None
        self.next_action_summary = None
        self.run_summary = None

    def display_prompt(self, prompt: Prompt) -> None:  # noqa: ARG002
        return

    def update_lineage_summary(self, text: str | None) -> None:
        self.lineage_summary = text

    def update_decision_summary(self, text: str | None) -> None:
        self.decision_summary = text

    def update_next_action_summary(self, text: str | None) -> None:
        self.next_action_summary = text

    def update_run_summary(self, text: str | None) -> None:
        self.run_summary = text


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
        description="Parent description",
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
    assert detail_widget.decision_summary == "Fork before editing"
    assert template_detail_widget.decision_summary == "Fork before editing"
    assert detail_widget.next_action_summary == "Fork before editing"
    assert template_detail_widget.next_action_summary == "Fork before editing"


def test_workspace_history_controller_shows_bounded_parent_difference_cue() -> None:
    """Forked prompts should show only the bounded changed-field labels."""
    parent_prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000211"),
        name="Source prompt",
        description="Parent description",
        category="General",
        tags=["baseline"],
        context="Draft the original summary.",
        source="import",
    )
    fork_prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000212"),
        name="Fork prompt",
        description="Parent description",
        category="General",
        tags=["baseline", "refined"],
        context="Draft the revised summary.",
        source="import",
    )
    parent_link = PromptForkLink(
        id=2,
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

    expected = "Forked from Source prompt | Changed from parent: body, tags"
    assert detail_widget.lineage_summary == expected
    assert template_detail_widget.lineage_summary == expected
    assert detail_widget.decision_summary == "Refine before reuse"
    assert template_detail_widget.decision_summary == "Refine before reuse"
    assert detail_widget.next_action_summary == "Refine candidate"
    assert template_detail_widget.next_action_summary == "Refine candidate"


def test_workspace_history_controller_surfaces_last_run_summary_for_prompt() -> None:
    """Inspect flow should expose one compact last-run provenance summary when history exists."""
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000231"),
        name="Reusable prompt",
        description="Description",
        category="General",
        context="Prompt body",
    )
    detail_widget = _PromptDetailWidgetStub()
    template_detail_widget = _PromptDetailWidgetStub()
    manager = _ManagerStub(
        execution_entries={
            prompt.id: [
                _ExecutionEntryStub(
                    status="success",
                    model="gpt-4o-mini",
                    duration_ms=120,
                    prompt_version=prompt.version,
                    conversation_messages=3,
                )
            ]
        }
    )
    controller = WorkspaceHistoryController(
        manager=_as_prompt_manager(manager),
        model=_as_prompt_list_model(_PromptListModelStub()),
        detail_widget=_as_prompt_detail_widget(detail_widget),
        list_view=_as_list_view(_ListViewStub()),
        current_prompt_supplier=lambda: prompt,
        template_detail_widget_supplier=_template_detail_supplier(template_detail_widget),
        template_preview_controller_supplier=_template_preview_supplier(),
        execution_controller_supplier=_execution_controller_supplier(),
    )

    controller.handle_selection_changed()

    assert detail_widget.run_summary is not None
    assert template_detail_widget.run_summary is not None
    assert "Last run" in detail_widget.run_summary
    assert "Last run" in template_detail_widget.run_summary
    assert "gpt-4o-mini" in detail_widget.run_summary
    assert "gpt-4o-mini" in template_detail_widget.run_summary
    assert str(prompt.version) in detail_widget.run_summary
    assert str(prompt.version) in template_detail_widget.run_summary
    assert "3 messages" in detail_widget.run_summary
    assert "120 ms" in detail_widget.run_summary


def test_workspace_history_controller_surfaces_candidate_vs_baseline_comparison_cue() -> None:
    """Inspect flow should show whether the latest run improved over the previous baseline."""
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000233"),
        name="Reusable prompt",
        description="Description",
        category="General",
        context="Prompt body",
    )
    detail_widget = _PromptDetailWidgetStub()
    template_detail_widget = _PromptDetailWidgetStub()
    manager = _ManagerStub(
        execution_entries={
            prompt.id: [
                _ExecutionEntryStub(
                    status="success",
                    model="gpt-4o-mini",
                    duration_ms=90,
                    prompt_version=int(prompt.version) + 1,
                    conversation_messages=3,
                    rating=5.0,
                ),
                _ExecutionEntryStub(
                    status="success",
                    model="gpt-4o-mini",
                    duration_ms=140,
                    prompt_version=int(prompt.version),
                    conversation_messages=3,
                    rating=4.0,
                ),
            ]
        }
    )
    controller = WorkspaceHistoryController(
        manager=_as_prompt_manager(manager),
        model=_as_prompt_list_model(_PromptListModelStub()),
        detail_widget=_as_prompt_detail_widget(detail_widget),
        list_view=_as_list_view(_ListViewStub()),
        current_prompt_supplier=lambda: prompt,
        template_detail_widget_supplier=_template_detail_supplier(template_detail_widget),
        template_preview_controller_supplier=_template_preview_supplier(),
        execution_controller_supplier=_execution_controller_supplier(),
    )

    controller.handle_selection_changed()

    assert detail_widget.run_summary is not None
    assert "Candidate vs baseline:" in detail_widget.run_summary
    assert "improved" in detail_widget.run_summary
    assert "rating 5.0 vs 4.0" in detail_widget.run_summary
    assert "90 ms vs 140 ms" in detail_widget.run_summary


def test_workspace_history_controller_surfaces_safe_to_compare_recommendation_for_two_compatible_runs(  # noqa: E501
) -> None:
    """Inspect flow should add one bounded recommendation cue when comparison evidence is ready."""
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000235"),
        name="Reusable prompt",
        description="Description",
        category="General",
        context="Prompt body",
    )
    detail_widget = _PromptDetailWidgetStub()
    template_detail_widget = _PromptDetailWidgetStub()
    manager = _ManagerStub(
        execution_entries={
            prompt.id: [
                _ExecutionEntryStub(
                    status="success",
                    model="gpt-4o-mini",
                    duration_ms=90,
                    prompt_version=int(prompt.version) + 1,
                    conversation_messages=3,
                    rating=5.0,
                ),
                _ExecutionEntryStub(
                    status="success",
                    model="gpt-4o-mini",
                    duration_ms=140,
                    prompt_version=int(prompt.version),
                    conversation_messages=3,
                    rating=4.0,
                ),
            ]
        }
    )
    controller = WorkspaceHistoryController(
        manager=_as_prompt_manager(manager),
        model=_as_prompt_list_model(_PromptListModelStub()),
        detail_widget=_as_prompt_detail_widget(detail_widget),
        list_view=_as_list_view(_ListViewStub()),
        current_prompt_supplier=lambda: prompt,
        template_detail_widget_supplier=_template_detail_supplier(template_detail_widget),
        template_preview_controller_supplier=_template_preview_supplier(),
        execution_controller_supplier=_execution_controller_supplier(),
    )

    controller.handle_selection_changed()

    assert detail_widget.decision_summary == "Safe to compare"
    assert template_detail_widget.decision_summary == "Safe to compare"
    assert detail_widget.run_summary is not None
    assert template_detail_widget.run_summary is not None
    assert "Candidate vs baseline:" in detail_widget.run_summary
    assert "Candidate vs baseline:" in template_detail_widget.run_summary


def test_workspace_history_controller_surfaces_compare_before_promoting_next_action_for_compatible_runs(  # noqa: E501
) -> None:
    """Inspect flow should surface one bounded next action when compatible comparison evidence
    exists.
    """
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000236"),
        name="Reusable prompt",
        description="Description",
        category="General",
        context="Prompt body",
    )
    detail_widget = _PromptDetailWidgetStub()
    template_detail_widget = _PromptDetailWidgetStub()
    manager = _ManagerStub(
        execution_entries={
            prompt.id: [
                _ExecutionEntryStub(
                    status="success",
                    model="gpt-4o-mini",
                    duration_ms=90,
                    prompt_version=int(prompt.version) + 1,
                    conversation_messages=3,
                    rating=5.0,
                ),
                _ExecutionEntryStub(
                    status="success",
                    model="gpt-4o-mini",
                    duration_ms=140,
                    prompt_version=int(prompt.version),
                    conversation_messages=3,
                    rating=4.0,
                ),
            ]
        }
    )
    controller = WorkspaceHistoryController(
        manager=_as_prompt_manager(manager),
        model=_as_prompt_list_model(_PromptListModelStub()),
        detail_widget=_as_prompt_detail_widget(detail_widget),
        list_view=_as_list_view(_ListViewStub()),
        current_prompt_supplier=lambda: prompt,
        template_detail_widget_supplier=_template_detail_supplier(template_detail_widget),
        template_preview_controller_supplier=_template_preview_supplier(),
        execution_controller_supplier=_execution_controller_supplier(),
    )

    controller.handle_selection_changed()

    assert detail_widget.next_action_summary == "Compare before promoting"
    assert template_detail_widget.next_action_summary == "Compare before promoting"


def test_workspace_history_controller_skips_comparison_cue_without_two_compatible_runs() -> None:
    """Comparison cue should stay absent when only one rated run is available."""
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000234"),
        name="Reusable prompt",
        description="Description",
        category="General",
        context="Prompt body",
    )
    detail_widget = _PromptDetailWidgetStub()
    template_detail_widget = _PromptDetailWidgetStub()
    manager = _ManagerStub(
        execution_entries={
            prompt.id: [
                _ExecutionEntryStub(
                    status="success",
                    model="gpt-4o-mini",
                    duration_ms=90,
                    prompt_version=int(prompt.version),
                    conversation_messages=3,
                    rating=5.0,
                )
            ]
        }
    )
    controller = WorkspaceHistoryController(
        manager=_as_prompt_manager(manager),
        model=_as_prompt_list_model(_PromptListModelStub()),
        detail_widget=_as_prompt_detail_widget(detail_widget),
        list_view=_as_list_view(_ListViewStub()),
        current_prompt_supplier=lambda: prompt,
        template_detail_widget_supplier=_template_detail_supplier(template_detail_widget),
        template_preview_controller_supplier=_template_preview_supplier(),
        execution_controller_supplier=_execution_controller_supplier(),
    )

    controller.handle_selection_changed()

    assert detail_widget.run_summary is not None
    assert "Candidate vs baseline:" not in detail_widget.run_summary
    assert template_detail_widget.run_summary is not None
    assert "Candidate vs baseline:" not in template_detail_widget.run_summary


def test_workspace_history_controller_maps_default_next_action_to_reuse_as_is() -> None:
    """Inspect flow should keep the default next action aligned with the default decision cue."""
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000238"),
        name="Reusable prompt",
        description="Description",
        category="General",
        context="Prompt body",
    )
    detail_widget = _PromptDetailWidgetStub()
    template_detail_widget = _PromptDetailWidgetStub()
    controller = WorkspaceHistoryController(
        manager=_as_prompt_manager(_ManagerStub()),
        model=_as_prompt_list_model(_PromptListModelStub()),
        detail_widget=_as_prompt_detail_widget(detail_widget),
        list_view=_as_list_view(_ListViewStub()),
        current_prompt_supplier=lambda: prompt,
        template_detail_widget_supplier=_template_detail_supplier(template_detail_widget),
        template_preview_controller_supplier=_template_preview_supplier(),
        execution_controller_supplier=_execution_controller_supplier(),
    )

    assert controller._map_decision_to_next_action("Safe to compare") == "Compare before promoting"  # noqa: SLF001
    assert controller._map_decision_to_next_action("Refine before reuse") == "Refine candidate"  # noqa: SLF001
    assert controller._map_decision_to_next_action("Fork before editing") == "Fork before editing"  # noqa: SLF001
    assert controller._map_decision_to_next_action("Anything else") == "Reuse as-is"  # noqa: SLF001

    controller.handle_selection_changed()

    assert detail_widget.decision_summary == "Reuse as-is"
    assert template_detail_widget.decision_summary == "Reuse as-is"
    assert detail_widget.next_action_summary == "Reuse as-is"
    assert template_detail_widget.next_action_summary == "Reuse as-is"


def test_workspace_history_controller_hides_last_run_summary_when_prompt_has_no_history() -> None:
    """Inspect flow should stay quiet when the prompt has no execution history yet."""
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000232"),
        name="Reusable prompt",
        description="Description",
        category="General",
        context="Prompt body",
    )
    detail_widget = _PromptDetailWidgetStub()
    template_detail_widget = _PromptDetailWidgetStub()
    controller = WorkspaceHistoryController(
        manager=_as_prompt_manager(_ManagerStub()),
        model=_as_prompt_list_model(_PromptListModelStub()),
        detail_widget=_as_prompt_detail_widget(detail_widget),
        list_view=_as_list_view(_ListViewStub()),
        current_prompt_supplier=lambda: prompt,
        template_detail_widget_supplier=_template_detail_supplier(template_detail_widget),
        template_preview_controller_supplier=_template_preview_supplier(),
        execution_controller_supplier=_execution_controller_supplier(),
    )

    controller.handle_selection_changed()

    assert detail_widget.run_summary is None
    assert template_detail_widget.run_summary is None


def test_workspace_history_controller_clears_next_action_summary_when_selection_becomes_empty(  # noqa: E501
) -> None:
    """Clearing selection should also clear the bounded next-action cue on both detail
    surfaces.
    """
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000239"),
        name="Reusable prompt",
        description="Description",
        category="General",
        context="Prompt body",
    )
    detail_widget = _PromptDetailWidgetStub()
    template_detail_widget = _PromptDetailWidgetStub()
    current_prompt: Prompt | None = prompt
    controller = WorkspaceHistoryController(
        manager=_as_prompt_manager(_ManagerStub()),
        model=_as_prompt_list_model(_PromptListModelStub()),
        detail_widget=_as_prompt_detail_widget(detail_widget),
        list_view=_as_list_view(_ListViewStub()),
        current_prompt_supplier=lambda: current_prompt,
        template_detail_widget_supplier=_template_detail_supplier(template_detail_widget),
        template_preview_controller_supplier=_template_preview_supplier(),
        execution_controller_supplier=_execution_controller_supplier(),
    )

    controller.handle_selection_changed()
    assert detail_widget.next_action_summary == "Reuse as-is"
    assert template_detail_widget.next_action_summary == "Reuse as-is"

    current_prompt = None
    controller.handle_selection_changed()

    assert detail_widget.next_action_summary is None
    assert template_detail_widget.next_action_summary is None
