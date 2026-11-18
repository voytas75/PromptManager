"""Focused tests for bounded prompt-list presenter retrieval cues.

Updates:
  v0.1.1 - 2026-11-18 - Exercise filter selection and detail state through Qt signals.
  v0.1.0 - 2026-04-27 - Cover similar-result path meaning cues without widening presenter workflow.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

import pytest

from gui.prompt_list_coordinator import PromptListCoordinator, PromptSortOrder
from gui.prompt_list_model import PromptListModel
from gui.prompt_list_presenter import PromptListCallbacks, PromptListPresenter
from gui.widgets.prompt_detail_widget import PromptDetailWidget
from gui.widgets.prompt_filter_panel import PromptFilterPanel
from gui.workspace_history_controller import WorkspaceHistoryController
from models.prompt_model import Prompt

try:
    from PySide6.QtCore import QModelIndex
    from PySide6.QtWidgets import QApplication, QListView, QWidget
except ImportError:  # pragma: no cover - optional dependency in test environments
    pytest.skip("PySide6 is not available", allow_module_level=True)


class _RepositoryStub:
    def __init__(self, prompts: list[Prompt]) -> None:
        self._prompts = list(prompts)

    def list(self) -> list[Prompt]:
        return list(self._prompts)


class _ManagerStub:
    def __init__(self, prompts: list[Prompt], similar_results: list[Prompt]) -> None:
        self.repository = _RepositoryStub(prompts)
        self._similar_results = list(similar_results)

    def list_categories(self) -> list[object]:
        return []

    def get_prompt_parent_fork(self, _prompt_id: uuid.UUID) -> None:
        return None

    def list_prompt_forks(self, _prompt_id: uuid.UUID) -> list[object]:
        return []

    def search_prompts(
        self,
        query_text: str,
        limit: int = 50,
        embedding: list[float] | None = None,
    ) -> list[Prompt]:
        return list(self._similar_results[:limit])


class _DetailWidgetStub:
    def __init__(self) -> None:
        self.cleared = False
        self.displayed_prompt: Prompt | None = None

    def clear(self) -> None:
        self.cleared = True

    def display_prompt(self, prompt: Prompt) -> None:
        self.displayed_prompt = prompt


class _ListViewStub:
    def __init__(self) -> None:
        self.clear_selection_calls = 0

    def clearSelection(self) -> None:
        self.clear_selection_calls += 1

    def currentIndex(self) -> QModelIndex:  # pragma: no cover - not used in these tests
        return QModelIndex()


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    """Provide a shared Qt application instance for presenter tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return cast("QApplication", app)


def _ignore_error(_title: str, _message: str) -> None:
    return None


def _ignore_toast(_message: str) -> None:
    return None


@dataclass
class _CallbackRecorder:
    statuses: list[tuple[str, int]]
    selected_ids: list[uuid.UUID]
    intent_counts: list[int]

    @classmethod
    def create(cls) -> _CallbackRecorder:
        return cls(statuses=[], selected_ids=[], intent_counts=[])

    def build(self) -> PromptListCallbacks:
        return PromptListCallbacks(
            update_intent_hint=lambda prompts: self.intent_counts.append(len(prompts)),
            select_prompt=lambda prompt_id: self.selected_ids.append(prompt_id),
            show_error=_ignore_error,
            show_status=lambda message, timeout: self.statuses.append((message, timeout)),
            show_toast=_ignore_toast,
        )


class _ParentStub(QWidget):
    def __init__(self) -> None:
        super().__init__()


class _FilterPanelStub:
    def category_slug(self) -> str | None:
        return None

    def tag_value(self) -> str | None:
        return None

    def favorites_only(self) -> bool:
        return False

    def min_quality(self) -> float:
        return 0.0

    def set_categories(
        self,
        _categories: list[object],
        _selected_slug: str | None = None,
    ) -> None:
        return

    def set_tags(
        self,
        _tags: list[str],
        _selected_tag: str | None = None,
    ) -> None:
        return


def _prompt(name: str, *, ext4: list[float] | None = None) -> Prompt:
    return Prompt(
        id=uuid.uuid4(),
        name=name,
        description=f"{name} description",
        category="General",
        context=f"{name} body",
        ext4=ext4,
        created_at=datetime(2026, 4, 27, 12, 0, tzinfo=UTC),
        last_modified=datetime(2026, 4, 27, 12, 0, tzinfo=UTC),
    )


def _build_presenter(
    *,
    qt_app: QApplication,
    source_prompt: Prompt,
    similar_results: list[Prompt],
) -> tuple[PromptListPresenter, _CallbackRecorder]:
    _ = qt_app
    manager = _ManagerStub([source_prompt, *similar_results], [source_prompt, *similar_results])
    coordinator = PromptListCoordinator(cast("Any", manager))
    model = PromptListModel()
    detail_widget = cast("Any", _DetailWidgetStub())
    list_view = cast("Any", _ListViewStub())
    filter_panel = cast("Any", _FilterPanelStub())
    callbacks = _CallbackRecorder.create()
    presenter = PromptListPresenter(
        manager=cast("Any", manager),
        coordinator=coordinator,
        model=model,
        detail_widget=detail_widget,
        list_view=list_view,
        filter_panel=filter_panel,
        toolbar=None,
        callbacks=callbacks.build(),
        parent=_ParentStub(),
    )
    return presenter, callbacks


@pytest.mark.parametrize(
    ("keep_selected", "keep_other"),
    [(True, False), (False, True), (False, False)],
    ids=["retained", "excluded", "empty"],
)
def test_filter_refresh_synchronizes_selection_and_details(
    qt_app: QApplication, keep_selected: bool, keep_other: bool
) -> None:
    selected, other = _prompt("Alpha"), _prompt("Beta")
    selected.is_favorite, other.is_favorite = keep_selected, keep_other
    manager = _ManagerStub([selected, other], [])
    model = PromptListModel()
    view = QListView()
    view.setModel(model)
    detail, template_detail = PromptDetailWidget(), PromptDetailWidget()
    panel = PromptFilterPanel(sort_options=[])
    callbacks = _CallbackRecorder.create().build()
    presenter = PromptListPresenter(
        manager=cast("Any", manager),
        coordinator=PromptListCoordinator(cast("Any", manager)),
        model=model,
        detail_widget=detail,
        list_view=view,
        filter_panel=panel,
        toolbar=None,
        callbacks=callbacks,
        parent=view,
    )
    history = WorkspaceHistoryController(
        manager=cast("Any", manager),
        model=model,
        detail_widget=detail,
        list_view=view,
        current_prompt_supplier=presenter.current_prompt,
        template_detail_widget_supplier=lambda: template_detail,
        template_preview_controller_supplier=lambda: None,
        execution_controller_supplier=lambda: None,
    )
    callbacks.select_prompt = history.select_prompt
    view.selectionModel().selectionChanged.connect(history.handle_selection_changed)
    presenter.display_prompt_collection([selected, other], preserve_order=False)
    presenter.update_sort_order(PromptSortOrder.NAME_DESC)
    assert view.currentIndex().row() == 1
    assert presenter.current_prompt() is detail.current_prompt() is selected
    assert template_detail.current_prompt() is selected

    panel.set_favorites_only(True)
    presenter.refresh_filtered_view()

    assert model.prompts() == tuple(p for p in (other, selected) if p.is_favorite)
    expected = selected if keep_selected else None
    assert presenter.current_prompt() is expected
    assert view.selectionModel().selectedRows() == ([model.index(0, 0)] if keep_selected else [])
    assert detail.current_prompt() is expected
    assert template_detail.current_prompt() is expected


def test_recent_handoff_does_not_restore_pending_filter_that_hides_target(
    qt_app: QApplication,
) -> None:
    """A cached startup filter must not obscure a hidden prompt reopened via Recent."""
    target = _prompt("Hidden target")
    tagged = _prompt("Tagged alternative")
    tagged.tags = ["ops"]
    manager = _ManagerStub([target, tagged], [])
    panel = PromptFilterPanel(sort_options=[])
    model = PromptListModel()
    callbacks = _CallbackRecorder.create()
    presenter = PromptListPresenter(
        manager=cast("Any", manager),
        coordinator=PromptListCoordinator(cast("Any", manager)),
        model=model,
        detail_widget=cast("Any", _DetailWidgetStub()),
        list_view=cast("Any", _ListViewStub()),
        filter_panel=panel,
        toolbar=None,
        callbacks=callbacks.build(),
        parent=_ParentStub(),
    )
    presenter.set_pending_filter_preferences(category_slug="ops", tag="ops", min_quality=9.0)
    panel.clear_narrowing()

    presenter.display_catalog_for_recent([target, tagged], target.id)

    assert panel.tag_value() is None
    assert panel.category_slug() is None
    assert panel.min_quality() == 0.0
    assert [prompt.id for prompt in model.prompts()] == [target.id, tagged.id]
    assert callbacks.selected_ids == [target.id]


def test_show_similar_prompts_surfaces_recommendation_state_cue(qt_app: QApplication) -> None:
    """Similar-result lists should read as recommendations, not ordinary search results."""
    source_prompt = _prompt("Alpha", ext4=[0.1, 0.2])
    similar_prompt = _prompt("Beta", ext4=[0.3, 0.4])
    presenter, callbacks = _build_presenter(
        qt_app=qt_app,
        source_prompt=source_prompt,
        similar_results=[similar_prompt],
    )

    presenter.show_similar_prompts(source_prompt)

    assert callbacks.statuses[-1] == (
        (
            "Showing similar prompts for 'Alpha'. Recommendation results only — "
            "inspect a prompt for reuse details."
        ),
        4000,
    )


def test_show_similar_prompts_adds_bounded_inspect_handoff_cue(qt_app: QApplication) -> None:
    """Recommendation-mode status should also hint that inspect is the next safe action."""
    source_prompt = _prompt("Alpha", ext4=[0.1, 0.2])
    similar_prompt = _prompt("Beta", ext4=[0.3, 0.4])
    presenter, callbacks = _build_presenter(
        qt_app=qt_app,
        source_prompt=source_prompt,
        similar_results=[similar_prompt],
    )

    presenter.show_similar_prompts(source_prompt)

    assert "inspect a prompt for reuse details" in callbacks.statuses[-1][0]


class _FavoritesOnlyPanelStub(_FilterPanelStub):
    def favorites_only(self) -> bool:
        return True


def test_search_matches_hidden_by_favorites_filter_explain_empty_view(
    qt_app: QApplication,
) -> None:
    """Do not call search empty when matches exist but a local filter hides them."""
    _ = qt_app
    matching_prompt = _prompt("Incident triage")
    matching_prompt.is_favorite = False
    manager = _ManagerStub([matching_prompt], [matching_prompt])
    model = PromptListModel()
    detail = _DetailWidgetStub()
    callbacks = _CallbackRecorder.create()
    presenter = PromptListPresenter(
        manager=cast("Any", manager),
        coordinator=PromptListCoordinator(cast("Any", manager)),
        model=model,
        detail_widget=cast("Any", detail),
        list_view=cast("Any", _ListViewStub()),
        filter_panel=cast("Any", _FavoritesOnlyPanelStub()),
        toolbar=None,
        callbacks=callbacks.build(),
        parent=_ParentStub(),
    )

    presenter.load_prompts("incident")

    assert not model.prompts()
    assert detail.cleared is True
    assert callbacks.selected_ids == []
    assert callbacks.statuses[-1] == (
        "Search found matches, but active filters hide them — adjust filters to inspect.",
        4000,
    )


def test_load_prompts_keeps_ordinary_search_status_calm_and_inspect_oriented(
    qt_app: QApplication,
) -> None:
    """Ordinary search results should hint inspect without becoming a second layer."""
    source_prompt = _prompt("Alpha", ext4=[0.1, 0.2])
    similar_prompt = _prompt("Beta", ext4=[0.3, 0.4])
    presenter, callbacks = _build_presenter(
        qt_app=qt_app,
        source_prompt=source_prompt,
        similar_results=[similar_prompt],
    )

    presenter.load_prompts("beta")

    assert callbacks.statuses[-1] == (
        "Showing search results — inspect a prompt for reuse details.",
        4000,
    )
