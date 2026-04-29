"""Focused tests for bounded prompt-filter panel visibility cues.

Updates:
  v0.1.1 - 2026-04-28 - Add sort-state continuity cue coverage for active search.
  v0.1.0 - 2026-04-28 - Cover tag filter visibility cue states without widening filter behavior.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QLabel

from gui.prompt_search_controller import PromptSearchController
from gui.widgets.prompt_filter_panel import PromptFilterPanel


@pytest.fixture
def qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _build_panel() -> PromptFilterPanel:
    return PromptFilterPanel(sort_options=[("Last modified", "last_modified")])


class _PresenterStub:
    def __init__(self) -> None:
        self.refresh_calls = 0

    def refresh_filtered_view(self) -> None:
        self.refresh_calls += 1


class _LayoutControllerStub:
    def __init__(self) -> None:
        self.persist_calls = 0

    def persist_filter_preferences(self) -> None:
        self.persist_calls += 1


def _build_search_controller(
    panel: PromptFilterPanel,
    *,
    presenter: _PresenterStub | None = None,
    layout_controller: _LayoutControllerStub | None = None,
) -> tuple[PromptSearchController, _PresenterStub, _LayoutControllerStub]:
    active_presenter = presenter or _PresenterStub()
    active_layout = layout_controller or _LayoutControllerStub()
    controller = PromptSearchController(
        parent=panel,
        manager=object(),
        presenter_supplier=lambda: active_presenter,
        filter_panel_supplier=lambda: panel,
        layout_controller=active_layout,
        load_prompts=lambda *_args, **_kwargs: None,
        current_search_text=lambda: "",
        select_prompt=lambda _prompt_id: None,
    )
    return controller, active_presenter, active_layout


def test_tag_filter_panel_shows_calm_all_tags_visibility_cue(qt_app) -> None:
    """Default tag state should be visible without selecting a tag."""
    panel = _build_panel()
    panel.set_tags(["docs", "ops"])

    visible_labels = [label.text() for label in panel.findChildren(QLabel)]

    assert "Tag filter: all tags" in visible_labels


def test_tag_filter_panel_updates_visibility_cue_for_active_tag(qt_app) -> None:
    """Selecting one tag should update the visible active-tag cue immediately."""
    panel = _build_panel()
    panel.set_tags(["docs", "ops"], selected_tag="ops")

    visible_labels = [label.text() for label in panel.findChildren(QLabel)]

    assert "Tag filter: ops" in visible_labels


def test_tag_filter_panel_user_selection_updates_cue_and_emits_filters_changed(qt_app) -> None:
    """Interactive tag changes should keep the helper cue and filter signal in sync."""
    panel = _build_panel()
    panel.set_tags(["docs", "ops"])
    emissions: list[str] = []

    def _record_emission() -> None:
        label = panel.findChild(QLabel, "tagFilterVisibilityLabel")
        assert label is not None
        emissions.append(label.text())

    panel.filters_changed.connect(_record_emission)

    panel._tag_combo.setCurrentIndex(2)  # noqa: SLF001
    QApplication.processEvents()

    label = panel.findChild(QLabel, "tagFilterVisibilityLabel")
    assert label is not None
    assert label.text() == "Tag filter: ops"
    assert emissions == ["Tag filter: ops"]


def test_tag_filter_panel_reset_restores_all_tags_visibility_cue(qt_app) -> None:
    """Returning to the neutral tag state should restore the calm all-tags cue."""
    panel = _build_panel()
    panel.set_tags(["docs", "ops"], selected_tag="ops")
    emissions: list[str] = []

    def _record_emission() -> None:
        label = panel.findChild(QLabel, "tagFilterVisibilityLabel")
        assert label is not None
        emissions.append(label.text())

    panel.filters_changed.connect(_record_emission)

    panel._tag_combo.setCurrentIndex(0)  # noqa: SLF001
    QApplication.processEvents()

    label = panel.findChild(QLabel, "tagFilterVisibilityLabel")
    assert label is not None
    assert label.text() == "Tag filter: all tags"
    assert emissions == ["Tag filter: all tags"]


def test_tag_filter_panel_active_search_disables_sort_with_visible_continuity_cue(qt_app) -> None:
    """Active search should make sort availability obvious on the same filter seam."""
    panel = _build_panel()
    controller, presenter, layout_controller = _build_search_controller(panel)

    controller.search_requested("ops", use_indicator=False)
    QApplication.processEvents()

    assert panel.is_sort_enabled() is False
    visible_labels = [label.text() for label in panel.findChildren(QLabel)]
    assert "Sort locked during search" in visible_labels
    assert presenter.refresh_calls == 0
    assert layout_controller.persist_calls == 0


def test_filter_panel_shows_active_narrowing_summary_for_combined_search_and_filters(
    qt_app,
) -> None:
    """Combined narrowing state should be visible without inspecting each control separately."""
    panel = _build_panel()
    panel.set_tags(["docs", "ops"], selected_tag="ops")
    panel.set_favorites_only(True)
    controller, _presenter, _layout_controller = _build_search_controller(panel)

    controller.search_requested("incident", use_indicator=False)
    QApplication.processEvents()

    summary = panel.findChild(QLabel, "activeNarrowingSummaryLabel")
    assert summary is not None
    assert summary.text() == (
        "Showing prompts narrowed by search: incident • tag: ops • favorites only"
    )


def test_filter_panel_search_summary_uses_live_query_text_and_resets_when_search_clears(
    qt_app,
) -> None:
    """Search continuity cue should reflect the live query and disappear when search clears."""
    panel = _build_panel()
    controller, _presenter, _layout_controller = _build_search_controller(panel)

    controller.search_requested("outage triage", use_indicator=False)
    QApplication.processEvents()

    summary = panel.findChild(QLabel, "activeNarrowingSummaryLabel")
    assert summary is not None
    assert summary.text() == "Showing prompts narrowed by search: outage triage"

    controller.search_changed("")
    QApplication.processEvents()

    assert summary.text() == "Showing all prompts"


def test_filter_panel_reset_restores_neutral_summary_after_search_and_filters_clear(
    qt_app,
) -> None:
    """Combined narrowing state should leave no stale summary after a full reset."""
    panel = _build_panel()
    panel.set_tags(["docs", "ops"], selected_tag="ops")
    panel.set_favorites_only(True)
    controller, _presenter, _layout_controller = _build_search_controller(panel)

    controller.search_requested("outage triage", use_indicator=False)
    QApplication.processEvents()

    summary = panel.findChild(QLabel, "activeNarrowingSummaryLabel")
    assert summary is not None
    assert summary.text() == (
        "Showing prompts narrowed by search: outage triage • tag: ops • favorites only"
    )

    controller.search_changed("")
    panel._tag_combo.setCurrentIndex(0)  # noqa: SLF001
    panel._favorites_only_checkbox.setChecked(False)  # noqa: SLF001
    QApplication.processEvents()

    assert summary.text() == "Showing all prompts"


def test_favorites_filter_panel_shows_active_visibility_cue(qt_app) -> None:
    """Favorites-only mode should expose a visible active-state cue on the filter seam."""
    panel = _build_panel()

    panel.set_favorites_only(True)
    QApplication.processEvents()

    visible_labels = [label.text() for label in panel.findChildren(QLabel)]

    assert "Favorites filter: favorites only" in visible_labels


def test_favorites_filter_panel_reset_restores_default_visibility_cue(qt_app) -> None:
    """Leaving favorites-only mode should restore the calm default cue exactly once."""
    panel = _build_panel()
    emissions: list[str] = []

    def _record_emission() -> None:
        label = panel.findChild(QLabel, "favoritesFilterVisibilityLabel")
        assert label is not None
        emissions.append(label.text())

    panel.set_favorites_only(True)
    panel.filters_changed.connect(_record_emission)

    panel._favorites_only_checkbox.setChecked(False)  # noqa: SLF001
    QApplication.processEvents()

    label = panel.findChild(QLabel, "favoritesFilterVisibilityLabel")
    assert label is not None
    assert label.text() == "Favorites filter: all prompts"
    assert emissions == ["Favorites filter: all prompts"]
