"""Focused tests for bounded prompt-filter panel visibility cues.

Updates:
  v0.1.0 - 2026-04-28 - Cover tag filter visibility cue states without widening filter behavior.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QLabel

from gui.widgets.prompt_filter_panel import PromptFilterPanel


@pytest.fixture
def qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _build_panel() -> PromptFilterPanel:
    return PromptFilterPanel(sort_options=[("Last modified", "last_modified")])


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
