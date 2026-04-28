"""Guard tests proving prompt-library filter cues stay GUI-local by design.

Updates:
  v0.1.1 - 2026-04-28 - Cover filter-panel locality guard alongside
    prompt-list cue parity.
  v0.1.0 - 2026-04-27 - Lock retrieval cues as GUI-local by design
    unless promoted into shared analytics.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QLabel

from gui.prompt_list_coordinator import PromptLoadResult
from gui.prompt_list_model import PromptListModel
from gui.widgets.prompt_filter_panel import PromptFilterPanel


@pytest.fixture
def qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_retrieval_prompt_list_cues_remain_gui_local_by_design() -> None:
    """Prompt-list retrieval cues should not be treated as shared/headless analytics fields."""
    assert hasattr(PromptListModel, "MatchReasonRole")
    result = PromptLoadResult(
        all_prompts=[],
        search_results=None,
        preserve_search_order=False,
        search_error=None,
        operator_state_label="Browsing all prompts",
    )

    assert result.operator_state_label == "Browsing all prompts"
    assert not hasattr(result, "decision_summary")
    assert not hasattr(result, "next_action_summary")
    assert not hasattr(result, "freshness_summary")


def test_prompt_filter_panel_cues_remain_local_widget_state(qt_app) -> None:
    """Filter-panel helper cues should stay on the widget seam, not shared analytics."""
    panel = PromptFilterPanel(sort_options=[("Last modified", "last_modified")])
    panel.set_tags(["docs", "ops"], selected_tag="ops")
    panel.set_sort_enabled(False)
    panel.set_favorites_only(True)

    tag_label = panel.findChild(QLabel, "tagFilterVisibilityLabel")
    sort_label = panel.findChild(QLabel, "sortFilterVisibilityLabel")
    favorites_label = panel.findChild(QLabel, "favoritesFilterVisibilityLabel")

    assert tag_label is not None
    assert tag_label.text() == "Tag filter: ops"
    assert sort_label is not None
    assert sort_label.text() == "Sort locked during search"
    assert favorites_label is not None
    assert favorites_label.text() == "Favorites filter: favorites only"
    assert not hasattr(panel, "decision_summary")
    assert not hasattr(panel, "next_action_summary")
    assert not hasattr(panel, "freshness_summary")
