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


def test_entry_dialog_clarity_cues_remain_local_widget_text(qt_app) -> None:
    """Entry-dialog clarity cues should stay dialog-local instead of shared analytics fields."""
    from datetime import UTC, datetime
    from uuid import UUID

    from gui.dialogs.draft_promote import DraftPromoteDialog
    from gui.dialogs.quick_capture import QuickCaptureDialog
    from gui.dialogs.recent_prompts import RecentPromptsDialog
    from models.prompt_model import Prompt

    quick_capture = QuickCaptureDialog()
    assert quick_capture._entry_guidance_label.text() == (  # noqa: SLF001
        "Paste a raw prompt or query. PromptManager only cleans obvious outer wrappers "
        "before saving the draft."
    )
    assert not hasattr(quick_capture, "decision_summary")
    assert not hasattr(quick_capture, "next_action_summary")
    assert not hasattr(quick_capture, "freshness_summary")

    recent_prompt = Prompt(
        id=UUID("00000000-0000-0000-0000-000000000601"),
        name="Recent Prompt",
        description="Recent prompt description",
        category="General",
        context="Recent prompt body",
        created_at=datetime(2026, 4, 28, 9, 0, tzinfo=UTC),
        last_modified=datetime(2026, 4, 29, 9, 0, tzinfo=UTC),
    )
    recent = RecentPromptsDialog([recent_prompt])
    recent_summary = recent.findChildren(QLabel)[0]
    assert recent_summary.text() == (
        "Reopen one of the prompts you touched most recently to continue refining it."
    )
    assert not hasattr(recent, "decision_summary")
    assert not hasattr(recent, "next_action_summary")
    assert not hasattr(recent, "freshness_summary")

    draft_prompt = Prompt(
        id=UUID("00000000-0000-0000-0000-000000000602"),
        name="Captured draft",
        description="Quick capture draft.",
        category="General",
        context="Draft body",
        ext2={"capture_state": "draft", "capture_method": "quick_capture"},
    )
    similar_prompt = Prompt(
        id=UUID("00000000-0000-0000-0000-000000000603"),
        name="Existing reusable prompt",
        description="Already curated.",
        category="Operations",
        context="Existing body",
        last_modified=datetime(2026, 4, 4, 18, 0, tzinfo=UTC),
    )
    similar_prompt.similarity = 0.72
    promote = DraftPromoteDialog(
        draft_prompt,
        categories=["General"],
        similar_prompts=[similar_prompt],
    )
    assert promote._similarity_summary.text() == (  # noqa: SLF001
        "Similar prompts already exist. Review an existing match or continue promoting "
        "this draft as a new prompt."
    )
    assert not hasattr(promote, "decision_summary")
    assert not hasattr(promote, "next_action_summary")
    assert not hasattr(promote, "freshness_summary")
