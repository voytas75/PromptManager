"""Focused tests for the draft promote dialog.

Updates:
  v0.1.4 - 2026-04-11 - Cover stronger open-existing button copy
  for very close promote-time matches.
  v0.1.3 - 2026-04-11 - Cover one bounded visible
  similarity-strength cue for very close promote-time matches.
  v0.1.2 - 2026-04-10 - Cover bounded similar-match preview cues in the advisory promote list.
  v0.1.1 - 2026-04-06 - Cover shared title-quality improvements for untouched draft titles.
  v0.1.0 - 2026-04-04 - Cover advisory similar-prompt rendering and actions.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import cast

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QPushButton

from gui.dialogs.draft_promote import DraftPromoteDialog, build_promoted_prompt
from models.prompt_model import Prompt


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    """Provide a shared Qt application instance for dialog tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return cast("QApplication", app)


def test_draft_promote_dialog_hides_similar_section_when_no_matches(qt_app: QApplication) -> None:
    """No-match promote flow should stay lightweight and advisory text should reflect it."""
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000501"),
        name="Captured draft",
        description="Quick capture draft.",
        category="General",
        context="Draft body",
        ext2={"capture_state": "draft", "capture_method": "quick_capture"},
    )

    dialog = DraftPromoteDialog(prompt, categories=["General"], similar_prompts=[])
    dialog.show()
    qt_app.processEvents()

    assert "No similar prompts found" in dialog._similarity_summary.text()  # noqa: SLF001
    assert "promote this draft as a new prompt" in dialog._similarity_summary.text()  # noqa: SLF001
    assert dialog._similar_prompts_list.isHidden()  # noqa: SLF001
    assert dialog.selected_existing_prompt_id is None


def test_draft_promote_dialog_lists_similar_prompts_and_opens_selection(
    qt_app: QApplication,
) -> None:
    """Some-match flow should show a compact list and support opening one existing prompt."""
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000511"),
        name="Captured draft",
        description="Quick capture draft.",
        category="General",
        context="Draft body",
        ext2={"capture_state": "draft", "capture_method": "quick_capture"},
    )
    similar = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000512"),
        name="Existing reusable prompt",
        description="Already curated.",
        category="Operations",
        tags=["ops", "reuse"],
        context="Existing body",
        last_modified=datetime(2026, 4, 4, 18, 0, tzinfo=UTC),
    )
    similar.similarity = 0.87

    dialog = DraftPromoteDialog(prompt, categories=["General"], similar_prompts=[similar])
    dialog.show()
    qt_app.processEvents()

    assert "A very close existing match may already exist" in dialog._similarity_summary.text()  # noqa: SLF001
    assert not dialog._similar_prompts_list.isHidden()  # noqa: SLF001
    assert dialog._similar_prompts_list.count() == 1  # noqa: SLF001
    item = dialog._similar_prompts_list.item(0)  # noqa: SLF001
    assert (
        item.text()
        == "Existing reusable prompt — Operations · Very close match · Already curated."
    )
    assert "Last modified: 2026-04-04 18:00 UTC" in item.toolTip()
    assert "Similarity: 0.87" in item.toolTip()
    open_button = next(
        button
        for button in dialog.findChildren(QPushButton)
        if button.text() == "Open Very Close Match"
    )
    assert open_button.isEnabled()

    dialog._similar_prompts_list.setCurrentRow(0)  # noqa: SLF001
    dialog._open_selected_existing_prompt()  # noqa: SLF001
    qt_app.processEvents()

    assert dialog.selected_existing_prompt_id == similar.id
    assert dialog.result_prompt is None
    assert dialog.result() == dialog.DialogCode.Accepted


def test_draft_promote_dialog_keeps_clean_label_for_weak_signal_match(
    qt_app: QApplication,
) -> None:
    """Weak-signal similar prompts should keep the compact label without extra filler text."""
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000519"),
        name="Captured draft",
        description="Quick capture draft.",
        category="General",
        context="Draft body",
        ext2={"capture_state": "draft", "capture_method": "quick_capture"},
    )
    similar = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000520"),
        name="Existing reusable prompt",
        description="",
        category="Operations",
        source="local",
        context="Existing body",
        last_modified=datetime(2026, 4, 4, 18, 0, tzinfo=UTC),
    )

    dialog = DraftPromoteDialog(prompt, categories=["General"], similar_prompts=[similar])
    dialog.show()
    qt_app.processEvents()

    item = dialog._similar_prompts_list.item(0)  # noqa: SLF001

    assert item.text() == "Existing reusable prompt — Operations"


def test_draft_promote_dialog_hides_visible_strength_cue_for_non_close_match(
    qt_app: QApplication,
) -> None:
    """Moderate similarity can stay in the tooltip without adding row-level strength wording."""
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000523"),
        name="Captured draft",
        description="Quick capture draft.",
        category="General",
        context="Draft body",
        ext2={"capture_state": "draft", "capture_method": "quick_capture"},
    )
    similar = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000524"),
        name="Existing reusable prompt",
        description="Already curated.",
        category="Operations",
        context="Existing body",
        last_modified=datetime(2026, 4, 4, 18, 0, tzinfo=UTC),
    )
    similar.similarity = 0.72

    dialog = DraftPromoteDialog(prompt, categories=["General"], similar_prompts=[similar])
    dialog.show()
    qt_app.processEvents()

    item = dialog._similar_prompts_list.item(0)  # noqa: SLF001

    assert item.text() == "Existing reusable prompt — Operations · Already curated."
    assert "Similarity: 0.72" in item.toolTip()
    assert "Similar prompts already exist" in dialog._similarity_summary.text()  # noqa: SLF001
    open_button = next(
        button
        for button in dialog.findChildren(QPushButton)
        if button.text() == "Open Existing Match"
    )
    assert open_button.isEnabled()


def test_draft_promote_dialog_shows_strength_cue_at_threshold(qt_app: QApplication) -> None:
    """Threshold-bound matches should still get the row-level very-close cue."""
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000525"),
        name="Captured draft",
        description="Quick capture draft.",
        category="General",
        context="Draft body",
        ext2={"capture_state": "draft", "capture_method": "quick_capture"},
    )
    similar = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000526"),
        name="Existing reusable prompt",
        description="Already curated.",
        category="Operations",
        context="Existing body",
        last_modified=datetime(2026, 4, 4, 18, 0, tzinfo=UTC),
    )
    similar.similarity = 0.85

    dialog = DraftPromoteDialog(prompt, categories=["General"], similar_prompts=[similar])
    dialog.show()
    qt_app.processEvents()

    item = dialog._similar_prompts_list.item(0)  # noqa: SLF001

    assert (
        item.text()
        == "Existing reusable prompt — Operations · Very close match · Already curated."
    )
    assert "A very close existing match may already exist" in dialog._similarity_summary.text()  # noqa: SLF001
    open_button = next(
        button
        for button in dialog.findChildren(QPushButton)
        if button.text() == "Open Very Close Match"
    )
    assert open_button.isEnabled()


def test_draft_promote_dialog_promote_as_new_keeps_existing_target_clear(
    qt_app: QApplication,
) -> None:
    """Continuing as new should return a promoted prompt and not select an existing one."""
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000521"),
        name="Captured draft",
        description="Quick capture draft.",
        category="General",
        context="Draft body",
        ext2={"capture_state": "draft", "capture_method": "quick_capture"},
    )
    similar = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000522"),
        name="Existing reusable prompt",
        description="Already curated.",
        category="Operations",
    )
    similar.similarity = 0.72

    dialog = DraftPromoteDialog(
        prompt,
        categories=["General", "Operations"],
        similar_prompts=[similar],
    )
    dialog.show()
    qt_app.processEvents()

    dialog._title_input.setText("Curated title")  # noqa: SLF001
    promote_button = next(
        button for button in dialog.findChildren(QPushButton) if button.text() == "Promote as New"
    )
    promote_button.click()
    qt_app.processEvents()

    assert dialog.selected_existing_prompt_id is None
    assert dialog.result_prompt is not None
    assert dialog.result_prompt.name == "Curated title"
    assert dialog.result() == dialog.DialogCode.Accepted


def test_draft_promote_dialog_prefills_improved_title_for_placeholder_draft(
    qt_app: QApplication,
) -> None:
    """Promote flow should replace placeholder/raw draft titles before the user edits them."""
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000531"),
        name="Quick Capture Draft",
        description="Quick capture draft.",
        category="General",
        context="Title:\n# Weekly deployment checklist\nList risky steps first.",
        ext2={"capture_state": "draft", "capture_method": "quick_capture"},
    )

    dialog = DraftPromoteDialog(prompt, categories=["General"])

    assert dialog._title_input.text() == "Weekly deployment checklist"  # noqa: SLF001


def test_build_promoted_prompt_improves_untouched_placeholder_title_only() -> None:
    """Promotion should improve untouched low-quality draft titles but keep manual titles intact."""
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000532"),
        name="Quick Capture Draft",
        description="Quick capture draft.",
        category="General",
        context="Prompt:\n## Weekly deployment checklist\nList risky steps first.",
        ext2={"capture_state": "draft", "capture_method": "quick_capture"},
    )

    improved = build_promoted_prompt(
        prompt,
        title="Quick Capture Draft",
        category="General",
        tags_text="",
        source="",
        description="",
        allow_title_improvement=True,
    )
    manual = build_promoted_prompt(
        prompt,
        title="# keep raw marker",
        category="General",
        tags_text="",
        source="",
        description="",
        allow_title_improvement=False,
    )

    assert improved.name == "Weekly deployment checklist"
    assert manual.name == "# keep raw marker"
