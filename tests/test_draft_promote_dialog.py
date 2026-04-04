"""Focused tests for the draft promote dialog.

Updates:
  v0.1.0 - 2026-04-04 - Cover advisory similar-prompt rendering and actions.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import cast

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QPushButton

from gui.dialogs.draft_promote import DraftPromoteDialog
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

    assert "Possible similar prompts already exist" in dialog._similarity_summary.text()  # noqa: SLF001
    assert not dialog._similar_prompts_list.isHidden()  # noqa: SLF001
    assert dialog._similar_prompts_list.count() == 1  # noqa: SLF001
    item = dialog._similar_prompts_list.item(0)  # noqa: SLF001
    assert item.text() == "Existing reusable prompt — Operations"
    assert "Last modified: 2026-04-04 18:00 UTC" in item.toolTip()
    assert "Similarity: 0.87" in item.toolTip()

    dialog._similar_prompts_list.setCurrentRow(0)  # noqa: SLF001
    dialog._open_selected_existing_prompt()  # noqa: SLF001
    qt_app.processEvents()

    assert dialog.selected_existing_prompt_id == similar.id
    assert dialog.result_prompt is None
    assert dialog.result() == dialog.DialogCode.Accepted


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
        button
        for button in dialog.findChildren(QPushButton)
        if button.text() == "Promote as New"
    )
    promote_button.click()
    qt_app.processEvents()

    assert dialog.selected_existing_prompt_id is None
    assert dialog.result_prompt is not None
    assert dialog.result_prompt.name == "Curated title"
    assert dialog.result() == dialog.DialogCode.Accepted
