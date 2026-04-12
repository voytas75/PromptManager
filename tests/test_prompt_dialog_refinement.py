"""Prompt dialog refinement controls tests.

Updates:
  v0.1.2 - 2025-12-08 - Cast Qt fixtures for Pyright and normalize docstring history.
  v0.1.1 - 2025-11-27 - Cover scenario metadata stripping helper.
  v0.1.0 - 2025-11-22 - Verify structure-only refinement button wiring.
"""

from __future__ import annotations

import uuid
from typing import cast

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox

from core.prompt_engineering import PromptRefinement
from gui.dialogs import PromptDialog, _strip_scenarios_metadata
from models.category_model import PromptCategory
from models.prompt_model import Prompt


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return cast("QApplication", app)


def _refinement_stub(*_: object, **__: object) -> PromptRefinement:
    return PromptRefinement(
        improved_prompt="Structured prompt",
        analysis="Structure improved",
        checklist=[],
        warnings=[],
        confidence=0.8,
    )


def test_structure_button_disabled_without_handler(qt_app: QApplication) -> None:
    dialog = PromptDialog(prompt_engineer=_refinement_stub, structure_refiner=None)
    try:
        assert not dialog._structure_refine_button.isEnabled()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_structure_button_enabled_with_handler(qt_app: QApplication) -> None:
    dialog = PromptDialog(prompt_engineer=_refinement_stub, structure_refiner=_refinement_stub)
    try:
        assert dialog._structure_refine_button.isEnabled()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_dialog_normalises_category_from_registry(qt_app: QApplication) -> None:
    categories = [PromptCategory(slug="documentation", label="Documentation", description="Docs")]

    dialog = PromptDialog(category_provider=lambda: categories)
    try:
        dialog._name_input.setText("Test Prompt")
        dialog._description_input.setPlainText("Summary")
        dialog._context_input.setPlainText("Body")
        dialog._category_input.setEditText("documentation")

        prompt = dialog._build_prompt()
        assert prompt is not None
        assert prompt.category == "Documentation"
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_dialog_shows_promote_shortcut_only_for_draft_prompts(qt_app: QApplication) -> None:
    draft_prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000611"),
        name="Captured draft",
        description="Quick capture draft.",
        category="General",
        context="Prompt body",
        ext2={"capture_state": "draft", "capture_method": "quick_capture"},
    )
    saved_prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000612"),
        name="Saved prompt",
        description="Curated prompt.",
        category="General",
        context="Prompt body",
    )

    draft_dialog = PromptDialog(prompt=draft_prompt)
    saved_dialog = PromptDialog(prompt=saved_prompt)
    try:
        assert draft_dialog._promote_draft_button is not None  # noqa: SLF001
        assert draft_dialog._promote_draft_button.text() == "Promote Draft…"  # noqa: SLF001
        assert saved_dialog._promote_draft_button is None  # noqa: SLF001
    finally:
        draft_dialog.close()
        draft_dialog.deleteLater()
        saved_dialog.close()
        saved_dialog.deleteLater()



def test_prompt_dialog_can_request_promote_with_unsaved_changes(
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000613"),
        name="Captured draft",
        description="Quick capture draft.",
        category="General",
        context="Prompt body",
        tags=["raw"],
        ext2={"capture_state": "draft", "capture_method": "quick_capture"},
    )
    dialog = PromptDialog(prompt=prompt)
    try:
        monkeypatch.setattr(
            QMessageBox,
            "question",
            lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
        )
        dialog._tags_input.setText("raw, reusable")  # noqa: SLF001

        dialog._on_promote_draft_clicked()  # noqa: SLF001
        qt_app.processEvents()

        assert dialog.promote_requested
        assert dialog.result_prompt is not None
        assert dialog.result_prompt.tags == ["raw", "reusable"]
        assert dialog.result() == dialog.DialogCode.Accepted
    finally:
        dialog.close()
        dialog.deleteLater()



def test_strip_scenarios_metadata_removes_entries() -> None:
    metadata = {"scenarios": ["Keep"], "ext": {"extra": True}}

    cleaned = _strip_scenarios_metadata(metadata)
    assert cleaned is not metadata
    assert cleaned == {"ext": {"extra": True}}
    assert metadata["scenarios"] == ["Keep"]

    assert _strip_scenarios_metadata({"scenarios": []}) is None
    assert _strip_scenarios_metadata(None) is None
