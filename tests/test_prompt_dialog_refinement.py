"""Prompt dialog refinement controls tests.

Updates:
  v0.1.3 - 2025-12-08 - Switch to public dialog seams and widget lookup helpers.
  v0.1.2 - 2025-12-08 - Cast Qt fixtures for Pyright and normalize docstring history.
  v0.1.1 - 2025-11-27 - Cover scenario metadata stripping helper.
  v0.1.0 - 2025-11-22 - Verify structure-only refinement button wiring.
"""

from __future__ import annotations

import uuid
from typing import cast

import pytest

pytest.importorskip("PySide6")
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
)

from core.prompt_engineering import PromptRefinement
from gui.dialogs import PromptDialog, strip_scenarios_metadata
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


def _message_box_yes(
    *_: object,
    **__: object,
) -> QMessageBox.StandardButton:
    return QMessageBox.StandardButton.Yes


def _message_box_no(*_: object, **__: object) -> QMessageBox.StandardButton:
    return QMessageBox.StandardButton.No


def _required_button(dialog: PromptDialog, name: str) -> QPushButton:
    button = dialog.findChild(QPushButton, name)
    assert button is not None
    return button


def _required_line_edit(dialog: PromptDialog, name: str) -> QLineEdit:
    widget = dialog.findChild(QLineEdit, name)
    assert widget is not None
    return widget


def _required_plain_text_edit(dialog: PromptDialog, name: str) -> QPlainTextEdit:
    widget = dialog.findChild(QPlainTextEdit, name)
    assert widget is not None
    return widget


def _required_combo_box(dialog: PromptDialog, name: str) -> QComboBox:
    widget = dialog.findChild(QComboBox, name)
    assert widget is not None
    return widget


def _promote_button(dialog: PromptDialog) -> QPushButton | None:
    return dialog.findChild(QPushButton, "promptDialogPromoteDraftButton")


def _build_prompt(dialog: PromptDialog) -> Prompt | None:
    return dialog.build_prompt()


def _request_draft_promotion(dialog: PromptDialog) -> None:
    dialog.request_draft_promotion()


def test_structure_button_disabled_without_handler(qt_app: QApplication) -> None:
    dialog = PromptDialog(prompt_engineer=_refinement_stub, structure_refiner=None)
    try:
        assert not _required_button(dialog, "promptDialogStructureRefineButton").isEnabled()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_structure_button_enabled_with_handler(qt_app: QApplication) -> None:
    dialog = PromptDialog(prompt_engineer=_refinement_stub, structure_refiner=_refinement_stub)
    try:
        assert _required_button(dialog, "promptDialogStructureRefineButton").isEnabled()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_dialog_normalises_category_from_registry(qt_app: QApplication) -> None:
    categories = [PromptCategory(slug="documentation", label="Documentation", description="Docs")]

    dialog = PromptDialog(category_provider=lambda: categories)
    try:
        _required_line_edit(dialog, "promptDialogNameInput").setText("Test Prompt")
        _required_plain_text_edit(dialog, "promptDialogDescriptionInput").setPlainText("Summary")
        _required_plain_text_edit(dialog, "promptDialogContextInput").setPlainText("Body")
        _required_combo_box(dialog, "promptDialogCategoryInput").setEditText("documentation")

        prompt = _build_prompt(dialog)
        assert prompt is not None
        assert prompt.category == "Documentation"
    finally:
        dialog.close()
        dialog.deleteLater()


def test_description_generation_is_explicit_and_uses_injected_generator(
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The operator can request a description before saving, not on text change."""
    calls: list[str] = []

    def generate(body: str) -> str:
        calls.append(body)
        return "Suitable for a deployment rollback review."

    dialog = PromptDialog(description_generator=generate)
    try:
        monkeypatch.setattr(QMessageBox, "question", _message_box_yes)
        button = _required_button(dialog, "promptDialogGenerateDescriptionButton")
        _required_line_edit(dialog, "promptDialogNameInput").setText("Rollback review")
        _required_plain_text_edit(dialog, "promptDialogContextInput").setPlainText(
            "Review deployment rollback steps and owners."
        )
        description = _required_plain_text_edit(dialog, "promptDialogDescriptionInput")
        assert description.toPlainText() == ""
        assert calls == []
        button.click()
        assert calls == ["Review deployment rollback steps and owners."]
        assert description.toPlainText() == "Suitable for a deployment rollback review."
    finally:
        dialog.close()
        dialog.deleteLater()


def test_description_model_consent_declined_preserves_manual_text(
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Declining a model request neither calls it nor overwrites existing text."""
    calls: list[str] = []

    def generate(body: str) -> str:
        calls.append(body)
        return "Should not appear"

    dialog = PromptDialog(description_generator=generate)
    try:
        monkeypatch.setattr(QMessageBox, "question", _message_box_no)
        _required_plain_text_edit(dialog, "promptDialogContextInput").setPlainText("Body")
        description = _required_plain_text_edit(dialog, "promptDialogDescriptionInput")
        description.setPlainText("Manual description")
        _required_button(dialog, "promptDialogGenerateDescriptionButton").click()
        assert calls == []
        assert description.toPlainText() == "Manual description"
    finally:
        dialog.close()
        dialog.deleteLater()


def test_name_model_consent_declined_keeps_local_suggestion(
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The existing name Generate control needs consent before a model request."""
    calls: list[str] = []

    def generate(body: str) -> str:
        calls.append(body)
        return "Model name"

    dialog = PromptDialog(name_generator=generate)
    try:
        monkeypatch.setattr(QMessageBox, "question", _message_box_no)
        _required_plain_text_edit(dialog, "promptDialogContextInput").setPlainText(
            "Review deployment rollback steps."
        )
        name = _required_line_edit(dialog, "promptDialogNameInput")
        local_name = name.text()
        assert local_name == ""
        assert calls == []
        button = next(
            item
            for item in dialog.findChildren(QPushButton)
            if item.text() == "Generate" and "name" in item.toolTip().lower()
        )
        button.click()
        assert calls == []
        assert name.text() == local_name
    finally:
        dialog.close()
        dialog.deleteLater()


def test_description_generate_button_uses_offline_excerpt_without_model(
    qt_app: QApplication,
) -> None:
    """The same explicit button works provider-free when no helper is supplied."""
    dialog = PromptDialog()
    try:
        _required_plain_text_edit(dialog, "promptDialogContextInput").setPlainText(
            "Review a deploy checklist before release."
        )
        _required_button(dialog, "promptDialogGenerateDescriptionButton").click()
        assert _required_plain_text_edit(dialog, "promptDialogDescriptionInput").toPlainText() == (
            "Review a deploy checklist before release."
        )
    finally:
        dialog.close()
        dialog.deleteLater()


def test_blank_description_on_save_uses_offline_excerpt_without_provider(
    qt_app: QApplication,
) -> None:
    """Saving blank description must not silently invoke a configured provider."""

    def unexpected_provider(_body: str) -> str:
        raise AssertionError("provider called during save")

    dialog = PromptDialog(description_generator=unexpected_provider)
    try:
        _required_line_edit(dialog, "promptDialogNameInput").setText("Rollback review")
        _required_plain_text_edit(dialog, "promptDialogContextInput").setPlainText(
            "Review deployment rollback steps and owners."
        )
        result = dialog.build_prompt()
        assert result is not None
        assert result.description == "Review deployment rollback steps and owners."
        assert _required_plain_text_edit(dialog, "promptDialogDescriptionInput").toPlainText() == (
            result.description
        )
    finally:
        dialog.close()
        dialog.deleteLater()


def test_manual_description_is_not_replaced_when_saving(
    qt_app: QApplication,
) -> None:
    """The local save fallback applies only when the description is empty."""

    def unexpected_provider(_body: str) -> str:
        raise AssertionError("provider called during save")

    dialog = PromptDialog(description_generator=unexpected_provider)
    try:
        _required_line_edit(dialog, "promptDialogNameInput").setText("Rollback review")
        _required_plain_text_edit(dialog, "promptDialogContextInput").setPlainText(
            "Review deployment rollback steps and owners."
        )
        _required_plain_text_edit(dialog, "promptDialogDescriptionInput").setPlainText(
            "Use this to assign rollback owners."
        )
        prompt = dialog.build_prompt()
        assert prompt is not None
        assert prompt.description == "Use this to assign rollback owners."
    finally:
        dialog.close()
        dialog.deleteLater()


def test_typing_body_and_saving_blank_metadata_never_calls_generators(
    qt_app: QApplication,
) -> None:
    """Provider-backed name and description helpers require explicit clicks."""

    def unexpected_provider(_body: str) -> str:
        raise AssertionError("provider called without clicking Generate")

    dialog = PromptDialog(
        name_generator=unexpected_provider, description_generator=unexpected_provider
    )
    try:
        _required_plain_text_edit(dialog, "promptDialogContextInput").setPlainText(
            "Summarise a deployment rollback."
        )
        prompt = dialog.build_prompt()
        assert prompt is not None
        assert prompt.name
        assert prompt.description == "Summarise a deployment rollback."
    finally:
        dialog.close()
        dialog.deleteLater()


def test_name_suggestion_waits_for_complete_body_when_typed_incrementally(
    qt_app: QApplication,
) -> None:
    """A first keystroke must never become the saved title."""
    dialog = PromptDialog()
    try:
        body = _required_plain_text_edit(dialog, "promptDialogContextInput")
        name = _required_line_edit(dialog, "promptDialogNameInput")
        for character in "Review deployment rollback steps and owners":
            body.moveCursor(QTextCursor.MoveOperation.End)
            body.insertPlainText(character)
        assert name.text() == ""
        prompt = dialog.build_prompt()
        assert prompt is not None
        assert prompt.name == "Review Deployment Rollback Steps And…"
    finally:
        dialog.close()
        dialog.deleteLater()


def test_edit_prompt_cleared_description_uses_local_body_excerpt(
    qt_app: QApplication,
) -> None:
    """Clearing a stored description and saving supplies required local metadata."""
    source = Prompt(
        id=uuid.uuid4(),
        name="Saved prompt",
        description="Old description",
        category="General",
        context="Updated prompt body",
    )

    def unexpected_provider(_body: str) -> str:
        raise AssertionError("provider called while editing")

    dialog = PromptDialog(prompt=source, description_generator=unexpected_provider)
    try:
        _required_plain_text_edit(dialog, "promptDialogContextInput").setPlainText(
            "Review this incident before closing it."
        )
        description = _required_plain_text_edit(dialog, "promptDialogDescriptionInput")
        description.clear()
        updated = dialog.build_prompt()
        assert updated is not None
        assert updated.id == source.id
        assert updated.description == "Review this incident before closing it."
        assert description.toPlainText() == updated.description
        assert source.description == "Old description"
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
        draft_button = _promote_button(draft_dialog)
        assert draft_button is not None
        assert draft_button.text() == "Promote Draft…"
        assert _promote_button(saved_dialog) is None
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
        monkeypatch.setattr(QMessageBox, "question", _message_box_yes)
        _required_line_edit(dialog, "promptDialogTagsInput").setText("raw, reusable")

        _request_draft_promotion(dialog)
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

    cleaned = strip_scenarios_metadata(metadata)
    assert cleaned is not metadata
    assert cleaned == {"ext": {"extra": True}}
    assert metadata["scenarios"] == ["Keep"]

    assert strip_scenarios_metadata({"scenarios": []}) is None
    assert strip_scenarios_metadata(None) is None
