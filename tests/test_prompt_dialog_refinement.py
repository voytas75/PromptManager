"""Prompt dialog refinement controls tests.

Updates: v0.1.0 - 2025-11-22 - Verify structure-only refinement button wiring.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from core.prompt_engineering import PromptRefinement
from models.category_model import PromptCategory
from gui.dialogs import PromptDialog


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


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
