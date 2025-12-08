"""Prompt version history dialog tests.

Updates:
  v0.1.2 - 2025-12-08 - Cast manager/prompt stubs to match dialog signatures for Pyright.
  v0.1.1 - 2025-11-29 - Wrap tab title comprehension for Ruff line length.
  v0.1.0 - 2025-11-22 - Ensure prompt body tab is default selection.
"""

from __future__ import annotations

import uuid
from typing import cast

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from core import PromptManager
from gui.dialogs import PromptVersionHistoryDialog
from models.prompt_model import Prompt


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    """Provide a shared Qt application instance for UI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return cast(QApplication, app)


class _ManagerStub:
    def list_prompt_versions(self, prompt_id: uuid.UUID, *, limit: int = 200) -> list[object]:  # noqa: ARG002
        return []


def _prompt_stub() -> Prompt:
    return Prompt(
        id=uuid.uuid4(),
        name="Demo prompt",
        description="Body",
        category="tests",
    )


def _as_prompt_manager(manager: _ManagerStub) -> PromptManager:
    return cast(PromptManager, manager)


def test_version_history_dialog_defaults_to_body_tab(qt_app: QApplication) -> None:
    """Ensure the prompt body tab is selected by default when opened."""
    dialog = PromptVersionHistoryDialog(_as_prompt_manager(_ManagerStub()), _prompt_stub())
    try:
        tab_titles = [
            dialog._tab_widget.tabText(index) for index in range(dialog._tab_widget.count())
        ]
        assert tab_titles[0] == "Prompt body"
        assert tab_titles[1] == "Diff vs previous"
        assert tab_titles[2] == "Diff vs current"
        assert dialog._tab_widget.currentIndex() == 0
        assert dialog._body_view.toPlainText() == PromptVersionHistoryDialog._BODY_PLACEHOLDER
        assert "No versions" in dialog._current_diff_view.toPlainText()
    finally:
        dialog.close()
        dialog.deleteLater()
