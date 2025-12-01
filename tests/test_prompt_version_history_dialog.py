"""Prompt version history dialog tests.

Updates:
  v0.1.1 - 2025-11-29 - Wrap tab title comprehension for Ruff line length.
  v0.1.0 - 2025-11-22 - Ensure prompt body tab is default selection.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from gui.dialogs import PromptVersionHistoryDialog


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _ManagerStub:
    def list_prompt_versions(self, prompt_id: uuid.UUID, *, limit: int = 200) -> list[object]:  # noqa: ARG002
        return []


def _prompt_stub() -> SimpleNamespace:
    return SimpleNamespace(id=uuid.uuid4(), name="Demo prompt")


def test_version_history_dialog_defaults_to_body_tab(qt_app: QApplication) -> None:
    dialog = PromptVersionHistoryDialog(_ManagerStub(), _prompt_stub())
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
