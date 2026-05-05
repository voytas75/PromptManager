"""Prompt version history dialog tests.

Updates:
  v0.2.0 - 2026-05-04 - Resolve private-usage checks via public widget lookup.
  v0.1.3 - 2026-04-06 - Lock body-only history copy controls to the shared Copy Prompt wording.
  v0.1.2 - 2025-12-08 - Cast manager/prompt stubs to match dialog signatures for Pyright.
  v0.1.1 - 2025-11-29 - Wrap tab title comprehension for Ruff line length.
  v0.1.0 - 2025-11-22 - Ensure prompt body tab is default selection.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, cast

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QPlainTextEdit, QPushButton, QTabWidget

from gui.dialogs import PromptVersionHistoryDialog
from models.prompt_model import Prompt

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from core import PromptManager
else:  # pragma: no cover - runtime placeholders
    from typing import Any as _Any

    PromptManager = _Any

_BODY_PLACEHOLDER = "Select a version to view the prompt body."
_NO_VERSIONS_MESSAGE = "No versions have been recorded for this prompt yet."


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    """Provide a shared Qt application instance for UI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return cast("QApplication", app)


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
    return cast("PromptManager", manager)


def test_version_history_dialog_defaults_to_body_tab(qt_app: QApplication) -> None:
    """Ensure the prompt body tab is selected by default when opened."""
    dialog = PromptVersionHistoryDialog(_as_prompt_manager(_ManagerStub()), _prompt_stub())
    try:
        tab_widget = dialog.findChild(QTabWidget, "promptVersionHistoryTabs")
        body_view = dialog.findChild(QPlainTextEdit, "promptVersionHistoryBodyView")
        current_diff_view = dialog.findChild(
            QPlainTextEdit,
            "promptVersionHistoryCurrentDiffView",
        )

        assert tab_widget is not None
        assert body_view is not None
        assert current_diff_view is not None

        tab_titles = [tab_widget.tabText(index) for index in range(tab_widget.count())]
        assert tab_titles[0] == "Prompt body"
        assert tab_titles[1] == "Diff vs previous"
        assert tab_titles[2] == "Diff vs current"
        assert tab_widget.currentIndex() == 0
        assert body_view.toPlainText() == _BODY_PLACEHOLDER
        assert _NO_VERSIONS_MESSAGE in current_diff_view.toPlainText()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_version_history_dialog_uses_copy_prompt_for_body_only_copy(
    qt_app: QApplication,
) -> None:
    """History dialog should keep snapshot copy distinct from body-only Copy Prompt."""
    dialog = PromptVersionHistoryDialog(_as_prompt_manager(_ManagerStub()), _prompt_stub())
    try:
        button_texts = [button.text() for button in dialog.findChildren(QPushButton)]
        assert "Copy Prompt" in button_texts
        assert "Copy Snapshot" in button_texts
        assert "Copy Prompt Body" not in button_texts
    finally:
        dialog.close()
        dialog.deleteLater()
