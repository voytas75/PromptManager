"""Focused tests for prompt toolbar front-door actions.

Updates:
  v0.2.0 - 2026-05-04 - Resolve private-usage checks via public widget lookup.
  v0.1.0 - 2026-04-11 - Lock the canonical front-door labels to Quick Capture and Recent.
"""

from __future__ import annotations

from typing import cast

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QPushButton, QToolButton

from gui.widgets.prompt_toolbar import PromptToolbar


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    """Provide a shared Qt application instance for toolbar tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return cast("QApplication", app)


def test_prompt_toolbar_exposes_canonical_front_door_actions(qt_app: QApplication) -> None:
    """Toolbar should keep the canonical front-door labels stable."""
    toolbar = PromptToolbar()

    toolbar.show()
    qt_app.processEvents()

    recent_button = toolbar.findChild(QPushButton, "promptToolbarRecentButton")
    new_button = toolbar.findChild(QToolButton, "promptToolbarQuickCaptureButton")

    assert recent_button is not None
    assert new_button is not None
    assert recent_button.text() == "Recent"
    assert new_button.text() == "Quick Capture"
    assert recent_button.toolTip() == "Reopen one of the prompts you touched most recently."
    assert new_button.toolTip() == (
        "Paste raw prompt text into a draft record, or open the full prompt/workbench flows."
    )

    menu = new_button.menu()
    assert menu is not None
    menu_texts = [action.text() for action in menu.actions()]
    assert "Quick Capture…" in menu_texts
    assert "New Prompt…" in menu_texts
    assert "Workbench Session…" in menu_texts
