"""Focused tests for prompt toolbar front-door actions.

Updates:
  v0.1.0 - 2026-04-11 - Lock the canonical front-door labels to Quick Capture and Recent.
"""

from __future__ import annotations

from typing import cast

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

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

    assert toolbar._recent_button.text() == "Recent"  # noqa: SLF001
    assert toolbar._new_button.text() == "Quick Capture"  # noqa: SLF001
    assert toolbar._recent_button.toolTip() == "Reopen one of the prompts you touched most recently."  # noqa: SLF001
    assert toolbar._new_button.toolTip() == (
        "Paste raw prompt text into a draft record, or open the full prompt/workbench flows."
    )  # noqa: SLF001

    menu_texts = [action.text() for action in toolbar._new_button.menu().actions()]  # noqa: SLF001
    assert "Quick Capture…" in menu_texts
    assert "New Prompt…" in menu_texts
    assert "Workbench Session…" in menu_texts
