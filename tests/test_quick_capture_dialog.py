"""Focused tests for the quick capture dialog and draft conversion.

Updates:
  v0.1.0 - 2026-04-05 - Cover source/provenance capture and prompt conversion.
"""

from __future__ import annotations

import uuid
from typing import cast

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from gui.dialogs.quick_capture import QuickCaptureDialog, resolve_quick_capture_source


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    """Provide a shared Qt application instance for dialog tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return cast("QApplication", app)


def test_quick_capture_dialog_builds_source_provenance_draft(qt_app: QApplication) -> None:
    """Quick capture should accept a simple source/provenance value from the form."""
    dialog = QuickCaptureDialog()
    dialog._title_input.setText("Incident summary")  # noqa: SLF001
    dialog._source_input.setText("ChatGPT thread / ops notes")  # noqa: SLF001
    dialog._tags_input.setText("incident, ops")  # noqa: SLF001
    dialog._body_input.setPlainText("Summarize the incident and highlight risk.")  # noqa: SLF001

    draft = dialog._build_draft()  # noqa: SLF001

    assert draft is not None
    assert draft.source_label == "ChatGPT thread / ops notes"
    prompt = draft.to_prompt()
    assert prompt.id != uuid.UUID(int=0)
    assert prompt.source == "ChatGPT thread / ops notes"
    assert prompt.context == "Summarize the incident and highlight risk."
    assert prompt.ext2 == {
        "capture_state": "draft",
        "capture_method": "quick_capture",
    }


def test_resolve_quick_capture_source_defaults_to_existing_storage_value() -> None:
    """Empty capture provenance should fall back to the existing quick-capture source marker."""
    assert resolve_quick_capture_source("  ") == "quick_capture"
    assert resolve_quick_capture_source("notes") == "notes"
