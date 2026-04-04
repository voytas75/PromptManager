"""Focused tests for prompt detail inspection cues.

Updates:
  v0.1.0 - 2026-04-04 - Cover always-visible provenance/status cues for captured drafts.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import cast

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from gui.widgets import PromptDetailWidget
from models.prompt_model import Prompt


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    """Provide a shared Qt application instance for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return cast("QApplication", app)


def test_prompt_detail_widget_shows_inspection_cues_for_captured_draft(qt_app: QApplication) -> None:
    """Captured drafts should expose draft/source/last-modified cues without metadata toggles."""
    widget = PromptDetailWidget()
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000123"),
        name="Captured draft",
        description="Quick capture draft.",
        category="General",
        context="Draft body",
        created_at=datetime(2026, 4, 4, 9, 0, tzinfo=UTC),
        last_modified=datetime(2026, 4, 4, 10, 30, tzinfo=UTC),
        source="chat thread",
        ext2={
            "capture_state": "draft",
            "capture_method": "quick_capture",
        },
    )

    widget.show()
    widget.display_prompt(prompt)
    qt_app.processEvents()

    assert widget._meta_label.isVisible()  # noqa: SLF001
    inspection_text = widget._meta_label.text()  # noqa: SLF001
    assert "Inspection:" in inspection_text
    assert "Draft (quick_capture)" in inspection_text
    assert "Source: chat thread" in inspection_text
    assert "Last modified: 2026-04-04T10:30:00+00:00" in inspection_text
    assert not widget._metadata_view.isVisible()  # noqa: SLF001
