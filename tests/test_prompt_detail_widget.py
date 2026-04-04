"""Focused tests for prompt detail inspection cues.

Updates:
  v0.1.2 - 2026-04-04 - Cover bounded quick-reuse actions in the shared detail widget.
  v0.1.1 - 2026-04-04 - Expect a human-readable UTC timestamp in inspection cues.
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


def test_prompt_detail_widget_shows_inspection_cues_for_captured_draft(
    qt_app: QApplication,
) -> None:
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
    assert "Last modified: 2026-04-04 10:30 UTC" in inspection_text
    assert not widget._metadata_view.isVisible()  # noqa: SLF001


def test_prompt_detail_widget_exposes_bounded_quick_reuse_actions(
    qt_app: QApplication,
) -> None:
    """Detail view should expose only the bounded reuse actions for direct handoff."""
    widget = PromptDetailWidget()
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000124"),
        name="Reusable prompt",
        description="Fallback description",
        category="General",
        context="Prompt body to reuse",
        created_at=datetime(2026, 4, 4, 9, 0, tzinfo=UTC),
        last_modified=datetime(2026, 4, 4, 10, 30, tzinfo=UTC),
    )
    copy_requests: list[str] = []
    open_requests: list[str] = []
    widget.copy_prompt_body_requested.connect(lambda: copy_requests.append("copy"))
    widget.open_in_workspace_requested.connect(lambda: open_requests.append("open"))

    widget.show()
    widget.display_prompt(prompt)
    qt_app.processEvents()

    assert widget._copy_prompt_body_button.text() == "Copy Prompt Body"  # noqa: SLF001
    assert widget._open_in_workspace_button.text() == "Open in Workspace"  # noqa: SLF001
    assert widget._copy_prompt_body_button.isEnabled()  # noqa: SLF001
    assert widget._open_in_workspace_button.isEnabled()  # noqa: SLF001

    widget._copy_prompt_body_button.click()  # noqa: SLF001
    widget._open_in_workspace_button.click()  # noqa: SLF001
    qt_app.processEvents()

    assert copy_requests == ["copy"]
    assert open_requests == ["open"]
