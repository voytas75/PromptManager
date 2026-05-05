"""Deterministic parity guard for the canonical operator path.

Updates:
  v0.1.0 - 2026-04-12 - Lock the README/docs path string to the current toolbar,
    detail-state, and recent-order seams.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QPushButton, QToolButton

from gui.dialogs.recent_prompts import recent_prompts
from gui.widgets import PromptDetailWidget
from gui.widgets.prompt_toolbar import PromptToolbar
from models.prompt_model import Prompt

CANONICAL_OPERATOR_PATH = (
    "`Quick Capture` → `Promote Draft` → `Recent` / search → inspect → "
    "`Copy Prompt` or `Open in Workspace`"
)

TOOLBAR_QUICK_CAPTURE_BUTTON = "promptToolbarQuickCaptureButton"
TOOLBAR_RECENT_BUTTON = "promptToolbarRecentButton"
DETAIL_PROMOTE_DRAFT_BUTTON = "promoteDraftButton"
DETAIL_COPY_PROMPT_BUTTON = "copyPromptBodyButton"
DETAIL_OPEN_IN_WORKSPACE_BUTTON = "openInWorkspaceButton"


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    """Provide a shared Qt application instance for parity tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return cast("QApplication", app)


def _project_file(relative_path: str) -> Path:
    """Return an absolute path inside the repository root."""
    return Path(__file__).resolve().parents[1] / relative_path


def _required_tool_button(widget: PromptToolbar, object_name: str) -> QToolButton:
    """Return a toolbar tool button by its public object name."""
    button = widget.findChild(QToolButton, object_name)
    assert button is not None
    return button


def _required_push_button(
    widget: PromptToolbar | PromptDetailWidget, object_name: str
) -> QPushButton:
    """Return a push button by its public object name."""
    button = widget.findChild(QPushButton, object_name)
    assert button is not None
    return button


def _build_prompt(
    *,
    prompt_id: str,
    name: str,
    last_modified: datetime,
    context: str | None,
    description: str = "Reusable prompt",
    ext2: dict[str, str] | None = None,
) -> Prompt:
    """Create a small prompt record for canonical path assertions."""
    return Prompt(
        id=uuid.UUID(prompt_id),
        name=name,
        description=description,
        category="General",
        context=context,
        created_at=datetime(2026, 4, 4, 9, 0, tzinfo=UTC),
        last_modified=last_modified,
        ext2=ext2 or {},
    )


def test_canonical_operator_path_docs_and_live_ui_stay_in_parity(
    qt_app: QApplication,
) -> None:
    """README/docs wording and shared UI seams should stay aligned for the v1 operator path."""
    readme_text = _project_file("README.md").read_text(encoding="utf-8")
    canonical_doc_text = _project_file("docs/canonical-usage-path-v1.md").read_text(
        encoding="utf-8"
    )

    assert CANONICAL_OPERATOR_PATH in readme_text
    assert CANONICAL_OPERATOR_PATH in canonical_doc_text

    toolbar = PromptToolbar()
    toolbar.show()
    qt_app.processEvents()

    quick_capture_button = _required_tool_button(toolbar, TOOLBAR_QUICK_CAPTURE_BUTTON)
    recent_button = _required_push_button(toolbar, TOOLBAR_RECENT_BUTTON)

    assert quick_capture_button.text() == "Quick Capture"
    assert recent_button.text() == "Recent"

    detail_widget = PromptDetailWidget()
    detail_widget.show()

    draft_prompt = _build_prompt(
        prompt_id="00000000-0000-0000-0000-000000000201",
        name="Draft prompt",
        context="Prompt body",
        last_modified=datetime(2026, 4, 4, 12, 0, tzinfo=UTC),
        ext2={"capture_state": "draft", "capture_method": "quick_capture"},
    )
    detail_widget.display_prompt(draft_prompt)
    qt_app.processEvents()

    promote_button = _required_push_button(detail_widget, DETAIL_PROMOTE_DRAFT_BUTTON)
    copy_button = _required_push_button(detail_widget, DETAIL_COPY_PROMPT_BUTTON)
    open_button = _required_push_button(detail_widget, DETAIL_OPEN_IN_WORKSPACE_BUTTON)

    assert promote_button.text() == "Promote Draft"
    assert promote_button.isVisible()
    assert promote_button.isEnabled()
    assert copy_button.text() == "Copy Prompt"
    assert copy_button.isEnabled()
    assert open_button.text() == "Open in Workspace"
    assert open_button.isEnabled()

    reusable_prompt = _build_prompt(
        prompt_id="00000000-0000-0000-0000-000000000202",
        name="Reusable prompt",
        context="Stored prompt body",
        last_modified=datetime(2026, 4, 5, 9, 0, tzinfo=UTC),
    )
    detail_widget.display_prompt(reusable_prompt)
    qt_app.processEvents()

    assert not promote_button.isVisible()
    assert copy_button.isEnabled()
    assert open_button.isEnabled()

    description_only_prompt = _build_prompt(
        prompt_id="00000000-0000-0000-0000-000000000203",
        name="Description only",
        context=None,
        description="Still reusable through the description seam.",
        last_modified=datetime(2026, 4, 6, 9, 0, tzinfo=UTC),
    )
    detail_widget.display_prompt(description_only_prompt)
    qt_app.processEvents()

    assert not promote_button.isVisible()
    assert not copy_button.isEnabled()
    assert open_button.isEnabled()

    beta = _build_prompt(
        prompt_id="00000000-0000-0000-0000-000000000205",
        name="Beta",
        context="Prompt body",
        last_modified=datetime(2026, 4, 7, 8, 30, tzinfo=UTC),
    )
    alpha = _build_prompt(
        prompt_id="00000000-0000-0000-0000-000000000204",
        name="Alpha",
        context="Prompt body",
        last_modified=datetime(2026, 4, 7, 8, 30, tzinfo=UTC),
    )
    newest = _build_prompt(
        prompt_id="00000000-0000-0000-0000-000000000206",
        name="Newest",
        context="Prompt body",
        last_modified=datetime(2026, 4, 8, 8, 30, tzinfo=UTC),
    )

    assert [prompt.name for prompt in recent_prompts([beta, newest, alpha])] == [
        "Newest",
        "Alpha",
        "Beta",
    ]
