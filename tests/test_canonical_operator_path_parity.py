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
from PySide6.QtWidgets import QApplication

from gui.dialogs.recent_prompts import recent_prompts
from gui.widgets import PromptDetailWidget
from gui.widgets.prompt_toolbar import PromptToolbar
from models.prompt_model import Prompt

CANONICAL_OPERATOR_PATH = (
    "`Quick Capture` → `Promote Draft` → `Recent` / search → inspect → "
    "`Copy Prompt` or `Open in Workspace`"
)


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

    assert toolbar._new_button.text() == "Quick Capture"  # noqa: SLF001
    assert toolbar._recent_button.text() == "Recent"  # noqa: SLF001

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

    assert detail_widget._promote_draft_button.text() == "Promote Draft"  # noqa: SLF001
    assert detail_widget._promote_draft_button.isVisible()  # noqa: SLF001
    assert detail_widget._promote_draft_button.isEnabled()  # noqa: SLF001
    assert detail_widget._copy_prompt_body_button.text() == "Copy Prompt"  # noqa: SLF001
    assert detail_widget._copy_prompt_body_button.isEnabled()  # noqa: SLF001
    assert detail_widget._open_in_workspace_button.text() == "Open in Workspace"  # noqa: SLF001
    assert detail_widget._open_in_workspace_button.isEnabled()  # noqa: SLF001

    reusable_prompt = _build_prompt(
        prompt_id="00000000-0000-0000-0000-000000000202",
        name="Reusable prompt",
        context="Stored prompt body",
        last_modified=datetime(2026, 4, 5, 9, 0, tzinfo=UTC),
    )
    detail_widget.display_prompt(reusable_prompt)
    qt_app.processEvents()

    assert not detail_widget._promote_draft_button.isVisible()  # noqa: SLF001
    assert detail_widget._copy_prompt_body_button.isEnabled()  # noqa: SLF001
    assert detail_widget._open_in_workspace_button.isEnabled()  # noqa: SLF001

    description_only_prompt = _build_prompt(
        prompt_id="00000000-0000-0000-0000-000000000203",
        name="Description only",
        context=None,
        description="Still reusable through the description seam.",
        last_modified=datetime(2026, 4, 6, 9, 0, tzinfo=UTC),
    )
    detail_widget.display_prompt(description_only_prompt)
    qt_app.processEvents()

    assert not detail_widget._promote_draft_button.isVisible()  # noqa: SLF001
    assert not detail_widget._copy_prompt_body_button.isEnabled()  # noqa: SLF001
    assert detail_widget._open_in_workspace_button.isEnabled()  # noqa: SLF001

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
