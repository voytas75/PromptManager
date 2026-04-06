"""Focused tests for prompt detail inspection cues.

Updates:
  v0.1.5 - 2026-04-06 - Expect Copy Prompt in the detail flow only when a body exists.
  v0.1.4 - 2026-04-05 - Cover bounded derived usage cues in the shared detail widget.
  v0.1.3 - 2026-04-05 - Keep source visible in the inspect path after draft metadata is gone.
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

    assert widget._copy_prompt_body_button.text() == "Copy Prompt"  # noqa: SLF001
    assert widget._open_in_workspace_button.text() == "Open in Workspace"  # noqa: SLF001
    assert widget._copy_prompt_body_button.isEnabled()  # noqa: SLF001
    assert widget._open_in_workspace_button.isEnabled()  # noqa: SLF001
    assert not widget._usage_cue_label.isVisible()  # noqa: SLF001

    widget._copy_prompt_body_button.click()  # noqa: SLF001
    widget._open_in_workspace_button.click()  # noqa: SLF001
    qt_app.processEvents()

    assert copy_requests == ["copy"]
    assert open_requests == ["open"]


def test_prompt_detail_widget_disables_copy_without_a_prompt_body(
    qt_app: QApplication,
) -> None:
    """Detail view should not offer Copy Prompt when only descriptive metadata exists."""
    widget = PromptDetailWidget()
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000129"),
        name="Description-only prompt",
        description="Helpful notes but no reusable body yet.",
        category="General",
        context=None,
        created_at=datetime(2026, 4, 4, 9, 0, tzinfo=UTC),
        last_modified=datetime(2026, 4, 4, 10, 35, tzinfo=UTC),
    )

    widget.show()
    widget.display_prompt(prompt)
    qt_app.processEvents()

    assert not widget._copy_prompt_body_button.isEnabled()  # noqa: SLF001
    assert widget._open_in_workspace_button.isEnabled()  # noqa: SLF001


def test_prompt_detail_widget_keeps_source_visible_for_promoted_prompt(
    qt_app: QApplication,
) -> None:
    """Inspect path should keep showing source even after the draft marker is cleared."""
    widget = PromptDetailWidget()
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000125"),
        name="Promoted prompt",
        description="Normalized for reuse",
        category="Operations",
        context="Reusable prompt body",
        created_at=datetime(2026, 4, 4, 9, 0, tzinfo=UTC),
        last_modified=datetime(2026, 4, 5, 8, 45, tzinfo=UTC),
        source="ops notebook",
        ext2={"capture_method": "quick_capture"},
    )

    widget.show()
    widget.display_prompt(prompt)
    qt_app.processEvents()

    inspection_text = widget._meta_label.text()  # noqa: SLF001
    assert "Inspection:" in inspection_text
    assert "Source: ops notebook" in inspection_text
    assert "Draft" not in inspection_text
    assert "Last modified: 2026-04-05 08:45 UTC" in inspection_text


def test_prompt_detail_widget_shows_usage_cue_when_saved_signal_exists(
    qt_app: QApplication,
) -> None:
    """Detail view should surface one compact usage cue from existing scenario text."""
    widget = PromptDetailWidget()
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000126"),
        name="Incident summary",
        description="Fallback description",
        category="Operations",
        context="Summarize the incident and call out operator-facing risks.",
        scenarios=["Use for quick summaries of incident notes before handoff."],
        created_at=datetime(2026, 4, 4, 9, 0, tzinfo=UTC),
        last_modified=datetime(2026, 4, 5, 9, 15, tzinfo=UTC),
    )

    widget.show()
    widget.display_prompt(prompt)
    qt_app.processEvents()

    assert widget._usage_cue_label.isVisible()  # noqa: SLF001
    usage_text = widget._usage_cue_label.text()  # noqa: SLF001
    assert "When to use:" in usage_text
    assert "Use for quick summaries of incident notes before handoff." in usage_text


def test_prompt_detail_widget_hides_usage_cue_when_no_credible_signal_exists(
    qt_app: QApplication,
) -> None:
    """Detail view should stay quiet when no short usage signal is already stored."""
    widget = PromptDetailWidget()
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000127"),
        name="Bare prompt",
        description="Fallback description",
        category="General",
        context="Prompt body without saved scenario or example text.",
        created_at=datetime(2026, 4, 4, 9, 0, tzinfo=UTC),
        last_modified=datetime(2026, 4, 5, 9, 20, tzinfo=UTC),
    )

    widget.show()
    widget.display_prompt(prompt)
    qt_app.processEvents()

    assert not widget._usage_cue_label.isVisible()  # noqa: SLF001
    assert widget._usage_cue_label.text() == ""  # noqa: SLF001


def test_prompt_detail_widget_keeps_usage_cue_bounded_in_existing_detail_flow(
    qt_app: QApplication,
) -> None:
    """Usage cue should stay in the shared detail flow without altering inspection behaviour."""
    widget = PromptDetailWidget()
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000128"),
        name="Review prompt",
        description="Review support notes before publishing the final response.",
        category="Support",
        context="Read the notes, identify the customer-visible issue, and draft the final reply.",
        source="support queue",
        created_at=datetime(2026, 4, 4, 9, 0, tzinfo=UTC),
        last_modified=datetime(2026, 4, 5, 9, 30, tzinfo=UTC),
    )

    widget.show()
    widget.display_prompt(prompt)
    qt_app.processEvents()

    assert widget._meta_label.isVisible()  # noqa: SLF001
    assert widget._usage_cue_label.isVisible()  # noqa: SLF001
    assert "When to use:" not in widget._meta_label.text()  # noqa: SLF001
    assert "Inspection:" in widget._meta_label.text()  # noqa: SLF001
    assert "Source: support queue" in widget._meta_label.text()  # noqa: SLF001
    assert "Description:" in widget._description.text()  # noqa: SLF001
    assert "Prompt Body (preview):" in widget._context.text()  # noqa: SLF001
    assert "Scenarios:" in widget._scenarios.text()  # noqa: SLF001
    assert not widget._metadata_view.isVisible()  # noqa: SLF001
