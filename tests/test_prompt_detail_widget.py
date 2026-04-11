"""Focused tests for prompt detail inspection cues.

Updates:
  v0.1.11 - 2026-04-11 - Cover template-aware workspace handoff tooltips in
    the shared detail widget.
  v0.1.10 - 2026-04-11 - Cover bounded template-variable cue rendering in the shared detail widget.
  v0.1.9 - 2026-04-11 - Cover bounded readability typography defaults in the shared detail widget.
  v0.1.8 - 2026-04-10 - Cover bounded credible-source filtering in shared inspection cues.
  v0.1.7 - 2026-04-10 - Cover bounded quick-reuse payload tooltips in the shared detail widget.
  v0.1.6 - 2026-04-10 - Cover bounded context-lead fallback for the shared usage cue.
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


def test_prompt_detail_widget_applies_readable_default_font_sizes(
    qt_app: QApplication,
) -> None:
    """Shared detail typography should make title/body text easier to read by default."""
    widget = PromptDetailWidget()

    widget.show()
    qt_app.processEvents()

    assert widget._name_label.font().pointSizeF() > widget.font().pointSizeF()  # noqa: SLF001
    assert widget._description.font().pointSizeF() > widget.font().pointSizeF()  # noqa: SLF001
    assert widget._template_variable_cue_label.font().pointSizeF() > widget.font().pointSizeF()  # noqa: SLF001
    assert widget._metadata_view.font().pointSizeF() > widget.font().pointSizeF()  # noqa: SLF001


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
    assert widget._copy_prompt_body_button.toolTip() == "Copy the stored prompt body."  # noqa: SLF001
    assert (
        widget._open_in_workspace_button.toolTip()
        == "Open the stored prompt body in the workspace without running it."
    )  # noqa: SLF001
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
    assert (
        widget._copy_prompt_body_button.toolTip()
        == "Copy Prompt is unavailable because this prompt has no stored prompt body."
    )  # noqa: SLF001
    assert (
        widget._open_in_workspace_button.toolTip()
        == "Open the saved description in the workspace without running it."
    )  # noqa: SLF001


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


def test_prompt_detail_widget_hides_low_signal_source_marker_in_inspection_cues(
    qt_app: QApplication,
) -> None:
    """Detail view should not surface low-signal technical source markers as provenance."""
    widget = PromptDetailWidget()
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000131"),
        name="Local draft",
        description="Fallback description",
        category="General",
        context="Prompt body",
        source="quick_capture",
        created_at=datetime(2026, 4, 4, 9, 0, tzinfo=UTC),
        last_modified=datetime(2026, 4, 5, 9, 12, tzinfo=UTC),
    )

    widget.show()
    widget.display_prompt(prompt)
    qt_app.processEvents()

    inspection_text = widget._meta_label.text()  # noqa: SLF001
    assert "Inspection:" in inspection_text
    assert "Source:" not in inspection_text
    assert "Last modified: 2026-04-05 09:12 UTC" in inspection_text


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


def test_prompt_detail_widget_uses_context_lead_for_usage_cue_when_saved_signals_are_absent(
    qt_app: QApplication,
) -> None:
    """Detail view should fall back to a compact prompt-body lead-in when it is credible."""
    widget = PromptDetailWidget()
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000130"),
        name="Release handoff",
        description="",
        category="Operations",
        context=(
            "Use when summarizing deployment risks for the release handoff.\n"
            "List blockers, rollback concerns, and owners."
        ),
        created_at=datetime(2026, 4, 4, 9, 0, tzinfo=UTC),
        last_modified=datetime(2026, 4, 5, 9, 18, tzinfo=UTC),
    )

    widget.show()
    widget.display_prompt(prompt)
    qt_app.processEvents()

    assert widget._usage_cue_label.isVisible()  # noqa: SLF001
    usage_text = widget._usage_cue_label.text()  # noqa: SLF001
    assert "When to use:" in usage_text
    assert "Use when summarizing deployment risks for the release handoff." in usage_text


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


def test_prompt_detail_widget_shows_template_variable_cue_when_prompt_requires_variables(
    qt_app: QApplication,
) -> None:
    """Detail view should expose one compact variable-requirement cue for template prompts."""
    widget = PromptDetailWidget()
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000132"),
        name="Templated prompt",
        description="Fallback description",
        category="Operations",
        context="Summarize {{ customer_name }} risk posture for {{ region }}.",
        created_at=datetime(2026, 4, 4, 9, 0, tzinfo=UTC),
        last_modified=datetime(2026, 4, 5, 9, 32, tzinfo=UTC),
    )

    widget.show()
    widget.display_prompt(prompt)
    qt_app.processEvents()

    assert widget._template_variable_cue_label.isVisible()  # noqa: SLF001
    cue_text = widget._template_variable_cue_label.text()  # noqa: SLF001
    assert "Template variables:" in cue_text
    assert "Requires variables: customer_name, region" in cue_text
    assert "When to use:" not in cue_text


def test_prompt_detail_widget_makes_workspace_tooltip_template_aware(
    qt_app: QApplication,
) -> None:
    """Template prompts should explain that Workspace is the next handoff path."""
    widget = PromptDetailWidget()
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000135"),
        name="Templated workspace handoff",
        description="Fallback description",
        category="Operations",
        context="Summarize {{ customer_name }} risk posture for {{ region }}.",
        created_at=datetime(2026, 4, 4, 9, 0, tzinfo=UTC),
        last_modified=datetime(2026, 4, 5, 9, 38, tzinfo=UTC),
    )

    widget.show()
    widget.display_prompt(prompt)
    qt_app.processEvents()

    assert widget._copy_prompt_body_button.toolTip() == "Copy the stored prompt body."  # noqa: SLF001
    assert (
        widget._open_in_workspace_button.toolTip()
        == "Open the prompt in Workspace to fill variables: customer_name, region."
    )  # noqa: SLF001


def test_prompt_detail_widget_hides_template_variable_cue_for_plain_prompt(
    qt_app: QApplication,
) -> None:
    """Plain prompt bodies should not gain template-variable noise in detail view."""
    widget = PromptDetailWidget()
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000133"),
        name="Plain prompt",
        description="Fallback description",
        category="General",
        context="Summarize the incident and call out operator-facing risks.",
        created_at=datetime(2026, 4, 4, 9, 0, tzinfo=UTC),
        last_modified=datetime(2026, 4, 5, 9, 34, tzinfo=UTC),
    )

    widget.show()
    widget.display_prompt(prompt)
    qt_app.processEvents()

    assert not widget._template_variable_cue_label.isVisible()  # noqa: SLF001
    assert widget._template_variable_cue_label.text() == ""  # noqa: SLF001


def test_prompt_detail_widget_bounds_template_variable_cue_summary(
    qt_app: QApplication,
) -> None:
    """Template variable cue should show at most two names before a count suffix."""
    widget = PromptDetailWidget()
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000134"),
        name="Large template",
        description="Fallback description",
        category="Operations",
        context=(
            "Summarize {{ customer_name }} risk posture for {{ region }} using {{ product_name }} "
            "and {{ severity_level }}."
        ),
        created_at=datetime(2026, 4, 4, 9, 0, tzinfo=UTC),
        last_modified=datetime(2026, 4, 5, 9, 36, tzinfo=UTC),
    )

    widget.show()
    widget.display_prompt(prompt)
    qt_app.processEvents()

    cue_text = widget._template_variable_cue_label.text()  # noqa: SLF001
    assert "Requires variables: customer_name, product_name +2" in cue_text


def test_prompt_detail_widget_bounds_template_aware_workspace_tooltip_summary(
    qt_app: QApplication,
) -> None:
    """Template-aware workspace handoff tooltips should stay bounded for large templates."""
    widget = PromptDetailWidget()
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000136"),
        name="Large template handoff",
        description="Fallback description",
        category="Operations",
        context=(
            "Summarize {{ customer_name }} risk posture for {{ region }} using {{ product_name }} "
            "and {{ severity_level }}."
        ),
        created_at=datetime(2026, 4, 4, 9, 0, tzinfo=UTC),
        last_modified=datetime(2026, 4, 5, 9, 40, tzinfo=UTC),
    )

    widget.show()
    widget.display_prompt(prompt)
    qt_app.processEvents()

    assert (
        widget._open_in_workspace_button.toolTip()
        == "Open the prompt in Workspace to fill variables: customer_name, product_name +2."
    )  # noqa: SLF001


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
    assert not widget._template_variable_cue_label.isVisible()  # noqa: SLF001
    assert "When to use:" not in widget._meta_label.text()  # noqa: SLF001
    assert "Inspection:" in widget._meta_label.text()  # noqa: SLF001
    assert "Source: support queue" in widget._meta_label.text()  # noqa: SLF001
    assert "Description:" in widget._description.text()  # noqa: SLF001
    assert "Prompt Body (preview):" in widget._context.text()  # noqa: SLF001
    assert "Scenarios:" in widget._scenarios.text()  # noqa: SLF001
    assert not widget._metadata_view.isVisible()  # noqa: SLF001
