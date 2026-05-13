"""Focused tests for prompt detail inspection cues.

Updates:
  v0.1.15 - 2026-04-27 - Expect shortened decision-provenance wording on inspect cues.
  v0.1.14 - 2026-04-27 - Expect action-oriented limited-evidence next-action wording.
  v0.1.13 - 2026-04-12 - Keep `Add Favorite` before `Promote Draft` in the
             shared draft detail action row.
  v0.1.12 - 2026-04-12 - Cover bounded usage-confidence cue rendering from usage counts.
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
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton

from gui.widgets import PromptDetailWidget
from models.prompt_model import Prompt


def _required_label(widget: PromptDetailWidget, name: str) -> QLabel:
    found = widget.findChild(QLabel, name)
    assert found is not None
    return found


def _required_button(widget: PromptDetailWidget, name: str) -> QPushButton:
    found = widget.findChild(QPushButton, name)
    assert found is not None
    return found


def _required_plain_text_edit(widget: PromptDetailWidget, name: str) -> QPlainTextEdit:
    found = widget.findChild(QPlainTextEdit, name)
    assert found is not None
    return found


def _run_summary_label(widget: PromptDetailWidget) -> QLabel:
    return _required_label(widget, "promptRunSummary")


def _decision_label(widget: PromptDetailWidget) -> QLabel:
    return _required_label(widget, "promptDecisionSummary")


def _next_action_label(widget: PromptDetailWidget) -> QLabel:
    return _required_label(widget, "promptNextActionSummary")


def _decision_provenance_label(widget: PromptDetailWidget) -> QLabel:
    return _required_label(widget, "promptDecisionProvenanceSummary")


def _name_label(widget: PromptDetailWidget) -> QLabel:
    return _required_label(widget, "promptTitle")


def _template_variable_cue_label(widget: PromptDetailWidget) -> QLabel:
    return _required_label(widget, "promptTemplateVariableCue")


def _meta_label(widget: PromptDetailWidget) -> QLabel:
    return _required_label(widget, "promptInspectionCues")


def _usage_cue_label(widget: PromptDetailWidget) -> QLabel:
    return _required_label(widget, "promptUsageCue")


def _reuse_signal_label(widget: PromptDetailWidget) -> QLabel:
    return _required_label(widget, "promptReuseSignalCue")


def _workspace_handoff_cue_label(widget: PromptDetailWidget) -> QLabel:
    return _required_label(widget, "promptWorkspaceHandoffCue")


def _description_label(widget: PromptDetailWidget) -> QLabel:
    return _required_label(widget, "promptDescription")


def _context_label(widget: PromptDetailWidget) -> QLabel:
    return _required_label(widget, "promptContext")


def _scenarios_label(widget: PromptDetailWidget) -> QLabel:
    return _required_label(widget, "promptScenarios")


def _metadata_view(widget: PromptDetailWidget) -> QPlainTextEdit:
    return _required_plain_text_edit(widget, "promptMetadata")


def _copy_prompt_body_button(widget: PromptDetailWidget) -> QPushButton:
    return _required_button(widget, "copyPromptBodyButton")


def _open_in_workspace_button(widget: PromptDetailWidget) -> QPushButton:
    return _required_button(widget, "openInWorkspaceButton")


def _favorite_button(widget: PromptDetailWidget) -> QPushButton:
    return _required_button(widget, "favoritePromptButton")


def _promote_draft_button(widget: PromptDetailWidget) -> QPushButton:
    return _required_button(widget, "promoteDraftButton")


def _edit_button_row(widget: PromptDetailWidget) -> QHBoxLayout:
    layout = widget.layout()
    assert layout is not None
    item = layout.itemAt(0)
    assert item is not None
    scroll_area = item.widget()
    assert scroll_area is not None
    content = scroll_area.findChild(QLabel, "promptTitle")
    assert content is not None
    parent = content.parentWidget()
    assert parent is not None
    parent_layout = parent.layout()
    assert parent_layout is not None
    row_item = parent_layout.itemAt(0)
    assert row_item is not None
    row = row_item.layout()
    assert isinstance(row, QHBoxLayout)
    return row


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    """Provide a shared Qt application instance for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return cast("QApplication", app)


def test_prompt_detail_widget_renders_last_run_summary_label(
    qt_app: QApplication,
) -> None:
    """Detail view should render one compact last-run cue when the controller provides it."""
    widget = PromptDetailWidget()

    widget.show()
    widget.update_run_summary(
        "Last run: success via gpt-4o-mini · v1 · 3 messages · 120 ms"
        " · Validation freshness: recent"
        " · Comparison readiness: limited"
    )
    qt_app.processEvents()

    assert _run_summary_label(widget).isVisible()  # noqa: SLF001
    summary_text = _run_summary_label(widget).text()  # noqa: SLF001
    assert "Last run:" in summary_text
    assert "gpt-4o-mini" in summary_text
    assert "Validation freshness: recent" in summary_text
    assert "Comparison readiness: limited" in summary_text


def test_prompt_detail_widget_hides_last_run_summary_when_empty(
    qt_app: QApplication,
) -> None:
    """Detail view should clear the last-run cue when the controller removes it."""
    widget = PromptDetailWidget()

    widget.show()
    widget.update_run_summary("Last run: success")
    widget.update_run_summary(None)
    qt_app.processEvents()

    assert not _run_summary_label(widget).isVisible()  # noqa: SLF001
    assert _run_summary_label(widget).text() == ""  # noqa: SLF001


def test_prompt_detail_widget_renders_decision_summary_label(
    qt_app: QApplication,
) -> None:
    """Detail view should render one compact decision cue when the controller provides it."""
    widget = PromptDetailWidget()

    widget.show()
    widget.update_decision_summary("Refine before reuse")
    qt_app.processEvents()

    assert _decision_label(widget).isVisible()  # noqa: SLF001
    decision_text = _decision_label(widget).text()  # noqa: SLF001
    assert "Decision:" in decision_text
    assert "Refine before reuse" in decision_text


def test_prompt_detail_widget_renders_next_action_summary_label(
    qt_app: QApplication,
) -> None:
    """Detail view should render one compact next-action cue when the controller provides it."""
    widget = PromptDetailWidget()

    widget.show()
    widget.update_next_action_summary("Compare before validating reuse")
    qt_app.processEvents()

    assert _next_action_label(widget).isVisible()  # noqa: SLF001
    next_action_text = _next_action_label(widget).text()  # noqa: SLF001
    assert "Recommended next action:" in next_action_text
    assert "Compare before validating reuse" in next_action_text


def test_prompt_detail_widget_renders_missing_evidence_next_action_label(
    qt_app: QApplication,
) -> None:
    """Detail view should keep missing-evidence guidance on the next-action surface."""
    widget = PromptDetailWidget()

    widget.show()
    widget.update_decision_summary("Reuse as-is")
    widget.update_next_action_summary("Validate before reuse")
    qt_app.processEvents()

    assert _decision_label(widget).isVisible()  # noqa: SLF001
    assert _next_action_label(widget).isVisible()  # noqa: SLF001
    next_action_text = _next_action_label(widget).text()  # noqa: SLF001
    assert "Recommended next action:" in next_action_text
    assert "Validate before reuse" in next_action_text


def test_prompt_detail_widget_shows_workspace_validation_handoff_for_validate_before_reuse(
    qt_app: QApplication,
) -> None:
    """Validation-first guidance should point plain prompts to Workspace first."""
    widget = PromptDetailWidget()
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000141"),
        name="Validation handoff prompt",
        description="Reusable prompt with limited confidence.",
        category="Operations",
        context="Summarize the incident and call out operator-facing risks.",
        created_at=datetime(2026, 4, 4, 9, 0, tzinfo=UTC),
        last_modified=datetime(2026, 4, 5, 9, 45, tzinfo=UTC),
    )

    widget.show()
    widget.display_prompt(prompt)
    widget.update_decision_summary("Reuse as-is")
    widget.update_next_action_summary("Validate before reuse")
    qt_app.processEvents()

    assert _open_in_workspace_button(widget).text() == "Open in Workspace"  # noqa: SLF001
    assert _workspace_handoff_cue_label(widget).isVisible()  # noqa: SLF001
    cue_text = _workspace_handoff_cue_label(widget).text()  # noqa: SLF001
    assert "Next step:" in cue_text
    assert "Open in Workspace before validating reuse." in cue_text


def test_prompt_detail_widget_shows_validate_decision_package_consistently(
    qt_app: QApplication,
) -> None:
    """Validate-path detail state should expose one coherent decision package."""
    widget = PromptDetailWidget()
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000149"),
        name="Validate package prompt",
        description="Prompt is reusable but should be validated before reuse.",
        category="Operations",
        context="Summarize the incident and call out operator-facing risks.",
        created_at=datetime(2026, 4, 6, 8, 0, tzinfo=UTC),
        last_modified=datetime(2026, 4, 6, 8, 45, tzinfo=UTC),
    )

    widget.show()
    widget.display_prompt(prompt)
    widget.update_decision_summary("Reuse as-is")
    widget.update_decision_provenance_summary("Based on latest 2 comparable runs")
    widget.update_next_action_summary("Validate before reuse")
    qt_app.processEvents()

    decision_text = _decision_label(widget).text()  # noqa: SLF001
    provenance_text = _decision_provenance_label(widget).text()  # noqa: SLF001
    next_action_text = _next_action_label(widget).text()  # noqa: SLF001
    cue_text = _workspace_handoff_cue_label(widget).text()  # noqa: SLF001

    assert _decision_label(widget).isVisible()  # noqa: SLF001
    assert _decision_provenance_label(widget).isVisible()  # noqa: SLF001
    assert _next_action_label(widget).isVisible()  # noqa: SLF001
    assert _workspace_handoff_cue_label(widget).isVisible()  # noqa: SLF001
    assert "Decision:</span> Reuse as-is" in decision_text
    assert "Decision basis: Based on latest 2 comparable runs." in provenance_text
    assert "Recommended next action:</span> Validate before reuse" in next_action_text
    assert cue_text.startswith("<span style=")
    assert "Next step:</span>" in cue_text
    assert "Open in Workspace before validating reuse." in cue_text


def test_prompt_detail_widget_renders_keep_baseline_decision_and_next_action(
    qt_app: QApplication,
) -> None:
    """Detail view should render replace-path guidance as bounded baseline-first wording."""
    widget = PromptDetailWidget()

    widget.show()
    widget.update_decision_summary("Keep baseline")
    widget.update_next_action_summary("Prefer baseline before reuse")
    qt_app.processEvents()

    assert _decision_label(widget).isVisible()  # noqa: SLF001
    assert _next_action_label(widget).isVisible()  # noqa: SLF001
    decision_text = _decision_label(widget).text()  # noqa: SLF001
    next_action_text = _next_action_label(widget).text()  # noqa: SLF001
    assert "Decision:" in decision_text
    assert "Keep baseline" in decision_text
    assert "Recommended next action:" in next_action_text
    assert "Prefer baseline before reuse" in next_action_text


def test_prompt_detail_widget_shows_baseline_handoff_for_keep_baseline(
    qt_app: QApplication,
) -> None:
    """Baseline-first detail guidance should surface one explicit baseline handoff cue."""
    widget = PromptDetailWidget()
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000145"),
        name="Baseline candidate",
        description="Prompt should defer to the stronger baseline before reuse.",
        category="Operations",
        context=(
            "Summarize the incident and compare the current candidate against "
            "the baseline guidance."
        ),
        created_at=datetime(2026, 4, 4, 9, 0, tzinfo=UTC),
        last_modified=datetime(2026, 4, 5, 10, 5, tzinfo=UTC),
    )

    widget.show()
    widget.display_prompt(prompt)
    widget.update_decision_summary("Keep baseline")
    widget.update_next_action_summary("Prefer baseline before reuse")
    qt_app.processEvents()

    assert _workspace_handoff_cue_label(widget).isVisible()  # noqa: SLF001
    cue_text = _workspace_handoff_cue_label(widget).text()  # noqa: SLF001
    assert "Next step:" in cue_text
    assert "Reuse the baseline prompt." in cue_text


def test_prompt_detail_widget_shows_refine_handoff_cue_for_edit_path(
    qt_app: QApplication,
) -> None:
    """Refine-first guidance should surface one visible next-step handoff on the detail seam."""
    widget = PromptDetailWidget()
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000142"),
        name="Refine candidate",
        description="Prompt likely needs a small local edit before reuse.",
        category="Operations",
        context="Summarize the incident and call out operator-facing risks.",
        created_at=datetime(2026, 4, 4, 9, 0, tzinfo=UTC),
        last_modified=datetime(2026, 4, 5, 10, 5, tzinfo=UTC),
    )

    widget.show()
    widget.display_prompt(prompt)
    widget.update_decision_summary("Refine before reuse")
    widget.update_next_action_summary("Edit Prompt before reuse")
    qt_app.processEvents()

    assert _workspace_handoff_cue_label(widget).isVisible()  # noqa: SLF001
    cue_text = _workspace_handoff_cue_label(widget).text()  # noqa: SLF001
    assert "Next step:" in cue_text
    assert "Edit Prompt before reuse." in cue_text


def test_prompt_detail_widget_shows_workspace_inspect_handoff_for_inspect_before_reuse(
    qt_app: QApplication,
) -> None:
    """Inspect-first detail guidance should surface one explicit inspect handoff cue."""
    widget = PromptDetailWidget()
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000144"),
        name="Inspect candidate",
        description="Prompt found through a contextual signal and should be checked before reuse.",
        category="Operations",
        context=(
            "Summarize the incident and confirm the recovery step still matches the operator goal."
        ),
        created_at=datetime(2026, 4, 4, 9, 0, tzinfo=UTC),
        last_modified=datetime(2026, 4, 5, 10, 5, tzinfo=UTC),
    )

    widget.show()
    widget.display_prompt(prompt)
    widget.update_decision_summary("Inspect before reuse")
    widget.update_next_action_summary("Inspect before reuse")
    qt_app.processEvents()

    assert _workspace_handoff_cue_label(widget).isVisible()  # noqa: SLF001
    cue_text = _workspace_handoff_cue_label(widget).text()  # noqa: SLF001
    assert "Next step:" in cue_text
    assert "Review prompt details before reusing." in cue_text


def test_prompt_detail_widget_shows_workspace_fork_handoff_for_fork_before_editing(
    qt_app: QApplication,
) -> None:
    """Fork-first detail guidance should surface one explicit fork handoff cue."""
    widget = PromptDetailWidget()
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000143"),
        name="Fork candidate",
        description="Prompt should branch before local edits.",
        category="Operations",
        context="Summarize the incident and preserve the original prompt.",
        created_at=datetime(2026, 4, 4, 9, 0, tzinfo=UTC),
        last_modified=datetime(2026, 4, 5, 10, 5, tzinfo=UTC),
    )

    widget.show()
    widget.display_prompt(prompt)
    widget.update_decision_summary("Fork before editing")
    widget.update_next_action_summary("Fork before editing")
    qt_app.processEvents()

    assert _workspace_handoff_cue_label(widget).isVisible()  # noqa: SLF001
    cue_text = _workspace_handoff_cue_label(widget).text()  # noqa: SLF001
    assert "Next step:" in cue_text
    assert "Fork Prompt to preserve the current version before editing." in cue_text


def test_prompt_detail_widget_renders_decision_provenance_label(
    qt_app: QApplication,
) -> None:
    """Detail view should render one compact provenance cue for the current decision."""
    widget = PromptDetailWidget()

    widget.show()
    widget.update_decision_provenance_summary("Based on latest 2 comparable runs")
    qt_app.processEvents()

    assert _decision_provenance_label(widget).isVisible()  # noqa: SLF001
    provenance_text = _decision_provenance_label(widget).text()  # noqa: SLF001
    assert "Based on latest 2 comparable runs" in provenance_text


def test_prompt_detail_widget_renders_limited_evidence_provenance_label(
    qt_app: QApplication,
) -> None:
    """Detail view should render a bounded provenance cue when only thin run evidence exists."""
    widget = PromptDetailWidget()

    widget.show()
    widget.update_decision_provenance_summary("Based on limited run evidence")
    qt_app.processEvents()

    assert _decision_provenance_label(widget).isVisible()  # noqa: SLF001
    provenance_text = _decision_provenance_label(widget).text()  # noqa: SLF001
    assert "Based on limited run evidence" in provenance_text


def test_prompt_detail_widget_formats_limited_evidence_provenance_as_cautionary_note(
    qt_app: QApplication,
) -> None:
    """Thin run evidence should read as an explicit cautionary provenance cue."""
    widget = PromptDetailWidget()

    widget.show()
    widget.update_decision_provenance_summary("Based on limited run evidence")
    qt_app.processEvents()

    assert _decision_provenance_label(widget).isVisible()  # noqa: SLF001
    provenance_text = _decision_provenance_label(widget).text()  # noqa: SLF001
    assert provenance_text.startswith("Note:")
    assert "Based on limited run evidence." in provenance_text


def test_prompt_detail_widget_formats_comparable_evidence_provenance_as_decision_basis(
    qt_app: QApplication,
) -> None:
    """Comparable runs should read as an explicit decision-basis provenance cue."""
    widget = PromptDetailWidget()

    widget.show()
    widget.update_decision_provenance_summary("Based on latest 2 comparable runs")
    qt_app.processEvents()

    assert _decision_provenance_label(widget).isVisible()  # noqa: SLF001
    provenance_text = _decision_provenance_label(widget).text()  # noqa: SLF001
    assert provenance_text.startswith("Decision basis:")
    assert "Based on latest 2 comparable runs." in provenance_text


def test_prompt_detail_widget_hides_next_action_summary_when_empty(
    qt_app: QApplication,
) -> None:
    """Detail view should clear the next-action cue when the controller removes it."""
    widget = PromptDetailWidget()

    widget.show()
    widget.update_next_action_summary("Compare before validating reuse")
    widget.update_next_action_summary(None)
    qt_app.processEvents()

    assert not _next_action_label(widget).isVisible()  # noqa: SLF001
    assert _next_action_label(widget).text() == ""  # noqa: SLF001


def test_prompt_detail_widget_reveals_next_action_after_decision_clears(
    qt_app: QApplication,
) -> None:
    """Hidden next-action cue should reappear once the duplicate decision cue is removed."""
    widget = PromptDetailWidget()

    widget.show()
    widget.update_decision_summary("Refine before reuse")
    widget.update_next_action_summary("Refine before reuse")
    widget.update_decision_summary(None)
    qt_app.processEvents()

    assert _next_action_label(widget).isVisible()  # noqa: SLF001
    next_action_text = _next_action_label(widget).text()  # noqa: SLF001
    assert "Recommended next action:" in next_action_text
    assert "Refine before reuse" in next_action_text


def test_prompt_detail_widget_hides_redundant_next_action_when_same_as_decision(
    qt_app: QApplication,
) -> None:
    """Hide duplicate next-action cue when it adds no new information."""
    widget = PromptDetailWidget()

    widget.show()
    widget.update_decision_summary("Refine before reuse")
    widget.update_next_action_summary("Refine before reuse")
    qt_app.processEvents()

    assert _decision_label(widget).isVisible()  # noqa: SLF001
    assert _next_action_label(widget).text() == ""  # noqa: SLF001


def test_prompt_detail_widget_applies_readable_default_font_sizes(
    qt_app: QApplication,
) -> None:
    """Shared detail typography should make title/body text easier to read by default."""
    widget = PromptDetailWidget()

    widget.show()
    qt_app.processEvents()

    assert _name_label(widget).font().pointSizeF() > widget.font().pointSizeF()  # noqa: SLF001
    assert _description_label(widget).font().pointSizeF() > widget.font().pointSizeF()  # noqa: SLF001
    assert _template_variable_cue_label(widget).font().pointSizeF() > widget.font().pointSizeF()  # noqa: SLF001
    assert _metadata_view(widget).font().pointSizeF() > widget.font().pointSizeF()  # noqa: SLF001


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

    assert _meta_label(widget).isVisible()  # noqa: SLF001
    inspection_text = _meta_label(widget).text()  # noqa: SLF001
    assert "Inspection:" in inspection_text
    assert "Draft (quick_capture)" in inspection_text
    assert "Source: chat thread" in inspection_text
    assert "Last modified: 2026-04-04 10:30 UTC" in inspection_text
    assert not _metadata_view(widget).isVisible()  # noqa: SLF001


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

    assert _copy_prompt_body_button(widget).text() == "Copy Prompt"  # noqa: SLF001
    assert _open_in_workspace_button(widget).text() == "Open in Workspace"  # noqa: SLF001
    assert _copy_prompt_body_button(widget).isEnabled()  # noqa: SLF001
    assert _open_in_workspace_button(widget).isEnabled()  # noqa: SLF001
    assert _copy_prompt_body_button(widget).toolTip() == "Copy the stored prompt body."  # noqa: SLF001
    assert (
        _open_in_workspace_button(widget).toolTip()
        == "Open the stored prompt body in the workspace without running it."
    )  # noqa: SLF001
    assert not _usage_cue_label(widget).isVisible()  # noqa: SLF001

    _copy_prompt_body_button(widget).click()  # noqa: SLF001
    _open_in_workspace_button(widget).click()  # noqa: SLF001
    qt_app.processEvents()

    assert copy_requests == ["copy"]
    assert open_requests == ["open"]


def test_prompt_detail_widget_shows_copy_first_handoff_when_direct_reuse_is_ready(
    qt_app: QApplication,
) -> None:
    """A reuse-ready prompt should expose copying as the direct next step."""
    widget = PromptDetailWidget()
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000136"),
        name="Direct reuse prompt",
        description="Fallback description",
        category="General",
        context="Prompt body to reuse",
        created_at=datetime(2026, 4, 4, 9, 0, tzinfo=UTC),
        last_modified=datetime(2026, 4, 4, 10, 30, tzinfo=UTC),
    )

    widget.show()
    widget.display_prompt(prompt)
    widget.update_decision_summary("Reuse as-is")
    qt_app.processEvents()

    assert _copy_prompt_body_button(widget).isEnabled()  # noqa: SLF001
    assert _open_in_workspace_button(widget).isEnabled()  # noqa: SLF001
    assert _workspace_handoff_cue_label(widget).isVisible()  # noqa: SLF001
    cue_text = _workspace_handoff_cue_label(widget).text()  # noqa: SLF001
    assert "Next step:" in cue_text
    assert "Copy Prompt for direct reuse." in cue_text


def test_prompt_detail_widget_toggles_favorite_action_from_detail_flow(
    qt_app: QApplication,
) -> None:
    """Detail flow should expose one bounded favorite toggle for the current prompt."""
    widget = PromptDetailWidget()
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000132"),
        name="Favorite candidate",
        description="Reusable and worth keeping close.",
        category="General",
        context="Prompt body",
        is_favorite=False,
        created_at=datetime(2026, 4, 4, 9, 0, tzinfo=UTC),
        last_modified=datetime(2026, 4, 5, 11, 0, tzinfo=UTC),
    )
    favorite_requests: list[str] = []
    widget.favorite_toggled_requested.connect(lambda: favorite_requests.append("toggle"))

    widget.show()
    widget.display_prompt(prompt)
    qt_app.processEvents()

    assert _favorite_button(widget).isEnabled()  # noqa: SLF001
    assert _favorite_button(widget).text() == "Add Favorite"  # noqa: SLF001
    assert (
        _favorite_button(widget).toolTip()
        == "Add this prompt to favorites so it stays easy to find later with Favorites only."
    )  # noqa: SLF001

    _favorite_button(widget).click()  # noqa: SLF001
    qt_app.processEvents()

    assert favorite_requests == ["toggle"]

    widget.display_prompt(Prompt.from_record({**prompt.to_record(), "is_favorite": True}))
    qt_app.processEvents()

    assert _favorite_button(widget).text() == "Remove Favorite"  # noqa: SLF001
    assert _favorite_button(widget).toolTip() == "Remove this prompt from favorites."  # noqa: SLF001


def test_prompt_detail_widget_keeps_favorite_before_promote_for_draft_prompts(
    qt_app: QApplication,
) -> None:
    """Draft detail actions should keep favorite to the left of the promote action."""
    widget = PromptDetailWidget()
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000133"),
        name="Draft favorite candidate",
        description="Reusable and worth keeping close.",
        category="General",
        context="Prompt body",
        is_favorite=False,
        ext2={"capture_state": "draft", "capture_method": "quick_capture"},
        created_at=datetime(2026, 4, 4, 9, 0, tzinfo=UTC),
        last_modified=datetime(2026, 4, 5, 11, 0, tzinfo=UTC),
    )

    widget.show()
    widget.display_prompt(prompt)
    qt_app.processEvents()

    assert _promote_draft_button(widget).isVisible()  # noqa: SLF001
    assert _edit_button_row(widget).indexOf(_favorite_button(widget)) < (
        _edit_button_row(widget).indexOf(_promote_draft_button(widget))
    )  # noqa: SLF001


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

    assert not _copy_prompt_body_button(widget).isEnabled()  # noqa: SLF001
    assert _open_in_workspace_button(widget).isEnabled()  # noqa: SLF001
    assert (
        _copy_prompt_body_button(widget).toolTip()
        == "Copy Prompt is unavailable because this prompt has no stored prompt body."
    )  # noqa: SLF001
    assert (
        _open_in_workspace_button(widget).toolTip()
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

    inspection_text = _meta_label(widget).text()  # noqa: SLF001
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

    inspection_text = _meta_label(widget).text()  # noqa: SLF001
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

    assert _usage_cue_label(widget).isVisible()  # noqa: SLF001
    usage_text = _usage_cue_label(widget).text()  # noqa: SLF001
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

    assert _usage_cue_label(widget).isVisible()  # noqa: SLF001
    usage_text = _usage_cue_label(widget).text()  # noqa: SLF001
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

    assert not _usage_cue_label(widget).isVisible()  # noqa: SLF001
    assert _usage_cue_label(widget).text() == ""  # noqa: SLF001


def test_prompt_detail_widget_shows_reuse_signal_when_prompt_has_usage_history(
    qt_app: QApplication,
) -> None:
    """Detail view should show one quiet reuse-confidence cue when usage exists."""
    widget = PromptDetailWidget()
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000136"),
        name="Reusable prompt",
        description="Fallback description",
        category="Operations",
        context="Prompt body",
        usage_count=4,
        created_at=datetime(2026, 4, 4, 9, 0, tzinfo=UTC),
        last_modified=datetime(2026, 4, 5, 9, 40, tzinfo=UTC),
    )

    widget.show()
    widget.display_prompt(prompt)
    qt_app.processEvents()

    assert _reuse_signal_label(widget).isVisible()  # noqa: SLF001
    cue_text = _reuse_signal_label(widget).text()  # noqa: SLF001
    assert "Reuse signal:" in cue_text
    assert "used 4 times" in cue_text


def test_prompt_detail_widget_uses_singular_reuse_signal_wording(
    qt_app: QApplication,
) -> None:
    """Detail view should use singular wording for one prior use."""
    widget = PromptDetailWidget()
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000137"),
        name="Reusable prompt",
        description="Fallback description",
        category="Operations",
        context="Prompt body",
        usage_count=1,
        created_at=datetime(2026, 4, 4, 9, 0, tzinfo=UTC),
        last_modified=datetime(2026, 4, 5, 9, 41, tzinfo=UTC),
    )

    widget.show()
    widget.display_prompt(prompt)
    qt_app.processEvents()

    assert _reuse_signal_label(widget).isVisible()  # noqa: SLF001
    cue_text = _reuse_signal_label(widget).text()  # noqa: SLF001
    assert "used 1 time" in cue_text
    assert "used 1 times" not in cue_text


def test_prompt_detail_widget_hides_reuse_signal_without_usage_history(
    qt_app: QApplication,
) -> None:
    """Detail view should stay quiet when no usage count exists yet."""
    widget = PromptDetailWidget()
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000138"),
        name="Stored prompt",
        description="Fallback description",
        category="Operations",
        context="Prompt body",
        usage_count=0,
        created_at=datetime(2026, 4, 4, 9, 0, tzinfo=UTC),
        last_modified=datetime(2026, 4, 5, 9, 42, tzinfo=UTC),
    )

    widget.show()
    widget.display_prompt(prompt)
    qt_app.processEvents()

    assert not _reuse_signal_label(widget).isVisible()  # noqa: SLF001
    assert _reuse_signal_label(widget).text() == ""  # noqa: SLF001


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

    assert _template_variable_cue_label(widget).isVisible()  # noqa: SLF001
    cue_text = _template_variable_cue_label(widget).text()  # noqa: SLF001
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

    assert _copy_prompt_body_button(widget).toolTip() == "Copy the stored prompt body."  # noqa: SLF001
    assert (
        _open_in_workspace_button(widget).toolTip()
        == "Open the prompt in Workspace to fill variables: customer_name, region."
    )  # noqa: SLF001


def test_prompt_detail_widget_shows_visible_workspace_handoff_cue_for_template_prompt(
    qt_app: QApplication,
) -> None:
    """Template prompts should get one visible reuse cue without adding a second CTA."""
    widget = PromptDetailWidget()
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000139"),
        name="Template reuse cue",
        description="Fallback description",
        category="Operations",
        context="Summarize {{ customer_name }} risk posture for {{ region }}.",
        created_at=datetime(2026, 4, 4, 9, 0, tzinfo=UTC),
        last_modified=datetime(2026, 4, 5, 9, 43, tzinfo=UTC),
    )

    widget.show()
    widget.display_prompt(prompt)
    qt_app.processEvents()

    assert _open_in_workspace_button(widget).text() == "Open in Workspace"  # noqa: SLF001
    assert _workspace_handoff_cue_label(widget).isVisible()  # noqa: SLF001
    cue_text = _workspace_handoff_cue_label(widget).text()  # noqa: SLF001
    assert "Next step:" in cue_text
    assert "Open in Workspace to fill variables before reuse." in cue_text


def test_prompt_detail_widget_hides_visible_workspace_handoff_cue_for_plain_prompt(
    qt_app: QApplication,
) -> None:
    """Plain prompts should not show the template-specific visible workspace cue."""
    widget = PromptDetailWidget()
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000140"),
        name="Plain reuse cue",
        description="Fallback description",
        category="General",
        context="Summarize the incident and call out operator-facing risks.",
        created_at=datetime(2026, 4, 4, 9, 0, tzinfo=UTC),
        last_modified=datetime(2026, 4, 5, 9, 44, tzinfo=UTC),
    )

    widget.show()
    widget.display_prompt(prompt)
    qt_app.processEvents()

    assert _open_in_workspace_button(widget).text() == "Open in Workspace"  # noqa: SLF001
    assert not _workspace_handoff_cue_label(widget).isVisible()  # noqa: SLF001
    assert _workspace_handoff_cue_label(widget).text() == ""  # noqa: SLF001


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

    assert not _template_variable_cue_label(widget).isVisible()  # noqa: SLF001
    assert _template_variable_cue_label(widget).text() == ""  # noqa: SLF001


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

    cue_text = _template_variable_cue_label(widget).text()  # noqa: SLF001
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
        _open_in_workspace_button(widget).toolTip()
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

    assert _meta_label(widget).isVisible()  # noqa: SLF001
    assert _usage_cue_label(widget).isVisible()  # noqa: SLF001
    assert not _template_variable_cue_label(widget).isVisible()  # noqa: SLF001
    assert "When to use:" not in _meta_label(widget).text()  # noqa: SLF001
    assert "Inspection:" in _meta_label(widget).text()  # noqa: SLF001
    assert "Source: support queue" in _meta_label(widget).text()  # noqa: SLF001
    assert "Description:" in _description_label(widget).text()  # noqa: SLF001
    assert "Prompt Body (preview):" in _context_label(widget).text()  # noqa: SLF001
    assert "Scenarios:" in _scenarios_label(widget).text()  # noqa: SLF001
    assert not _metadata_view(widget).isVisible()  # noqa: SLF001
