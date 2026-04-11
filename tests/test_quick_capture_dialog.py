"""Focused tests for the quick capture dialog and draft conversion.

Updates:
  v0.1.3 - 2026-04-11 - Cover one allowed outer prompt label strip and unchanged ambiguous input.
  v0.1.2 - 2026-04-10 - Cover bounded outer-fence unwrapping for messy quick-capture input.
  v0.1.1 - 2026-04-06 - Cover shared draft-title normalization for quick capture defaults.
  v0.1.0 - 2026-04-05 - Cover source/provenance capture and prompt conversion.
"""

from __future__ import annotations

import uuid
from typing import cast

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from gui.dialogs.quick_capture import (
    QuickCaptureDialog,
    QuickCaptureDraft,
    derive_quick_capture_title,
    resolve_quick_capture_source,
    strip_quick_capture_prompt_label,
    unwrap_quick_capture_body,
)


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


def test_derive_quick_capture_title_skips_placeholder_lines_and_strips_raw_markers() -> None:
    """Default quick-capture titles should come from the first meaningful normalized line."""
    body = "\nPrompt:\n## Incident summary for handoff\n- Highlight operator-facing risks"

    assert derive_quick_capture_title(body) == "Incident summary for handoff"


def test_quick_capture_draft_unwraps_one_outer_markdown_fence() -> None:
    """Quick capture should remove one obvious outer markdown fence before storing body text."""
    draft = QuickCaptureDraft(body="```text\nSummarize deployment risks for the handoff.\n```")

    prompt = draft.to_prompt()

    assert prompt.context == "Summarize deployment risks for the handoff."


def test_quick_capture_draft_strips_one_allowed_outer_prompt_label() -> None:
    """Quick capture should remove one obvious top-level prompt label before storing."""
    draft = QuickCaptureDraft(body="Prompt: Summarize deployment risks for this release.")

    prompt = draft.to_prompt()

    assert prompt.context == "Summarize deployment risks for this release."


def test_quick_capture_draft_strips_one_allowed_multiline_system_prompt_label() -> None:
    """Quick capture should remove one allowed label line when real prompt text follows."""
    draft = QuickCaptureDraft(body="System prompt:\nKeep the summary terse and operator-focused.")

    prompt = draft.to_prompt()

    assert prompt.context == "Keep the summary terse and operator-focused."


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("Prompt:", "Prompt:"),
        (
            "Meeting notes mention prompt: summarize risks later.",
            "Meeting notes mention prompt: summarize risks later.",
        ),
        (
            "Prompt:\nUser: summarize the release risk\nAssistant: acknowledged",
            "Prompt:\nUser: summarize the release risk\nAssistant: acknowledged",
        ),
    ],
)
def test_strip_quick_capture_prompt_label_keeps_non_obvious_input_unchanged(
    body: str, expected: str
) -> None:
    """Ambiguous, incomplete, or transcript-like input should remain unchanged."""
    assert strip_quick_capture_prompt_label(body) == expected


def test_unwrap_quick_capture_body_keeps_non_wrapped_or_mixed_input_intact() -> None:
    """Mixed prose plus fenced content should stay unchanged without one outer fence."""
    body = "Intro note\n```text\nSummarize deployment risks.\n```"

    assert unwrap_quick_capture_body(body) == body.strip()


def test_quick_capture_dialog_uses_shared_title_quality_heuristic_when_title_missing(
    qt_app: QApplication,
) -> None:
    """Saving without a manual title should reuse the shared draft-title heuristic."""
    dialog = QuickCaptureDialog()
    dialog._body_input.setPlainText(  # noqa: SLF001
        "Title:\n> Weekly deployment checklist\nList the risky steps first."
    )

    draft = dialog._build_draft()  # noqa: SLF001

    assert draft is not None
    prompt = draft.to_prompt()
    assert prompt.name == "Weekly deployment checklist"
