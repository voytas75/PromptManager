"""Minimal dialog and helpers for quick prompt capture.

Updates:
  v0.1.2 - 2026-04-06 - Use shared draft-title heuristics when no manual title is supplied.
  v0.1.1 - 2026-04-05 - Clarify source/provenance copy while keeping prompt.source storage.
  v0.1.0 - 2026-04-04 - Add quick capture draft dialog and deterministic prompt conversion.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from models.prompt_model import Prompt

from .base import resolve_draft_origin_title

_DEFAULT_CATEGORY = "General"
_DEFAULT_DESCRIPTION = "Quick capture draft."
_DEFAULT_LANGUAGE = "en"
_DEFAULT_SOURCE = "quick_capture"
_DRAFT_METADATA = {
    "capture_state": "draft",
    "capture_method": "quick_capture",
}
_TITLE_LIMIT = 80


def derive_quick_capture_title(body: str, *, max_length: int = _TITLE_LIMIT) -> str:
    """Return a deterministic title derived from the first meaningful body line."""
    return resolve_draft_origin_title("", body, max_length=max_length)


def parse_quick_capture_tags(raw_tags: str) -> list[str]:
    """Split comma-separated tags into a trimmed list."""
    return [tag.strip() for tag in raw_tags.split(",") if tag.strip()]


def resolve_quick_capture_source(source_label: str) -> str:
    """Return the persisted source/provenance value for a quick-captured prompt."""
    return source_label.strip() or _DEFAULT_SOURCE


@dataclass(slots=True)
class QuickCaptureDraft:
    """Structured payload captured from the quick-capture dialog."""

    body: str
    title: str = ""
    source_label: str = ""
    tags_text: str = ""
    description: str = ""

    def to_prompt(self) -> Prompt:
        """Build a draft prompt record using existing catalog fields."""
        body = self.body.strip()
        if not body:
            raise ValueError("Prompt body is required.")
        name = self.title.strip() or derive_quick_capture_title(body)
        description = self.description.strip() or _DEFAULT_DESCRIPTION
        source = resolve_quick_capture_source(self.source_label)
        tags = parse_quick_capture_tags(self.tags_text)
        timestamp = datetime.now(UTC)
        return Prompt(
            id=uuid.uuid4(),
            name=name,
            description=description,
            category=_DEFAULT_CATEGORY,
            tags=tags,
            language=_DEFAULT_LANGUAGE,
            context=body,
            created_at=timestamp,
            last_modified=timestamp,
            source=source,
            ext2=dict(_DRAFT_METADATA),
        )


class QuickCaptureDialog(QDialog):
    """Collect raw prompt text plus minimal metadata for quick capture."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialise the compact capture form."""
        super().__init__(parent)
        self._result_draft: QuickCaptureDraft | None = None
        self.setWindowTitle("Quick Capture")
        self.setMinimumSize(720, 520)
        self.resize(820, 620)
        self._build_ui()

    @property
    def result_draft(self) -> QuickCaptureDraft | None:
        """Return the captured payload once the dialog is accepted."""
        return self._result_draft

    def _build_ui(self) -> None:
        """Construct the form controls and dialog buttons."""
        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setContentsMargins(16, 16, 16, 8)
        form.setSpacing(10)

        self._title_input = QLineEdit(self)
        self._title_input.setPlaceholderText(
            "Optional title; defaults to the first meaningful line"
        )
        form.addRow("Title", self._title_input)

        self._source_input = QLineEdit(self)
        self._source_input.setPlaceholderText(
            "Optional source or provenance, e.g. chat thread, notes, or script"
        )
        form.addRow("Source / Provenance", self._source_input)

        self._tags_input = QLineEdit(self)
        self._tags_input.setPlaceholderText("Optional tags, comma-separated")
        form.addRow("Tags", self._tags_input)

        self._description_input = QPlainTextEdit(self)
        self._description_input.setPlaceholderText("Optional short note or description")
        self._description_input.setFixedHeight(80)
        form.addRow("Note", self._description_input)

        self._body_input = QPlainTextEdit(self)
        self._body_input.setPlaceholderText("Paste the raw prompt or LLM query here…")
        self._body_input.setMinimumHeight(320)
        form.addRow("Prompt Body", self._body_input)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        save_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        if save_button is not None:
            save_button.setText("Save Draft")
        buttons.accepted.connect(self._on_accept)  # type: ignore[arg-type]
        buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        layout.addWidget(buttons)

        self._body_input.setFocus()

    def _on_accept(self) -> None:
        """Validate the form and keep the structured draft for callers."""
        draft = self._build_draft()
        if draft is None:
            return
        self._result_draft = draft
        self.accept()

    def _build_draft(self) -> QuickCaptureDraft | None:
        """Return a validated draft payload from the current form state."""
        body = self._body_input.toPlainText().strip()
        if not body:
            QMessageBox.warning(
                self,
                "Prompt body required",
                "Paste prompt or query text before saving a draft.",
            )
            return None
        return QuickCaptureDraft(
            body=body,
            title=self._title_input.text().strip(),
            source_label=self._source_input.text().strip(),
            tags_text=self._tags_input.text().strip(),
            description=self._description_input.toPlainText().strip(),
        )


__all__ = [
    "QuickCaptureDialog",
    "QuickCaptureDraft",
    "derive_quick_capture_title",
    "parse_quick_capture_tags",
    "resolve_quick_capture_source",
]
