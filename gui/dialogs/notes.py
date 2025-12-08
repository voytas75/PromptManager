"""Dialogs for creating and editing prompt notes.

Updates:
  v0.1.0 - 2025-12-03 - Extract prompt note dialog from gui.dialogs.
"""

from __future__ import annotations

import uuid
from dataclasses import replace

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QMessageBox,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from models.prompt_note import PromptNote


class PromptNoteDialog(QDialog):
    """Modal dialog for creating or editing prompt notes."""

    def __init__(self, parent: QWidget | None = None, *, note: PromptNote | None = None) -> None:
        """Initialise the note editor and optionally preload existing text."""
        super().__init__(parent)
        self._source_note = note
        self._result_note: PromptNote | None = None
        self.setWindowTitle("New Note" if note is None else "Edit Note")
        self.resize(520, 320)
        self._build_ui()
        if note is not None:
            self._note_input.setPlainText(note.note)

    @property
    def result_note(self) -> PromptNote | None:
        """Return the resulting note after dialog acceptance."""
        return self._result_note

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self._note_input = QPlainTextEdit(self)
        self._note_input.setPlaceholderText("Write your prompt note here…")
        self._note_input.setMinimumHeight(200)
        layout.addWidget(self._note_input)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self._on_accept)  # type: ignore[arg-type]
        buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        text = self._note_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Missing note", "Enter note text before saving.")
            return
        if self._source_note is None:
            self._result_note = PromptNote(id=uuid.uuid4(), note=text)
        else:
            updated = replace(self._source_note, note=text)
            updated.touch()
            self._result_note = updated
        self.accept()


__all__ = ["PromptNoteDialog"]
