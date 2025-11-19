"""Notes tab for storing simple prompt notes.

Updates: v0.1.0 - 2025-12-06 - Initial implementation with CRUD actions.
"""

from __future__ import annotations

import uuid
from typing import Callable, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QDialog,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QGuiApplication

from core import (
    PromptManager,
    PromptNoteError,
    PromptNoteNotFoundError,
    PromptNoteStorageError,
)
from models.prompt_note import PromptNote

from .dialogs import PromptNoteDialog, MarkdownPreviewDialog


class NotesPanel(QWidget):
    """Display and manage prompt notes."""

    def __init__(
        self,
        manager: PromptManager,
        parent: Optional[QWidget] = None,
        *,
        status_callback: Optional[Callable[[str, int], None]] = None,
    ) -> None:
        super().__init__(parent)
        self._manager = manager
        self._status_callback = status_callback
        self._notes: List[PromptNote] = []
        self._list = QListWidget(self)
        self._note_view = QPlainTextEdit(self)
        self._note_view.setReadOnly(True)
        self._note_view.setPlaceholderText("Select a note to view its contents…")
        self._build_ui()
        self._load_notes()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        controls = QHBoxLayout()
        self._add_button = QPushButton("Add", self)
        self._add_button.clicked.connect(self._on_add_clicked)  # type: ignore[arg-type]
        self._edit_button = QPushButton("Edit", self)
        self._edit_button.setEnabled(False)
        self._edit_button.clicked.connect(self._on_edit_clicked)  # type: ignore[arg-type]
        self._delete_button = QPushButton("Delete", self)
        self._delete_button.setEnabled(False)
        self._delete_button.clicked.connect(self._on_delete_clicked)  # type: ignore[arg-type]
        self._copy_button = QPushButton("Copy", self)
        self._copy_button.setEnabled(False)
        self._copy_button.clicked.connect(self._on_copy_clicked)  # type: ignore[arg-type]
        self._markdown_button = QPushButton("Markdown", self)
        self._markdown_button.setEnabled(False)
        self._markdown_button.clicked.connect(self._on_markdown_clicked)  # type: ignore[arg-type]
        self._export_button = QPushButton("Export", self)
        self._export_button.clicked.connect(self._on_export_clicked)  # type: ignore[arg-type]
        self._refresh_button = QPushButton("Refresh", self)
        self._refresh_button.clicked.connect(self._load_notes)  # type: ignore[arg-type]
        controls.addWidget(self._add_button)
        controls.addWidget(self._edit_button)
        controls.addWidget(self._delete_button)
        controls.addWidget(self._copy_button)
        controls.addWidget(self._markdown_button)
        controls.addWidget(self._export_button)
        controls.addStretch(1)
        controls.addWidget(self._refresh_button)
        layout.addLayout(controls)

        self._list.setAlternatingRowColors(True)
        self._list.itemSelectionChanged.connect(self._on_selection_changed)  # type: ignore[arg-type]
        self._list.itemDoubleClicked.connect(lambda _: self._on_edit_clicked())  # type: ignore[arg-type]
        layout.addWidget(self._list, 2)
        layout.addWidget(self._note_view, 3)

    def _load_notes(self) -> None:
        try:
            self._notes = self._manager.list_prompt_notes()
        except PromptNoteStorageError as exc:
            QMessageBox.critical(self, "Unable to load notes", str(exc))
            self._notes = []
        self._list.clear()
        for note in self._notes:
            title = note.note.splitlines()[0] if note.note else "Untitled note"
            item = QListWidgetItem(title.strip() or "Untitled note", self._list)
            item.setData(Qt.UserRole, str(note.id))
        self._note_view.clear()
        self._edit_button.setEnabled(False)
        self._delete_button.setEnabled(False)
        self._copy_button.setEnabled(False)
        self._markdown_button.setEnabled(False)
        if self._notes:
            self._list.setCurrentRow(0)

    def _selected_note(self) -> Optional[PromptNote]:
        item = self._list.currentItem()
        if item is None:
            return None
        raw_id = item.data(Qt.UserRole)
        try:
            note_id = uuid.UUID(str(raw_id))
        except (TypeError, ValueError):
            return None
        for note in self._notes:
            if note.id == note_id:
                return note
        try:
            return self._manager.get_prompt_note(note_id)
        except PromptNoteError:
            return None

    def _on_selection_changed(self) -> None:
        note = self._selected_note()
        self._edit_button.setEnabled(note is not None)
        self._delete_button.setEnabled(note is not None)
        self._copy_button.setEnabled(note is not None)
        self._markdown_button.setEnabled(note is not None)
        self._note_view.setPlainText(note.note if note else "")

    def _on_add_clicked(self) -> None:
        dialog = PromptNoteDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        result = dialog.result_note
        if result is None:
            return
        try:
            self._manager.create_prompt_note(result)
        except PromptNoteError as exc:
            QMessageBox.critical(self, "Unable to create note", str(exc))
            return
        self._load_notes()
        self._show_status("Note created.")

    def _on_edit_clicked(self) -> None:
        note = self._selected_note()
        if note is None:
            QMessageBox.information(self, "Edit note", "Select a note first.")
            return
        dialog = PromptNoteDialog(self, note=note)
        if dialog.exec() != QDialog.Accepted:
            return
        result = dialog.result_note
        if result is None:
            return
        try:
            self._manager.update_prompt_note(result)
        except PromptNoteError as exc:
            QMessageBox.critical(self, "Unable to update note", str(exc))
            return
        self._load_notes()
        self._show_status("Note updated.")

    def _on_delete_clicked(self) -> None:
        note = self._selected_note()
        if note is None:
            QMessageBox.information(self, "Delete note", "Select a note first.")
            return
        confirmation = QMessageBox.question(
            self,
            "Delete note",
            "Delete the selected note?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirmation != QMessageBox.Yes:
            return
        try:
            self._manager.delete_prompt_note(note.id)
        except PromptNoteError as exc:
            QMessageBox.critical(self, "Unable to delete note", str(exc))
            return
        self._load_notes()
        self._show_status("Note deleted.")

    def _on_copy_clicked(self) -> None:
        note = self._selected_note()
        if note is None:
            QMessageBox.information(self, "Copy note", "Select a note first.")
            return
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(note.note)
        self._show_status("Note copied to clipboard.")

    def _on_export_clicked(self) -> None:
        if not self._notes:
            QMessageBox.information(self, "Export notes", "There are no notes to export yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export notes",
            "prompt_notes.txt",
            "Text Files (*.txt);;All Files (*.*)",
        )
        if not path:
            return
        lines = []
        for note in self._notes:
            lines.append("---")
            lines.append(f"Created: {note.created_at.isoformat()}")
            lines.append(f"Last modified: {note.last_modified.isoformat()}")
            lines.append("Note:")
            lines.append(note.note)
            lines.append("")
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines).strip() + "\n")
        except OSError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self._show_status(f"Exported {len(self._notes)} notes.")

    def _on_markdown_clicked(self) -> None:
        note = self._selected_note()
        if note is None:
            QMessageBox.information(self, "Markdown preview", "Select a note first.")
            return
        if not note.note.strip():
            QMessageBox.information(self, "Markdown preview", "The selected note is empty.")
            return
        dialog = MarkdownPreviewDialog(note.note, self, title="Note Preview")
        dialog.exec()

    def _show_status(self, message: str, duration_ms: int = 3000) -> None:
        if self._status_callback is not None:
            self._status_callback(message, duration_ms)


__all__ = ["NotesPanel"]
