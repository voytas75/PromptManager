"""Prompt part management tab.

Updates: v0.15.3 - 2025-11-27 - Add toast callback for clipboard operations.
Updates: v0.15.2 - 2025-11-27 - Add prompt part classification column and rename UI strings accordingly.
Updates: v0.15.1 - 2025-11-25 - Document module metadata for AGENTS compliance.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core import (
    PromptManager,
    ResponseStyleError,
    ResponseStyleStorageError,
)
from models.response_style import ResponseStyle

from .dialogs import MarkdownPreviewDialog, ResponseStyleDialog


class ResponseStylesPanel(QWidget):
    """Display and manage prompt parts (including response styles) inside their own tab."""

    def __init__(
        self,
        manager: PromptManager,
        parent: QWidget | None = None,
        *,
        status_callback: Callable[[str, int], None] | None = None,
        toast_callback: Callable[[str, int], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._manager = manager
        self._status_callback = status_callback
        self._toast_callback = toast_callback
        self._styles: list[ResponseStyle] = []
        self._list = QListWidget(self)
        self._detail_view = QPlainTextEdit(self)
        self._detail_view.setReadOnly(True)
        self._detail_view.setPlaceholderText("Select a prompt part to view its details…")
        self._build_ui()
        self._load_styles()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        controls = QHBoxLayout()
        self._new_button = QPushButton("New", self)
        self._new_button.clicked.connect(self._on_new_clicked)  # type: ignore[arg-type]
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
        self._refresh_button.clicked.connect(self._load_styles)  # type: ignore[arg-type]
        controls.addWidget(self._new_button)
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
        layout.addWidget(self._detail_view, 3)

    def _load_styles(self) -> None:
        try:
            self._styles = self._manager.list_response_styles(include_inactive=True)
        except ResponseStyleStorageError as exc:
            QMessageBox.critical(self, "Unable to load prompt parts", str(exc))
            self._styles = []
        self._list.clear()
        for style in self._styles:
            label = style.name or "Untitled"
            item = QListWidgetItem(label, self._list)
            item.setData(Qt.UserRole, str(style.id))
        self._detail_view.clear()
        self._edit_button.setEnabled(False)
        self._delete_button.setEnabled(False)
        self._copy_button.setEnabled(False)
        self._markdown_button.setEnabled(False)
        if self._styles:
            self._list.setCurrentRow(0)

    def _selected_style(self) -> ResponseStyle | None:
        item = self._list.currentItem()
        if item is None:
            return None
        raw_id = item.data(Qt.UserRole)
        try:
            style_id = uuid.UUID(str(raw_id))
        except (TypeError, ValueError):
            return None
        for style in self._styles:
            if style.id == style_id:
                return style
        try:
            return self._manager.get_response_style(style_id)
        except ResponseStyleError:
            return None

    def _on_selection_changed(self) -> None:
        style = self._selected_style()
        enabled = style is not None
        self._edit_button.setEnabled(enabled)
        self._delete_button.setEnabled(enabled)
        self._copy_button.setEnabled(enabled)
        self._markdown_button.setEnabled(enabled)
        self._detail_view.setPlainText(self._format_style(style))

    def _format_style(self, style: ResponseStyle | None) -> str:
        if style is None:
            return ""
        lines = [
            f"Name: {style.name}",
            f"Prompt part: {style.prompt_part}",
            f"Description:\n{style.description or 'n/a'}",
            "",
        ]
        lines.append(f"Tone: {style.tone or 'n/a'}")
        lines.append(f"Voice: {style.voice or 'n/a'}")
        if style.tags:
            lines.append(f"Tags: {', '.join(style.tags)}")
        if style.examples:
            lines.append("Examples:")
            for example in style.examples:
                lines.append(f"  - {example}")
        if style.format_instructions:
            lines.append("")
            lines.append("Format instructions:")
            lines.append(style.format_instructions)
        if style.guidelines:
            lines.append("")
            lines.append("Guidelines:")
            lines.append(style.guidelines)
        return "\n".join(lines)

    def _on_new_clicked(self) -> None:
        dialog = ResponseStyleDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        result = dialog.result_style
        if result is None:
            return
        try:
            self._manager.create_response_style(result)
        except ResponseStyleError as exc:
            QMessageBox.critical(self, "Unable to create prompt part", str(exc))
            return
        self._load_styles()
        self._show_status("Prompt part created.")

    def _on_edit_clicked(self) -> None:
        style = self._selected_style()
        if style is None:
            QMessageBox.information(self, "Edit prompt part", "Select a prompt part first.")
            return
        dialog = ResponseStyleDialog(self, style=style)
        if dialog.exec() != QDialog.Accepted:
            return
        result = dialog.result_style
        if result is None:
            return
        try:
            self._manager.update_response_style(result)
        except ResponseStyleError as exc:
            QMessageBox.critical(self, "Unable to update prompt part", str(exc))
            return
        self._load_styles()
        self._show_status("Prompt part updated.")

    def _on_delete_clicked(self) -> None:
        style = self._selected_style()
        if style is None:
            QMessageBox.information(self, "Delete prompt part", "Select a prompt part first.")
            return
        confirmation = QMessageBox.question(
            self,
            "Delete prompt part",
            f"Delete prompt part '{style.name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirmation != QMessageBox.Yes:
            return
        try:
            self._manager.delete_response_style(style.id)
        except ResponseStyleError as exc:
            QMessageBox.critical(self, "Unable to delete prompt part", str(exc))
            return
        self._load_styles()
        self._show_status("Prompt part deleted.")

    def _on_copy_clicked(self) -> None:
        style = self._selected_style()
        if style is None:
            QMessageBox.information(self, "Copy prompt part", "Select a prompt part first.")
            return
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(self._format_style(style))
        self._show_toast("Prompt part copied to clipboard.")

    def _on_markdown_clicked(self) -> None:
        style = self._selected_style()
        if style is None:
            QMessageBox.information(self, "Markdown preview", "Select a prompt part first.")
            return
        content = self._format_style(style)
        if not content.strip():
            QMessageBox.information(self, "Markdown preview", "The selected prompt part is empty.")
            return
        dialog = MarkdownPreviewDialog(content, self, title="Prompt Part Preview")
        dialog.exec()

    def _on_export_clicked(self) -> None:
        if not self._styles:
            QMessageBox.information(self, "Export prompt parts", "There are no prompt parts to export yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export prompt parts",
            "prompt_parts.txt",
            "Text Files (*.txt);;All Files (*.*)",
        )
        if not path:
            return
        lines = []
        for style in self._styles:
            lines.append("---")
            lines.append(f"Name: {style.name}")
            lines.append(f"Prompt part: {style.prompt_part}")
            lines.append(f"Description: {style.description}")
            lines.append(f"Tone: {style.tone or 'n/a'}")
            lines.append(f"Voice: {style.voice or 'n/a'}")
            if style.tags:
                lines.append(f"Tags: {', '.join(style.tags)}")
            if style.examples:
                lines.append("Examples:")
                for example in style.examples:
                    lines.append(f"  - {example}")
            if style.format_instructions:
                lines.append("Format:")
                lines.append(style.format_instructions)
            if style.guidelines:
                lines.append("Guidelines:")
                lines.append(style.guidelines)
            lines.append("")
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines).strip() + "\n")
        except OSError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self._show_status(f"Exported {len(self._styles)} prompt parts.")

    def _show_status(self, message: str, duration_ms: int = 3000) -> None:
        if self._status_callback is not None:
            self._status_callback(message, duration_ms)

    def _show_toast(self, message: str, duration_ms: int = 2500) -> None:
        if self._toast_callback is not None:
            self._toast_callback(message, duration_ms)
        else:
            self._show_status(message, duration_ms)


__all__ = ["ResponseStylesPanel"]
