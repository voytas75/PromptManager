"""Execution history panel for Prompt Manager.

Updates: v0.3.0 - 2025-11-12 - Display chat conversations alongside execution details.
Updates: v0.2.0 - 2025-11-09 - Surface execution ratings in tables, details, and exports.
Updates: v0.1.0 - 2025-11-08 - Provide filterable, editable execution history workspace.
"""

from __future__ import annotations

import csv
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core import (
    PromptHistoryError,
    PromptManager,
    PromptManagerError,
    PromptNotFoundError,
)
from models.prompt_model import ExecutionStatus, PromptExecution

from .dialogs import SaveResultDialog


@dataclass(slots=True)
class _ExecutionRow:
    execution: PromptExecution
    prompt_name: str


class HistoryPanel(QWidget):
    """Filterable, editable prompt execution history pane."""

    def __init__(
        self,
        manager: PromptManager,
        parent: QWidget | None = None,
        *,
        limit: int = 100,
        on_note_updated: Callable[[uuid.UUID, str], None] | None = None,
        on_export: Callable[[int, str], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._manager = manager
        self._limit = max(1, limit)
        self._note_callback = on_note_updated
        self._export_callback = on_export
        self._rows: list[_ExecutionRow] = []
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        header = QLabel(
            "History of prompt executions. Filter by status, prompt, or search across request, response, and notes.",
            self,
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(8)

        self._status_filter = QComboBox(self)
        self._status_filter.addItem("All statuses", None)
        for status in ExecutionStatus:
            self._status_filter.addItem(status.value.title(), status.value)
        self._status_filter.currentIndexChanged.connect(self.refresh)  # type: ignore[arg-type]
        filter_layout.addWidget(QLabel("Status:", self))
        filter_layout.addWidget(self._status_filter)

        self._prompt_filter = QComboBox(self)
        self._prompt_filter.addItem("All prompts", None)
        self._populate_prompt_filter()
        self._prompt_filter.currentIndexChanged.connect(self.refresh)  # type: ignore[arg-type]
        filter_layout.addWidget(QLabel("Prompt:", self))
        filter_layout.addWidget(self._prompt_filter)

        self._search_input = QLineEdit(self)
        self._search_input.setPlaceholderText("Search text...")
        self._search_input.setClearButtonEnabled(True)
        self._search_input.returnPressed.connect(self.refresh)  # type: ignore[arg-type]
        filter_layout.addWidget(QLabel("Search:", self))
        filter_layout.addWidget(self._search_input)

        filter_layout.addStretch(1)

        self._export_button = QPushButton("Export CSV", self)
        self._export_button.clicked.connect(self._export_history)  # type: ignore[arg-type]
        filter_layout.addWidget(self._export_button)

        layout.addLayout(filter_layout)

        self._table = QTableWidget(self)
        self._table.setColumnCount(7)
        self._table.setHorizontalHeaderLabels(
            ["Prompt", "Status", "Rating", "Duration (ms)", "Executed At", "Error", "Note"]
        )
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSortingEnabled(True)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)  # type: ignore[arg-type]
        layout.addWidget(self._table, stretch=1)

        detail_layout = QVBoxLayout()
        self._detail_label = QLabel("Select an execution to view details.", self)
        self._detail_label.setObjectName("historyDetailLabel")
        self._detail_view = QPlainTextEdit(self)
        self._detail_view.setReadOnly(True)
        detail_layout.addWidget(self._detail_label)
        detail_layout.addWidget(self._detail_view, stretch=1)
        layout.addLayout(detail_layout)

        footer = QHBoxLayout()
        self._summary_label = QLabel("", self)
        footer.addWidget(self._summary_label)
        footer.addStretch(1)
        self._edit_button = QPushButton("Edit Note", self)
        self._edit_button.clicked.connect(self._on_edit_note_clicked)  # type: ignore[arg-type]
        footer.addWidget(self._edit_button)
        self._refresh_button = QPushButton("Refresh", self)
        self._refresh_button.clicked.connect(self.refresh)  # type: ignore[arg-type]
        footer.addWidget(self._refresh_button)
        layout.addLayout(footer)

    def _populate_prompt_filter(self) -> None:
        self._prompt_filter.blockSignals(True)
        current_data = self._prompt_filter.currentData()
        self._prompt_filter.clear()
        self._prompt_filter.addItem("All prompts", None)
        try:
            prompts = self._manager.repository.list()
        except PromptManagerError:
            prompts = []
        for prompt in prompts:
            self._prompt_filter.addItem(prompt.name, prompt.id)
        if current_data is not None:
            index = self._prompt_filter.findData(current_data)
            self._prompt_filter.setCurrentIndex(index if index >= 0 else 0)
        self._prompt_filter.blockSignals(False)

    def refresh(self) -> None:
        """Reload executions with current filters."""

        self._populate_prompt_filter()
        status_value = self._status_filter.currentData()
        prompt_id = self._prompt_filter.currentData()
        search_term = self._search_input.text().strip()
        if search_term == "":
            search_term = None

        entries = self._manager.query_executions(
            status=status_value,
            prompt_id=prompt_id,
            search=search_term,
            limit=self._limit,
        )

        rows: list[_ExecutionRow] = []
        for entry in entries:
            name = self._resolve_prompt_name(entry.prompt_id)
            rows.append(_ExecutionRow(entry, name))
        self._rows = rows
        self._populate_table()
        total = len(rows)
        if total:
            self._summary_label.setText(f"{total} executions shown (limit {self._limit}).")
            self._table.selectRow(0)
        else:
            self._summary_label.setText("No executions match the current filters.")
            self._detail_view.clear()
            self._detail_label.setText("Select an execution to view details.")

    def _populate_table(self) -> None:
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(self._rows))
        for row_index, row in enumerate(self._rows):
            execution = row.execution
            executed_at_display = self._format_timestamp(execution.executed_at)
            duration_text = (
                str(execution.duration_ms) if execution.duration_ms is not None else "n/a"
            )
            error_preview = execution.error_message or ""
            note_preview = ""
            if execution.metadata:
                note_preview = str(execution.metadata.get("note") or "")
                if len(note_preview) > 80:
                    note_preview = note_preview[:77] + "..."
            rating_text = (
                f"{execution.rating:.1f}" if execution.rating is not None else "-"
            )
            values = [
                row.prompt_name,
                execution.status.value.title(),
                rating_text,
                duration_text,
                executed_at_display,
                error_preview[:120],
                note_preview,
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col == 1:
                    item.setData(Qt.UserRole, execution.status.value)
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                self._table.setItem(row_index, col, item)
        self._table.resizeColumnsToContents()
        self._table.setSortingEnabled(True)

    def _on_selection_changed(self) -> None:
        indexes = self._table.selectionModel().selectedRows()
        if not indexes:
            self._detail_view.clear()
            self._detail_label.setText("Select an execution to view details.")
            return

        row = self._rows[indexes[0].row()]
        execution = row.execution
        timestamp = self._format_timestamp(execution.executed_at)
        detail_lines = [
            f"Prompt: {row.prompt_name}",
            f"Status: {execution.status.value.title()}",
            f"Executed: {timestamp}",
        ]
        if execution.duration_ms is not None:
            detail_lines.append(f"Duration: {execution.duration_ms} ms")
        if execution.rating is not None:
            detail_lines.append(f"Rating: {execution.rating:.1f}/10")
        note_text = ""
        if execution.metadata:
            note_text = str(execution.metadata.get("note") or "")
            if note_text:
                detail_lines.append(f"Note: {note_text}")
        detail_lines.append("")
        detail_lines.append("Request:")
        detail_lines.append(execution.request_text or "(empty)")
        detail_lines.append("")
        if execution.error_message:
            detail_lines.append("Error:")
            detail_lines.append(execution.error_message)
            detail_lines.append("")
        detail_lines.append("Response:")
        detail_lines.append(execution.response_text or "(empty)")
        conversation_lines: list[str] = []
        if execution.metadata:
            raw_conversation = execution.metadata.get("conversation")
            if isinstance(raw_conversation, list):
                conversation_lines = self._format_conversation(raw_conversation)
        if conversation_lines:
            detail_lines.append("")
            detail_lines.append("Conversation:")
            detail_lines.append("")
            detail_lines.extend(conversation_lines)
        self._detail_label.setText(f"Execution detail — {row.prompt_name}")
        self._detail_view.setPlainText("\n".join(detail_lines))

    def _resolve_prompt_name(self, prompt_id: uuid.UUID) -> str:
        try:
            prompt = self._manager.get_prompt(prompt_id)
            return prompt.name
        except (PromptNotFoundError, PromptManagerError):
            return str(prompt_id)
        except Exception:
            return str(prompt_id)

    @staticmethod
    def _format_timestamp(value: datetime) -> str:
        return value.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    def _selected_row(self) -> int | None:
        indexes = self._table.selectionModel().selectedRows()
        if not indexes:
            return None
        return indexes[0].row()

    def _on_edit_note_clicked(self) -> None:
        row_index = self._selected_row()
        if row_index is None:
            QMessageBox.information(self, "Select execution", "Select a history entry to edit.")
            return
        entry = self._rows[row_index]
        current_note = ""
        if entry.execution.metadata:
            current_note = str(entry.execution.metadata.get("note") or "")
        dialog = SaveResultDialog(
            self,
            prompt_name=entry.prompt_name,
            default_text=current_note,
            button_text="Update",
            enable_rating=False,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        note = dialog.note
        try:
            updated_execution = self._manager.update_execution_note(entry.execution.id, note)
        except (PromptHistoryError, PromptManagerError) as exc:
            QMessageBox.warning(self, "Unable to update note", str(exc))
            return
        self._rows[row_index] = _ExecutionRow(updated_execution, entry.prompt_name)
        note_preview = note if len(note) <= 80 else note[:77] + "..."
        self._table.item(row_index, 6).setText(note_preview)
        self._on_selection_changed()
        if self._note_callback:
            self._note_callback(entry.execution.id, note)

    def _export_history(self) -> None:
        if not self._rows:
            QMessageBox.information(self, "Nothing to export", "There are no rows to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export execution history",
            str(Path.cwd() / "execution_history.csv"),
            "CSV Files (*.csv)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "prompt",
                        "status",
                        "rating",
                        "duration_ms",
                        "executed_at",
                        "error",
                        "note",
                        "request",
                        "response",
                    ]
                )
                for row in self._rows:
                    execution = row.execution
                    note = ""
                    if execution.metadata:
                        note = str(execution.metadata.get("note") or "")
                    writer.writerow(
                            [
                                row.prompt_name,
                                execution.status.value,
                                execution.rating if execution.rating is not None else "",
                                execution.duration_ms if execution.duration_ms is not None else "",
                                execution.executed_at.isoformat(),
                                execution.error_message or "",
                                note,
                                execution.request_text,
                            execution.response_text,
                        ]
                    )
        except OSError as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        if self._export_callback:
            self._export_callback(len(self._rows), path)

    def row_count(self) -> int:
        """Return the number of rows currently displayed."""

        return len(self._rows)

    @staticmethod
    def _format_conversation(messages: list[Any]) -> list[str]:
        """Convert stored chat messages into plain text lines."""

        lines: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content", "") or "")
            if role == "user":
                speaker = "You"
            elif role == "assistant":
                speaker = "Assistant"
            elif role == "system":
                speaker = "System"
            else:
                speaker = str(message.get("role", "Message"))
            lines.append(f"{speaker}:")
            if content:
                lines.extend(content.splitlines())
            else:
                lines.append("(empty)")
            lines.append("")
        if lines and lines[-1] == "":
            lines.pop()
        return lines
