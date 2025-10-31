"""Execution history dialog for Prompt Manager.

Updates: v0.1.0 - 2025-11-08 - Provide read-only view of recent prompt executions.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core import PromptManager, PromptManagerError, PromptNotFoundError
from models.prompt_model import PromptExecution


@dataclass(slots=True)
class _ExecutionRow:
    execution: PromptExecution
    prompt_name: str


class ExecutionHistoryDialog(QDialog):
    """Display recent prompt executions with request/response detail."""

    def __init__(
        self,
        manager: PromptManager,
        parent: Optional[QWidget] = None,
        *,
        limit: int = 50,
    ) -> None:
        super().__init__(parent)
        self._manager = manager
        self._limit = max(1, limit)
        self._rows: List[_ExecutionRow] = []
        self.setWindowTitle("Execution History")
        self.resize(760, 520)
        self._build_ui()
        self._load_entries()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        header = QLabel(
            "Recent prompt executions (most recent first). Select a row to view request and response bodies.",
            self,
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        self._table = QTableWidget(self)
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(["Prompt", "Status", "Duration (ms)", "Executed At", "Error"])
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)  # type: ignore[arg-type]
        self._table.horizontalHeader().setStretchLastSection(True)
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
        refresh_button = QPushButton("Refresh", self)
        refresh_button.clicked.connect(self._load_entries)  # type: ignore[arg-type]
        footer.addWidget(refresh_button)
        layout.addLayout(footer)

    def _load_entries(self) -> None:
        entries = self._manager.list_recent_executions(limit=self._limit)
        rows: List[_ExecutionRow] = []
        for entry in entries:
            prompt_name = self._resolve_prompt_name(entry.prompt_id)
            rows.append(_ExecutionRow(execution=entry, prompt_name=prompt_name))
        self._rows = rows
        self._populate_table()
        if rows:
            self._summary_label.setText(f"{len(rows)} executions shown (limit {self._limit}).")
        else:
            self._summary_label.setText("No executions recorded yet.")
        if self._rows:
            self._table.selectRow(0)
        else:
            self._detail_view.clear()
            self._detail_label.setText("Select an execution to view details.")

    def _populate_table(self) -> None:
        self._table.setRowCount(len(self._rows))
        for row_index, row in enumerate(self._rows):
            execution = row.execution
            executed_at_display = self._format_timestamp(execution.executed_at)
            duration_text = str(execution.duration_ms) if execution.duration_ms is not None else "n/a"
            error_preview = execution.error_message or ""
            values = [
                row.prompt_name,
                execution.status.value.title(),
                duration_text,
                executed_at_display,
                error_preview[:120],
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col == 1:
                    item.setData(Qt.UserRole, execution.status.value)
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                self._table.setItem(row_index, col, item)
        self._table.resizeColumnsToContents()

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


__all__ = ["ExecutionHistoryDialog"]
