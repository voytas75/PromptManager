"""Dialog for viewing and restoring prompt version history.

Updates:
  v0.1.0 - 2025-12-03 - Extract version history dialog from gui.dialogs.
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core import PromptManager, PromptManagerError

try:
    from ..toast import show_toast
except ImportError:  # pragma: no cover - fallback when loaded outside package
    from gui.toast import show_toast

StatusReporter = Callable[[str, int], None]


def _default_status_reporter(_message: str, _duration: int = 0) -> None:
    return None


if TYPE_CHECKING:
    from models.prompt_model import Prompt, PromptVersion


class PromptVersionHistoryDialog(QDialog):
    """Display committed prompt versions with diff/restore controls."""

    _BODY_PLACEHOLDER = "Select a version to view the prompt body."
    _EMPTY_BODY_TEXT = "Prompt body is empty."

    def __init__(
        self,
        manager: PromptManager,
        prompt: Prompt,
        parent: QWidget | None = None,
        *,
        status_callback: Callable[[str, int], None] | None = None,
        limit: int = 200,
    ) -> None:
        """Create the history dialog for *prompt* with optional status callbacks."""
        super().__init__(parent)
        self._manager = manager
        self._prompt = prompt
        self._limit = max(1, limit)
        self._status_callback: StatusReporter = status_callback or _default_status_reporter
        self._versions: list[PromptVersion] = []
        self.last_restored_prompt: Prompt | None = None
        self.setWindowTitle(f"{prompt.name} – Version History")
        self.resize(820, 520)
        self._build_ui()
        self._load_versions()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        header = QLabel(
            (
                "Every save creates a version. Select an entry to inspect the snapshot, "
                "diff against the prior version, or restore it."
            ),
            self,
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        self._table = QTableWidget(self)
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["Version", "Created", "Message"])
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        header_view = self._table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header_view.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header_view.setSectionResizeMode(2, QHeaderView.Stretch)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)  # type: ignore[arg-type]
        layout.addWidget(self._table, stretch=1)

        self._tab_widget = QTabWidget(self)
        self._body_view = QPlainTextEdit(self)
        self._body_view.setReadOnly(True)
        self._tab_widget.addTab(self._body_view, "Prompt body")
        self._diff_view = QPlainTextEdit(self)
        self._diff_view.setReadOnly(True)
        self._tab_widget.addTab(self._diff_view, "Diff vs previous")
        self._current_diff_view = QPlainTextEdit(self)
        self._current_diff_view.setReadOnly(True)
        self._tab_widget.addTab(self._current_diff_view, "Diff vs current")
        self._snapshot_view = QPlainTextEdit(self)
        self._snapshot_view.setReadOnly(True)
        self._tab_widget.addTab(self._snapshot_view, "Snapshot JSON")
        self._tab_widget.setCurrentIndex(0)
        layout.addWidget(self._tab_widget, stretch=1)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        refresh_button = QPushButton("Refresh", self)
        refresh_button.clicked.connect(self._load_versions)  # type: ignore[arg-type]
        button_row.addWidget(refresh_button)

        copy_snapshot_button = QPushButton("Copy Snapshot", self)
        copy_snapshot_button.clicked.connect(self._copy_snapshot_to_clipboard)  # type: ignore[arg-type]
        button_row.addWidget(copy_snapshot_button)

        copy_body_button = QPushButton("Copy Prompt Body", self)
        copy_body_button.clicked.connect(self._copy_body_to_clipboard)  # type: ignore[arg-type]
        button_row.addWidget(copy_body_button)

        restore_button = QPushButton("Restore Version", self)
        restore_button.clicked.connect(self._on_restore_clicked)  # type: ignore[arg-type]
        button_row.addWidget(restore_button)

        close_button = QPushButton("Close", self)
        close_button.clicked.connect(self.reject)  # type: ignore[arg-type]
        button_row.addWidget(close_button)
        layout.addLayout(button_row)

    def _load_versions(self) -> None:
        try:
            versions = self._manager.list_prompt_versions(self._prompt.id, limit=self._limit)
        except PromptManagerError as exc:
            QMessageBox.critical(self, "Unable to load versions", str(exc))
            versions = []
        self._versions = versions
        self._table.setRowCount(len(versions))
        for row, version in enumerate(versions):
            timestamp = self._format_timestamp(version.created_at)
            self._table.setItem(row, 0, QTableWidgetItem(f"v{version.version_number}"))
            self._table.setItem(row, 1, QTableWidgetItem(timestamp))
            message = version.commit_message or "Auto-snapshot"
            self._table.setItem(row, 2, QTableWidgetItem(message))
        if versions:
            self._table.selectRow(0)
        else:
            self._body_view.setPlainText(self._BODY_PLACEHOLDER)
            empty_message = "No versions have been recorded for this prompt yet."
            self._diff_view.setPlainText(empty_message)
            self._current_diff_view.setPlainText(empty_message)
            self._snapshot_view.clear()

    def _on_selection_changed(self) -> None:
        version = self._selected_version()
        if version is None:
            self._body_view.setPlainText(self._BODY_PLACEHOLDER)
            self._diff_view.clear()
            self._current_diff_view.clear()
            self._snapshot_view.clear()
            return
        snapshot_text = json.dumps(version.snapshot, ensure_ascii=False, indent=2)
        self._snapshot_view.setPlainText(snapshot_text)

        body_text = self._body_text_for_version(version)
        self._body_view.setPlainText(body_text)
        self._current_diff_view.setPlainText(self._format_diff_against_current(version, body_text))

        previous_version = self._previous_version(version)
        if previous_version is None:
            self._diff_view.setPlainText("No previous version available for diffing.")
            return
        try:
            diff = self._manager.diff_prompt_versions(previous_version.id, version.id)
        except PromptManagerError as exc:
            self._diff_view.setPlainText(f"Unable to compute diff: {exc}")
            return
        diff_text = diff.body_diff or json.dumps(diff.changed_fields, ensure_ascii=False, indent=2)
        if not diff_text.strip():
            diff_text = "Snapshots are identical."
        self._diff_view.setPlainText(diff_text)

    def _selected_version(self) -> PromptVersion | None:
        selection = self._table.selectionModel()
        if selection is None:
            return None
        indexes = selection.selectedRows()
        if not indexes:
            return None
        row = int(indexes[0].row())
        if 0 <= row < len(self._versions):
            return self._versions[row]
        return None

    def _previous_version(self, version: PromptVersion) -> PromptVersion | None:
        try:
            current_index = self._versions.index(version)
        except ValueError:
            return None
        next_index = current_index + 1
        if next_index < len(self._versions):
            return self._versions[next_index]
        return None

    def _body_text_for_version(self, version: PromptVersion) -> str:
        """Return the prompt body stored in the snapshot or a placeholder."""
        raw_body = version.snapshot.get("context")
        if isinstance(raw_body, str) and raw_body.strip():
            return raw_body
        if raw_body:
            return str(raw_body)
        return self._EMPTY_BODY_TEXT

    def _current_prompt_body(self) -> str:
        raw_body = getattr(self._prompt, "context", None)
        if isinstance(raw_body, str):
            return raw_body
        if raw_body:
            return str(raw_body)
        return ""

    def _format_diff_against_current(self, version: PromptVersion, version_body: str) -> str:
        current_body = self._current_prompt_body()
        current_text = current_body.strip()
        version_text = version_body.strip()
        if not current_text and not version_text:
            return "Current prompt and selected version bodies are empty."
        if current_body == version_body:
            return "Version body matches the current prompt."
        current_lines = current_body.splitlines()
        version_lines = version_body.splitlines()
        diff = difflib.unified_diff(
            current_lines,
            version_lines,
            fromfile="Current prompt",
            tofile=f"Version v{version.version_number}",
            lineterm="",
        )
        diff_text = "\n".join(diff)
        if not diff_text.strip():
            return "Version body matches the current prompt."
        return diff_text

    def _copy_snapshot_to_clipboard(self) -> None:
        version = self._selected_version()
        if version is None:
            return
        snapshot_text = json.dumps(version.snapshot, ensure_ascii=False, indent=2)
        QGuiApplication.clipboard().setText(snapshot_text)
        self._status_callback("Snapshot copied to clipboard", 2000)
        show_toast(self, "Snapshot copied to clipboard.")

    def _copy_body_to_clipboard(self) -> None:
        """Copy the selected prompt version body to the clipboard."""
        version = self._selected_version()
        if version is None:
            return
        body_text = self._body_text_for_version(version)
        QGuiApplication.clipboard().setText(body_text)
        self._status_callback("Prompt body copied to clipboard", 2000)
        show_toast(self, "Prompt body copied to clipboard.")

    def _on_restore_clicked(self) -> None:
        version = self._selected_version()
        if version is None:
            return
        confirm = QMessageBox.question(
            self,
            "Restore Version",
            "This will replace the current prompt contents with the selected snapshot. Continue?",
        )
        if confirm != QMessageBox.Yes:
            return
        try:
            restored = self._manager.restore_prompt_version(version.id)
        except PromptManagerError as exc:
            QMessageBox.critical(self, "Restore failed", str(exc))
            return
        self.last_restored_prompt = restored
        self._status_callback("Prompt restored to selected version", 4000)
        self._load_versions()

    @staticmethod
    def _format_timestamp(value: datetime) -> str:
        timestamp = value.astimezone(UTC)
        return timestamp.strftime("%Y-%m-%d %H:%M UTC")


__all__ = ["PromptVersionHistoryDialog"]
