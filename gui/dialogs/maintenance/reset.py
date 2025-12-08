"""Data reset helpers for the maintenance dialog.

Updates:
  v0.1.0 - 2025-12-04 - Extract snapshot and reset workflows into a mixin.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core import PromptManagerError


class ResetMaintenanceMixin:
    """Provide snapshot creation and destructive reset routines."""

    _manager: Any
    _reset_log_view: QPlainTextEdit

    def _build_reset_tab(self, parent: QWidget) -> QWidget:
        reset_tab = QWidget(parent)
        reset_layout = QVBoxLayout(reset_tab)

        reset_intro = QLabel(
            (
                "Use these actions to clear application data while leaving configuration "
                "and settings untouched."
            ),
            reset_tab,
        )
        reset_intro.setWordWrap(True)
        reset_layout.addWidget(reset_intro)

        reset_warning = QLabel(
            (
                "<b>Warning:</b> these operations permanently delete existing prompts, "
                "histories, and embeddings."
            ),
            reset_tab,
        )
        reset_warning.setWordWrap(True)
        reset_layout.addWidget(reset_warning)

        reset_buttons_container = QWidget(reset_tab)
        reset_buttons_layout = QVBoxLayout(reset_buttons_container)
        reset_buttons_layout.setContentsMargins(0, 0, 0, 0)
        reset_buttons_layout.setSpacing(8)

        snapshot_button = QPushButton("Create Backup Snapshot", reset_buttons_container)
        snapshot_button.setToolTip(
            "Zip the SQLite database, Chroma store, and manifest before running resets."
        )
        snapshot_button.clicked.connect(self._on_snapshot_clicked)  # type: ignore[arg-type]
        reset_buttons_layout.addWidget(snapshot_button)

        reset_sqlite_button = QPushButton("Clear Prompt Database", reset_buttons_container)
        reset_sqlite_button.setToolTip("Delete all prompts and execution history from SQLite.")
        reset_sqlite_button.clicked.connect(self._on_reset_prompts_clicked)  # type: ignore[arg-type]
        reset_buttons_layout.addWidget(reset_sqlite_button)

        reset_chroma_button = QPushButton("Clear Embedding Store", reset_buttons_container)
        reset_chroma_button.setToolTip(
            "Remove all vectors from the ChromaDB collection used for semantic search."
        )
        reset_chroma_button.clicked.connect(self._on_reset_chroma_clicked)  # type: ignore[arg-type]
        reset_buttons_layout.addWidget(reset_chroma_button)

        reset_all_button = QPushButton("Reset Application Data", reset_buttons_container)
        reset_all_button.setToolTip(
            "Clear prompts, histories, embeddings, and usage logs in one step. "
            "Settings remain unchanged."
        )
        reset_all_button.clicked.connect(self._on_reset_application_clicked)  # type: ignore[arg-type]
        reset_buttons_layout.addWidget(reset_all_button)

        reset_layout.addWidget(reset_buttons_container)

        self._reset_log_view = QPlainTextEdit(reset_tab)
        self._reset_log_view.setReadOnly(True)
        reset_layout.addWidget(self._reset_log_view, stretch=1)

        return reset_tab

    def _append_reset_log(self, message: str) -> None:
        timestamp = datetime.now(UTC).strftime("%H:%M:%S")
        self._reset_log_view.appendPlainText(f"[{timestamp}] {message}")

    def _confirm_destructive_action(self, prompt: str) -> bool:
        result = QMessageBox.question(
            self,
            "Confirm Data Reset",
            f"{prompt}\n\nThis action cannot be undone. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return result == QMessageBox.StandardButton.Yes

    def _on_snapshot_clicked(self) -> None:
        default_name = datetime.now(UTC).strftime("prompt-manager-snapshot-%Y%m%d-%H%M%S.zip")
        default_path = str(Path.home() / default_name)
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Maintenance Snapshot",
            default_path,
            "Zip Archives (*.zip)",
        )
        if not file_path:
            return
        destination = Path(file_path)

        self._append_reset_log("Creating maintenance snapshot…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            archive_path = self._manager.create_data_snapshot(destination)
        except PromptManagerError as exc:
            QMessageBox.critical(self, "Snapshot failed", str(exc))
            self._append_reset_log(f"Snapshot failed: {exc}")
        else:
            self._append_reset_log(f"Snapshot saved to {archive_path}")
            QMessageBox.information(
                self,
                "Snapshot created",
                f"Backup stored at:\n{archive_path}",
            )
        finally:
            QApplication.restoreOverrideCursor()

    def _on_reset_prompts_clicked(self) -> None:
        if not self._confirm_destructive_action(
            "Clear the prompt database and execution history?",
        ):
            return
        try:
            self._manager.reset_prompt_repository()
        except PromptManagerError as exc:
            QMessageBox.critical(self, "Reset failed", str(exc))
            self._append_reset_log(f"Prompt database reset failed: {exc}")
            return
        self._append_reset_log("Prompt database cleared.")
        QMessageBox.information(
            self,
            "Prompt database cleared",
            "All prompts and execution history have been removed.",
        )
        self._refresh_catalogue_stats()
        self._refresh_storage_info()
        self.maintenance_applied.emit("Prompt database cleared.")

    def _on_reset_chroma_clicked(self) -> None:
        if not self._confirm_destructive_action(
            "Remove all embeddings from the ChromaDB vector store?",
        ):
            return
        try:
            self._manager.reset_vector_store()
        except PromptManagerError as exc:
            QMessageBox.critical(self, "Reset failed", str(exc))
            self._append_reset_log(f"Embedding store reset failed: {exc}")
            return
        self._append_reset_log("Chroma vector store cleared.")
        QMessageBox.information(
            self,
            "Embedding store cleared",
            "All stored embeddings have been removed.",
        )
        self._refresh_chroma_info()
        self.maintenance_applied.emit("Embedding store cleared.")

    def _on_reset_application_clicked(self) -> None:
        if not self._confirm_destructive_action(
            "Reset all application data (prompts, history, embeddings, and logs)?",
        ):
            return
        try:
            self._manager.reset_application_data(clear_logs=True)
        except PromptManagerError as exc:
            QMessageBox.critical(self, "Reset failed", str(exc))
            self._append_reset_log(f"Application reset failed: {exc}")
            return
        self._append_reset_log("Application data reset completed.")
        QMessageBox.information(
            self,
            "Application data reset",
            "Prompt data, embeddings, and usage logs have been cleared.",
        )
        self._refresh_catalogue_stats()
        self._refresh_storage_info()
        self._refresh_chroma_info()
        self.maintenance_applied.emit("Application data reset.")
