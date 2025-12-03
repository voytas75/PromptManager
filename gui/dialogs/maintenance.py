"""Maintenance dialogs and utilities for prompt catalogue hygiene.

Updates:
  v0.1.0 - 2025-12-03 - Split maintenance dialog from the dialogs monolith.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QEvent, QSettings, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core import PromptManager, PromptManagerError, RepositoryError

from .base import logger

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from models.prompt_model import Prompt


class PromptMaintenanceDialog(QDialog):
    """Expose bulk metadata maintenance utilities."""

    maintenance_applied = Signal(str)

    def __init__(
        self,
        parent: QWidget,
        manager: PromptManager,
        *,
        category_generator: Callable[[str], str] | None = None,
        tags_generator: Callable[[str], Sequence[str]] | None = None,
    ) -> None:
        """Create the maintenance dialog with optional LiteLLM helpers."""
        super().__init__(parent)
        self._manager = manager
        self._category_generator = category_generator
        self._tags_generator = tags_generator
        self._stats_labels: dict[str, QLabel] = {}
        self._log_view: QPlainTextEdit
        self._redis_status_label: QLabel
        self._redis_connection_label: QLabel
        self._redis_stats_view: QPlainTextEdit
        self._redis_refresh_button: QPushButton
        self._tab_widget: QTabWidget
        self._chroma_status_label: QLabel
        self._chroma_path_label: QLabel
        self._chroma_stats_view: QPlainTextEdit
        self._chroma_refresh_button: QPushButton
        self._chroma_compact_button: QPushButton
        self._chroma_optimize_button: QPushButton
        self._chroma_verify_button: QPushButton
        self._storage_status_label: QLabel
        self._storage_path_label: QLabel
        self._storage_stats_view: QPlainTextEdit
        self._storage_refresh_button: QPushButton
        self._sqlite_compact_button: QPushButton
        self._sqlite_optimize_button: QPushButton
        self._sqlite_verify_button: QPushButton
        self._stats_refresh_button: QPushButton
        self._category_table: QTableWidget
        self._category_refresh_button: QPushButton
        self._reset_log_view: QPlainTextEdit
        self._settings = QSettings("PromptManager", "PromptMaintenanceDialog")
        self.setWindowTitle("Prompt Maintenance")
        self.resize(780, 360)
        self._restore_window_size()
        self._build_ui()
        self._refresh_catalogue_stats()
        self._refresh_category_health()
        self._refresh_redis_info()
        self._refresh_chroma_info()
        self._refresh_storage_info()

    def _restore_window_size(self) -> None:
        """Resize the dialog using the last persisted geometry if available."""
        width = self._settings.value("width", type=int)
        height = self._settings.value("height", type=int)
        if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
            self.resize(width, height)

    def closeEvent(self, event: QEvent) -> None:  # type: ignore[override]
        """Persist the current window size before closing."""
        self._settings.setValue("width", self.width())
        self._settings.setValue("height", self.height())
        super().closeEvent(event)

    def _build_ui(self) -> None:
        outer_layout = QVBoxLayout(self)

        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer_layout.addWidget(scroll_area, stretch=1)

        scroll_contents = QWidget(self)
        scroll_area.setWidget(scroll_contents)
        layout = QVBoxLayout(scroll_contents)

        self._tab_widget = QTabWidget(scroll_contents)
        layout.addWidget(self._tab_widget, stretch=1)

        metadata_tab = QWidget(self)
        metadata_layout = QVBoxLayout(metadata_tab)

        stats_group = QGroupBox("Prompt Catalogue Overview", metadata_tab)
        stats_layout = QGridLayout(stats_group)
        stats_layout.setContentsMargins(12, 12, 12, 12)
        stats_layout.setHorizontalSpacing(16)
        stats_layout.setVerticalSpacing(6)

        stat_rows = [
            ("total_prompts", "Total prompts"),
            ("active_prompts", "Active prompts"),
            ("inactive_prompts", "Inactive prompts"),
            ("distinct_categories", "Distinct categories"),
            ("prompts_without_category", "Prompts without category"),
            ("distinct_tags", "Distinct tags"),
            ("prompts_without_tags", "Prompts without tags"),
            ("average_tags_per_prompt", "Average tags per prompt"),
            ("stale_prompts", "Stale prompts (> 4 weeks)"),
            ("last_modified_at", "Last prompt update"),
        ]

        for row_index, (key, label_text) in enumerate(stat_rows):
            label_widget = QLabel(label_text, stats_group)
            value_widget = QLabel("—", stats_group)
            value_widget.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            stats_layout.addWidget(label_widget, row_index, 0)
            stats_layout.addWidget(value_widget, row_index, 1)
            self._stats_labels[key] = value_widget

        stats_layout.setColumnStretch(0, 1)
        stats_layout.setColumnStretch(1, 0)

        self._stats_refresh_button = QPushButton("Refresh Overview", stats_group)
        self._stats_refresh_button.clicked.connect(self._refresh_catalogue_stats)  # type: ignore[arg-type]
        stats_layout.addWidget(
            self._stats_refresh_button,
            len(stat_rows),
            0,
            1,
            2,
            alignment=Qt.AlignRight,
        )

        helper_label = QLabel(
            "Stale prompts have not been updated in the last 30 days.",
            stats_group,
        )
        helper_label.setWordWrap(True)
        stats_layout.addWidget(helper_label, len(stat_rows) + 1, 0, 1, 2)

        metadata_layout.addWidget(stats_group)

        health_group = QGroupBox("Category Health", metadata_tab)
        health_layout = QVBoxLayout(health_group)
        health_layout.setContentsMargins(12, 12, 12, 12)
        health_layout.setSpacing(8)

        self._category_table = QTableWidget(0, 5, health_group)
        self._category_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._category_table.setSelectionMode(QTableWidget.NoSelection)
        self._category_table.setHorizontalHeaderLabels(
            [
                "Category",
                "Prompts",
                "Active",
                "Success rate",
                "Last executed",
            ]
        )
        self._category_table.horizontalHeader().setStretchLastSection(True)
        self._category_table.verticalHeader().setVisible(False)
        health_layout.addWidget(self._category_table)

        self._category_refresh_button = QPushButton("Refresh Category Health", health_group)
        self._category_refresh_button.clicked.connect(self._refresh_category_health)  # type: ignore[arg-type]
        health_layout.addWidget(self._category_refresh_button, alignment=Qt.AlignRight)

        metadata_layout.addWidget(health_group)

        description = QLabel(
            "Run maintenance tasks to enrich prompt metadata. Only prompts missing the "
            "selected metadata are updated.",
            metadata_tab,
        )
        description.setWordWrap(True)
        metadata_layout.addWidget(description)

        button_container = QWidget(metadata_tab)
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(12)

        self._categories_button = QPushButton("Generate Missing Categories", metadata_tab)
        self._categories_button.clicked.connect(self._on_generate_categories_clicked)  # type: ignore[arg-type]
        self._categories_button.setEnabled(self._category_generator is not None)
        if self._category_generator is None:
            self._categories_button.setToolTip("Category suggestions are unavailable.")
        button_layout.addWidget(self._categories_button)

        self._tags_button = QPushButton("Generate Missing Tags", metadata_tab)
        self._tags_button.clicked.connect(self._on_generate_tags_clicked)  # type: ignore[arg-type]
        self._tags_button.setEnabled(self._tags_generator is not None)
        if self._tags_generator is None:
            self._tags_button.setToolTip("Tag suggestions are unavailable.")
        button_layout.addWidget(self._tags_button)

        button_layout.addStretch(1)
        metadata_layout.addWidget(button_container)

        self._log_view = QPlainTextEdit(metadata_tab)
        self._log_view.setReadOnly(True)
        self._log_view.setMinimumHeight(120)
        self._log_view.setMaximumHeight(200)
        metadata_layout.addWidget(self._log_view)

        self._tab_widget.addTab(metadata_tab, "Metadata")

        redis_tab = QWidget(self)
        redis_layout = QVBoxLayout(redis_tab)

        redis_description = QLabel("Inspect the Redis cache used for prompt caching.", redis_tab)
        redis_description.setWordWrap(True)
        redis_layout.addWidget(redis_description)

        status_container = QWidget(redis_tab)
        status_layout = QHBoxLayout(status_container)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(12)

        self._redis_status_label = QLabel("Checking…", status_container)
        status_layout.addWidget(self._redis_status_label)

        self._redis_connection_label = QLabel("", status_container)
        self._redis_connection_label.setWordWrap(True)
        status_layout.addWidget(self._redis_connection_label, stretch=1)

        self._redis_refresh_button = QPushButton("Refresh", status_container)
        self._redis_refresh_button.clicked.connect(self._refresh_redis_info)  # type: ignore[arg-type]
        status_layout.addWidget(self._redis_refresh_button)

        redis_layout.addWidget(status_container)

        self._redis_stats_view = QPlainTextEdit(redis_tab)
        self._redis_stats_view.setReadOnly(True)
        redis_layout.addWidget(self._redis_stats_view, stretch=1)

        self._tab_widget.addTab(redis_tab, "Redis")

        chroma_tab = QWidget(self)
        chroma_layout = QVBoxLayout(chroma_tab)

        chroma_description = QLabel(
            "Review the ChromaDB vector store used for semantic search.", chroma_tab
        )
        chroma_description.setWordWrap(True)
        chroma_layout.addWidget(chroma_description)

        chroma_status_container = QWidget(chroma_tab)
        chroma_status_layout = QHBoxLayout(chroma_status_container)
        chroma_status_layout.setContentsMargins(0, 0, 0, 0)
        chroma_status_layout.setSpacing(12)

        self._chroma_status_label = QLabel("Checking…", chroma_status_container)
        chroma_status_layout.addWidget(self._chroma_status_label)

        self._chroma_path_label = QLabel("", chroma_status_container)
        self._chroma_path_label.setWordWrap(True)
        chroma_status_layout.addWidget(self._chroma_path_label, stretch=1)

        self._chroma_refresh_button = QPushButton("Refresh", chroma_status_container)
        self._chroma_refresh_button.clicked.connect(self._refresh_chroma_info)  # type: ignore[arg-type]
        chroma_status_layout.addWidget(self._chroma_refresh_button)

        chroma_layout.addWidget(chroma_status_container)

        self._chroma_stats_view = QPlainTextEdit(chroma_tab)
        self._chroma_stats_view.setReadOnly(True)
        chroma_layout.addWidget(self._chroma_stats_view, stretch=1)

        chroma_actions_container = QWidget(chroma_tab)
        chroma_actions_layout = QHBoxLayout(chroma_actions_container)
        chroma_actions_layout.setContentsMargins(0, 0, 0, 0)
        chroma_actions_layout.setSpacing(12)

        self._chroma_compact_button = QPushButton(
            "Compact Persistent Store",
            chroma_actions_container,
        )
        self._chroma_compact_button.setToolTip(
            "Reclaim disk space by vacuuming the Chroma SQLite store."
        )
        self._chroma_compact_button.clicked.connect(self._on_chroma_compact_clicked)  # type: ignore[arg-type]
        chroma_actions_layout.addWidget(self._chroma_compact_button)

        self._chroma_optimize_button = QPushButton(
            "Optimize Persistent Store",
            chroma_actions_container,
        )
        self._chroma_optimize_button.setToolTip(
            "Refresh query statistics to improve Chroma performance."
        )
        self._chroma_optimize_button.clicked.connect(self._on_chroma_optimize_clicked)  # type: ignore[arg-type]
        chroma_actions_layout.addWidget(self._chroma_optimize_button)

        self._chroma_verify_button = QPushButton(
            "Verify Index Integrity",
            chroma_actions_container,
        )
        self._chroma_verify_button.setToolTip(
            "Run integrity checks against the Chroma index files."
        )
        self._chroma_verify_button.clicked.connect(self._on_chroma_verify_clicked)  # type: ignore[arg-type]
        chroma_actions_layout.addWidget(self._chroma_verify_button)

        chroma_actions_layout.addStretch(1)

        chroma_layout.addWidget(chroma_actions_container)

        self._tab_widget.addTab(chroma_tab, "ChromaDB")

        storage_tab = QWidget(self)
        storage_layout = QVBoxLayout(storage_tab)

        storage_description = QLabel(
            "Inspect the SQLite repository backing prompt storage.", storage_tab
        )
        storage_description.setWordWrap(True)
        storage_layout.addWidget(storage_description)

        storage_status_container = QWidget(storage_tab)
        storage_status_layout = QHBoxLayout(storage_status_container)
        storage_status_layout.setContentsMargins(0, 0, 0, 0)
        storage_status_layout.setSpacing(12)

        self._storage_status_label = QLabel("Checking…", storage_status_container)
        storage_status_layout.addWidget(self._storage_status_label)

        self._storage_path_label = QLabel("", storage_status_container)
        self._storage_path_label.setWordWrap(True)
        storage_status_layout.addWidget(self._storage_path_label, stretch=1)

        self._storage_refresh_button = QPushButton("Refresh", storage_status_container)
        self._storage_refresh_button.clicked.connect(self._refresh_storage_info)  # type: ignore[arg-type]
        storage_status_layout.addWidget(self._storage_refresh_button)

        storage_layout.addWidget(storage_status_container)

        self._storage_stats_view = QPlainTextEdit(storage_tab)
        self._storage_stats_view.setReadOnly(True)
        storage_layout.addWidget(self._storage_stats_view, stretch=1)

        storage_actions_container = QWidget(storage_tab)
        storage_actions_layout = QHBoxLayout(storage_actions_container)
        storage_actions_layout.setContentsMargins(0, 0, 0, 0)
        storage_actions_layout.setSpacing(12)

        self._sqlite_compact_button = QPushButton(
            "Compact Database",
            storage_actions_container,
        )
        self._sqlite_compact_button.setToolTip(
            "Run VACUUM on the prompt database to reclaim space."
        )
        self._sqlite_compact_button.clicked.connect(self._on_sqlite_compact_clicked)  # type: ignore[arg-type]
        storage_actions_layout.addWidget(self._sqlite_compact_button)

        self._sqlite_optimize_button = QPushButton("Optimize Database", storage_actions_container)
        self._sqlite_optimize_button.setToolTip("Refresh SQLite statistics for prompt lookups.")
        self._sqlite_optimize_button.clicked.connect(self._on_sqlite_optimize_clicked)  # type: ignore[arg-type]
        storage_actions_layout.addWidget(self._sqlite_optimize_button)

        self._sqlite_verify_button = QPushButton(
            "Verify Index Integrity",
            storage_actions_container,
        )
        self._sqlite_verify_button.setToolTip(
            "Run integrity checks against the prompt database indexes."
        )
        self._sqlite_verify_button.clicked.connect(self._on_sqlite_verify_clicked)  # type: ignore[arg-type]
        storage_actions_layout.addWidget(self._sqlite_verify_button)

        storage_actions_layout.addStretch(1)

        storage_layout.addWidget(storage_actions_container)

        self._tab_widget.addTab(storage_tab, "SQLite")

        reset_tab = QWidget(self)
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

        self._tab_widget.addTab(reset_tab, "Data Reset")

        self._buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
        self._buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        outer_layout.addWidget(self._buttons)

    def _set_stat_value(self, key: str, value: str) -> None:
        label = self._stats_labels.get(key)
        if label is not None:
            label.setText(value)

    @staticmethod
    def _format_timestamp(value: datetime | None) -> str:
        if value is None:
            return "—"
        return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")

    def _refresh_catalogue_stats(self) -> None:
        try:
            stats = self._manager.get_prompt_catalogue_stats()
        except PromptManagerError as exc:
            logger.warning("Prompt catalogue stats refresh failed", exc_info=True)
            self._append_log(f"Failed to load prompt statistics: {exc}")
            for label in self._stats_labels.values():
                label.setText("—")
            return

        self._set_stat_value("total_prompts", str(stats.total_prompts))
        self._set_stat_value("active_prompts", str(stats.active_prompts))
        self._set_stat_value("inactive_prompts", str(stats.inactive_prompts))
        self._set_stat_value("distinct_categories", str(stats.distinct_categories))
        self._set_stat_value("prompts_without_category", str(stats.prompts_without_category))
        self._set_stat_value("distinct_tags", str(stats.distinct_tags))
        self._set_stat_value("prompts_without_tags", str(stats.prompts_without_tags))
        self._set_stat_value(
            "average_tags_per_prompt",
            f"{stats.average_tags_per_prompt:.2f}",
        )
        self._set_stat_value("stale_prompts", str(stats.stale_prompts))
        self._set_stat_value(
            "last_modified_at",
            self._format_timestamp(stats.last_modified_at),
        )

    def _refresh_category_health(self) -> None:
        try:
            health_entries = self._manager.get_category_health()
        except PromptManagerError as exc:
            QMessageBox.warning(self, "Category health", str(exc))
            self._category_table.setRowCount(0)
            return

        self._category_table.setRowCount(len(health_entries))
        for row, entry in enumerate(health_entries):
            success_text = "—"
            if entry.success_rate is not None:
                success_text = f"{entry.success_rate * 100:.1f}%"
            executed_text = self._format_timestamp(entry.last_executed_at)
            self._category_table.setItem(row, 0, QTableWidgetItem(entry.label))
            self._category_table.setItem(row, 1, QTableWidgetItem(str(entry.total_prompts)))
            self._category_table.setItem(row, 2, QTableWidgetItem(str(entry.active_prompts)))
            self._category_table.setItem(row, 3, QTableWidgetItem(success_text))
            self._category_table.setItem(row, 4, QTableWidgetItem(executed_text or "—"))

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now(UTC).strftime("%H:%M:%S")
        self._log_view.appendPlainText(f"[{timestamp}] {message}")

    def _append_reset_log(self, message: str) -> None:
        timestamp = datetime.now(UTC).strftime("%H:%M:%S")
        self._reset_log_view.appendPlainText(f"[{timestamp}] {message}")

    def _confirm_destructive_action(self, prompt: str) -> bool:
        result = QMessageBox.question(
            self,
            "Confirm Data Reset",
            f"{prompt}\n\nThis action cannot be undone. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return result == QMessageBox.Yes

    def _collect_prompts(self) -> list[Prompt]:
        try:
            return self._manager.repository.list()
        except RepositoryError as exc:
            self._append_log(f"Unable to load prompts: {exc}")
            return []

    @staticmethod
    def _prompt_context(prompt: Prompt) -> str:
        return (prompt.context or prompt.description or prompt.example_input or "").strip()

    def _on_generate_categories_clicked(self) -> None:
        if self._category_generator is None:
            self._append_log("Category generator is unavailable.")
            return
        prompts = self._collect_prompts()
        if not prompts:
            return
        updated = 0
        for prompt in prompts:
            category_value = (prompt.category or "").strip()
            if category_value and category_value.lower() != "general":
                continue
            context = self._prompt_context(prompt)
            if not context:
                continue
            try:
                suggestion = (self._category_generator(context) or "").strip()
            except Exception as exc:  # noqa: BLE001
                self._append_log(f"Failed to suggest category for '{prompt.name}': {exc}")
                continue
            if not suggestion:
                continue
            updated_prompt = replace(
                prompt,
                category=suggestion,
                last_modified=datetime.now(UTC),
            )
            try:
                self._manager.update_prompt(updated_prompt)
            except PromptManagerError as exc:
                self._append_log(f"Unable to update '{prompt.name}': {exc}")
                continue
            updated += 1
            self._append_log(f"Set category '{suggestion}' for '{prompt.name}'.")
        self._append_log(f"Category task completed. Updated {updated} prompt(s).")
        if updated:
            self.maintenance_applied.emit(f"Generated categories for {updated} prompt(s).")

    def _on_generate_tags_clicked(self) -> None:
        if self._tags_generator is None:
            self._append_log("Tag generator is unavailable.")
            return
        prompts = self._collect_prompts()
        if not prompts:
            return
        updated = 0
        for prompt in prompts:
            existing_tags = [tag.strip() for tag in (prompt.tags or []) if tag.strip()]
            if existing_tags:
                continue
            context = self._prompt_context(prompt)
            if not context:
                continue
            try:
                suggestions = self._tags_generator(context) or []
            except Exception as exc:  # noqa: BLE001
                self._append_log(f"Failed to suggest tags for '{prompt.name}': {exc}")
                continue
            tags = [tag.strip() for tag in suggestions if str(tag).strip()]
            if not tags:
                continue
            updated_prompt = replace(
                prompt,
                tags=tags,
                last_modified=datetime.now(UTC),
            )
            try:
                self._manager.update_prompt(updated_prompt)
            except PromptManagerError as exc:
                self._append_log(f"Unable to update '{prompt.name}': {exc}")
                continue
            updated += 1
            self._append_log(f"Assigned tags {tags} to '{prompt.name}'.")
        self._append_log(f"Tag task completed. Updated {updated} prompt(s).")
        if updated:
            self.maintenance_applied.emit(f"Generated tags for {updated} prompt(s).")

    def _on_snapshot_clicked(self) -> None:
        """Prompt for a destination and create a maintenance snapshot."""
        default_name = datetime.now(UTC).strftime("prompt-manager-snapshot-%Y%m%d-%H%M%S.zip")
        default_path = str(Path.home() / default_name)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Maintenance Snapshot",
            default_path,
            "Zip Archives (*.zip)",
        )
        if not path:
            return

        self._append_reset_log("Creating maintenance snapshot…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            archive_path = self._manager.create_data_snapshot(path)
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
        """Clear the SQLite prompt repository."""
        if not self._confirm_destructive_action("Clear the prompt database and execution history?"):
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
        """Clear the ChromaDB vector store."""
        if not self._confirm_destructive_action(
            "Remove all embeddings from the ChromaDB vector store?"
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

    def _set_chroma_actions_busy(self, busy: bool) -> None:
        """Disable Chroma maintenance buttons while a task is running."""
        if not busy:
            return
        for button in (
            self._chroma_refresh_button,
            self._chroma_compact_button,
            self._chroma_optimize_button,
            self._chroma_verify_button,
        ):
            button.setEnabled(False)

    def _on_chroma_compact_clicked(self) -> None:
        """Run VACUUM maintenance on the Chroma persistent store."""
        self._set_chroma_actions_busy(True)
        try:
            self._manager.compact_vector_store()
        except PromptManagerError as exc:
            QMessageBox.critical(self, "Compaction failed", str(exc))
            return
        else:
            QMessageBox.information(
                self,
                "Chroma store compacted",
                "The persistent Chroma store has been vacuumed and reclaimed space.",
            )
        finally:
            self._refresh_chroma_info()

    def _on_chroma_optimize_clicked(self) -> None:
        """Refresh query statistics for the Chroma persistent store."""
        self._set_chroma_actions_busy(True)
        try:
            self._manager.optimize_vector_store()
        except PromptManagerError as exc:
            QMessageBox.critical(self, "Optimization failed", str(exc))
            return
        else:
            QMessageBox.information(
                self,
                "Chroma store optimized",
                "Chroma query statistics have been refreshed for better performance.",
            )
        finally:
            self._refresh_chroma_info()

    def _on_chroma_verify_clicked(self) -> None:
        """Verify integrity of the Chroma persistent store."""
        self._set_chroma_actions_busy(True)
        try:
            summary = self._manager.verify_vector_store()
        except PromptManagerError as exc:
            QMessageBox.critical(self, "Verification failed", str(exc))
            return
        else:
            message = summary or "Chroma store integrity verified successfully."
            QMessageBox.information(self, "Chroma store verified", message)
        finally:
            self._refresh_chroma_info()

    def _set_storage_actions_busy(self, busy: bool) -> None:
        """Disable SQLite maintenance buttons while a task is running."""
        if not busy:
            return
        for button in (
            self._storage_refresh_button,
            self._sqlite_compact_button,
            self._sqlite_optimize_button,
            self._sqlite_verify_button,
        ):
            button.setEnabled(False)

    def _on_sqlite_compact_clicked(self) -> None:
        """Run VACUUM on the prompt repository."""
        self._set_storage_actions_busy(True)
        try:
            self._manager.compact_repository()
        except PromptManagerError as exc:
            QMessageBox.critical(self, "Compaction failed", str(exc))
            return
        else:
            QMessageBox.information(
                self,
                "Prompt database compacted",
                "The SQLite repository has been vacuumed and reclaimed space.",
            )
        finally:
            self._refresh_storage_info()

    def _on_sqlite_optimize_clicked(self) -> None:
        """Refresh SQLite statistics for the prompt repository."""
        self._set_storage_actions_busy(True)
        try:
            self._manager.optimize_repository()
        except PromptManagerError as exc:
            QMessageBox.critical(self, "Optimization failed", str(exc))
            return
        else:
            QMessageBox.information(
                self,
                "Prompt database optimized",
                "SQLite statistics have been refreshed for prompt lookups.",
            )
        finally:
            self._refresh_storage_info()

    def _on_sqlite_verify_clicked(self) -> None:
        """Verify integrity of the prompt repository."""
        self._set_storage_actions_busy(True)
        try:
            summary = self._manager.verify_repository()
        except PromptManagerError as exc:
            QMessageBox.critical(self, "Verification failed", str(exc))
            return
        else:
            message = summary or "SQLite repository integrity verified successfully."
            QMessageBox.information(self, "Prompt database verified", message)
        finally:
            self._refresh_storage_info()

    def _on_reset_application_clicked(self) -> None:
        """Clear prompts, embeddings, and usage logs."""
        if not self._confirm_destructive_action(
            "Reset all application data (prompts, history, embeddings, and logs)?"
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

    def _refresh_redis_info(self) -> None:
        """Update the Redis tab with the latest cache status."""
        details = self._manager.get_redis_details()
        enabled = details.get("enabled", False)
        if not enabled:
            self._redis_status_label.setText("Redis caching is disabled.")
            self._redis_connection_label.setText("")
            self._redis_stats_view.setPlainText("")
            self._redis_refresh_button.setEnabled(False)
            return

        self._redis_refresh_button.setEnabled(True)

        status = details.get("status", "unknown").capitalize()
        if details.get("error"):
            status = f"Error: {details['error']}"
        self._redis_status_label.setText(f"Status: {status}")

        connection = details.get("connection", {})
        connection_parts = []
        if connection.get("host"):
            host = connection["host"]
            port = connection.get("port")
            if port is not None:
                connection_parts.append(f"{host}:{port}")
            else:
                connection_parts.append(str(host))
        if connection.get("database") is not None:
            connection_parts.append(f"DB {connection['database']}")
        if connection.get("ssl"):
            connection_parts.append("SSL")
        if not connection_parts:
            self._redis_connection_label.setText("")
        else:
            self._redis_connection_label.setText("Connection: " + ", ".join(connection_parts))

        stats = details.get("stats", {})
        lines: list[str] = []
        for key, label in (
            ("keys", "Keys"),
            ("used_memory_human", "Used memory"),
            ("used_memory_peak_human", "Peak memory"),
            ("maxmemory_human", "Configured max memory"),
            ("hits", "Keyspace hits"),
            ("misses", "Keyspace misses"),
            ("hit_rate", "Hit rate (%)"),
        ):
            if stats.get(key) is not None:
                lines.append(f"{label}: {stats[key]}")
        if not lines and details.get("error"):
            lines.append(details["error"])
        elif not lines and stats.get("info_error"):
            lines.append(f"Unable to fetch stats: {stats['info_error']}")
        redis_text = "\n".join(lines) if lines else "No Redis statistics available."
        self._redis_stats_view.setPlainText(redis_text)

    def _refresh_chroma_info(self) -> None:
        """Update the ChromaDB tab with vector store information."""
        details = self._manager.get_chroma_details()
        enabled = details.get("enabled", False)
        path = details.get("path") or ""
        collection = details.get("collection") or ""
        if not enabled:
            self._chroma_status_label.setText("ChromaDB is not initialised.")
            self._chroma_path_label.setText(f"Path: {path}" if path else "")
            self._chroma_stats_view.setPlainText("")
            self._chroma_refresh_button.setEnabled(False)
            self._chroma_compact_button.setEnabled(False)
            self._chroma_optimize_button.setEnabled(False)
            self._chroma_verify_button.setEnabled(False)
            return

        self._chroma_refresh_button.setEnabled(True)

        status = details.get("status", "unknown").capitalize()
        if details.get("error"):
            status = f"Error: {details['error']}"
        self._chroma_status_label.setText(f"Status: {status}")

        has_error = bool(details.get("error"))
        self._chroma_compact_button.setEnabled(not has_error)
        self._chroma_optimize_button.setEnabled(not has_error)
        self._chroma_verify_button.setEnabled(not has_error)

        path_parts = []
        if path:
            path_parts.append(f"Path: {path}")
        if collection:
            path_parts.append(f"Collection: {collection}")
        self._chroma_path_label.setText(" | ".join(path_parts))

        stats = details.get("stats", {})
        lines: list[str] = []
        for key, label in (
            ("documents", "Documents"),
            ("disk_usage_bytes", "Disk usage (bytes)"),
        ):
            value = stats.get(key)
            if value is not None:
                lines.append(f"{label}: {value}")
        chroma_text = "\n".join(lines) if lines else "No ChromaDB statistics available."
        self._chroma_stats_view.setPlainText(chroma_text)

    def _refresh_storage_info(self) -> None:
        """Update the SQLite tab with repository information."""
        repository = self._manager.repository
        db_path_obj = getattr(repository, "_db_path", None)
        if isinstance(db_path_obj, Path):
            db_path = str(db_path_obj)
        else:
            db_path = str(db_path_obj) if db_path_obj is not None else ""

        self._storage_path_label.setText(f"Path: {db_path}" if db_path else "Path: unknown")
        self._storage_refresh_button.setEnabled(True)

        stats_lines: list[str] = []
        healthy = True

        size_bytes = None
        if db_path:
            try:
                path_obj = Path(db_path)
                if path_obj.exists():
                    size_bytes = path_obj.stat().st_size
                else:
                    healthy = False
                    stats_lines.append("Database file not found.")
            except OSError as exc:
                healthy = False
                stats_lines.append(f"File size: error ({exc})")
        else:
            healthy = False

        if size_bytes is not None:
            stats_lines.append(f"File size: {size_bytes} bytes")

        try:
            prompt_count = len(repository.list())
            stats_lines.append(f"Prompts: {prompt_count}")
        except RepositoryError as exc:
            healthy = False
            stats_lines.append(f"Prompts: error ({exc})")

        try:
            execution_count = len(repository.list_executions())
            stats_lines.append(f"Executions: {execution_count}")
        except RepositoryError as exc:
            healthy = False
            stats_lines.append(f"Executions: error ({exc})")

        storage_text = "\n".join(stats_lines) if stats_lines else "No SQLite statistics available."
        self._storage_stats_view.setPlainText(storage_text)

        if healthy:
            self._storage_status_label.setText("Status: ready")
        else:
            self._storage_status_label.setText("Status: unavailable")

        self._sqlite_compact_button.setEnabled(healthy)
        self._sqlite_optimize_button.setEnabled(healthy)
        self._sqlite_verify_button.setEnabled(healthy)


__all__ = ["PromptMaintenanceDialog"]
