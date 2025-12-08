"""Catalogue statistics and metadata helpers for the maintenance dialog.

Updates:
  v0.1.0 - 2025-12-04 - Extract metadata tab builders and refresh logic.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Callable, Sequence, cast

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core import PromptManager, PromptManagerError, RepositoryError

from ..base import logger

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from PySide6.QtWidgets import QWidget as _QWidget
    from models.prompt_model import Prompt


class CatalogueMaintenanceMixin:
    """Provide catalogue overview and metadata generation helpers."""

    _manager: PromptManager
    _stats_labels: dict[str, QLabel]
    _log_view: QPlainTextEdit
    _category_table: QTableWidget
    _category_refresh_button: QPushButton
    _categories_button: QPushButton
    _tags_button: QPushButton
    _category_generator: Callable[[str], str] | None
    _tags_generator: Callable[[str], Sequence[str]] | None
    maintenance_applied: Any

    def _parent_widget(self) -> "_QWidget":
        return cast("_QWidget", self)

    def _build_metadata_tab(self, parent: QWidget) -> QWidget:
        metadata_tab = QWidget(parent)
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

        self._stats_labels = {}
        for row_index, (key, label_text) in enumerate(stat_rows):
            label_widget = QLabel(label_text, stats_group)
            value_widget = QLabel("—", stats_group)
            value_widget.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
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
            alignment=Qt.AlignmentFlag.AlignRight,
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
        self._category_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._category_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
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
        health_layout.addWidget(self._category_refresh_button, alignment=Qt.AlignmentFlag.AlignRight)

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

        return metadata_tab

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
        parent = self._parent_widget()
        try:
            health_entries = self._manager.get_category_health()
        except PromptManagerError as exc:
            QMessageBox.warning(parent, "Category health", str(exc))
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
