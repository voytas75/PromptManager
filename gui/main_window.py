"""Main window widgets and models for the Prompt Manager GUI.

Updates: v0.5.0 - 2025-11-08 - Add prompt execution workflow with result pane.
Updates: v0.4.0 - 2025-11-07 - Add intent workspace with detect/suggest/copy actions and usage logging.
Updates: v0.3.0 - 2025-11-06 - Surface intent-aware search hints and recommendations.
Updates: v0.2.0 - 2025-11-05 - Add catalogue filters, LiteLLM name generation, and settings UI.
Updates: v0.1.0 - 2025-11-04 - Provide list/search/detail panes with CRUD controls.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from PySide6.QtCore import QAbstractListModel, QModelIndex, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from config import PromptManagerSettings
from core import (
    IntentLabel,
    NameGenerationError,
    PromptManager,
    PromptManagerError,
    PromptExecutionError,
    PromptExecutionUnavailable,
    PromptNotFoundError,
    PromptStorageError,
    RepositoryError,
    diff_prompt_catalog,
    export_prompt_catalog,
    import_prompt_catalog,
)
from models.prompt_model import Prompt

from .dialogs import CatalogPreviewDialog, PromptDialog
from .history_dialog import ExecutionHistoryDialog
from .settings_dialog import SettingsDialog, persist_settings_to_config
from .usage_logger import IntentUsageLogger


class PromptListModel(QAbstractListModel):
    """List model providing prompt summaries for the QListView."""

    def __init__(self, prompts: Optional[Sequence[Prompt]] = None, parent=None) -> None:
        super().__init__(parent)
        self._prompts: List[Prompt] = list(prompts or [])

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802 - Qt API
        if parent.isValid():
            return 0
        return len(self._prompts)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Optional[str]:  # noqa: N802
        if not index.isValid() or index.row() >= len(self._prompts):
            return None
        prompt = self._prompts[index.row()]
        if role in {Qt.DisplayRole, Qt.EditRole}:
            category = f" ({prompt.category})" if prompt.category else ""
            return f"{prompt.name}{category}"
        return None

    def prompt_at(self, row: int) -> Optional[Prompt]:
        """Return the prompt at the given list index."""

        if 0 <= row < len(self._prompts):
            return self._prompts[row]
        return None

    def set_prompts(self, prompts: Iterable[Prompt]) -> None:
        """Replace the backing prompt list and notify listeners."""

        self.beginResetModel()
        self._prompts = list(prompts)
        self.endResetModel()

    def prompts(self) -> Sequence[Prompt]:
        """Expose the underlying prompts for selection helpers."""

        return tuple(self._prompts)


class PromptDetailWidget(QWidget):
    """Panel that summarises the currently selected prompt."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        self._title = QLabel("Select a prompt to view details", self)
        self._title.setObjectName("promptTitle")
        self._description = QLabel("", self)
        self._description.setWordWrap(True)

        self._context = QLabel("", self)
        self._context.setWordWrap(True)
        self._examples = QLabel("", self)
        self._examples.setWordWrap(True)

        layout.addWidget(self._title)
        layout.addSpacing(8)
        layout.addWidget(self._description)
        layout.addSpacing(8)
        layout.addWidget(self._context)
        layout.addSpacing(8)
        layout.addWidget(self._examples)
        layout.addStretch(1)

    def display_prompt(self, prompt: Prompt) -> None:
        """Populate labels using the provided prompt."""

        tags = ", ".join(prompt.tags) if prompt.tags else "No tags"
        language = prompt.language or "en"
        header = f"{prompt.name} — {prompt.category or 'Uncategorised'}\nLanguage: {language}\nTags: {tags}"
        self._title.setText(header)
        self._description.setText(prompt.description)
        context_text = prompt.context or "No prompt text provided."
        self._context.setText(f"Prompt Body:\n{context_text}")
        example_lines = []
        if prompt.example_input:
            example_lines.append(f"Example input:\n{prompt.example_input}")
        if prompt.example_output:
            example_lines.append(f"Example output:\n{prompt.example_output}")
        self._examples.setText("\n\n".join(example_lines) or "")

    def clear(self) -> None:
        """Reset the panel to its empty state."""

        self._title.setText("Select a prompt to view details")
        self._description.clear()
        self._context.clear()
        self._examples.clear()


class MainWindow(QMainWindow):
    """Primary window exposing prompt CRUD operations."""

    def __init__(
        self,
        prompt_manager: PromptManager,
        settings: Optional[PromptManagerSettings] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._manager = prompt_manager
        self._settings = settings
        self._model = PromptListModel(parent=self)
        self._detail_widget = PromptDetailWidget(self)
        self._all_prompts: List[Prompt] = []
        self._current_prompts: List[Prompt] = []
        self._suggestions: Optional[PromptManager.IntentSuggestions] = None
        self._last_execution: Optional[PromptManager.ExecutionOutcome] = None
        self._history_limit = 50
        self._runtime_settings = self._initial_runtime_settings(settings)
        self._usage_logger = IntentUsageLogger()
        self.setWindowTitle("Prompt Manager")
        self.resize(1024, 640)
        self._build_ui()
        self._load_prompts()

    def _build_ui(self) -> None:
        """Create the main layout with list/search/detail panes."""

        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)

        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(8)

        self._search_input = QLineEdit(self)
        self._search_input.setPlaceholderText("Search prompts…")
        self._search_input.textChanged.connect(self._on_search_changed)  # type: ignore[arg-type]
        controls_layout.addWidget(self._search_input)

        self._refresh_button = QPushButton("Refresh", self)
        self._refresh_button.clicked.connect(self._on_refresh_clicked)  # type: ignore[arg-type]
        controls_layout.addWidget(self._refresh_button)

        self._add_button = QPushButton("Add", self)
        self._add_button.clicked.connect(self._on_add_clicked)  # type: ignore[arg-type]
        controls_layout.addWidget(self._add_button)

        self._edit_button = QPushButton("Edit", self)
        self._edit_button.clicked.connect(self._on_edit_clicked)  # type: ignore[arg-type]
        controls_layout.addWidget(self._edit_button)

        self._delete_button = QPushButton("Delete", self)
        self._delete_button.clicked.connect(self._on_delete_clicked)  # type: ignore[arg-type]
        controls_layout.addWidget(self._delete_button)

        self._import_button = QPushButton("Import", self)
        self._import_button.clicked.connect(self._on_import_clicked)  # type: ignore[arg-type]
        controls_layout.addWidget(self._import_button)

        self._export_button = QPushButton("Export", self)
        self._export_button.clicked.connect(self._on_export_clicked)  # type: ignore[arg-type]
        controls_layout.addWidget(self._export_button)

        self._history_button = QPushButton("History", self)
        self._history_button.clicked.connect(self._on_history_clicked)  # type: ignore[arg-type]
        controls_layout.addWidget(self._history_button)

        self._settings_button = QPushButton("Settings", self)
        self._settings_button.clicked.connect(self._on_settings_clicked)  # type: ignore[arg-type]
        controls_layout.addWidget(self._settings_button)

        layout.addLayout(controls_layout)

        self._query_input = QPlainTextEdit(self)
        self._query_input.setPlaceholderText("Paste code or text to analyse and suggest prompts…")
        self._query_input.setMinimumHeight(120)
        layout.addWidget(self._query_input)

        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(8)

        self._detect_button = QPushButton("Detect Need", self)
        self._detect_button.clicked.connect(self._on_detect_intent_clicked)  # type: ignore[arg-type]
        actions_layout.addWidget(self._detect_button)

        self._suggest_button = QPushButton("Suggest Prompt", self)
        self._suggest_button.clicked.connect(self._on_suggest_prompt_clicked)  # type: ignore[arg-type]
        actions_layout.addWidget(self._suggest_button)

        self._run_button = QPushButton("Run Prompt", self)
        self._run_button.clicked.connect(self._on_run_prompt_clicked)  # type: ignore[arg-type]
        actions_layout.addWidget(self._run_button)

        self._copy_button = QPushButton("Copy Prompt", self)
        self._copy_button.clicked.connect(self._on_copy_prompt_clicked)  # type: ignore[arg-type]
        actions_layout.addWidget(self._copy_button)

        self._copy_result_button = QPushButton("Copy Result", self)
        self._copy_result_button.clicked.connect(self._on_copy_result_clicked)  # type: ignore[arg-type]
        actions_layout.addWidget(self._copy_result_button)

        actions_layout.addStretch(1)
        layout.addLayout(actions_layout)

        self._intent_hint = QLabel("", self)
        self._intent_hint.setObjectName("intentHintLabel")
        self._intent_hint.setStyleSheet("color: #5b5b5b; font-style: italic;")
        self._intent_hint.setVisible(False)
        layout.addWidget(self._intent_hint)

        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(8)

        self._category_filter = QComboBox(self)
        self._category_filter.addItem("All categories", None)
        self._category_filter.currentIndexChanged.connect(self._on_filters_changed)

        self._tag_filter = QComboBox(self)
        self._tag_filter.addItem("All tags", None)
        self._tag_filter.currentIndexChanged.connect(self._on_filters_changed)

        self._quality_filter = QDoubleSpinBox(self)
        self._quality_filter.setRange(0.0, 10.0)
        self._quality_filter.setDecimals(1)
        self._quality_filter.setSingleStep(0.1)
        self._quality_filter.valueChanged.connect(self._on_filters_changed)

        filter_layout.addWidget(QLabel("Category:", self))
        filter_layout.addWidget(self._category_filter)
        filter_layout.addWidget(QLabel("Tag:", self))
        filter_layout.addWidget(self._tag_filter)
        filter_layout.addWidget(QLabel("Quality ≥", self))
        filter_layout.addWidget(self._quality_filter)
        filter_layout.addStretch(1)

        layout.addLayout(filter_layout)

        splitter = QSplitter(Qt.Horizontal, self)

        self._list_view = QListView(splitter)
        self._list_view.setModel(self._model)
        self._list_view.setSelectionMode(QAbstractItemView.SingleSelection)
        self._list_view.selectionModel().selectionChanged.connect(self._on_selection_changed)  # type: ignore[arg-type]
        splitter.addWidget(self._list_view)

        splitter.addWidget(self._detail_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, stretch=1)

        result_layout = QVBoxLayout()
        result_layout.setContentsMargins(0, 12, 0, 0)

        self._result_label = QLabel("No prompt executed yet", self)
        self._result_label.setObjectName("resultTitle")
        self._result_meta = QLabel("", self)
        self._result_meta.setStyleSheet("color: #5b5b5b; font-style: italic;")
        self._result_view = QPlainTextEdit(self)
        self._result_view.setReadOnly(True)
        self._result_view.setPlaceholderText("Run a prompt to see output here.")

        result_layout.addWidget(self._result_label)
        result_layout.addWidget(self._result_meta)
        result_layout.addWidget(self._result_view, 1)
        layout.addLayout(result_layout)

        self.setCentralWidget(container)

        has_executor = self._manager.executor is not None
        self._run_button.setEnabled(has_executor)
        self._copy_result_button.setEnabled(False)
        self._history_button.setEnabled(True)

    def _load_prompts(self, search_text: str = "") -> None:
        """Populate the list model from the repository or semantic search."""

        try:
            self._all_prompts = self._manager.repository.list()
        except RepositoryError as exc:
            self._show_error("Unable to load prompts", str(exc))
            return

        self._populate_filters(self._all_prompts)

        stripped = search_text.strip()
        if stripped:
            try:
                suggestions = self._manager.suggest_prompts(stripped, limit=50)
            except PromptManagerError as exc:
                self._show_error("Unable to search prompts", str(exc))
            else:
                self._apply_suggestions(suggestions)
                return

        self._suggestions = None
        self._current_prompts = list(self._all_prompts)

        filtered = self._apply_filters(self._current_prompts)
        self._model.set_prompts(filtered)
        self._list_view.clearSelection()
        if not filtered:
            self._detail_widget.clear()
        self._update_intent_hint(filtered)

    def _populate_filters(self, prompts: Sequence[Prompt]) -> None:
        """Refresh category and tag filters based on available prompts."""

        categories = sorted({prompt.category for prompt in prompts if prompt.category})
        current_category = self._category_filter.currentData()
        self._category_filter.blockSignals(True)
        self._category_filter.clear()
        self._category_filter.addItem("All categories", None)
        for category in categories:
            self._category_filter.addItem(category, category)
        category_index = self._category_filter.findData(current_category)
        self._category_filter.setCurrentIndex(category_index if category_index != -1 else 0)
        self._category_filter.blockSignals(False)

        tags = sorted({tag for prompt in prompts for tag in prompt.tags})
        current_tag = self._tag_filter.currentData()
        self._tag_filter.blockSignals(True)
        self._tag_filter.clear()
        self._tag_filter.addItem("All tags", None)
        for tag in tags:
            self._tag_filter.addItem(tag, tag)
        tag_index = self._tag_filter.findData(current_tag)
        self._tag_filter.setCurrentIndex(tag_index if tag_index != -1 else 0)
        self._tag_filter.blockSignals(False)

    def _apply_filters(self, prompts: Sequence[Prompt]) -> List[Prompt]:
        """Apply category, tag, and quality filters to a prompt sequence."""

        selected_category = self._category_filter.currentData()
        selected_tag = self._tag_filter.currentData()
        min_quality = self._quality_filter.value()

        filtered: List[Prompt] = []
        for prompt in prompts:
            if selected_category and prompt.category != selected_category:
                continue
            if selected_tag and selected_tag not in prompt.tags:
                continue
            if min_quality > 0.0:
                quality = prompt.quality_score or 0.0
                if quality < min_quality:
                    continue
            filtered.append(prompt)
        return filtered

    def _on_filters_changed(self, *_: object) -> None:
        """Refresh the prompt list when filter widgets change."""

        self._refresh_filtered_view()

    def _refresh_filtered_view(self) -> None:
        filtered = self._apply_filters(self._current_prompts)
        self._model.set_prompts(filtered)
        self._list_view.clearSelection()
        if not filtered:
            self._detail_widget.clear()
        self._update_intent_hint(filtered)

    def _update_intent_hint(self, prompts: Sequence[Prompt]) -> None:
        """Update the hint label with detected intent context and matches."""

        if self._suggestions is None:
            self._intent_hint.clear()
            self._intent_hint.setVisible(False)
            return

        prompt_list = list(prompts)
        prediction = self._suggestions.prediction
        label_text = prediction.label.value.replace("_", " ").title()
        confidence_pct = int(round(prediction.confidence * 100))

        summary_parts: List[str] = []
        if prediction.label is not IntentLabel.GENERAL or prediction.category_hints:
            summary_parts.append(f"Detected intent: {label_text} ({confidence_pct}%).")
        if prediction.category_hints:
            summary_parts.append(f"Focus: {', '.join(prediction.category_hints)}")
        if prediction.language_hints:
            summary_parts.append(f"Lang: {', '.join(prediction.language_hints)}")

        top_names = [prompt.name for prompt in prompt_list[:3]]
        if top_names:
            summary_parts.append(f"Top matches: {', '.join(top_names)}")
        if self._suggestions.fallback_used:
            summary_parts.append("Fallback ranking applied")

        if summary_parts:
            message = " | ".join(summary_parts)
        elif top_names:
            message = "Top matches: " + ", ".join(top_names)
        else:
            self._intent_hint.clear()
            self._intent_hint.setVisible(False)
            return

        self._intent_hint.setText(message)
        self._intent_hint.setVisible(True)

    def _display_execution_result(
        self,
        prompt: Prompt,
        outcome: PromptManager.ExecutionOutcome,
    ) -> None:
        """Render the most recent execution result in the result pane."""

        self._last_execution = outcome
        self._result_label.setText(f"Last Result — {prompt.name}")
        meta_parts: List[str] = [f"Duration: {outcome.result.duration_ms} ms"]
        history_entry = outcome.history_entry
        if history_entry is not None:
            executed_at = history_entry.executed_at.astimezone()
            meta_parts.append(f"Logged: {executed_at.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        self._result_meta.setText(" | ".join(meta_parts))
        self._result_view.setPlainText(outcome.result.response_text or "")
        self._copy_result_button.setEnabled(bool(outcome.result.response_text))

    def _clear_execution_result(self) -> None:
        """Reset the result pane to its default state."""

        self._last_execution = None
        self._result_label.setText("No prompt executed yet")
        self._result_meta.clear()
        self._result_view.clear()
        self._copy_result_button.setEnabled(False)

    def _generate_prompt_name(self, context: str) -> str:
        """Delegate name generation to PromptManager, surfacing errors."""

        if not context.strip():
            return ""
        return self._manager.generate_prompt_name(context)

    def _generate_prompt_description(self, context: str) -> str:
        """Delegate description generation to PromptManager."""

        if not context.strip():
            return ""
        return self._manager.generate_prompt_description(context)

    def _on_detect_intent_clicked(self) -> None:
        """Run intent detection on the free-form query input."""

        text = self._query_input.toPlainText().strip()
        if not text:
            self.statusBar().showMessage("Paste some text or code to analyse.", 4000)
            return
        classifier = self._manager.intent_classifier
        if classifier is None:
            self._show_error("Intent detection unavailable", "No intent classifier is configured.")
            return
        prediction = classifier.classify(text)
        current_prompts = list(self._model.prompts())
        self._suggestions = PromptManager.IntentSuggestions(
            prediction=prediction,
            prompts=list(current_prompts),
            fallback_used=False,
        )
        self._update_intent_hint(current_prompts)
        label = prediction.label.value.replace("_", " ").title()
        self.statusBar().showMessage(f"Detected intent: {label} ({int(prediction.confidence * 100)}%)", 5000)
        self._usage_logger.log_detect(prediction=prediction, query_text=text)

    def _on_suggest_prompt_clicked(self) -> None:
        """Generate prompt suggestions from the free-form query input."""

        query = self._query_input.toPlainText().strip() or self._search_input.text().strip()
        if not query:
            self.statusBar().showMessage("Provide text or use search to fetch suggestions.", 4000)
            return
        try:
            suggestions = self._manager.suggest_prompts(query, limit=20)
        except PromptManagerError as exc:
            self._show_error("Unable to suggest prompts", str(exc))
            return
        self._usage_logger.log_suggest(
            prediction=suggestions.prediction,
            query_text=query,
            prompts=suggestions.prompts,
            fallback_used=suggestions.fallback_used,
        )
        self._apply_suggestions(suggestions)
        top_name = suggestions.prompts[0].name if suggestions.prompts else None
        if top_name:
            self.statusBar().showMessage(f"Top suggestion: {top_name}", 5000)

    def _on_history_clicked(self) -> None:
        """Open the execution history dialog."""

        dialog = ExecutionHistoryDialog(self._manager, self, limit=self._history_limit)
        dialog.exec()
        total = len(self._manager.list_recent_executions(limit=self._history_limit))
        self._usage_logger.log_history_view(total=total)

    def _on_run_prompt_clicked(self) -> None:
        """Execute the selected prompt via the manager."""

        request_text = self._query_input.toPlainText().strip()
        if not request_text:
            self.statusBar().showMessage("Paste some text or code before executing a prompt.", 4000)
            return
        prompt = self._current_prompt()
        if prompt is None:
            prompts = self._model.prompts()
            prompt = prompts[0] if prompts else None
        if prompt is None:
            self.statusBar().showMessage("Select a prompt to execute first.", 4000)
            return

        try:
            outcome = self._manager.execute_prompt(prompt.id, request_text)
        except PromptExecutionUnavailable as exc:
            self._run_button.setEnabled(False)
            self._usage_logger.log_execute(
                prompt_name=prompt.name,
                success=False,
                duration_ms=None,
                error=str(exc),
            )
            self._show_error("Prompt execution unavailable", str(exc))
            return
        except (PromptExecutionError, PromptManagerError) as exc:
            self._usage_logger.log_execute(
                prompt_name=prompt.name,
                success=False,
                duration_ms=None,
                error=str(exc),
            )
            self._show_error("Prompt execution failed", str(exc))
            return

        self._display_execution_result(prompt, outcome)
        self._usage_logger.log_execute(
            prompt_name=prompt.name,
            success=True,
            duration_ms=outcome.result.duration_ms,
        )
        self.statusBar().showMessage(
            f"Executed '{prompt.name}' in {outcome.result.duration_ms} ms.",
            5000,
        )

    def _on_copy_prompt_clicked(self) -> None:
        """Copy the currently selected prompt body to the clipboard."""

        prompt = self._current_prompt()
        if prompt is None:
            prompts = self._model.prompts()
            prompt = prompts[0] if prompts else None
        if prompt is None:
            self.statusBar().showMessage("Select a prompt to copy first.", 3000)
            return
        payload = prompt.context or prompt.description
        if not payload:
            self.statusBar().showMessage("Selected prompt does not include a body to copy.", 3000)
            return
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(payload)
        self.statusBar().showMessage(f"Copied '{prompt.name}' to the clipboard.", 4000)
        self._usage_logger.log_copy(prompt_name=prompt.name, prompt_has_body=bool(prompt.context))

    def _on_copy_result_clicked(self) -> None:
        """Copy the latest execution result to the clipboard."""

        if self._last_execution is None:
            self.statusBar().showMessage("Run a prompt to generate a result first.", 3000)
            return
        response = self._last_execution.result.response_text
        if not response:
            self.statusBar().showMessage("Latest result is empty; nothing to copy.", 3000)
            return
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(response)
        self.statusBar().showMessage("Prompt result copied to the clipboard.", 4000)

    def _apply_suggestions(self, suggestions: PromptManager.IntentSuggestions) -> None:
        """Apply intent suggestions to the list view and update filters."""

        self._suggestions = suggestions
        self._current_prompts = list(suggestions.prompts)
        filtered = self._apply_filters(self._current_prompts)
        self._model.set_prompts(filtered)
        self._list_view.clearSelection()
        if filtered:
            self._select_prompt(filtered[0].id)
        else:
            self._detail_widget.clear()
        self._update_intent_hint(filtered)

    def _current_prompt(self) -> Optional[Prompt]:
        """Return the prompt currently selected in the list."""

        index = self._list_view.currentIndex()
        if not index.isValid():
            return None
        return self._model.prompt_at(index.row())

    def _on_search_changed(self, text: str) -> None:
        """Trigger prompt search with debounce-friendly minimum length."""

        if not text or len(text.strip()) >= 2:
            self._load_prompts(text)

    def _on_refresh_clicked(self) -> None:
        """Reload prompts from storage, respecting current search text."""

        self._load_prompts(self._search_input.text())

    def _on_add_clicked(self) -> None:
        """Open the creation dialog and persist a new prompt."""

        dialog = PromptDialog(
            self,
            name_generator=self._generate_prompt_name,
            description_generator=self._generate_prompt_description,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        prompt = dialog.result_prompt
        if prompt is None:
            return
        try:
            created = self._manager.create_prompt(prompt)
        except PromptStorageError as exc:
            self._show_error("Unable to create prompt", str(exc))
            return
        self._load_prompts()
        self._select_prompt(created.id)

    def _on_edit_clicked(self) -> None:
        """Open the edit dialog for the selected prompt and persist changes."""

        prompt = self._current_prompt()
        if prompt is None:
            return
        dialog = PromptDialog(
            self,
            prompt,
            name_generator=self._generate_prompt_name,
            description_generator=self._generate_prompt_description,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        updated = dialog.result_prompt
        if updated is None:
            return
        try:
            stored = self._manager.update_prompt(updated)
        except PromptNotFoundError:
            self._show_error("Prompt missing", "The prompt cannot be located. Refresh and try again.")
            self._load_prompts()
            return
        except PromptStorageError as exc:
            self._show_error("Unable to update prompt", str(exc))
            return
        self._load_prompts(self._search_input.text())
        self._select_prompt(stored.id)

    def _on_delete_clicked(self) -> None:
        """Remove the selected prompt after confirmation."""

        prompt = self._current_prompt()
        if prompt is None:
            return
        confirmation = QMessageBox.question(
            self,
            "Delete prompt",
            f"Are you sure you want to delete '{prompt.name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirmation != QMessageBox.Yes:
            return
        try:
            self._manager.delete_prompt(prompt.id)
        except PromptNotFoundError:
            self._show_error("Prompt missing", "The prompt was already removed.")
        except PromptStorageError as exc:
            self._show_error("Unable to delete prompt", str(exc))
            return
        self._load_prompts(self._search_input.text())

    def _on_import_clicked(self) -> None:
        """Preview catalogue diff and optionally apply updates."""

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select catalogue file",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        catalog_path: Optional[Path]
        if file_path:
            catalog_path = Path(file_path)
        else:
            directory = QFileDialog.getExistingDirectory(self, "Select catalogue directory", "")
            if not directory:
                return
            catalog_path = Path(directory)

        catalog_path = catalog_path.expanduser()

        try:
            preview = diff_prompt_catalog(self._manager, catalog_path)
        except Exception as exc:
            QMessageBox.warning(self, "Catalogue preview failed", str(exc))
            return

        dialog = CatalogPreviewDialog(preview, self)
        if dialog.exec() != QDialog.Accepted or not dialog.apply_requested:
            return

        try:
            result = import_prompt_catalog(self._manager, catalog_path)
        except Exception as exc:
            QMessageBox.critical(self, "Catalogue import failed", str(exc))
            return

        message = (
            f"Catalogue applied (added {result.added}, updated {result.updated}, "
            f"skipped {result.skipped}, errors {result.errors})"
        )
        if result.errors:
            QMessageBox.warning(self, "Catalogue applied with errors", message)
        else:
            self.statusBar().showMessage(message, 5000)
        self._load_prompts(self._search_input.text())

    def _on_export_clicked(self) -> None:
        """Export current prompts to JSON or YAML."""

        default_path = str(Path.home() / "prompt_catalog.json")
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export prompt catalogue",
            default_path,
            "JSON Files (*.json);;YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if not file_path:
            return
        export_path = Path(file_path)
        lower_suffix = export_path.suffix.lower()
        if selected_filter.startswith("YAML") or lower_suffix in {".yaml", ".yml"}:
            fmt = "yaml"
        else:
            fmt = "json"
        try:
            resolved = export_prompt_catalog(self._manager, export_path, fmt=fmt)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.statusBar().showMessage(f"Catalogue exported to {resolved}", 5000)
    def _on_settings_clicked(self) -> None:
        """Open configuration dialog and apply updates."""

        dialog = SettingsDialog(
            self,
            catalog_path=self._runtime_settings.get("catalog_path"),
            litellm_model=self._runtime_settings.get("litellm_model"),
            litellm_api_key=self._runtime_settings.get("litellm_api_key"),
            litellm_api_base=self._runtime_settings.get("litellm_api_base"),
            litellm_api_version=self._runtime_settings.get("litellm_api_version"),
        )
        if dialog.exec() != QDialog.Accepted:
            return
        updates = dialog.result_settings()
        self._apply_settings(updates)

    def _apply_settings(self, updates: dict[str, Optional[str]]) -> None:
        """Persist settings, refresh catalogue, and update name generator."""

        catalog_path = updates.get("catalog_path")
        normalized_catalog = (
            str(Path(catalog_path).expanduser()) if catalog_path else None
        )
        self._runtime_settings["catalog_path"] = normalized_catalog
        self._runtime_settings["litellm_model"] = updates.get("litellm_model")
        self._runtime_settings["litellm_api_key"] = updates.get("litellm_api_key")
        self._runtime_settings["litellm_api_base"] = updates.get("litellm_api_base")
        self._runtime_settings["litellm_api_version"] = updates.get("litellm_api_version")
        persist_settings_to_config(self._runtime_settings)

        if self._settings is not None:
            self._settings.catalog_path = (
                Path(normalized_catalog).expanduser()
                if normalized_catalog
                else None
            )
            self._settings.litellm_model = updates.get("litellm_model")
            self._settings.litellm_api_key = updates.get("litellm_api_key")
            self._settings.litellm_api_base = updates.get("litellm_api_base")
            self._settings.litellm_api_version = updates.get("litellm_api_version")

        try:
            self._manager.set_name_generator(
                self._runtime_settings.get("litellm_model"),
                self._runtime_settings.get("litellm_api_key"),
                self._runtime_settings.get("litellm_api_base"),
                self._runtime_settings.get("litellm_api_version"),
            )
        except NameGenerationError as exc:
            QMessageBox.warning(self, "LiteLLM configuration", str(exc))

        try:
            result = import_prompt_catalog(
                self._manager,
                Path(normalized_catalog).expanduser() if normalized_catalog else None,
            )
            self.statusBar().showMessage(
                f"Catalogue synced (added {result.added}, updated {result.updated}, skipped {result.skipped})",
                5000,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Catalogue import failed", str(exc))

        self._load_prompts(self._search_input.text())

    def _initial_runtime_settings(
        self, settings: Optional[PromptManagerSettings]
    ) -> dict[str, Optional[str]]:
        """Load current settings snapshot from configuration."""

        runtime = {
            "catalog_path": str(settings.catalog_path)
            if settings and settings.catalog_path is not None
            else None,
            "litellm_model": settings.litellm_model if settings else None,
            "litellm_api_key": settings.litellm_api_key if settings else None,
            "litellm_api_base": settings.litellm_api_base if settings else None,
            "litellm_api_version": settings.litellm_api_version if settings else None,
        }

        config_path = Path("config/config.json")
        if config_path.exists():
            try:
                data = json.loads(config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
            else:
                for key in (
                    "catalog_path",
                    "litellm_model",
                    "litellm_api_key",
                    "litellm_api_base",
                    "litellm_api_version",
                ):
                    if isinstance(data.get(key), str):
                        runtime[key] = data[key]
        return runtime

    def _on_selection_changed(self, *_: object) -> None:
        """Update the detail panel to reflect the new selection."""

        prompt = self._current_prompt()
        if prompt is None:
            self._detail_widget.clear()
            return
        self._detail_widget.display_prompt(prompt)

    def _select_prompt(self, prompt_id: uuid.UUID) -> None:
        """Highlight the given prompt in the list view when present."""

        for row, prompt in enumerate(self._model.prompts()):
            if prompt.id == prompt_id:
                index = self._model.index(row, 0)
                self._list_view.setCurrentIndex(index)
                break

    def _show_error(self, title: str, message: str) -> None:
        """Display an error dialog and log to status bar."""

        QMessageBox.critical(self, title, message)
        self.statusBar().showMessage(message, 5000)


__all__ = ["MainWindow", "PromptListModel", "PromptDetailWidget"]
