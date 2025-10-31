"""Main window widgets and models for the Prompt Manager GUI.

Updates: v0.1.0 - 2025-11-04 - Provide list/search/detail panes with CRUD controls.
"""

from __future__ import annotations

import uuid
from typing import Iterable, List, Optional, Sequence

from PySide6.QtCore import QAbstractListModel, QModelIndex, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core import (
    PromptManager,
    PromptManagerError,
    PromptNotFoundError,
    PromptStorageError,
    RepositoryError,
)
from models.prompt_model import Prompt

from .dialogs import PromptDialog


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
        context_text = prompt.context or "No additional context provided."
        self._context.setText(f"Context:\n{context_text}")
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

    def __init__(self, prompt_manager: PromptManager, parent=None) -> None:
        super().__init__(parent)
        self._manager = prompt_manager
        self._model = PromptListModel(parent=self)
        self._detail_widget = PromptDetailWidget(self)
        self._all_prompts: List[Prompt] = []
        self._current_prompts: List[Prompt] = []
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

        layout.addLayout(controls_layout)

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

        self.setCentralWidget(container)

    def _load_prompts(self, search_text: str = "") -> None:
        """Populate the list model from the repository or semantic search."""

        try:
            self._all_prompts = self._manager.repository.list()
        except RepositoryError as exc:
            self._show_error("Unable to load prompts", str(exc))
            return

        self._populate_filters(self._all_prompts)

        try:
            if search_text.strip():
                self._current_prompts = self._manager.search_prompts(search_text.strip(), limit=50)
            else:
                self._current_prompts = list(self._all_prompts)
        except PromptManagerError as exc:
            self._show_error("Unable to search prompts", str(exc))
            self._current_prompts = list(self._all_prompts)

        filtered = self._apply_filters(self._current_prompts)
        self._model.set_prompts(filtered)
        self._list_view.clearSelection()
        if not filtered:
            self._detail_widget.clear()

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

        dialog = PromptDialog(self)
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
        dialog = PromptDialog(self, prompt)
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
