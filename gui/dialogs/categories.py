"""Category management dialogs used across the GUI.

Updates:
  v0.1.0 - 2025-12-03 - Extract category dialogs from the gui.dialogs monolith.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core import CategoryNotFoundError, CategoryStorageError, PromptManager
from models.category_model import PromptCategory, slugify_category

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from PySide6.QtGui import QCloseEvent


class CategoryPayload(TypedDict):
    label: str
    slug: str
    description: str
    parent_slug: str | None
    color: str | None
    icon: str | None
    min_quality: float | None
    default_tags: list[str]
    is_active: bool


class CategoryEditorDialog(QDialog):
    """Collect category details for creation or editing workflows."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        category: PromptCategory | None = None,
    ) -> None:
        """Create the editor and optionally preload an existing category."""
        super().__init__(parent)
        self._source = category
        self._payload: CategoryPayload | None = None
        self.setWindowTitle("Add Category" if category is None else "Edit Category")
        self.resize(460, 360)
        self._build_ui()
        if category is not None:
            self._populate(category)

    @property
    def payload(self) -> CategoryPayload:
        """Return the collected form data."""
        if self._payload is None:
            raise ValueError("Category payload is not available before acceptance.")
        return self._payload

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._label_input = QLineEdit(self)
        self._label_input.setPlaceholderText("Documentation, Refactoring…")
        form.addRow("Label*", self._label_input)

        self._slug_input = QLineEdit(self)
        self._slug_input.setPlaceholderText("documentation, refactoring…")
        form.addRow("Slug*", self._slug_input)

        self._description_input = QLineEdit(self)
        self._description_input.setPlaceholderText("Short sentence describing the scope.")
        form.addRow("Description", self._description_input)

        self._parent_input = QLineEdit(self)
        self._parent_input.setPlaceholderText("Parent category slug (optional)")
        form.addRow("Parent slug", self._parent_input)

        self._color_input = QLineEdit(self)
        self._color_input.setPlaceholderText("#RRGGBB")
        form.addRow("Colour", self._color_input)

        self._icon_input = QLineEdit(self)
        self._icon_input.setPlaceholderText("mdi-icon-name")
        form.addRow("Icon", self._icon_input)

        self._min_quality_input = QLineEdit(self)
        self._min_quality_input.setPlaceholderText("Minimum quality score (optional)")
        form.addRow("Min quality", self._min_quality_input)

        self._tags_input = QLineEdit(self)
        self._tags_input.setPlaceholderText("Comma-separated default tags")
        form.addRow("Default tags", self._tags_input)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttons.accepted.connect(self._on_accept)  # type: ignore[arg-type]
        buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        layout.addWidget(buttons)

    def _populate(self, category: PromptCategory) -> None:
        """Populate the form with an existing category."""
        self._label_input.setText(category.label)
        self._slug_input.setText(category.slug)
        self._slug_input.setReadOnly(True)
        self._description_input.setText(category.description)
        self._parent_input.setText(category.parent_slug or "")
        self._color_input.setText(category.color or "")
        self._icon_input.setText(category.icon or "")
        min_quality = "" if category.min_quality is None else str(category.min_quality)
        self._min_quality_input.setText(min_quality)
        self._tags_input.setText(", ".join(category.default_tags))

    def _on_accept(self) -> None:
        """Validate inputs and persist them in payload."""
        label = self._label_input.text().strip()
        if not label:
            QMessageBox.warning(self, "Invalid category", "Label is required.")
            return
        slug_source = self._slug_input.text().strip() or label
        slug = slugify_category(slug_source)
        if not slug:
            QMessageBox.warning(self, "Invalid category", "Slug is required.")
            return
        description = self._description_input.text().strip() or label
        min_quality_text = self._min_quality_input.text().strip()
        min_quality: float | None = None
        if min_quality_text:
            try:
                min_quality = float(min_quality_text)
            except ValueError:
                QMessageBox.warning(self, "Invalid category", "Minimum quality must be a number.")
                return
        tags = [tag.strip() for tag in self._tags_input.text().split(",") if tag.strip()]
        is_active = self._source.is_active if self._source is not None else True
        payload: CategoryPayload = {
            "label": label,
            "slug": slug,
            "description": description,
            "parent_slug": self._parent_input.text().strip() or None,
            "color": self._color_input.text().strip() or None,
            "icon": self._icon_input.text().strip() or None,
            "min_quality": min_quality,
            "default_tags": tags,
            "is_active": is_active,
        }
        self._payload = payload
        self.accept()


class CategoryManagerDialog(QDialog):
    """Provide CRUD workflows for prompt categories."""

    def __init__(self, manager: PromptManager, parent: QWidget | None = None) -> None:
        """Create the manager dialog backed by *manager*."""
        super().__init__(parent)
        self._manager = manager
        self._has_changes = False
        self._categories: list[PromptCategory] = []
        self._settings = QSettings("PromptManager", "CategoryManagerDialog")
        self.setWindowTitle("Manage Categories")
        self.resize(760, 520)
        self._build_ui()
        self._restore_window_size()
        self._load_categories()

    @property
    def has_changes(self) -> bool:
        """Return True when categories were created or updated."""
        return self._has_changes

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        helper = QLabel(
            "Create, edit, and archive prompt categories. Changes apply immediately "
            "and update linked prompts.",
            self,
        )
        helper.setWordWrap(True)
        layout.addWidget(helper)

        self._table = QTableWidget(self)
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels(
            ["Label", "Slug", "Description", "Parent", "Min quality", "Status"]
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.itemSelectionChanged.connect(self._update_button_states)  # type: ignore[arg-type]
        layout.addWidget(self._table, stretch=1)

        button_row = QHBoxLayout()
        self._add_button = QPushButton("Add", self)
        self._add_button.clicked.connect(self._on_add_category)  # type: ignore[arg-type]
        button_row.addWidget(self._add_button)

        self._edit_button = QPushButton("Edit", self)
        self._edit_button.clicked.connect(self._on_edit_category)  # type: ignore[arg-type]
        button_row.addWidget(self._edit_button)

        self._toggle_button = QPushButton("Archive", self)
        self._toggle_button.clicked.connect(self._on_toggle_category)  # type: ignore[arg-type]
        button_row.addWidget(self._toggle_button)

        button_row.addStretch(1)

        self._refresh_button = QPushButton("Refresh", self)
        self._refresh_button.clicked.connect(self._on_refresh_clicked)  # type: ignore[arg-type]
        button_row.addWidget(self._refresh_button)
        layout.addLayout(button_row)

        close_buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        close_buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        layout.addWidget(close_buttons)

        self._update_button_states()

    def _restore_window_size(self) -> None:
        width = self._settings.value("width", type=int)
        height = self._settings.value("height", type=int)
        if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
            self.resize(width, height)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Persist the current geometry when closing the manager."""
        self._settings.setValue("width", self.width())
        self._settings.setValue("height", self.height())
        super().closeEvent(event)

    def _load_categories(self) -> None:
        """Populate table with repository categories."""
        try:
            categories = self._manager.list_categories(include_archived=True)
        except CategoryStorageError as exc:
            QMessageBox.critical(self, "Unable to load categories", str(exc))
            categories = []
        self._categories = categories
        self._table.setRowCount(len(categories))
        for row, category in enumerate(categories):
            self._table.setItem(row, 0, QTableWidgetItem(category.label))
            self._table.setItem(row, 1, QTableWidgetItem(category.slug))
            self._table.setItem(row, 2, QTableWidgetItem(category.description))
            self._table.setItem(row, 3, QTableWidgetItem(category.parent_slug or "—"))
            min_quality = "—" if category.min_quality is None else f"{category.min_quality:.1f}"
            self._table.setItem(row, 4, QTableWidgetItem(min_quality))
            status = "Active" if category.is_active else "Archived"
            self._table.setItem(row, 5, QTableWidgetItem(status))
        self._table.resizeColumnsToContents()
        self._update_button_states()

    def _selected_category(self) -> PromptCategory | None:
        """Return the currently selected category."""
        selected_rows = self._table.selectionModel().selectedRows()
        if not selected_rows:
            return None
        index = int(selected_rows[0].row())
        if index < 0 or index >= len(self._categories):
            return None
        return self._categories[index]

    def _update_button_states(self) -> None:
        """Enable/disable actions based on selection."""
        category = self._selected_category()
        has_selection = category is not None
        self._edit_button.setEnabled(has_selection)
        self._toggle_button.setEnabled(has_selection)
        if category is None:
            self._toggle_button.setText("Archive")
        else:
            self._toggle_button.setText("Archive" if category.is_active else "Activate")

    def _on_refresh_clicked(self) -> None:
        """Reload categories from the registry."""
        try:
            self._manager.refresh_categories()
        except CategoryStorageError as exc:
            QMessageBox.warning(self, "Refresh failed", str(exc))
        self._load_categories()

    def _on_add_category(self) -> None:
        """Open the category editor and persist a new category."""
        dialog = CategoryEditorDialog(self)
        if not dialog.exec():
            return
        data = dialog.payload
        try:
            self._manager.create_category(**data)
        except CategoryStorageError as exc:
            QMessageBox.critical(self, "Create failed", str(exc))
            return
        self._has_changes = True
        self._load_categories()

    def _on_edit_category(self) -> None:
        """Edit the selected category."""
        category = self._selected_category()
        if category is None:
            return
        dialog = CategoryEditorDialog(self, category=category)
        if not dialog.exec():
            return
        data = dialog.payload
        try:
            self._manager.update_category(
                category.slug,
                label=data["label"],
                description=data["description"],
                parent_slug=data["parent_slug"],
                color=data["color"],
                icon=data["icon"],
                min_quality=data["min_quality"],
                default_tags=data["default_tags"],
                is_active=data["is_active"],
            )
        except CategoryNotFoundError as exc:
            QMessageBox.warning(self, "Edit failed", str(exc))
            return
        except CategoryStorageError as exc:
            QMessageBox.critical(self, "Edit failed", str(exc))
            return
        self._has_changes = True
        self._load_categories()

    def _on_toggle_category(self) -> None:
        """Archive or activate the selected category."""
        category = self._selected_category()
        if category is None:
            return
        try:
            self._manager.set_category_active(category.slug, not category.is_active)
        except CategoryNotFoundError as exc:
            QMessageBox.warning(self, "Update failed", str(exc))
            return
        except CategoryStorageError as exc:
            QMessageBox.critical(self, "Update failed", str(exc))
            return
        self._has_changes = True
        self._load_categories()


__all__ = ["CategoryEditorDialog", "CategoryManagerDialog"]
