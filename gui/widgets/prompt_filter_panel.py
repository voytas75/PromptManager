"""Filter panel widget for category, tag, quality, and sort controls.

Updates:
  v0.1.0 - 2025-11-30 - Extract reusable prompt filter panel widget.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QWidget,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from models.category_model import PromptCategory
else:  # pragma: no cover - runtime fallback for type checking aids
    PromptCategory = Any


class PromptFilterPanel(QWidget):
    """Expose filter inputs and emit signals when the user changes them."""

    filters_changed = Signal()
    sort_changed = Signal(str)
    manage_categories_requested = Signal()

    def __init__(
        self,
        *,
        sort_options: Sequence[tuple[str, str]],
        parent: QWidget | None = None,
    ) -> None:
        """Build filter controls and wire their change signals."""
        super().__init__(parent)
        if not isinstance(sort_options, Sequence):  # pragma: no cover - defensive
            raise TypeError("sort_options must be a sequence of (label, value) tuples.")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Category:", self))
        self._category_combo = QComboBox(self)
        self._category_combo.addItem("All categories", None)
        self._category_combo.currentIndexChanged.connect(self.filters_changed)  # type: ignore[arg-type]
        layout.addWidget(self._category_combo)

        self._manage_button = QToolButton(self)
        self._manage_button.setText("Manage")
        self._manage_button.setToolTip("Manage prompt categories.")
        self._manage_button.clicked.connect(self.manage_categories_requested)  # type: ignore[arg-type]
        layout.addWidget(self._manage_button)

        layout.addWidget(QLabel("Tag:", self))
        self._tag_combo = QComboBox(self)
        self._tag_combo.addItem("All tags", None)
        self._tag_combo.currentIndexChanged.connect(self.filters_changed)  # type: ignore[arg-type]
        layout.addWidget(self._tag_combo)

        layout.addWidget(QLabel("Quality ≥", self))
        self._quality_spin = QDoubleSpinBox(self)
        self._quality_spin.setRange(0.0, 10.0)
        self._quality_spin.setDecimals(1)
        self._quality_spin.setSingleStep(0.1)
        self._quality_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self._quality_spin.setAlignment(Qt.AlignRight)
        self._quality_spin.setMinimumWidth(
            self._quality_spin.fontMetrics().horizontalAdvance("10.0") + 32
        )
        self._quality_spin.valueChanged.connect(self.filters_changed)  # type: ignore[arg-type]
        layout.addWidget(self._quality_spin)

        layout.addWidget(QLabel("Sort:", self))
        self._sort_combo = QComboBox(self)
        for label, value in sort_options:
            self._sort_combo.addItem(label, value)
        self._sort_combo.currentIndexChanged.connect(self._emit_sort_changed)  # type: ignore[arg-type]
        layout.addWidget(self._sort_combo)

        layout.addStretch(1)

    def set_categories(
        self,
        categories: Sequence[PromptCategory],
        selected_slug: str | None = None,
    ) -> None:
        """Populate the category combo box with *categories*."""
        self._category_combo.blockSignals(True)
        try:
            self._category_combo.clear()
            self._category_combo.addItem("All categories", None)
            for category in categories:
                self._category_combo.addItem(category.label, category.slug)
            target = selected_slug or ""
            index = self._category_combo.findData(target) if target else 0
            self._category_combo.setCurrentIndex(index if index != -1 else 0)
        finally:
            self._category_combo.blockSignals(False)

    def set_tags(self, tags: Sequence[str], selected_tag: str | None = None) -> None:
        """Populate the tag combo box with *tags*."""
        self._tag_combo.blockSignals(True)
        try:
            self._tag_combo.clear()
            self._tag_combo.addItem("All tags", None)
            for tag in tags:
                self._tag_combo.addItem(tag, tag)
            target = selected_tag or ""
            index = self._tag_combo.findData(target) if target else 0
            self._tag_combo.setCurrentIndex(index if index != -1 else 0)
        finally:
            self._tag_combo.blockSignals(False)

    def set_min_quality(self, value: float) -> None:
        """Set the numeric quality threshold without emitting signals."""
        previous = self._quality_spin.blockSignals(True)
        try:
            self._quality_spin.setValue(value)
        finally:
            self._quality_spin.blockSignals(previous)

    def category_slug(self) -> str | None:
        """Return the currently selected category slug."""
        return self._clean_text(self._category_combo.currentData())

    def tag_value(self) -> str | None:
        """Return the currently selected tag value."""
        return self._clean_text(self._tag_combo.currentData())

    def min_quality(self) -> float:
        """Return the minimum quality threshold."""
        return float(self._quality_spin.value())

    def sort_value(self) -> str | None:
        """Return the active sort option identifier."""
        return self._clean_text(self._sort_combo.currentData())

    def set_sort_value(self, value: str | None) -> None:
        """Select the sort option identified by *value*."""
        target = value or ""
        index = self._sort_combo.findData(target)
        if index >= 0:
            self._sort_combo.setCurrentIndex(index)

    def set_sort_enabled(self, enabled: bool) -> None:
        """Enable or disable manual sort selection."""
        self._sort_combo.setEnabled(enabled)

    def is_sort_enabled(self) -> bool:
        """Return True when the sort combo box accepts user input."""
        return self._sort_combo.isEnabled()

    def _emit_sort_changed(self) -> None:
        value = self.sort_value()
        if value is None:
            return
        self.sort_changed.emit(value)

    @staticmethod
    def _clean_text(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None
