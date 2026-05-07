"""Filter panel widget for category, tag, favorite, quality, and sort controls.

Updates:
  v0.1.1 - 2025-12-08 - Adopt ButtonSymbols enum for quality spin box typing.
  v0.1.0 - 2025-11-30 - Extract reusable prompt filter panel widget.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QWidget,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

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
        self._tag_combo.setObjectName("tagFilterCombo")
        self._tag_combo.addItem("All tags", None)
        self._tag_combo.currentIndexChanged.connect(self._handle_tag_changed)  # type: ignore[arg-type]
        layout.addWidget(self._tag_combo)

        self._tag_visibility_label = QLabel("Tag filter: all tags", self)
        self._tag_visibility_label.setObjectName("tagFilterVisibilityLabel")
        layout.addWidget(self._tag_visibility_label)

        self._active_narrowing_summary_label = QLabel("Showing all prompts", self)
        self._active_narrowing_summary_label.setObjectName("activeNarrowingSummaryLabel")
        layout.addWidget(self._active_narrowing_summary_label)
        self._active_search_text: str = ""

        self._sort_visibility_label = QLabel("Sort: manual", self)
        self._sort_visibility_label.setObjectName("sortFilterVisibilityLabel")

        self._favorites_only_checkbox = QCheckBox("Favorites only", self)
        self._favorites_only_checkbox.setObjectName("favoritesOnlyFilterCheckbox")
        self._favorites_only_checkbox.toggled.connect(self._handle_favorites_toggled)  # type: ignore[arg-type]
        layout.addWidget(self._favorites_only_checkbox)

        self._favorites_visibility_label = QLabel("Favorites filter: all prompts", self)
        self._favorites_visibility_label.setObjectName("favoritesFilterVisibilityLabel")
        layout.addWidget(self._favorites_visibility_label)

        layout.addWidget(QLabel("Quality ≥", self))
        self._quality_spin = QDoubleSpinBox(self)
        self._quality_spin.setRange(0.0, 10.0)
        self._quality_spin.setDecimals(1)
        self._quality_spin.setSingleStep(0.1)
        self._quality_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self._quality_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
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
        layout.addWidget(self._sort_visibility_label)

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
        self._update_tag_visibility_label()
        self._update_active_narrowing_summary_label()

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

    def favorites_only(self) -> bool:
        """Return True when the list should show favorite prompts only."""
        return self._favorites_only_checkbox.isChecked()

    def set_favorites_only(self, value: bool) -> None:
        """Toggle the favorites-only filter without emitting signals."""
        previous = self._favorites_only_checkbox.blockSignals(True)
        try:
            self._favorites_only_checkbox.setChecked(bool(value))
        finally:
            self._favorites_only_checkbox.blockSignals(previous)
        self._update_favorites_visibility_label()
        self._update_active_narrowing_summary_label()

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
        if enabled:
            self._active_search_text = ""
        self._update_sort_visibility_label()
        self._update_active_narrowing_summary_label()

    def set_active_search_text(self, text: str | None) -> None:
        """Store the current active search text for continuity cues."""
        self._active_search_text = str(text or "").strip()
        self._update_active_narrowing_summary_label()

    def is_sort_enabled(self) -> bool:
        """Return True when the sort combo box accepts user input."""
        return self._sort_combo.isEnabled()

    def _emit_sort_changed(self) -> None:
        value = self.sort_value()
        if value is None:
            return
        self.sort_changed.emit(value)

    def _handle_tag_changed(self) -> None:
        self._update_tag_visibility_label()
        self._update_active_narrowing_summary_label()
        self.filters_changed.emit()

    def _handle_favorites_toggled(self) -> None:
        self._update_favorites_visibility_label()
        self._update_active_narrowing_summary_label()
        self.filters_changed.emit()

    def _update_tag_visibility_label(self) -> None:
        active_tag = self.tag_value()
        if active_tag is None:
            self._tag_visibility_label.setText("Tag filter: all tags")
            return
        self._tag_visibility_label.setText(f"Tag filter: {active_tag}")

    def _update_favorites_visibility_label(self) -> None:
        if self.favorites_only():
            self._favorites_visibility_label.setText("Favorites filter: favorites only")
            return
        self._favorites_visibility_label.setText("Favorites filter: all prompts")

    def _update_active_narrowing_summary_label(self) -> None:
        parts: list[str] = []
        if self._active_search_text:
            parts.append(f"search: {self._active_search_text}")
        active_tag = self.tag_value()
        if active_tag is not None:
            parts.append(f"tag: {active_tag}")
        if self.favorites_only():
            parts.append("favorites only")
        if not parts:
            self._active_narrowing_summary_label.setText("Showing all prompts")
            return
        summary = " • ".join(parts)
        self._active_narrowing_summary_label.setText(f"Showing prompts narrowed by {summary}")

    def _update_sort_visibility_label(self) -> None:
        if self.is_sort_enabled():
            self._sort_visibility_label.setText("Sort: manual")
            return
        self._sort_visibility_label.setText("Sort locked during search")

    @staticmethod
    def _clean_text(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None
