"""Flow layout helper used by tabs with wrap-around action buttons.

Updates:
  v0.1.1 - 2025-12-08 - Align layout overrides with PySide6 typing requirements.
  v0.1.0 - 2025-11-30 - Extract FlowLayout from main window module.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout, QLayoutItem, QWidget, QWidgetItem


class FlowLayout(QLayout):
    """Layout that arranges widgets left-to-right and wraps on overflow."""

    def __init__(
        self, parent: QWidget | None = None, *, margin: int = 0, spacing: int = -1
    ) -> None:
        """Initialise the layout with optional *margin* and *spacing* overrides."""
        super().__init__(parent)
        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)
        self._item_list: list[QLayoutItem] = []
        default_spacing = spacing if spacing >= 0 else self.spacing()
        self.setSpacing(default_spacing if default_spacing >= 0 else 0)

    def addItem(self, item: QLayoutItem) -> None:
        """Qt hook that inserts a layout item."""
        self._item_list.append(item)

    def addWidget(self, widget: QWidget) -> None:
        """Convenience helper to add a QWidget."""
        self.addChildWidget(widget)
        self.addItem(QWidgetItem(widget))

    def count(self) -> int:
        """Return the number of managed items."""
        return len(self._item_list)

    def itemAt(self, index: int) -> QLayoutItem | None:
        """Return the item at *index* if it exists."""
        if 0 <= index < len(self._item_list):
            return self._item_list[index]
        return None

    def takeAt(self, index: int) -> QLayoutItem:
        """Remove and return the item at *index*."""
        if 0 <= index < len(self._item_list):
            return self._item_list.pop(index)
        raise IndexError("FlowLayout index out of range")

    def expandingDirections(self) -> Qt.Orientation:
        """FlowLayout never expands to fill space."""
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        """Report that height depends on width."""
        return True

    def heightForWidth(self, width: int) -> int:
        """Return required height for a given *width*."""
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:
        """Position child widgets within *rect*."""
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:
        """Return the preferred size for the layout."""
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        """Return the minimum size derived from child widgets."""
        size = QSize()
        for item in self._item_list:
            size = size.expandedTo(item.minimumSize())
        left, top, right, bottom = self.getContentsMargins()
        size += QSize(left + right, top + bottom)
        return size

    def _do_layout(self, rect: QRect, *, test_only: bool) -> int:
        """Lay out items either virtually (when *test_only*) or for rendering."""
        left, top, right, bottom = self.getContentsMargins()
        effective_rect = rect.adjusted(left, top, -right, -bottom)
        x = effective_rect.x()
        y = effective_rect.y()
        line_height = 0
        space_x = self.spacing()
        space_y = self.spacing()

        for item in self._item_list:
            widget = item.widget()
            if widget is None or not widget.isVisible():
                continue
            next_x = x + item.sizeHint().width() + space_x
            if next_x - space_x > effective_rect.right() and line_height > 0:
                x = effective_rect.x()
                y = y + line_height + space_y
                next_x = x + item.sizeHint().width() + space_x
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))
            x = next_x
            line_height = max(line_height, item.sizeHint().height())

        return y + line_height - rect.y() + bottom


__all__ = ["FlowLayout"]
