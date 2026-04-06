"""Custom delegate for bounded retrieval previews in the main prompt list.

Updates:
  v0.1.0 - 2026-04-06 - Render a muted second preview line when the model exposes one.
"""

from __future__ import annotations

from typing import Any, cast

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPalette
from PySide6.QtWidgets import QApplication, QStyle, QStyledItemDelegate, QStyleOptionViewItem

from .prompt_list_model import PromptListModel


class PromptListDelegate(QStyledItemDelegate):
    """Render the main prompt list with one optional muted preview line."""

    _VERTICAL_PADDING = 6
    _LINE_SPACING = 2

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index,
    ) -> None:
        """Draw the standard row plus one compact preview line when available."""
        preview = index.data(PromptListModel.PreviewRole)
        if not isinstance(preview, str) or not preview:
            super().paint(painter, option, index)
            return

        item_option = QStyleOptionViewItem(option)
        self.initStyleOption(item_option, index)
        title = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        item_option_any = cast("Any", item_option)
        item_option_any.text = ""

        style = (
            item_option_any.widget.style()
            if item_option_any.widget is not None
            else QApplication.style()
        )
        painter.save()
        style.drawControl(
            QStyle.ControlElement.CE_ItemViewItem,
            item_option,
            painter,
            item_option_any.widget,
        )

        text_rect = style.subElementRect(
            QStyle.SubElement.SE_ItemViewItemText,
            item_option,
            item_option_any.widget,
        )
        title_font = cast("QFont", item_option_any.font)
        preview_font = self._preview_font(title_font)
        title_metrics = QFontMetrics(title_font)
        preview_metrics = QFontMetrics(preview_font)

        title_rect = QRect(text_rect)
        title_rect.setHeight(title_metrics.height())
        preview_rect = QRect(text_rect)
        preview_rect.setTop(title_rect.bottom() + self._LINE_SPACING)
        preview_rect.setHeight(preview_metrics.height())

        painter.setClipRect(text_rect)
        painter.setFont(title_font)
        painter.setPen(self._title_color(item_option_any.palette, item_option_any.state))
        painter.drawText(
            title_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            title_metrics.elidedText(title, Qt.TextElideMode.ElideRight, title_rect.width()),
        )

        painter.setFont(preview_font)
        painter.setPen(self._preview_color(item_option_any.palette, item_option_any.state))
        painter.drawText(
            preview_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            preview_metrics.elidedText(preview, Qt.TextElideMode.ElideRight, preview_rect.width()),
        )
        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index) -> QSize:
        """Return a taller row height only when preview text is available."""
        base_size = super().sizeHint(option, index)
        preview = index.data(PromptListModel.PreviewRole)
        if not isinstance(preview, str) or not preview:
            return base_size

        option_any = cast("Any", option)
        title_font = cast("QFont", option_any.font)
        title_metrics = QFontMetrics(title_font)
        preview_metrics = QFontMetrics(self._preview_font(title_font))
        height = (
            self._VERTICAL_PADDING * 2
            + title_metrics.height()
            + self._LINE_SPACING
            + preview_metrics.height()
        )
        return QSize(base_size.width(), max(base_size.height(), height))

    @staticmethod
    def _preview_font(font: QFont) -> QFont:
        """Return a slightly smaller font for preview text."""
        preview_font = QFont(font)
        preview_font.setPointSizeF(max(1.0, preview_font.pointSizeF() - 1.0))
        return preview_font

    @staticmethod
    def _title_color(palette: QPalette, state: QStyle.StateFlag) -> QColor:
        """Return the correct title color for selected and unselected rows."""
        role = (
            QPalette.ColorRole.HighlightedText
            if state & QStyle.StateFlag.State_Selected
            else QPalette.ColorRole.Text
        )
        return palette.color(role)

    @staticmethod
    def _preview_color(palette: QPalette, state: QStyle.StateFlag) -> QColor:
        """Return a muted preview color that still respects selection state."""
        role = (
            QPalette.ColorRole.HighlightedText
            if state & QStyle.StateFlag.State_Selected
            else QPalette.ColorRole.Text
        )
        color = QColor(palette.color(role))
        color.setAlpha(170 if state & QStyle.StateFlag.State_Selected else 180)
        return color


__all__ = ["PromptListDelegate"]
