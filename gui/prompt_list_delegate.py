"""Custom delegate for bounded retrieval previews in the main prompt list.

Updates:
  v0.2.0 - 2026-04-12 - Add subtle active-search emphasis for matching title and preview text.
  v0.1.1 - 2026-04-11 - Keep the second preview line
  at the row base font size for better readability.
  v0.1.0 - 2026-04-06 - Render a muted second preview line when the model exposes one.
"""

from __future__ import annotations

from typing import Any, cast

from PySide6.QtCore import QPointF, QRect, QSize, Qt
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
        title_spans = self._coerce_match_spans(index.data(PromptListModel.TitleMatchRole))
        preview_spans = self._coerce_match_spans(index.data(PromptListModel.PreviewMatchRole))
        has_preview = isinstance(preview, str) and bool(preview)
        if not has_preview and not title_spans:
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
        title_text, title_spans = self._elide_text_and_spans(
            title,
            title_spans,
            title_metrics,
            title_rect.width(),
        )
        self._draw_text_runs(
            painter,
            title_rect,
            title_font,
            self._title_color(item_option_any.palette, item_option_any.state),
            title_text,
            title_spans,
        )

        if has_preview:
            preview_text, preview_spans = self._elide_text_and_spans(
                preview,
                preview_spans,
                preview_metrics,
                preview_rect.width(),
            )
            self._draw_text_runs(
                painter,
                preview_rect,
                preview_font,
                self._preview_color(item_option_any.palette, item_option_any.state),
                preview_text,
                preview_spans,
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
        """Return the same readable base font for preview text."""
        return QFont(font)

    @staticmethod
    def _highlight_font(font: QFont) -> QFont:
        """Return a subtle emphasis font that stays selection-safe."""
        highlighted = QFont(font)
        highlighted.setWeight(QFont.Weight.DemiBold)
        return highlighted

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

    @staticmethod
    def _coerce_match_spans(value: object | None) -> tuple[tuple[int, int], ...]:
        """Normalize model-provided match spans into a validated tuple."""
        if not isinstance(value, (list, tuple)):
            return ()
        spans: list[tuple[int, int]] = []
        for candidate in value:
            if (
                isinstance(candidate, tuple)
                and len(candidate) == 2
                and isinstance(candidate[0], int)
                and isinstance(candidate[1], int)
                and candidate[0] >= 0
                and candidate[1] > 0
            ):
                spans.append(candidate)
        return tuple(spans)

    @staticmethod
    def _elide_text_and_spans(
        text: str,
        spans: tuple[tuple[int, int], ...],
        metrics: QFontMetrics,
        width: int,
    ) -> tuple[str, tuple[tuple[int, int], ...]]:
        """Elide text to the available width and clip match spans to the visible prefix."""
        if width <= 0:
            return "", ()

        elided = metrics.elidedText(text, Qt.TextElideMode.ElideRight, width)
        if elided == text:
            return text, spans

        visible_length = len(elided) - 1 if elided.endswith("…") else len(elided)
        clipped: list[tuple[int, int]] = []
        for start, length in spans:
            if start >= visible_length:
                continue
            clipped_length = min(length, visible_length - start)
            if clipped_length > 0:
                clipped.append((start, clipped_length))
        return elided, tuple(clipped)

    @staticmethod
    def _build_text_runs(
        text: str,
        spans: tuple[tuple[int, int], ...],
    ) -> tuple[tuple[str, bool], ...]:
        """Split text into plain and emphasized fragments."""
        if not text:
            return ()
        if not spans:
            return ((text, False),)

        runs: list[tuple[str, bool]] = []
        cursor = 0
        for start, length in spans:
            if start > cursor:
                runs.append((text[cursor:start], False))
            end = min(len(text), start + length)
            if end > start:
                runs.append((text[start:end], True))
            cursor = max(cursor, end)
        if cursor < len(text):
            runs.append((text[cursor:], False))
        return tuple(run for run in runs if run[0])

    def _draw_text_runs(
        self,
        painter: QPainter,
        rect: QRect,
        font: QFont,
        color: QColor,
        text: str,
        spans: tuple[tuple[int, int], ...],
    ) -> None:
        """Draw text fragments with subtle emphasis on matching spans."""
        base_metrics = QFontMetrics(font)
        y = rect.top() + ((rect.height() - base_metrics.height()) / 2) + base_metrics.ascent()
        x = float(rect.left())

        painter.setPen(color)
        for fragment, is_highlighted in self._build_text_runs(text, spans):
            current_font = self._highlight_font(font) if is_highlighted else font
            painter.setFont(current_font)
            painter.drawText(QPointF(x, y), fragment)
            x += QFontMetrics(current_font).horizontalAdvance(fragment)


__all__ = ["PromptListDelegate"]
