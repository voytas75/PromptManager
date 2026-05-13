"""Custom delegate for bounded retrieval previews in the main prompt list.

Updates:
  v0.2.0 - 2026-04-12 - Add subtle active-search emphasis for matching title and preview text.
  v0.1.1 - 2026-04-11 - Keep the second preview line
  at the row base font size for better readability.
  v0.1.0 - 2026-04-06 - Render a muted second preview line when the model exposes one.
"""

from __future__ import annotations

from typing import Any, cast

from PySide6.QtCore import (
    QModelIndex,
    QPersistentModelIndex as _QPersistentModelIndex,
    QPointF,
    QRect,
    QSize,
    Qt,
)
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPalette
from PySide6.QtWidgets import QApplication, QStyle, QStyledItemDelegate, QStyleOptionViewItem

from .prompt_list_model import PromptListModel

type IndexLike = QModelIndex | _QPersistentModelIndex


class PromptListDelegate(QStyledItemDelegate):
    """Render the main prompt list with one optional muted preview line."""

    _VERTICAL_PADDING = 6
    _LINE_SPACING = 2

    @staticmethod
    def preview_font(font: QFont) -> QFont:
        """Public helper exposing the preview font choice for bounded tests."""
        return PromptListDelegate._preview_font(font)

    @staticmethod
    def _data_as_text(index: IndexLike, role: int) -> str | None:
        """Return one role value as visible text when the model exposes a non-empty string."""
        value = cast("object", index.data(role))
        return value if isinstance(value, str) and value else None

    @staticmethod
    def handoff_cue_text(index: IndexLike) -> str | None:
        """Public helper exposing the visible row handoff cue for bounded tests."""
        return PromptListDelegate._data_as_text(index, PromptListModel.HandoffCueRole)

    @staticmethod
    def build_text_runs(
        text: str,
        spans: tuple[tuple[int, int], ...],
    ) -> tuple[tuple[str, bool], ...]:
        """Public helper exposing bounded emphasis-run building for tests."""
        return PromptListDelegate._build_text_runs(text, spans)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: IndexLike,
    ) -> None:
        """Draw the standard row plus bounded preview and handoff cue lines when available."""
        preview = self._data_as_text(index, PromptListModel.PreviewRole)
        handoff_cue = self.handoff_cue_text(index)
        title_spans = self._coerce_match_spans(
            cast("object", index.data(PromptListModel.TitleMatchRole))
        )
        preview_spans = self._coerce_match_spans(
            cast("object", index.data(PromptListModel.PreviewMatchRole))
        )
        has_preview = preview is not None
        has_handoff_cue = handoff_cue is not None
        if not has_preview and not title_spans and not has_handoff_cue:
            super().paint(painter, option, index)
            return

        item_option = QStyleOptionViewItem(option)
        self.initStyleOption(item_option, index)
        title = str(cast("object", index.data(Qt.ItemDataRole.DisplayRole)) or "")
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
        handoff_font = self._handoff_cue_font(title_font)
        title_metrics = QFontMetrics(title_font)
        preview_metrics = QFontMetrics(preview_font)
        handoff_metrics = QFontMetrics(handoff_font)

        title_rect = QRect(text_rect)
        title_rect.setHeight(title_metrics.height())
        preview_rect = QRect(text_rect)
        preview_rect.setTop(title_rect.bottom() + self._LINE_SPACING)
        preview_rect.setHeight(preview_metrics.height())
        handoff_rect = QRect(text_rect)
        handoff_rect.setTop(preview_rect.bottom() + self._LINE_SPACING)
        handoff_rect.setHeight(handoff_metrics.height())

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

        if has_handoff_cue:
            handoff_text, handoff_spans = self._elide_text_and_spans(
                handoff_cue,
                (),
                handoff_metrics,
                handoff_rect.width(),
            )
            self._draw_text_runs(
                painter,
                handoff_rect,
                handoff_font,
                self._handoff_cue_color(item_option_any.palette, item_option_any.state),
                handoff_text,
                handoff_spans,
            )
        painter.restore()

    def sizeHint(
        self,
        option: QStyleOptionViewItem,
        index: IndexLike,
    ) -> QSize:
        """Return a taller row height when preview text or a visible handoff cue is available."""
        base_size = super().sizeHint(option, index)
        preview = self._data_as_text(index, PromptListModel.PreviewRole)
        handoff_cue = self.handoff_cue_text(index)
        has_preview = preview is not None
        has_handoff_cue = handoff_cue is not None
        if not has_preview and not has_handoff_cue:
            return base_size

        option_any = cast("Any", option)
        title_font = cast("QFont", option_any.font)
        title_metrics = QFontMetrics(title_font)
        height = self._VERTICAL_PADDING * 2 + title_metrics.height()
        if has_preview:
            preview_metrics = QFontMetrics(self._preview_font(title_font))
            height += self._LINE_SPACING + preview_metrics.height()
        if has_handoff_cue:
            handoff_metrics = QFontMetrics(self._handoff_cue_font(title_font))
            height += self._LINE_SPACING + handoff_metrics.height()
        return QSize(base_size.width(), max(base_size.height(), height))

    @staticmethod
    def _preview_font(font: QFont) -> QFont:
        """Return the same readable base font for preview text."""
        return QFont(font)

    @staticmethod
    def _handoff_cue_font(font: QFont) -> QFont:
        """Return a subtle readable font for the bounded handoff cue line."""
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
    def _handoff_cue_color(palette: QPalette, state: QStyle.StateFlag) -> QColor:
        """Return a muted color for the bounded handoff cue line."""
        role = (
            QPalette.ColorRole.HighlightedText
            if state & QStyle.StateFlag.State_Selected
            else QPalette.ColorRole.Text
        )
        color = QColor(palette.color(role))
        color.setAlpha(150 if state & QStyle.StateFlag.State_Selected else 160)
        return color

    @staticmethod
    def _coerce_match_spans(value: object | None) -> tuple[tuple[int, int], ...]:
        """Normalize model-provided match spans into a validated tuple."""
        if not isinstance(value, (list, tuple)):
            return ()
        spans: list[tuple[int, int]] = []
        for candidate_any in cast("tuple[object, ...]", tuple(cast("Any", value))):
            if not isinstance(candidate_any, tuple):
                continue
            candidate_items = cast("tuple[object, ...]", candidate_any)
            if len(candidate_items) != 2:
                continue
            start = candidate_items[0]
            length = candidate_items[1]
            if isinstance(start, int) and isinstance(length, int) and start >= 0 and length > 0:
                spans.append((start, length))
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
