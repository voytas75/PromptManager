"""Overlay manager that keeps result action buttons aligned with the output view.

Updates:
  v0.1.1 - 2025-11-30 - Ensure module is packaged and docstring formatting matches guidelines.
  v0.1.0 - 2025-11-30 - Introduce ResultActionsOverlay helper for workspace output buttons.
"""
from __future__ import annotations

from typing import Sequence

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QHBoxLayout, QTextEdit, QWidget


class ResultActionsOverlay(QObject):
    """Manage the floating overlay that contains result action buttons."""

    def __init__(self, text_edit: QTextEdit | None) -> None:
        super().__init__(text_edit)
        self._text_edit = text_edit
        self._overlay: QWidget | None = None
        if text_edit is not None:
            text_edit.viewport().installEventFilter(self)
            text_edit.installEventFilter(self)

    def rebuild(self, buttons: Sequence[QWidget]) -> None:
        """Create or refresh the overlay with the provided *buttons*."""

        if self._text_edit is None:
            return
        if self._overlay is not None:
            self._overlay.deleteLater()
        overlay_parent = self._text_edit
        overlay = QWidget(overlay_parent)
        overlay.setObjectName("resultActionsOverlay")
        overlay.setAttribute(Qt.WA_StyledBackground, True)
        layout = QHBoxLayout(overlay)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        for button in buttons:
            layout.addWidget(button)
        self._overlay = overlay
        overlay.show()
        self._position_overlay()

    def delete(self) -> None:
        """Remove the overlay and detach event filters."""

        if self._overlay is not None:
            self._overlay.deleteLater()
            self._overlay = None
        if self._text_edit is not None:
            self._text_edit.viewport().removeEventFilter(self)
            self._text_edit.removeEventFilter(self)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """Keep the overlay anchored whenever the viewport updates."""

        if self._text_edit is not None:
            if obj in {self._text_edit.viewport(), self._text_edit}:
                if event.type() in {QEvent.Resize, QEvent.Show}:
                    self._position_overlay()
        return super().eventFilter(obj, event)

    def _position_overlay(self) -> None:
        if self._text_edit is None or self._overlay is None:
            return
        viewport = self._text_edit.viewport()
        overlay = self._overlay
        if viewport is None:
            return
        overlay.adjustSize()
        desired_width = overlay.width()
        desired_height = overlay.height()
        margin = 12
        viewport_geometry = viewport.geometry()
        x = viewport_geometry.x() + viewport_geometry.width() - desired_width - margin
        y = viewport_geometry.y() + viewport_geometry.height() - desired_height - margin
        if x < viewport_geometry.x():
            x = viewport_geometry.x() + max(0, viewport_geometry.width() - desired_width)
        if y < viewport_geometry.y():
            y = viewport_geometry.y() + max(0, viewport_geometry.height() - desired_height)
        overlay.move(x, y)
        overlay.raise_()


__all__ = ["ResultActionsOverlay"]
