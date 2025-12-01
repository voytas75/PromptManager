"""Overlay manager that keeps result action buttons aligned with the output view.

Updates:
  v0.1.3 - 2025-12-01 - Prevent deleted QTextEdit references during shutdown.
  v0.1.2 - 2025-12-01 - Guard overlay event filter when QTextEdit is destroyed.
  v0.1.1 - 2025-11-30 - Ensure module is packaged and docstring formatting matches guidelines.
  v0.1.0 - 2025-11-30 - Introduce ResultActionsOverlay helper for workspace output buttons.
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QHBoxLayout, QTextEdit, QWidget
from shiboken6 import Shiboken


class ResultActionsOverlay(QObject):
    """Manage the floating overlay that contains result action buttons."""

    def __init__(self, text_edit: QTextEdit | None) -> None:
        """Initialize the overlay manager for *text_edit*.

        Args:
            text_edit: Output widget that hosts the floating action buttons.
        """
        super().__init__(text_edit)
        self._text_edit = text_edit
        self._overlay: QWidget | None = None
        if text_edit is not None:
            text_edit.viewport().installEventFilter(self)
            text_edit.installEventFilter(self)
            text_edit.destroyed.connect(self._handle_text_edit_destroyed)

    def rebuild(self, buttons: tuple[QWidget, ...] | list[QWidget]) -> None:
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
        text_edit = self._text_edit
        if self._is_widget_alive(text_edit):
            text_edit.viewport().removeEventFilter(self)
            text_edit.removeEventFilter(self)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """Keep the overlay anchored whenever the viewport updates."""
        text_edit = self._text_edit
        if not self._is_widget_alive(text_edit):
            return False
        viewport = text_edit.viewport()
        if not self._is_widget_alive(viewport):
            return False
        if obj is viewport or obj is text_edit:
            if event.type() in {QEvent.Resize, QEvent.Show}:
                self._position_overlay()
        return super().eventFilter(obj, event)

    def _position_overlay(self) -> None:
        if not self._is_widget_alive(self._text_edit) or self._overlay is None:
            return
        viewport = self._text_edit.viewport()
        overlay = self._overlay
        if not self._is_widget_alive(viewport) or not self._is_widget_alive(overlay):
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

    def _handle_text_edit_destroyed(self, _obj: QObject | None = None) -> None:
        """Reset references once Qt deletes the observed QTextEdit."""
        self._text_edit = None
        self._overlay = None

    def _is_widget_alive(self, widget: QWidget | QObject | None) -> bool:
        """Return True if *widget* still wraps a living Qt object."""
        return widget is not None and Shiboken.isValid(widget)


__all__ = ["ResultActionsOverlay"]
