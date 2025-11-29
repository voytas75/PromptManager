"""Toast helpers used by the Prompt Manager GUI.

Updates: v0.1.0 - 2025-11-27 - Provide reusable toast notifications for transient feedback."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QLabel, QWidget


def show_toast(parent: QWidget | None, message: str, duration_ms: int = 2500) -> None:
    """Display a brief toast message anchored to the supplied parent widget."""

    if parent is None or not message:
        return
    toast = QLabel(message, parent)
    toast.setObjectName("toastNotification")
    toast.setStyleSheet(
        "background-color: rgba(17, 24, 39, 0.92);"
        "color: #f8fafc;"
        "padding: 8px 16px;"
        "border-radius: 8px;"
        "font-weight: 500;"
    )
    toast.setAttribute(Qt.WA_TransparentForMouseEvents)
    toast.adjustSize()
    parent_rect = parent.rect()
    x_pos = max((parent_rect.width() - toast.width()) // 2, 0)
    y_pos = max(parent_rect.height() - toast.height() - 24, 0)
    toast.move(x_pos, y_pos)
    toast.show()
    toast.raise_()
    QTimer.singleShot(duration_ms, toast.deleteLater)


__all__ = ["show_toast"]
