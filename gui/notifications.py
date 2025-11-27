"""Qt helpers for surfacing core notification events in the GUI.

Updates: v0.1.1 - 2025-11-27 - Add toast confirmation for copying notification details.
Updates: v0.1.0 - 2025-11-11 - Introduce notification bridge and history dialog.
"""

from __future__ import annotations

from typing import Sequence

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QClipboard, QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
)

from core.notifications import Notification, NotificationCenter
from .toast import show_toast


class QtNotificationBridge(QObject):
    """Subscribe to core notifications and forward them via Qt signals."""

    notification_received: Signal = Signal(object)

    def __init__(self, center: NotificationCenter, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._center = center
        self._subscription = center.subscribe(self._forward)

    def _forward(self, notification: Notification) -> None:
        self.notification_received.emit(notification)

    def close(self) -> None:
        self._subscription.close()


class NotificationHistoryDialog(QDialog):
    """Modal dialog presenting recent notification events."""

    def __init__(self, notifications: Sequence[Notification], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Notifications")
        self.resize(520, 360)

        layout = QVBoxLayout(self)
        self._list = QListWidget(self)
        layout.addWidget(self._list)

        for notification in reversed(list(notifications)):
            text = _format_notification(notification)
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, notification)
            self._list.addItem(item)

        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        copy_button = button_box.addButton("Copy Details", QDialogButtonBox.ActionRole)
        copy_button.clicked.connect(self._copy_selected)  # type: ignore[arg-type]
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _copy_selected(self) -> None:
        selected = self._list.currentItem()
        if selected is None:
            return
        notification: Notification = selected.data(Qt.UserRole)
        clipboard: QClipboard = QGuiApplication.clipboard()
        clipboard.setText(_format_notification(notification, include_metadata=True))
        show_toast(self, "Notification details copied to clipboard.")


def _format_notification(notification: Notification, *, include_metadata: bool = False) -> str:
    status = notification.status.value.replace("_", " ").title()
    parts = [
        f"[{notification.timestamp.astimezone().strftime('%H:%M:%S')}] {notification.title}",
        f"Status: {status}",
        f"Message: {notification.message}",
    ]
    if notification.task_id:
        parts.append(f"Task ID: {notification.task_id}")
    if notification.duration_ms is not None:
        parts.append(f"Duration: {notification.duration_ms} ms")
    if include_metadata and notification.metadata:
        parts.append(f"Metadata: {notification.metadata}")
    return " | ".join(parts)


__all__ = [
    "QtNotificationBridge",
    "NotificationHistoryDialog",
]
