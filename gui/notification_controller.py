"""Notification aggregation and task center coordination.

Updates:
  v0.1.0 - 2025-12-01 - Extracted notification indicator management from MainWindow.
"""
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from core.notifications import Notification, NotificationStatus

from .notifications import BackgroundTaskCenterDialog, QtNotificationBridge

if TYPE_CHECKING:
    from collections.abc import Iterable

    from PySide6.QtWidgets import QLabel, QWidget


class NotificationController:
    """Bridge core notifications into the Qt UI with history tracking."""
    def __init__(
        self,
        *,
        parent: QWidget,
        indicator: QLabel,
        notification_center,
        status_callback,
        toast_callback,
    ) -> None:
        """Bind Qt widgets to the shared notification center."""
        self._parent = parent
        self._indicator = indicator
        self._notification_center = notification_center
        self._status_callback = status_callback
        self._toast_callback = toast_callback
        self._history: deque[Notification] = deque(maxlen=200)
        self._active_notifications: dict[str, Notification] = {}
        self._task_center_dialog: BackgroundTaskCenterDialog | None = None
        self._bridge = QtNotificationBridge(notification_center, parent)
        self._bridge.notification_received.connect(self._handle_notification)  # type: ignore[attr-defined]

    def bootstrap_history(self, history: Iterable[Notification]) -> None:
        """Seed the indicator with existing notification history."""
        for note in history:
            self._history.append(note)
            self._update_active_notification(note)
        self._update_indicator()

    def show_task_center(self) -> None:
        """Present the background task center dialog."""
        dialog = self._ensure_task_center_dialog()
        dialog.set_history(tuple(self._history))
        dialog.set_active_notifications(tuple(self._active_notifications.values()))
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def close(self) -> None:
        """Release signal subscriptions and close dialogs."""
        if self._task_center_dialog is not None:
            self._task_center_dialog.close()
        self._bridge.close()
        self._toast_callback = lambda *_: None
        self._status_callback = lambda *_: None

    def _handle_notification(self, notification: Notification) -> None:
        self._history.append(notification)
        self._update_active_notification(notification)
        self._update_indicator()
        if self._task_center_dialog is not None:
            self._task_center_dialog.handle_notification(notification)
        message = self._format_notification_message(notification)
        duration = 0 if notification.status is NotificationStatus.STARTED else 5000
        self._status_callback(message, duration)
        if notification.task_id and notification.status in {
            NotificationStatus.SUCCEEDED,
            NotificationStatus.FAILED,
        }:
            toast_duration = 3500 if notification.status is NotificationStatus.SUCCEEDED else 4500
            self._toast_callback(message, toast_duration)

    def _update_active_notification(self, notification: Notification) -> None:
        task_id = notification.task_id
        if not task_id:
            return
        if notification.status is NotificationStatus.STARTED:
            self._active_notifications[task_id] = notification
        elif notification.status in {NotificationStatus.SUCCEEDED, NotificationStatus.FAILED}:
            self._active_notifications.pop(task_id, None)

    def _format_notification_message(self, notification: Notification) -> str:
        status = notification.status.value.replace("_", " ").title()
        return f"{notification.title}: {status} — {notification.message}"

    def _update_indicator(self) -> None:
        active = len(self._active_notifications)
        if active:
            self._indicator.setText(f"Tasks: {active}")
            self._indicator.setVisible(True)
        else:
            self._indicator.clear()
            self._indicator.setVisible(False)

    def _ensure_task_center_dialog(self) -> BackgroundTaskCenterDialog:
        if self._task_center_dialog is None:
            self._task_center_dialog = BackgroundTaskCenterDialog(self._parent)
        return self._task_center_dialog


__all__ = ["NotificationController"]
