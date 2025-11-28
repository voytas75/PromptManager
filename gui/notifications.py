"""Qt helpers for surfacing core notification events in the GUI.

Updates: v0.2.0 - 2025-11-28 - Add background task center with progress indicators and live feed.
Updates: v0.1.1 - 2025-11-27 - Add toast confirmation for copying notification details.
Updates: v0.1.0 - 2025-11-11 - Introduce notification bridge and history dialog.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QClipboard, QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from core.notifications import Notification, NotificationCenter, NotificationStatus
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


class BackgroundTaskItemWidget(QWidget):
    """Compact widget showing task title, message, and progress."""

    def __init__(self, notification: Notification, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setSpacing(8)
        self._title = QLabel(notification.title, self)
        self._title.setStyleSheet("font-weight: 600;")
        self._status = QLabel("", self)
        self._status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header.addWidget(self._title)
        header.addWidget(self._status)
        layout.addLayout(header)

        self._message = QLabel(notification.message, self)
        self._message.setWordWrap(True)
        layout.addWidget(self._message)

        self._progress = QProgressBar(self)
        self._progress.setTextVisible(False)
        layout.addWidget(self._progress)

        self.update_notification(notification)

    def update_notification(self, notification: Notification) -> None:
        """Refresh labels and progress bar for the supplied notification."""

        self._title.setText(notification.title)
        self._message.setText(notification.message)
        status_text = notification.status.value.replace("_", " ").title()
        self._status.setText(status_text)
        if notification.status is NotificationStatus.STARTED:
            self._progress.setRange(0, 0)
            self._status.setStyleSheet("color: #2563eb;")
        elif notification.status is NotificationStatus.SUCCEEDED:
            self._progress.setRange(0, 1)
            self._progress.setValue(1)
            self._status.setStyleSheet("color: #059669;")
        elif notification.status is NotificationStatus.FAILED:
            self._progress.setRange(0, 1)
            self._progress.setValue(1)
            self._status.setStyleSheet("color: #dc2626;")
        else:
            self._progress.setRange(0, 1)
            self._progress.setValue(1)
            self._status.setStyleSheet("")


@dataclass(slots=True)
class _ActiveTaskEntry:
    item: QListWidgetItem
    widget: BackgroundTaskItemWidget


class BackgroundTaskCenterDialog(QDialog):
    """Live feed of background tasks with progress indicators."""

    _HISTORY_LIMIT = 200

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Background Tasks")
        self.resize(560, 420)

        self._active_items: Dict[str, _ActiveTaskEntry] = {}

        layout = QVBoxLayout(self)

        active_label = QLabel("Active Tasks", self)
        active_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(active_label)

        self._active_placeholder = QLabel("No active tasks", self)
        self._active_placeholder.setAlignment(Qt.AlignCenter)
        self._active_placeholder.setStyleSheet("color: #6b7280;")
        layout.addWidget(self._active_placeholder)

        self._active_list = QListWidget(self)
        self._active_list.setSelectionMode(QAbstractItemView.NoSelection)
        self._active_list.setSpacing(6)
        layout.addWidget(self._active_list)

        history_label = QLabel("Recent Activity", self)
        history_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(history_label)

        self._history_list = QListWidget(self)
        self._history_list.setSelectionMode(QAbstractItemView.NoSelection)
        layout.addWidget(self._history_list)

        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self._update_active_placeholder()

    def set_history(self, notifications: Sequence[Notification]) -> None:
        """Populate the activity feed from persisted notifications."""

        self._history_list.clear()
        ordered = list(notifications)[-self._HISTORY_LIMIT :]
        for notification in reversed(ordered):
            self._history_list.addItem(_format_notification(notification))

    def set_active_notifications(self, notifications: Sequence[Notification]) -> None:
        """Rebuild the active task list from current manager state."""

        self._active_list.clear()
        self._active_items.clear()
        for notification in notifications:
            if notification.task_id:
                self._create_or_update_task(notification)
        self._update_active_placeholder()

    def handle_notification(self, notification: Notification) -> None:
        """Update feed and task widgets for a new notification event."""

        self._prepend_history(notification)
        if notification.task_id:
            self._create_or_update_task(notification)
        self._update_active_placeholder()

    def _prepend_history(self, notification: Notification) -> None:
        entry = QListWidgetItem(_format_notification(notification))
        self._history_list.insertItem(0, entry)
        while self._history_list.count() > self._HISTORY_LIMIT:
            self._history_list.takeItem(self._history_list.count() - 1)

    def _create_or_update_task(self, notification: Notification) -> None:
        assert notification.task_id is not None  # for type-checkers
        existing = self._active_items.get(notification.task_id)
        if existing is None:
            widget = BackgroundTaskItemWidget(notification, self._active_list)
            item = QListWidgetItem(self._active_list)
            item.setSizeHint(widget.sizeHint())
            self._active_list.addItem(item)
            self._active_list.setItemWidget(item, widget)
            self._active_items[notification.task_id] = _ActiveTaskEntry(item=item, widget=widget)
            existing = self._active_items[notification.task_id]
        existing.widget.update_notification(notification)
        existing.item.setSizeHint(existing.widget.sizeHint())
        if notification.status in {NotificationStatus.SUCCEEDED, NotificationStatus.FAILED}:
            self._remove_task(notification.task_id)

    def _remove_task(self, task_id: str) -> None:
        entry = self._active_items.pop(task_id, None)
        if entry is None:
            return
        row = self._active_list.row(entry.item)
        if row >= 0:
            self._active_list.takeItem(row)

    def _update_active_placeholder(self) -> None:
        has_active = bool(self._active_items)
        self._active_placeholder.setVisible(not has_active)
        self._active_list.setVisible(has_active)


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
    "BackgroundTaskCenterDialog",
]
