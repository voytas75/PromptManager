"""Application-wide notification infrastructure for long-running tasks.

Updates:
  v0.1.2 - 2025-11-29 - Move Callable/Iterator imports under TYPE_CHECKING per lint.
  v0.1.1 - 2025-11-29 - Reformat docstring and wrap subscription constructor signature.
  v0.1.0 - 2025-11-11 - Introduce notification hub with task tracking helpers.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

logger = logging.getLogger("prompt_manager.notifications")


class NotificationLevel(str, Enum):
    """Severity levels communicated to listeners."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


class NotificationStatus(str, Enum):
    """High-level lifecycle stage for a task notification."""
    STARTED = "started"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(slots=True)
class Notification:
    """Immutable payload describing a notification event."""
    id: uuid.UUID
    title: str
    message: str
    level: NotificationLevel
    status: NotificationStatus
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    task_id: str | None = None
    duration_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable representation of the notification."""
        return {
            "id": str(self.id),
            "title": self.title,
            "message": self.message,
            "level": self.level.value,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "task_id": self.task_id,
            "duration_ms": self.duration_ms,
            "metadata": dict(self.metadata),
        }


class NotificationSubscription:
    """Disposable handle that removes its callback when closed."""
    def __init__(
        self,
        center: NotificationCenter,
        callback: Callable[[Notification], None],
    ) -> None:
        """Store *center* subscription metadata for later cleanup."""
        self._center = center
        self._callback = callback
        self._closed = False

    def close(self) -> None:
        """Detach the stored callback if it is still active."""
        if self._closed:
            return
        self._closed = True
        self._center.unsubscribe(self._callback)

    def __enter__(self) -> NotificationSubscription:
        """Return the subscription so it can be used as a context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Ensure the callback is removed when leaving the context."""
        self.close()


class NotificationCenter:
    """Thread-safe publish/subscribe hub for sending notifications to listeners."""
    def __init__(self, history_limit: int = 200) -> None:
        """Initialise the subscriber registry and bounded history queue."""
        self._subscribers: list[Callable[[Notification], None]] = []
        self._lock = threading.RLock()
        self._history: deque[Notification] = deque(maxlen=history_limit)

    def subscribe(self, callback: Callable[[Notification], None]) -> NotificationSubscription:
        """Register *callback* to receive future notifications."""
        with self._lock:
            self._subscribers.append(callback)
        return NotificationSubscription(self, callback)

    def unsubscribe(self, callback: Callable[[Notification], None]) -> None:
        """Remove a previously subscribed callback if present."""
        with self._lock:
            try:
                self._subscribers.remove(callback)
            except ValueError:
                pass

    def publish(self, notification: Notification) -> None:
        """Deliver *notification* to all registered subscribers."""
        with self._lock:
            self._history.append(notification)
            subscribers = list(self._subscribers)

        logger.debug(
            "Notification event",
            extra={
                "title": notification.title,
                "status": notification.status.value,
                "level": notification.level.value,
                "task_id": notification.task_id,
            },
        )

        for callback in subscribers:
            try:
                callback(notification)
            except Exception:  # pragma: no cover - defensive log to avoid cascading failures
                logger.exception("Notification subscriber raised an exception")

    def history(self) -> tuple[Notification, ...]:
        """Return a snapshot of stored notifications."""
        with self._lock:
            return tuple(self._history)

    @contextmanager
    def track_task(
        self,
        *,
        title: str,
        task_id: str | None = None,
        start_message: str,
        success_message: str,
        failure_message: str | None = None,
        metadata: dict[str, Any] | None = None,
        level: NotificationLevel = NotificationLevel.INFO,
        failure_level: NotificationLevel = NotificationLevel.ERROR,
    ) -> Iterator[None]:
        """Convenience context manager emitting start/success/failure events."""
        resolved_task_id = task_id or f"task:{uuid.uuid4()}"
        started_at = time.perf_counter()
        self.publish(
            Notification(
                id=uuid.uuid4(),
                title=title,
                message=start_message,
                level=level,
                status=NotificationStatus.STARTED,
                task_id=resolved_task_id,
                metadata=dict(metadata or {}),
            )
        )

        try:
            yield
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            message = failure_message or f"{title} failed"
            self.publish(
                Notification(
                    id=uuid.uuid4(),
                    title=title,
                    message=f"{message}: {exc}",
                    level=failure_level,
                    status=NotificationStatus.FAILED,
                    task_id=resolved_task_id,
                    duration_ms=duration_ms,
                    metadata=dict(metadata or {}),
                )
            )
            raise
        else:
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            self.publish(
                Notification(
                    id=uuid.uuid4(),
                    title=title,
                    message=success_message,
                    level=NotificationLevel.SUCCESS,
                    status=NotificationStatus.SUCCEEDED,
                    task_id=resolved_task_id,
                    duration_ms=duration_ms,
                    metadata=dict(metadata or {}),
                )
            )


notification_center = NotificationCenter()


__all__ = [
    "NotificationCenter",
    "Notification",
    "NotificationLevel",
    "NotificationStatus",
    "NotificationSubscription",
    "notification_center",
]
