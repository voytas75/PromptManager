"""Transient processing overlays for long-running GUI tasks.

Updates:
  v0.1.1 - 2025-11-27 - Run blocking tasks on a worker thread so the UI stays responsive.
  v0.1.0 - 2025-11-27 - Add reusable busy indicator for prompt workflows.
"""

from __future__ import annotations

import threading
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Callable, Optional, Type, TypeVar

from PySide6.QtCore import QEventLoop, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QProgressDialog, QWidget

_T = TypeVar("_T")


class ProcessingIndicator(AbstractContextManager["ProcessingIndicator"]):
    """Display a modal busy dialog while scoped work executes."""

    def __init__(self, parent: QWidget, message: str, *, title: str = "Processing") -> None:
        self._dialog = QProgressDialog(parent)
        self._dialog.setWindowTitle(title)
        self._dialog.setLabelText(message)
        self._dialog.setRange(0, 0)
        self._dialog.setCancelButton(None)
        self._dialog.setAutoClose(False)
        self._dialog.setAutoReset(False)
        self._dialog.setMinimumDuration(0)
        self._dialog.setWindowModality(Qt.ApplicationModal)
        self._dialog.setWindowFlag(Qt.WindowTitleHint, True)
        self._dialog.setWindowFlag(Qt.WindowSystemMenuHint, False)

    def __enter__(self) -> "ProcessingIndicator":
        self._dialog.show()
        QGuiApplication.processEvents(QEventLoop.AllEvents)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        self._dialog.hide()
        self._dialog.deleteLater()
        return False

    def run(self, func: Callable[..., _T], *args, **kwargs) -> _T:
        """Execute *func* on a worker thread while keeping the UI responsive."""

        with self:
            return self._run_in_thread(func, *args, **kwargs)

    def _run_in_thread(self, func: Callable[..., _T], *args, **kwargs) -> _T:
        event = threading.Event()
        result: dict[str, _T] = {}
        error: list[BaseException] = []

        def _worker() -> None:
            try:
                result["value"] = func(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - propagate original failure
                error.append(exc)
            finally:
                event.set()

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        try:
            while not event.is_set():
                QGuiApplication.processEvents(QEventLoop.AllEvents, 50)
                event.wait(0.01)
        finally:
            thread.join()
        if error:
            raise error[0]
        if "value" not in result:  # pragma: no cover - defensive guard
            raise RuntimeError("Processing result missing")
        return result["value"]

    def update_message(self, message: str) -> None:
        """Refresh the indicator text while the dialog is visible."""

        self._dialog.setLabelText(message)
        QGuiApplication.processEvents(QEventLoop.AllEvents)


__all__ = ["ProcessingIndicator"]
