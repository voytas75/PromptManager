"""Transient processing overlays for long-running GUI tasks.

Updates:
  v0.1.2 - 2025-11-27 - Center the busy progress bar to avoid stretching across the dialog.
  v0.1.1 - 2025-11-27 - Run blocking tasks on a worker thread so the UI stays responsive.
  v0.1.0 - 2025-11-27 - Add reusable busy indicator for prompt workflows.
"""

from __future__ import annotations

import threading
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, TypeVar

from PySide6.QtCore import QEventLoop, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QDialog, QLabel, QProgressBar, QVBoxLayout, QWidget

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

_T = TypeVar("_T")


class ProcessingIndicator(AbstractContextManager["ProcessingIndicator"]):
    """Display a modal busy dialog while scoped work executes."""

    def __init__(self, parent: QWidget, message: str, *, title: str = "Processing") -> None:
        """Create an indicator bound to *parent* with the provided message."""
        self._dialog = _ProcessingDialog(parent, title=title, message=message)

    def __enter__(self) -> ProcessingIndicator:
        """Show the dialog and pump the event loop before returning self."""
        self._dialog.show()
        QGuiApplication.processEvents(QEventLoop.AllEvents)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Hide and dispose of the dialog when exiting the context."""
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


class _ProcessingDialog(QDialog):
    """Compact dialog that centers an indeterminate progress bar."""

    def __init__(self, parent: QWidget, *, title: str, message: str) -> None:
        super().__init__(parent, Qt.WindowType.Dialog | Qt.WindowType.CustomizeWindowHint | Qt.WindowType.WindowTitleHint)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        self._label = QLabel(message, self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._label, alignment=Qt.AlignmentFlag.AlignCenter)

        self._progress = QProgressBar(self)
        self._progress.setRange(0, 0)
        self._progress.setFixedWidth(220)
        self._progress.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._progress, alignment=Qt.AlignmentFlag.AlignCenter)

    def setLabelText(self, text: str) -> None:
        """Update the message displayed above the progress bar."""
        self._label.setText(text)


__all__ = ["ProcessingIndicator"]
