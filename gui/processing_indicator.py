"""Transient processing overlays for long-running GUI tasks.

Updates:
  v0.1.0 - 2025-11-27 - Add reusable busy indicator for prompt workflows.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from types import TracebackType
from typing import Optional, Type

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QProgressDialog, QWidget


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
        self._dialog.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        self._dialog.setWindowModality(Qt.ApplicationModal)

    def __enter__(self) -> "ProcessingIndicator":
        self._dialog.show()
        QGuiApplication.processEvents()
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

    def update_message(self, message: str) -> None:
        """Refresh the indicator text while the dialog is visible."""

        self._dialog.setLabelText(message)
        QGuiApplication.processEvents()


__all__ = ["ProcessingIndicator"]
