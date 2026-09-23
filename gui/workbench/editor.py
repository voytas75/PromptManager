"""Editor widget customisations for the Enhanced Prompt Workbench.

Updates:
  v0.1.0 - 2025-12-04 - Extract WorkbenchPromptEditor into a dedicated module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QPlainTextEdit

if TYPE_CHECKING:
    from PySide6.QtGui import QMouseEvent

from .utils import variable_at_cursor

__all__ = ["WorkbenchPromptEditor"]


class WorkbenchPromptEditor(QPlainTextEdit):
    """Custom prompt editor that surfaces variable tokens on double-click."""

    variableActivated = Signal(str)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Emit the variable token under the cursor when the user double-clicks."""
        super().mouseDoubleClickEvent(event)
        name = variable_at_cursor(self.cursorForPosition(event.position().toPoint()))
        if name:
            self.variableActivated.emit(name)
