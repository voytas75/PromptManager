"""Standalone dialog for displaying prompt refinement summaries.

Updates:
  v0.1.0 - 2025-12-04 - Extracted from dialog module for reuse and clarity.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)


class PromptRefinedDialog(QDialog):
    """Modal dialog presenting prompt refinement output in a resizable view."""

    def __init__(
        self,
        content: str,
        parent: QWidget | None = None,
        *,
        title: str = "Prompt refined",
    ) -> None:
        """Initialize the dialog with the refinement summary content.

        Args:
            content: Text produced by the refinement workflow.
            parent: Optional parent widget that owns the dialog.
            title: Window title describing the refinement action.
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setSizeGripEnabled(True)

        layout = QVBoxLayout(self)
        header = QLabel("Review the refinement details below.", self)
        header.setWordWrap(True)
        layout.addWidget(header)

        self._body = QPlainTextEdit(self)
        self._body.setReadOnly(True)
        self._body.setPlainText(content)
        self._body.setMinimumHeight(200)
        layout.addWidget(self._body)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok, Qt.Orientation.Horizontal, self
        )
        button_box.accepted.connect(self.accept)  # type: ignore[arg-type]
        layout.addWidget(button_box)

        self._apply_initial_size()

    def _apply_initial_size(self) -> None:
        """Resize the dialog to fit comfortably under the active screen size."""
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(720, 480)
            return

        geometry = screen.availableGeometry()
        screen_height = geometry.height()
        screen_width = geometry.width()

        height_buffer = 60
        max_height = max(screen_height - height_buffer, int(screen_height * 0.8))
        preferred_height = max(320, int(screen_height * 0.6))
        height = min(preferred_height, max_height)

        width_buffer = 120
        max_width = max(screen_width - width_buffer, int(screen_width * 0.7))
        preferred_width = max(560, int(screen_width * 0.45))
        width = min(preferred_width, max_width)

        self.resize(width, height)


__all__ = ["PromptRefinedDialog"]
