"""Standalone dialog for displaying prompt refinement summaries.

Updates:
  v0.1.0 - 2025-12-04 - Extracted from dialog module for reuse and clarity.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
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
        self.resize(720, 480)


__all__ = ["PromptRefinedDialog"]
