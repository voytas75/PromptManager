"""Execute-as-context dialog with history selection.

Updates: v0.1.0 - 2025-11-26 - Add execute-as-context history picker dialog.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QVBoxLayout,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class ExecuteContextDialog(QDialog):
    """Collect a task description and expose quick history selection."""

    def __init__(
        self,
        *,
        last_task: str = "",
        history: Sequence[str] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Execute as Context")
        self.setModal(True)
        self.resize(520, 420)
        self._history: Sequence[str] = history or ()
        self._description_input: QPlainTextEdit
        self._history_list: QListWidget | None = None
        self._build_ui(last_task)

    def task_text(self) -> str:
        """Return the task description provided by the user."""

        return self._description_input.toPlainText()

    def _build_ui(self, last_task: str) -> None:
        """Create the dialog layout and populate controls."""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        prompt_label = QLabel(
            "Describe the task to perform using this prompt's body as context:",
            self,
        )
        prompt_label.setWordWrap(True)
        layout.addWidget(prompt_label)

        self._description_input = QPlainTextEdit(self)
        self._description_input.setObjectName("executeContextTaskInput")
        self._description_input.setPlaceholderText("Summarise logs, outline incidents, …")
        self._description_input.setTabChangesFocus(True)
        self._description_input.setMinimumHeight(150)
        if last_task:
            self._description_input.setPlainText(last_task)
            cursor = self._description_input.textCursor()
            cursor.movePosition(QTextCursor.End)
            self._description_input.setTextCursor(cursor)
        layout.addWidget(self._description_input)

        if self._history:
            history_label = QLabel("Recent descriptions", self)
            history_label.setObjectName("executeContextHistoryLabel")
            layout.addWidget(history_label)

            history_list = QListWidget(self)
            history_list.setObjectName("executeContextHistoryList")
            history_list.setSelectionMode(QAbstractItemView.SingleSelection)
            history_list.setAlternatingRowColors(True)
            history_list.setMinimumHeight(140)
            for entry in self._history:
                item = QListWidgetItem(entry)
                item.setToolTip(entry)
                history_list.addItem(item)
            history_list.itemClicked.connect(self._on_history_item_clicked)  # type: ignore[arg-type]
            history_list.itemDoubleClicked.connect(self._on_history_item_double_clicked)  # type: ignore[arg-type]
            layout.addWidget(history_list)
            self._history_list = history_list

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok, self)
        buttons.button(QDialogButtonBox.Ok).setText("Execute")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_history_item_clicked(self, item: QListWidgetItem) -> None:
        """Populate the task input with the clicked history entry."""

        self._apply_history_text(item.text())

    def _on_history_item_double_clicked(self, item: QListWidgetItem) -> None:
        """Populate the task input and accept the dialog on double click."""

        self._apply_history_text(item.text())
        self.accept()

    def _apply_history_text(self, text: str) -> None:
        """Set the description field to *text* and focus the editor."""

        self._description_input.setPlainText(text)
        cursor = self._description_input.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._description_input.setTextCursor(cursor)
        self._description_input.setFocus(Qt.TabFocusReason)
