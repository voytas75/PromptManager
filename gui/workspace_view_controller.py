"""Workspace text/clear helpers for Prompt Manager GUI.

Updates:
  v0.1.0 - 2025-12-01 - Added WorkspaceViewController for text + clear actions.
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QLabel, QPlainTextEdit, QTabWidget

from .controllers.execution_controller import ExecutionController
from .quick_action_controller import QuickActionController


class WorkspaceViewController:
    """Manage workspace text edits and clearing logic."""
    def __init__(
        self,
        query_input: QPlainTextEdit,
        result_tabs: QTabWidget,
        intent_hint_label: QLabel,
        *,
        status_callback: Callable[[str, int], None],
        execution_controller_supplier: Callable[[], ExecutionController | None],
        quick_action_controller_supplier: Callable[[], QuickActionController | None],
    ) -> None:
        self._query_input = query_input
        self._result_tabs = result_tabs
        self._intent_hint = intent_hint_label
        self._show_status = status_callback
        self._execution_controller_supplier = execution_controller_supplier
        self._quick_action_controller_supplier = quick_action_controller_supplier

    def set_text(self, text: str, *, focus: bool = False) -> None:
        """Populate the workspace editor with *text* and optionally focus the field."""
        self._query_input.setPlainText(text)
        cursor = self._query_input.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._query_input.setTextCursor(cursor)
        if focus:
            self._query_input.setFocus(Qt.ShortcutFocusReason)

    def clear(self) -> None:
        """Clear the workspace editor, output tab, and chat transcript."""
        controller = self._execution_controller_supplier()
        if controller is not None:
            controller.abort_streaming()
            controller.clear_execution_result()

        self._query_input.clear()
        self._result_tabs.setCurrentIndex(0)
        self._intent_hint.clear()
        self._intent_hint.setVisible(False)
        quick_actions = self._quick_action_controller_supplier()
        if quick_actions is not None:
            quick_actions.clear_workspace_seed()
        self._show_status("Workspace cleared.", 3000)


__all__ = ["WorkspaceViewController"]
