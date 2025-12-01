"""Prompt-specific actions separated from the main window.

Updates:
  v0.1.0 - 2025-12-01 - Extracted context menu, clipboard, and execute-as-context workflows.
"""
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QGuiApplication, QTextCursor
from PySide6.QtWidgets import QListView, QMenu, QMessageBox, QPlainTextEdit, QWidget

from .execute_context_dialog import ExecuteContextDialog

if TYPE_CHECKING:
    from collections.abc import Callable

    from models.prompt_model import Prompt

    from .controllers.execution_controller import ExecutionController
    from .layout_state import WindowStateManager
    from .prompt_list_model import PromptListModel
    from .usage_logger import IntentUsageLogger
    from .workspace_view_controller import WorkspaceViewController


class PromptActionsController:
    """Bundle prompt actions such as duplication, copying, and execute-as-context."""
    def __init__(
        self,
        *,
        parent: QWidget,
        model: PromptListModel,
        list_view: QListView,
        query_input: QPlainTextEdit,
        layout_state: WindowStateManager,
        workspace_view: WorkspaceViewController | None,
        execution_controller_supplier: Callable[[], ExecutionController | None],
        current_prompt_supplier: Callable[[], Prompt | None],
        edit_callback: Callable[[], None],
        duplicate_callback: Callable[[Prompt], None],
        fork_callback: Callable[[Prompt], None],
        similar_callback: Callable[[Prompt], None] | None,
        status_callback: Callable[[str, int], None],
        error_callback: Callable[[str, str], None],
        toast_callback: Callable[[str, int], None],
        usage_logger: IntentUsageLogger,
    ) -> None:
        """Wire UI widgets, callbacks, and helpers used for prompt actions."""
        self._parent = parent
        self._model = model
        self._list_view = list_view
        self._query_input = query_input
        self._layout_state = layout_state
        self._workspace_view = workspace_view
        self._execution_controller_supplier = execution_controller_supplier
        self._current_prompt_supplier = current_prompt_supplier
        self._edit_callback = edit_callback
        self._duplicate_callback = duplicate_callback
        self._fork_callback = fork_callback
        self._similar_callback = similar_callback
        self._status_callback = status_callback
        self._error_callback = error_callback
        self._toast_callback = toast_callback
        self._usage_logger = usage_logger
        execute_state = self._layout_state.load_execute_context_state()
        self._last_execute_context_task = execute_state.last_task
        self._execute_context_history: deque[str] = deque(execute_state.history)

    def show_context_menu(self, point: QPoint) -> None:
        """Display the prompt context menu anchored to the list view."""
        index = self._list_view.indexAt(point)
        prompt: Prompt | None = None
        if index.isValid():
            self._list_view.setCurrentIndex(index)
            prompt = self._model.prompt_at(index.row())
        else:
            prompt = self._current_prompt_supplier()

        menu = QMenu(self._parent)
        edit_action = menu.addAction("Edit Prompt")
        duplicate_action = menu.addAction("Duplicate Prompt")
        fork_action = menu.addAction("Fork Prompt")
        similar_action = menu.addAction("Similar Prompts")
        execute_action = menu.addAction("Execute Prompt")
        execute_context_action = menu.addAction("Execute as Context…")
        copy_action = menu.addAction("Copy Prompt Text")
        description_action = menu.addAction("Show Description")

        if prompt is None:
            for action in (
                edit_action,
                duplicate_action,
                fork_action,
                similar_action,
                execute_action,
                execute_context_action,
                copy_action,
                description_action,
            ):
                action.setEnabled(False)
        else:
            controller = self._execution_controller_supplier()
            can_execute = bool((prompt.context or prompt.description) and controller)
            execute_action.setEnabled(can_execute)
            has_context_body = bool((prompt.context or "").strip())
            execute_context_action.setEnabled(bool(controller) and has_context_body)
            if not (prompt.context or prompt.description):
                copy_action.setEnabled(False)
            if not (prompt.description and prompt.description.strip()):
                description_action.setEnabled(False)
            if self._similar_callback is None:
                similar_action.setEnabled(False)

        selected_action = menu.exec(self._list_view.viewport().mapToGlobal(point))
        if selected_action is None:
            return
        if selected_action is edit_action:
            self._edit_callback()
        elif selected_action is duplicate_action and prompt is not None:
            self._duplicate_callback(prompt)
        elif selected_action is fork_action and prompt is not None:
            self._fork_callback(prompt)
        elif (
            selected_action is similar_action
            and prompt is not None
            and self._similar_callback is not None
        ):
            self._similar_callback(prompt)
        elif selected_action is execute_action and prompt is not None:
            self.execute_prompt_from_body(prompt)
        elif selected_action is execute_context_action and prompt is not None:
            self.execute_prompt_as_context(prompt)
        elif selected_action is copy_action and prompt is not None:
            self.copy_prompt_to_clipboard(prompt)
        elif selected_action is description_action and prompt is not None:
            self.show_prompt_description(prompt)

    def execute_prompt_from_body(self, prompt: Prompt) -> None:
        """Populate the workspace with the prompt body and execute immediately."""
        raw_payload = prompt.context or prompt.description or ""
        if not raw_payload.strip():
            self._status_callback(
                "Selected prompt does not include any text to execute.",
                4000,
            )
            return
        self._query_input.setPlainText(raw_payload)
        cursor = self._query_input.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._query_input.setTextCursor(cursor)
        self._query_input.setFocus(Qt.ShortcutFocusReason)
        controller = self._execution_controller_supplier()
        if controller is None:
            self._error_callback("Workspace unavailable", "Execution controller is not ready.")
            return
        controller.execute_prompt_with_text(
            prompt,
            raw_payload,
            status_prefix="Executed",
            empty_text_message="Selected prompt does not include any text to execute.",
            keep_text_after=True,
        )

    def execute_prompt_as_context(
        self,
        prompt: Prompt,
        *,
        parent: QWidget | None = None,
        context_override: str | None = None,
    ) -> None:
        """Ask for a task and run the prompt using its body as contextual input."""
        context_text = context_override if context_override is not None else prompt.context or ""
        cleaned_context = context_text.strip()
        if not cleaned_context:
            message = "Selected prompt does not include a prompt body to use as context."
            if parent is not None:
                QMessageBox.information(parent, "Execute as context", message)
            else:
                self._status_callback(message, 5000)
            return
        parent_widget = parent or self._parent
        if self._workspace_view is not None:
            self._workspace_view.set_text(context_text, focus=True)
        dialog = ExecuteContextDialog(
            parent=parent_widget,
            last_task=self._last_execute_context_task,
            history=tuple(self._execute_context_history),
        )
        if dialog.exec() != ExecuteContextDialog.Accepted:
            return
        task_text = dialog.task_text()
        cleaned_task = task_text.strip()
        if not cleaned_task:
            QMessageBox.warning(parent_widget, "Task required", "Enter a task before executing.")
            return
        self._last_execute_context_task = task_text
        self._layout_state.persist_last_execute_task(task_text)
        self._layout_state.record_execute_task(task_text, self._execute_context_history)
        request_payload = (
            "You will receive a task and a context block. "
            "Use the context exclusively when fulfilling the task.\n\n"
            f"Task:\n{cleaned_task}\n\n"
            f"Context:\n{cleaned_context}"
        )
        controller = self._execution_controller_supplier()
        if controller is None:
            self._error_callback("Workspace unavailable", "Execution controller is not ready.")
            return
        try:
            controller.execute_prompt_with_text(
                prompt,
                request_payload,
                status_prefix="Executed context",
                empty_text_message="Provide context text before executing.",
                keep_text_after=False,
            )
        finally:
            if self._workspace_view is not None:
                self._workspace_view.set_text(context_text, focus=False)

    def copy_prompt_to_clipboard(self, prompt: Prompt) -> None:
        """Copy a prompt's primary text to the clipboard with status feedback."""
        payload = prompt.context or prompt.description
        if not payload:
            self._status_callback("Selected prompt does not include a body to copy.", 3000)
            return
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(payload)
        self._toast_callback(f"Copied '{prompt.name}' to the clipboard.")
        self._usage_logger.log_copy(prompt_name=prompt.name, prompt_has_body=bool(prompt.context))

    def show_prompt_description(self, prompt: Prompt) -> None:
        """Display the prompt description in a dialog for quick reference."""
        description = (prompt.description or "").strip()
        if not description:
            QMessageBox.information(
                self._parent,
                "No description available",
                "The selected prompt does not have a description yet.",
            )
            return
        QMessageBox.information(
            self._parent,
            f"{prompt.name} — Description",
            description,
        )


__all__ = ["PromptActionsController"]
