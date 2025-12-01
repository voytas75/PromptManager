"""Workspace-oriented operations extracted from the main window.

Updates:
  v0.1.0 - 2025-12-01 - Introduced controller to manage execution, history, and chat actions.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from PySide6.QtWidgets import QDialog, QMessageBox, QPlainTextEdit, QMainWindow

from core import PromptExecutionUnavailable, PromptHistoryError

from .dialogs import SaveResultDialog

if TYPE_CHECKING:
    from uuid import UUID

    from core import PromptManager
    from models.prompt_model import Prompt

    from .controllers.execution_controller import ExecutionController
    from .workspace_view_controller import WorkspaceViewController
    from .history_panel import HistoryPanel
    from .usage_logger import IntentUsageLogger


class WorkspaceActionsController:
    """Coordinate workspace actions such as run, save, chat, and analytics logging."""
    def __init__(
        self,
        *,
        parent: QMainWindow,
        manager: PromptManager,
        query_input: QPlainTextEdit,
        execution_controller_supplier: Callable[[], ExecutionController | None],
        current_prompt_supplier: Callable[[], Prompt | None],
        prompt_list_supplier: Callable[[], Sequence[Prompt]],
        workspace_view: WorkspaceViewController | None,
        history_panel: HistoryPanel,
        usage_logger: IntentUsageLogger,
        status_callback: Callable[[str, int], None],
        error_callback: Callable[[str, str], None],
        disable_save_button: Callable[[bool], None],
        refresh_prompt_after_rating: Callable[[UUID], None],
        execute_from_prompt_body: Callable[[Prompt], None],
    ) -> None:
        self._parent = parent
        self._manager = manager
        self._query_input = query_input
        self._execution_controller_supplier = execution_controller_supplier
        self._current_prompt_supplier = current_prompt_supplier
        self._prompt_list_supplier = prompt_list_supplier
        self._workspace_view = workspace_view
        self._history_panel = history_panel
        self._usage_logger = usage_logger
        self._status_callback = status_callback
        self._error_callback = error_callback
        self._disable_save_button = disable_save_button
        self._refresh_prompt_after_rating = refresh_prompt_after_rating
        self._execute_from_prompt_body = execute_from_prompt_body

    def run_prompt(self) -> None:
        """Execute the selected prompt using the workspace query text."""
        prompt = self._current_prompt()
        if prompt is None:
            prompts = self._prompt_list_supplier()
            prompt = prompts[0] if prompts else None
        if prompt is None:
            self._status_callback("Select a prompt to execute first.", 4000)
            return
        request_text = self._query_input.toPlainText()
        if not request_text.strip():
            self._execute_from_prompt_body(prompt)
            return
        controller = self._execution_controller()
        if controller is None:
            self._error_callback("Workspace unavailable", "Execution controller is not ready.")
            return
        controller.execute_prompt_with_text(
            prompt,
            request_text,
            status_prefix="Executed",
            empty_text_message="Paste some text or code before executing a prompt.",
            keep_text_after=False,
        )

    def save_result(self) -> None:
        """Persist the latest execution result with optional user notes."""
        controller = self._execution_controller()
        if controller is None:
            self._error_callback("Workspace unavailable", "Execution controller is not ready.")
            return
        outcome = controller.last_execution
        if outcome is None:
            self._status_callback("Run a prompt before saving the result.", 3000)
            return
        prompt = self._current_prompt()
        if prompt is None:
            prompts = self._prompt_list_supplier()
            prompt = prompts[0] if prompts else None
        if prompt is None:
            self._status_callback("Select a prompt to associate with the result.", 3000)
            return
        default_summary = outcome.result.response_text[:200] if outcome.result.response_text else ""
        dialog = SaveResultDialog(
            self._parent,
            prompt_name=prompt.name,
            default_text=default_summary,
            button_text="Save",
            enable_rating=True,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        note = dialog.note
        rating_value = dialog.rating
        metadata: dict[str, Any] = {"source": "manual-save"}
        if note:
            metadata["note"] = note
        if outcome.history_entry is not None:
            metadata["source_execution_id"] = str(outcome.history_entry.id)

        usage_metadata = outcome.result.usage if outcome.result.usage else None
        try:
            saved_entry = self._manager.save_execution_result(
                prompt.id,
                outcome.result.request_text,
                outcome.result.response_text,
                duration_ms=outcome.result.duration_ms,
                usage=usage_metadata,
                metadata=metadata,
                rating=rating_value,
            )
        except PromptExecutionUnavailable as exc:
            self._disable_save_button(False)
            self._error_callback("History unavailable", str(exc))
            return
        except PromptHistoryError as exc:
            self._error_callback("Unable to save result", str(exc))
            return

        self._usage_logger.log_save(
            prompt_name=prompt.name,
            note_length=len(note),
            rating=rating_value,
        )
        executed_at = saved_entry.executed_at.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        self._status_callback(f"Result saved ({executed_at}).", 5000)
        if rating_value is not None:
            self._refresh_prompt_after_rating(prompt.id)
        self._history_panel.refresh()

    def continue_chat(self) -> None:
        """Send a follow-up message within the active chat session."""
        controller = self._execution_controller()
        if controller is None:
            self._error_callback("Workspace unavailable", "Execution controller is not ready.")
            return
        controller.continue_chat()

    def end_chat(self) -> None:
        """Terminate the active chat session without clearing the transcript."""
        controller = self._execution_controller()
        if controller is None:
            self._error_callback("Workspace unavailable", "Execution controller is not ready.")
            return
        controller.end_chat()

    def clear_workspace(self) -> None:
        """Clear the workspace editor, output tab, and chat transcript."""
        if self._workspace_view is not None:
            self._workspace_view.clear()

    def copy_result_to_clipboard(self) -> None:
        """Copy the latest execution result to the clipboard."""
        controller = self._execution_controller()
        if controller is None:
            self._error_callback("Workspace unavailable", "Execution controller is not ready.")
            return
        controller.copy_result_to_clipboard()

    def copy_result_to_workspace(self) -> None:
        """Populate the workspace text window with the latest execution result."""
        controller = self._execution_controller()
        if controller is None:
            self._error_callback("Workspace unavailable", "Execution controller is not ready.")
            return
        controller.copy_result_to_workspace()

    def handle_note_update(self, execution_id: UUID, note: str) -> None:  # noqa: ARG002
        """Record analytics when execution notes are edited."""
        self._usage_logger.log_note_edit(note_length=len(note))

    def handle_history_export(self, entries: int, path: str) -> None:
        """Record analytics when history is exported."""
        self._usage_logger.log_history_export(entries=entries, path=path)

    def _current_prompt(self) -> Prompt | None:
        return self._current_prompt_supplier()

    def _execution_controller(self) -> ExecutionController | None:
        return self._execution_controller_supplier()


__all__ = ["WorkspaceActionsController"]
