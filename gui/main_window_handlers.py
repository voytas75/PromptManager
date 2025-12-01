"""Handlers extracted from :mod:`gui.main_window` for clarity.

Updates:
  v0.15.82 - 2025-12-01 - Introduce prompt, workspace, and template handlers.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QMessageBox

from core import PromptManager, PromptNotFoundError, PromptStorageError

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Callable, Sequence
    from uuid import UUID

    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QWidget

    from models.prompt_model import Prompt

    from .catalog_workflow_controller import CatalogWorkflowController
    from .dialog_launcher import DialogLauncher
    from .prompt_actions_controller import PromptActionsController
    from .prompt_editor_flow import PromptEditorFlow
    from .prompt_search_controller import PromptSearchController
    from .settings_workflow import SettingsWorkflow
    from .share_workflow import ShareWorkflowCoordinator
    from .template_preview_controller import TemplatePreviewController
    from .widgets import PromptDetailWidget
    from .workspace_command_router import WorkspaceCommandRouter


class PromptActionsHandler:
    """Bundle prompt CRUD, catalog, and sharing event handlers."""

    def __init__(
        self,
        *,
        parent: QWidget,
        manager: PromptManager,
        model_prompts_supplier: Callable[[], Sequence[Prompt]],
        current_prompt_supplier: Callable[[], Prompt | None],
        detail_widget: PromptDetailWidget,
        prompt_search_controller: PromptSearchController,
        prompt_actions_controller_supplier: Callable[[], PromptActionsController | None],
        prompt_editor_flow_supplier: Callable[[], PromptEditorFlow | None],
        catalog_controller_supplier: Callable[[], CatalogWorkflowController | None],
        settings_workflow_supplier: Callable[[], SettingsWorkflow | None],
        dialog_launcher_supplier: Callable[[], DialogLauncher | None],
        share_workflow_supplier: Callable[[], ShareWorkflowCoordinator | None],
        load_prompts: Callable[[str], None],
        current_search_text: Callable[[], str],
        status_callback: Callable[[str, int], None],
        exit_callback: Callable[[], None],
    ) -> None:
        """Store collaborators for prompt CRUD, catalog, and sharing actions."""
        self._parent = parent
        self._manager = manager
        self._model_prompts_supplier = model_prompts_supplier
        self._current_prompt_supplier = current_prompt_supplier
        self._detail_widget = detail_widget
        self._prompt_search_controller = prompt_search_controller
        self._prompt_actions_controller_supplier = prompt_actions_controller_supplier
        self._prompt_editor_flow_supplier = prompt_editor_flow_supplier
        self._catalog_controller_supplier = catalog_controller_supplier
        self._settings_workflow_supplier = settings_workflow_supplier
        self._dialog_launcher_supplier = dialog_launcher_supplier
        self._share_workflow_supplier = share_workflow_supplier
        self._load_prompts = load_prompts
        self._current_search_text = current_search_text
        self._status_callback = status_callback
        self._exit_callback = exit_callback

    # ------------------------------------------------------------------
    # Prompt collection management
    # ------------------------------------------------------------------
    def manage_categories(self) -> None:
        """Open the category management dialog via the search controller."""
        self._prompt_search_controller.manage_categories()

    def handle_refresh_scenarios_request(self, detail_widget: PromptDetailWidget) -> None:
        """Trigger scenario regeneration for the provided detail widget."""
        self._prompt_search_controller.handle_refresh_scenarios_request(detail_widget)

    def refresh_prompts(self) -> None:
        """Reload prompts while preserving the current search text."""
        self._load_prompts(self._current_search_text())

    def add_prompt(self) -> None:
        """Launch the creation flow for a new prompt."""
        flow = self._prompt_editor_flow_supplier()
        if flow is None:
            return
        flow.create_prompt()

    def edit_prompt(self, prompt: Prompt | None = None) -> None:
        """Edit *prompt* or the currently selected prompt when omitted."""
        target = prompt or self._current_prompt_supplier()
        if target is None:
            return
        flow = self._prompt_editor_flow_supplier()
        if flow is None:
            return
        flow.edit_prompt(target)

    def fork_prompt(self) -> None:
        """Fork the currently selected prompt."""
        prompt = self._current_prompt_supplier()
        if prompt is None:
            return
        self.fork_prompt_direct(prompt)

    def fork_prompt_direct(self, prompt: Prompt) -> None:
        """Fork *prompt* using the editor flow."""
        flow = self._prompt_editor_flow_supplier()
        if flow is None:
            return
        flow.fork_prompt(prompt)

    def duplicate_prompt(self, prompt: Prompt) -> None:
        """Duplicate *prompt* within the editor workflow."""
        flow = self._prompt_editor_flow_supplier()
        if flow is None:
            return
        flow.duplicate_prompt(prompt)

    def open_version_history_dialog(self, prompt: Prompt | None = None) -> None:
        """Open the version history dialog for *prompt* when provided."""
        dialog_launcher = self._dialog_launcher_supplier()
        if dialog_launcher is None:
            return
        dialog_launcher.open_version_history_dialog(prompt)

    # ------------------------------------------------------------------
    # Prompt actions + clipboard helpers
    # ------------------------------------------------------------------
    def copy_prompt(self) -> None:
        """Copy the active prompt, falling back to the list head when needed."""
        prompt = self._current_prompt_supplier()
        if prompt is None:
            prompts = self._model_prompts_supplier()
            prompt = prompts[0] if prompts else None
        if prompt is None:
            self._status_callback("Select a prompt to copy first.", 3000)
            return
        self.copy_prompt_to_clipboard(prompt)

    def copy_prompt_to_clipboard(self, prompt: Prompt) -> None:
        """Copy *prompt* to the clipboard via the actions controller."""
        controller = self._prompt_actions_controller_supplier()
        if controller is None:
            return
        controller.copy_prompt_to_clipboard(prompt)

    def show_prompt_description(self, prompt: Prompt) -> None:
        """Display the details dialog for *prompt*."""
        controller = self._prompt_actions_controller_supplier()
        if controller is None:
            return
        controller.show_prompt_description(prompt)

    def delete_prompt(self, prompt: Prompt, *, skip_confirmation: bool = False) -> None:
        """Delete *prompt* after optional confirmation and refresh the list."""
        if not skip_confirmation:
            confirmation = QMessageBox.question(
                self._parent,
                "Delete prompt",
                f"Are you sure you want to delete '{prompt.name}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if confirmation != QMessageBox.Yes:
                return
        try:
            self._manager.delete_prompt(prompt.id)
        except PromptNotFoundError:
            QMessageBox.critical(self._parent, "Prompt missing", "The prompt was already removed.")
        except PromptStorageError as exc:
            QMessageBox.critical(self._parent, "Unable to delete prompt", str(exc))
            return
        self._detail_widget.clear()
        self._load_prompts(self._current_search_text())

    def delete_current_prompt(self) -> None:
        """Delete the currently selected prompt."""
        prompt = self._current_prompt_supplier()
        if prompt is None:
            return
        self.delete_prompt(prompt)

    # ------------------------------------------------------------------
    # Catalog + dialogs
    # ------------------------------------------------------------------
    def import_catalog(self) -> None:
        """Open the import dialog through the catalog controller."""
        controller = self._catalog_controller_supplier()
        if controller is None:
            return
        controller.open_import_dialog()

    def export_catalog(self) -> None:
        """Trigger catalog export."""
        controller = self._catalog_controller_supplier()
        if controller is None:
            return
        controller.export_catalog()

    def open_maintenance_dialog(self) -> None:
        """Open the catalog maintenance dialog."""
        controller = self._catalog_controller_supplier()
        if controller is None:
            return
        controller.open_maintenance_dialog()

    def open_info_dialog(self) -> None:
        """Display the application info dialog."""
        launcher = self._dialog_launcher_supplier()
        if launcher is None:
            return
        launcher.show_info_dialog()

    def open_settings_dialog(self, settings) -> None:
        """Launch the settings workflow with the provided settings object."""
        workflow = self._settings_workflow_supplier()
        if workflow is None:
            return
        workflow.open_settings_dialog(settings)

    def open_prompt_templates_dialog(self) -> None:
        """Open the prompt templates dialog."""
        workflow = self._settings_workflow_supplier()
        if workflow is None:
            return
        workflow.open_prompt_templates_dialog()

    def open_workbench(self) -> None:
        """Launch the Enhanced Prompt Workbench."""
        launcher = self._dialog_launcher_supplier()
        if launcher is None:
            return
        launcher.open_workbench()

    # ------------------------------------------------------------------
    # Share + execution helpers
    # ------------------------------------------------------------------
    def share_prompt(self) -> None:
        """Share the active prompt via the configured workflow."""
        workflow = self._share_workflow_supplier()
        if workflow is None:
            return
        workflow.share_prompt()

    def share_result(self) -> None:
        """Share the latest workspace result text."""
        workflow = self._share_workflow_supplier()
        if workflow is None:
            return
        workflow.share_result()

    def execute_prompt_from_body(self, prompt: Prompt) -> None:
        """Run *prompt* using its saved body text."""
        controller = self._prompt_actions_controller_supplier()
        if controller is None:
            return
        controller.execute_prompt_from_body(prompt)

    def execute_prompt_as_context(
        self,
        prompt: Prompt,
        *,
        parent: QWidget | None = None,
        context_override: str | None = None,
    ) -> None:
        """Execute *prompt* as context, optionally overriding with extra text."""
        controller = self._prompt_actions_controller_supplier()
        if controller is None:
            return
        controller.execute_prompt_as_context(
            prompt,
            parent=parent,
            context_override=context_override,
        )

    def show_prompt_context_menu(self, point: QPoint) -> None:
        """Display the prompt context menu at the requested point."""
        controller = self._prompt_actions_controller_supplier()
        if controller is None:
            return
        controller.show_context_menu(point)

    def close_application(self) -> None:
        """Dismiss the app after surfacing a status message."""
        self._status_callback("Closing Prompt Manager…", 2000)
        self._exit_callback()


class WorkspaceInputHandler:
    """Delegate workspace router interactions to a focused helper."""

    def __init__(self, *, workspace_router: WorkspaceCommandRouter) -> None:
        """Store the workspace router dependency."""
        self._workspace_router = workspace_router

    def save_result(self) -> None:
        """Persist the latest execution result."""
        self._workspace_router.save_result()

    def continue_chat(self) -> None:
        """Send the next chat message."""
        self._workspace_router.continue_chat()

    def end_chat(self) -> None:
        """Terminate the current chat session."""
        self._workspace_router.end_chat()

    def run_prompt(self) -> None:
        """Execute the selected prompt."""
        self._workspace_router.run_prompt()

    def clear_workspace(self) -> None:
        """Clear workspace inputs and outputs."""
        self._workspace_router.clear_workspace()

    def copy_result(self) -> None:
        """Copy the latest result to the clipboard."""
        self._workspace_router.copy_result_to_clipboard()

    def copy_result_to_workspace(self) -> None:
        """Copy the latest result back into the workspace input."""
        self._workspace_router.copy_result_to_workspace()

    def handle_note_update(self, execution_id: UUID, note: str) -> None:
        """Record changes to execution notes."""
        self._workspace_router.handle_note_update(execution_id, note)

    def handle_history_export(self, entries: int, path: str) -> None:
        """Record history export analytics."""
        self._workspace_router.handle_history_export(entries, path)


class TemplatePreviewHandler:
    """Encapsulate template preview/run interactions."""

    def __init__(
        self,
        *,
        controller_supplier: Callable[[], TemplatePreviewController | None],
    ) -> None:
        """Persist the supplier for the template preview controller."""
        self._controller_supplier = controller_supplier

    def handle_run_requested(self, rendered_text: str, variables: dict[str, str]) -> None:
        """Execute a template preview with the rendered inputs."""
        controller = self._controller_supplier()
        if controller is None:
            return
        controller.handle_run_requested(rendered_text, variables)

    def handle_run_state_changed(self, can_run: bool) -> None:
        """Propagate preview run state changes."""
        controller = self._controller_supplier()
        if controller is None:
            return
        controller.handle_run_state_changed(can_run)

    def handle_tab_run_clicked(self) -> None:
        """Trigger the template tab run action."""
        controller = self._controller_supplier()
        if controller is None:
            return
        controller.handle_template_tab_run_clicked()


__all__ = [
    "PromptActionsHandler",
    "TemplatePreviewHandler",
    "WorkspaceInputHandler",
]
