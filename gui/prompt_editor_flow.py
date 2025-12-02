"""Prompt dialog factory and editor flow helpers.

Updates:
  v0.1.1 - 2025-12-02 - Ensure delete flow bypasses confirmation via keyword arg.
  v0.1.0 - 2025-12-01 - Introduce shared dialog factory and CRUD flow coordinator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from PySide6.QtWidgets import QDialog, QWidget

from core import PromptManager, PromptManagerError, PromptNotFoundError, PromptStorageError

from .dialogs import PromptDialog
from .processing_indicator import ProcessingIndicator

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Callable
    from uuid import UUID

    from models.prompt_model import Prompt


class _DeletePromptCallable(Protocol):
    def __call__(self, prompt: Prompt, *, skip_confirmation: bool = False) -> None:
        """Delete *prompt* with optional confirmation overrides."""


class PromptDialogFactory:
    """Build pre-configured :class:`PromptDialog` instances for editors."""

    def __init__(
        self,
        *,
        manager: PromptManager,
        name_generator: Callable[[str], str],
        description_generator: Callable[[str], str],
        category_generator: Callable[[str], str],
        tags_generator: Callable[[str], list[str]],
        scenario_generator: Callable[[str], list[str]],
        prompt_engineer,
        structure_refiner,
        version_history_handler: Callable[[Prompt | None], None],
        execute_context_handler: Callable[[Prompt, str, QWidget | None], None],
    ) -> None:
        """Capture dependencies for subsequent dialog creation."""
        self._manager = manager
        self._name_generator = name_generator
        self._description_generator = description_generator
        self._category_generator = category_generator
        self._tags_generator = tags_generator
        self._scenario_generator = scenario_generator
        self._prompt_engineer = prompt_engineer
        self._structure_refiner = structure_refiner
        self._version_history_handler = version_history_handler
        self._execute_context_handler = execute_context_handler

    def build(self, parent: QWidget, prompt: Prompt | None = None) -> PromptDialog:
        """Return a configured prompt dialog for the requested context."""
        dialog = PromptDialog(
            parent,
            prompt,
            category_provider=self._manager.list_categories,
            name_generator=self._name_generator,
            description_generator=self._description_generator,
            category_generator=self._category_generator,
            tags_generator=self._tags_generator,
            scenario_generator=self._scenario_generator,
            prompt_engineer=self._prompt_engineer,
            structure_refiner=self._structure_refiner,
            version_history_handler=self._version_history_handler,
        )

        def _handle_execute_context(
            prompt_obj: Prompt,
            context_text: str,
            dlg: QWidget | None = dialog,
        ) -> None:
            self._execute_context_handler(prompt_obj, context_text, dlg)

        dialog.execute_context_requested.connect(_handle_execute_context)  # type: ignore[arg-type]
        return dialog


class PromptEditorFlow:
    """Coordinate add/edit/fork flows using :class:`PromptDialogFactory`."""

    def __init__(
        self,
        *,
        parent: QWidget,
        manager: PromptManager,
        dialog_factory: PromptDialogFactory,
        load_prompts: Callable[[str], None],
        current_search_text: Callable[[], str],
        select_prompt: Callable[[UUID], None],
        delete_prompt: _DeletePromptCallable,
        status_callback: Callable[[str, int], None],
        error_callback: Callable[[str, str], None],
    ) -> None:
        """Store dependencies needed for prompt CRUD flows."""
        self._parent = parent
        self._manager = manager
        self._dialog_factory = dialog_factory
        self._load_prompts = load_prompts
        self._current_search_text = current_search_text
        self._select_prompt = select_prompt
        self._delete_prompt = delete_prompt
        self._status_callback = status_callback
        self._error_callback = error_callback

    def create_prompt(self) -> None:
        """Open a creation dialog and persist the resulting prompt."""
        dialog = self._dialog_factory.build(self._parent)
        if dialog.exec() != QDialog.Accepted:
            return
        prompt = dialog.result_prompt
        if prompt is None:
            return
        try:
            created = self._manager.create_prompt(prompt)
        except PromptStorageError as exc:
            self._error_callback("Unable to create prompt", str(exc))
            return
        self._load_prompts("")
        self._select_prompt(created.id)

    def edit_prompt(self, prompt: Prompt) -> None:
        """Show the edit dialog for *prompt* and persist any changes."""
        dialog = self._dialog_factory.build(self._parent, prompt)
        dialog.applied.connect(  # type: ignore[arg-type]
            lambda updated_prompt: self._handle_prompt_applied(updated_prompt, dialog)
        )
        if dialog.exec() != QDialog.Accepted:
            return
        if dialog.delete_requested:
            self._delete_prompt(prompt, skip_confirmation=True)
            return
        updated = dialog.result_prompt
        if updated is None:
            return
        try:
            stored = ProcessingIndicator(self._parent, "Saving prompt changes…").run(
                self._manager.update_prompt,
                updated,
            )
        except PromptNotFoundError:
            self._error_callback(
                "Prompt missing",
                "The prompt cannot be located. Refresh and try again.",
            )
            self._load_prompts(self._current_search_text())
            return
        except PromptStorageError as exc:
            self._error_callback("Unable to update prompt", str(exc))
            return
        self._load_prompts(self._current_search_text())
        self._select_prompt(stored.id)

    def duplicate_prompt(self, prompt: Prompt) -> None:
        """Duplicate *prompt* via a pre-filled creation dialog."""
        dialog = self._dialog_factory.build(self._parent)
        dialog.prefill_from_prompt(prompt)
        if dialog.exec() != QDialog.Accepted:
            return
        duplicate = dialog.result_prompt
        if duplicate is None:
            return
        try:
            created = self._manager.create_prompt(duplicate)
        except PromptStorageError as exc:
            self._error_callback("Unable to duplicate prompt", str(exc))
            return
        self._load_prompts(self._current_search_text())
        self._select_prompt(created.id)
        self._status_callback("Prompt duplicated.", 4000)

    def fork_prompt(self, prompt: Prompt) -> None:
        """Fork *prompt* and allow optional edits before saving."""
        try:
            forked = self._manager.fork_prompt(prompt.id)
        except PromptManagerError as exc:
            self._error_callback("Unable to fork prompt", str(exc))
            return

        dialog = self._dialog_factory.build(self._parent, forked)
        dialog.setWindowTitle("Edit Forked Prompt")
        dialog.applied.connect(  # type: ignore[arg-type]
            lambda updated_prompt: self._handle_prompt_applied(updated_prompt, dialog)
        )
        if dialog.exec() == QDialog.Accepted and dialog.result_prompt is not None:
            try:
                forked = self._manager.update_prompt(dialog.result_prompt)
            except PromptManagerError as exc:
                self._error_callback("Unable to save forked prompt", str(exc))
                return
        self._load_prompts(self._current_search_text())
        self._select_prompt(forked.id)
        self._status_callback("Prompt fork created.", 4000)

    def _handle_prompt_applied(self, prompt: Prompt, dialog: QDialog) -> None:
        try:
            stored = ProcessingIndicator(dialog, "Saving prompt changes…").run(
                self._manager.update_prompt,
                prompt,
            )
        except PromptNotFoundError:
            self._error_callback(
                "Prompt missing",
                "The prompt cannot be located. Refresh and try again.",
            )
            self._load_prompts("")
            dialog.reject()
            return
        except PromptStorageError as exc:
            self._error_callback("Unable to update prompt", str(exc))
            return

        dialog.update_source_prompt(stored)
        self._load_prompts(self._current_search_text())
        self._select_prompt(stored.id)
        self._status_callback("Prompt changes applied.", 4000)


__all__ = ["PromptDialogFactory", "PromptEditorFlow"]
