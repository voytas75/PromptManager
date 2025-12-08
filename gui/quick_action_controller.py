"""Quick action orchestration for the Prompt Manager workspace.

Updates:
  v0.1.1 - 2025-12-08 - Use QTextCursor.MoveOperation enums for cursor positioning.
  v0.1.0 - 2025-12-01 - Extract controller to manage palette, shortcuts, and execution.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut, QTextCursor
from PySide6.QtWidgets import QDialog, QPlainTextEdit, QPushButton

from core import PromptManager, RepositoryError

from .command_palette import CommandPaletteDialog, QuickAction, rank_prompts_for_action

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Callable, Iterable

    from models.prompt_model import Prompt

    from .prompt_list_presenter import PromptListPresenter
    from .widgets import PromptDetailWidget

logger = logging.getLogger(__name__)


class QuickActionController:
    """Encapsulate quick action palette, shortcuts, and execution wiring."""

    def __init__(
        self,
        *,
        parent,
        manager: PromptManager,
        presenter: PromptListPresenter,
        detail_widget: PromptDetailWidget,
        query_input: QPlainTextEdit,
        quick_actions_button: QPushButton,
        button_default_text: str,
        button_default_tooltip: str,
        status_callback: Callable[[str, int], None],
        error_callback: Callable[[str, str], None],
        exit_callback: Callable[[], None],
        quick_actions_config: object | None,
    ) -> None:
        """Capture dependencies and register palette shortcuts."""
        self._parent = parent
        self._manager = manager
        self._presenter = presenter
        self._detail_widget = detail_widget
        self._query_input = query_input
        self._status_callback = status_callback
        self._error_callback = error_callback
        self._exit_callback = exit_callback

        self._quick_actions_button = quick_actions_button
        self._button_default_text = button_default_text
        self._button_default_tooltip = button_default_tooltip

        self._quick_actions: list[QuickAction] = self._build_quick_actions(quick_actions_config)
        self._quick_shortcuts: list[QShortcut] = []
        self._active_quick_action_id: str | None = None
        self._workspace_seed_action_id: str | None = None
        self._suppress_workspace_signal = False

        self._register_shortcuts()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def refresh_actions(self, quick_actions_config: object | None) -> None:
        """Rebuild quick actions from runtime settings and re-register shortcuts."""
        self._quick_actions = self._build_quick_actions(quick_actions_config)
        self._register_shortcuts()
        self._sync_active_button()

    def show_command_palette(self) -> None:
        """Display the quick action palette dialog."""
        dialog = CommandPaletteDialog(self._quick_actions, self._parent)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        action = dialog.selected_action
        if action is not None:
            self.execute_quick_action(action)

    def execute_quick_action(self, action: QuickAction) -> None:
        """Run *action* and update the workspace accordingly."""
        try:
            prompts = self._manager.repository.list()
        except RepositoryError as exc:
            self._error_callback("Unable to load prompts", str(exc))
            return

        ranked = rank_prompts_for_action(prompts, action)
        selected_prompt = self._resolve_quick_action_prompt(action, prompts, ranked)

        if selected_prompt is None:
            self._status_callback(
                f"No prompts matched quick action '{action.title}'.",
                5000,
            )
            return

        self._presenter.display_prompt_collection(
            prompts,
            preserve_order=False,
            selected_prompt_id=selected_prompt.id,
        )
        self._detail_widget.display_prompt(selected_prompt)
        self._apply_quick_action_template(action)
        self._query_input.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self._set_active_quick_action(action)
        self._status_callback(f"Quick action applied: {action.title}", 4000)

    def register_palette_shortcuts(self) -> None:
        """Ensure palette shortcuts stay in sync with the action list."""
        self._register_shortcuts()

    def show_palette_for_shortcut(self) -> None:
        """Shortcut handler that opens the palette dialog."""
        self.show_command_palette()

    def clear_workspace_seed(self) -> None:
        """Reset the quick-action template marker when user edits the workspace."""
        self._workspace_seed_action_id = None

    def is_workspace_signal_suppressed(self) -> bool:
        """Return ``True`` while the controller updates the workspace text."""
        return self._suppress_workspace_signal

    def sync_button(self) -> None:
        """Update the quick actions button label to reflect the active action."""
        self._sync_active_button()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _register_shortcuts(self) -> None:
        for shortcut in self._quick_shortcuts:
            shortcut.setParent(None)
        self._quick_shortcuts.clear()

        for seq in ("Ctrl+K", "Ctrl+Shift+P"):
            shortcut = QShortcut(QKeySequence(seq), self._parent)
            shortcut.activated.connect(self.show_palette_for_shortcut)  # type: ignore[arg-type]
            self._quick_shortcuts.append(shortcut)

        exit_shortcut = QShortcut(QKeySequence("Ctrl+Q"), self._parent)
        exit_shortcut.activated.connect(self._exit_callback)  # type: ignore[arg-type]
        self._quick_shortcuts.append(exit_shortcut)

        for action in self._quick_actions:
            if not action.shortcut:
                continue
            shortcut = QShortcut(QKeySequence(action.shortcut), self._parent)
            shortcut.activated.connect(lambda a=action: self.execute_quick_action(a))  # type: ignore[arg-type]
            self._quick_shortcuts.append(shortcut)

    def _apply_quick_action_template(self, action: QuickAction) -> None:
        template = action.template
        if not template:
            return

        current_text = self._query_input.toPlainText()
        if current_text.strip() and self._workspace_seed_action_id is None:
            return

        self._suppress_workspace_signal = True
        try:
            self._query_input.setPlainText(template)
            cursor = self._query_input.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self._query_input.setTextCursor(cursor)
        finally:
            self._suppress_workspace_signal = False
        self._workspace_seed_action_id = action.identifier

    def _set_active_quick_action(self, action: QuickAction | None) -> None:
        if action is None:
            self._active_quick_action_id = None
            self._quick_actions_button.setText(self._button_default_text)
            self._quick_actions_button.setToolTip(self._button_default_tooltip)
            self._workspace_seed_action_id = None
            return

        self._active_quick_action_id = action.identifier
        label = action.title
        if action.shortcut:
            label = f"{label} ({action.shortcut})"
        self._quick_actions_button.setText(label)
        self._quick_actions_button.setToolTip(action.description or action.title)

    def _sync_active_button(self) -> None:
        if not self._active_quick_action_id:
            self._set_active_quick_action(None)
            return

        for action in self._quick_actions:
            if action.identifier == self._active_quick_action_id:
                self._set_active_quick_action(action)
                return
        self._set_active_quick_action(None)

    def _default_quick_actions(self) -> list[QuickAction]:
        return [
            QuickAction(
                identifier="explain",
                title="Explain This Code",
                description="Select analysis prompts to describe behaviour and intent.",
                category_hint="Code Analysis",
                tag_hints=("analysis", "code-review"),
                template="Explain what this code does and highlight any risks:\n",
                shortcut="Ctrl+1",
            ),
            QuickAction(
                identifier="fix-errors",
                title="Fix Errors",
                description="Surface debugging prompts to diagnose and resolve failures.",
                category_hint="Reasoning / Debugging",
                tag_hints=("debugging", "incident-response"),
                template="Identify and fix the issues in this snippet:\n",
                shortcut="Ctrl+2",
            ),
            QuickAction(
                identifier="add-comments",
                title="Add Comments",
                description=(
                    "Jump to documentation prompts that generate docstrings and commentary."
                ),
                category_hint="Documentation",
                tag_hints=("documentation", "docstrings"),
                template="Add detailed docstrings and inline comments explaining this code:\n",
                shortcut="Ctrl+3",
            ),
            QuickAction(
                identifier="enhance",
                title="Suggest Improvements",
                description="Open enhancement prompts that brainstorm new ideas or edge cases.",
                category_hint="Enhancement",
                tag_hints=("enhancement", "product"),
                template="Suggest improvements, safeguards, and edge cases for this work:\n",
                shortcut="Ctrl+4",
            ),
        ]

    def _build_quick_actions(self, custom_actions: object | None) -> list[QuickAction]:
        actions_by_id: dict[str, QuickAction] = {
            action.identifier: action for action in self._default_quick_actions()
        }
        if not custom_actions:
            return list(actions_by_id.values())

        data: Iterable[dict[str, object]]
        if isinstance(custom_actions, list):
            data = [entry for entry in custom_actions if isinstance(entry, dict)]
        else:
            logger.warning("Ignoring invalid quick_actions settings value: %s", custom_actions)
            return list(actions_by_id.values())

        for entry in data:
            try:
                action = QuickAction.from_mapping(entry)
            except ValueError:
                continue
            actions_by_id[action.identifier] = action
        return list(actions_by_id.values())

    def _resolve_quick_action_prompt(
        self,
        action: QuickAction,
        prompts: Iterable[Prompt],
        ranked: list[Prompt],
    ) -> Prompt | None:
        if action.prompt_id:
            prompt_id = action.prompt_id
            try:
                target_uuid = uuid.UUID(prompt_id)
            except ValueError:
                for prompt in prompts:
                    if prompt.name.lower() == prompt_id.lower():
                        return prompt
            else:
                for prompt in prompts:
                    if prompt.id == target_uuid:
                        return prompt
        return ranked[0] if ranked else None


__all__ = ["QuickActionController"]
