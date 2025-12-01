"""Workspace command routing helpers for :class:`gui.main_window.MainWindow`.

Updates:
  v0.15.81 - 2025-12-01 - Extracted workspace signal handlers from gui.main_window.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Callable
    from uuid import UUID

    from .workspace_actions_controller import WorkspaceActionsController
else:  # pragma: no cover - runtime placeholder for type-only imports
    from typing import Any as _Any

    Callable = _Any
    UUID = _Any
    WorkspaceActionsController = _Any  # type: ignore[assignment]


class WorkspaceCommandRouter:
    """Proxy Qt signal handlers to :class:`WorkspaceActionsController`."""

    def __init__(
        self,
        supplier: Callable[[], WorkspaceActionsController | None],
    ) -> None:
        """Store a supplier hook that yields the active workspace actions controller."""
        self._supplier = supplier

    def save_result(self) -> None:
        """Persist the latest execution result if the controller is available."""
        actions = self._supplier()
        if actions is not None:
            actions.save_result()

    def continue_chat(self) -> None:
        """Continue the current chat session when the controller is ready."""
        actions = self._supplier()
        if actions is not None:
            actions.continue_chat()

    def end_chat(self) -> None:
        """End the active chat session when available."""
        actions = self._supplier()
        if actions is not None:
            actions.end_chat()

    def run_prompt(self) -> None:
        """Execute the selected prompt through the workspace controller."""
        actions = self._supplier()
        if actions is not None:
            actions.run_prompt()

    def clear_workspace(self) -> None:
        """Clear workspace inputs and outputs."""
        actions = self._supplier()
        if actions is not None:
            actions.clear_workspace()

    def handle_note_update(self, execution_id: UUID, note: str) -> None:
        """Propagate note updates to the workspace history."""
        actions = self._supplier()
        if actions is not None:
            actions.handle_note_update(execution_id, note)

    def handle_history_export(self, entries: int, path: str) -> None:
        """Track exports initiated from the workspace history."""
        actions = self._supplier()
        if actions is not None:
            actions.handle_history_export(entries, path)

    def copy_result_to_clipboard(self) -> None:
        """Copy the latest result to the clipboard."""
        actions = self._supplier()
        if actions is not None:
            actions.copy_result_to_clipboard()

    def copy_result_to_workspace(self) -> None:
        """Populate the workspace editor with the last execution result."""
        actions = self._supplier()
        if actions is not None:
            actions.copy_result_to_workspace()


__all__ = ["WorkspaceCommandRouter"]
