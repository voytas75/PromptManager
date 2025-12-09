"""Dialog helpers used by the Prompt Manager main window.

Updates:
  v0.15.85 - 2025-12-07 - Focus embedded Workbench tab instead of spawning windows.
  v0.15.84 - 2025-12-05 - Pipe prompt edit callbacks through the chain dialog launcher.
  v0.15.83 - 2025-12-04 - Add prompt chain dialog launcher wiring.
  v0.15.82 - 2025-12-01 - Extract info/workbench/version dialogs from gui.main_window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from PySide6.QtWidgets import QDialog, QMainWindow, QWidget

from core import PromptManager, RepositoryError

from .dialogs import InfoDialog, PromptChainManagerDialog, PromptVersionHistoryDialog
from .workbench import WorkbenchModeDialog

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Callable
    from uuid import UUID

    from models.prompt_model import Prompt
else:  # pragma: no cover - runtime placeholders for type-only imports
    from typing import Any as _Any

    Prompt = _Any


class DialogLauncher:
    """Launch common dialogs on behalf of :class:`gui.main_window.MainWindow`."""

    def __init__(
        self,
        *,
        parent: QWidget,
        manager: PromptManager,
        current_prompt_supplier: Callable[[], Prompt | None],
        load_prompts: Callable[[str], None],
        current_search_text: Callable[[], str],
        select_prompt: Callable[[Prompt], None],
        prompt_edit_callback: Callable[[UUID], None] | None = None,
    ) -> None:
        """Store shared collaborators used by dialog helpers."""
        self._parent = parent
        self._manager = manager
        self._current_prompt_supplier = current_prompt_supplier
        self._load_prompts = load_prompts
        self._current_search_text = current_search_text
        self._select_prompt = select_prompt
        self._prompt_edit_callback = prompt_edit_callback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def show_info_dialog(self) -> None:
        """Display basic application info."""
        dialog = InfoDialog(self._parent)
        dialog.exec()

    def open_workbench(self) -> None:
        """Launch the Enhanced Prompt Workbench."""
        try:
            templates = self._manager.repository.list(limit=200)
        except RepositoryError:
            templates = []
        dialog = WorkbenchModeDialog(templates, self._parent)
        if dialog.exec() != int(QDialog.DialogCode.Accepted):
            return
        selection = dialog.result_selection()
        activator = getattr(self._parent, "activate_workbench_tab", None)
        if callable(activator):
            activator(selection.mode, selection.template_prompt)

    def open_version_history_dialog(self, prompt: Prompt | None = None) -> None:
        """Display the version history for *prompt* or the current selection."""
        target = prompt or self._current_prompt_supplier()
        if target is None:
            return
        window = cast("QMainWindow", self._parent)
        dialog = PromptVersionHistoryDialog(
            self._manager,
            target,
            self._parent,
            status_callback=lambda message, duration=3000: window.statusBar().showMessage(
                message, duration
            ),
        )
        dialog.exec()
        if dialog.last_restored_prompt is not None:
            self._load_prompts(self._current_search_text())
            self._select_prompt(dialog.last_restored_prompt)
            window.statusBar().showMessage("Prompt restored to selected version.", 4000)

    def open_prompt_chains_dialog(self) -> None:
        """Launch the prompt chain management dialog."""
        dialog = PromptChainManagerDialog(
            self._manager,
            self._parent,
            prompt_edit_callback=self._prompt_edit_callback,
        )
        dialog.exec()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------


__all__ = ["DialogLauncher"]
