"""Dialog helpers used by the Prompt Manager main window.

Updates:
  v0.15.82 - 2025-12-01 - Extract info/workbench/version dialogs from gui.main_window.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from core import PromptManager, RepositoryError

from .dialogs import InfoDialog, PromptVersionHistoryDialog
from .workbench.workbench_window import WorkbenchModeDialog, WorkbenchWindow

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Callable

    from PySide6.QtWidgets import QWidget

    from models.prompt_model import Prompt
else:  # pragma: no cover - runtime placeholders for type-only imports
    from typing import Any as _Any

    Callable = _Any
    QWidget = _Any
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
    ) -> None:
        """Store shared collaborators used by dialog helpers."""
        self._parent = parent
        self._manager = manager
        self._current_prompt_supplier = current_prompt_supplier
        self._load_prompts = load_prompts
        self._current_search_text = current_search_text
        self._select_prompt = select_prompt
        self._workbench_windows: list[WorkbenchWindow] = []

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
        if dialog.exec() != WorkbenchModeDialog.Accepted:
            return
        selection = dialog.result_selection()
        window = WorkbenchWindow(
            self._manager,
            mode=selection.mode,
            template_prompt=selection.template_prompt,
            parent=self._parent,
        )
        self._workbench_windows.append(window)
        window.destroyed.connect(  # type: ignore[arg-type]
            lambda *_: self._remove_workbench_window(window)
        )
        window.show()

    def open_version_history_dialog(self, prompt: Prompt | None = None) -> None:
        """Display the version history for *prompt* or the current selection."""
        target = prompt or self._current_prompt_supplier()
        if target is None:
            return
        dialog = PromptVersionHistoryDialog(
            self._manager,
            target,
            self._parent,
            status_callback=lambda message, duration=3000: self._parent.statusBar().showMessage(
                message, duration
            ),
        )
        dialog.exec()
        if dialog.last_restored_prompt is not None:
            self._load_prompts(self._current_search_text())
            self._select_prompt(dialog.last_restored_prompt)
            self._parent.statusBar().showMessage("Prompt restored to selected version.", 4000)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _remove_workbench_window(self, window: WorkbenchWindow) -> None:
        try:
            self._workbench_windows.remove(window)
        except ValueError:
            return


__all__ = ["DialogLauncher"]
