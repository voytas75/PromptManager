"""Controller responsible for prompt filtering, sorting, and search workflows.

Updates:
  v0.15.82 - 2025-12-08 - Model load_prompts callbacks with keyword-aware Protocol.
  v0.15.81 - 2025-12-01 - Extracted search/filter orchestration from gui.main_window.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from .prompt_list_coordinator import PromptSortOrder

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Callable
    from uuid import UUID

    from PySide6.QtWidgets import QWidget

    from core import PromptManager

    from .layout_controller import LayoutController
    from .prompt_list_presenter import PromptListPresenter
    from .widgets import PromptDetailWidget, PromptFilterPanel
else:  # pragma: no cover - runtime placeholders for type-only imports
    from typing import Any as _Any

    Callable = _Any
    PromptListPresenter = _Any
    QWidget = object
    PromptManager = _Any
    LayoutController = _Any
    PromptDetailWidget = _Any
    PromptFilterPanel = _Any

logger = logging.getLogger(__name__)


class LoadPromptsCallable(Protocol):
    def __call__(self, search_text: str = "", *, use_indicator: bool = ...) -> None:
        ...


class PromptSearchController:
    """Coordinate search, filtering, and catalog-related workflows."""

    def __init__(
        self,
        *,
        parent: QWidget,
        manager: PromptManager,
        presenter_supplier: Callable[[], PromptListPresenter | None],
        filter_panel_supplier: Callable[[], PromptFilterPanel | None],
        layout_controller: LayoutController,
        load_prompts: LoadPromptsCallable,
        current_search_text: Callable[[], str],
        select_prompt: Callable[[UUID], None],
    ) -> None:
        """Store collaborators used for managing search/filter workflows."""
        self._parent = parent
        self._manager = manager
        self._presenter_supplier = presenter_supplier
        self._filter_panel_supplier = filter_panel_supplier
        self._layout_controller = layout_controller
        self._load_prompts = load_prompts
        self._current_search_text = current_search_text
        self._select_prompt = select_prompt
        self._search_active = False

    # ------------------------------------------------------------------
    # Dialog orchestration
    # ------------------------------------------------------------------
    def manage_categories(self) -> None:
        """Open the category manager dialog and refresh filters as needed."""
        from .dialogs import CategoryManagerDialog  # Local import to avoid cycles

        dialog = CategoryManagerDialog(self._manager, self._parent)
        dialog.exec()
        if dialog.has_changes:
            self._load_prompts(self._current_search_text())
            return
        presenter = self._presenter_supplier()
        if presenter is not None:
            presenter.refresh_filtered_view()

    # ------------------------------------------------------------------
    # Filters and sorting
    # ------------------------------------------------------------------
    def filters_changed(self) -> None:
        """Propagate filter changes to the presenter and persist preferences."""
        presenter = self._presenter_supplier()
        if presenter is not None:
            presenter.refresh_filtered_view()
        self._layout_controller.persist_filter_preferences()

    def sort_changed(self, raw_order: str) -> None:
        """Adjust the presenter sort order when the combo box changes."""
        try:
            sort_order = PromptSortOrder(str(raw_order))
        except ValueError:
            logger.warning("Unknown sort order selection: %s", raw_order)
            return
        presenter = self._presenter_supplier()
        if presenter is not None:
            presenter.update_sort_order(sort_order)
        self._layout_controller.persist_sort_preference(sort_order)

    def refresh_prompt_after_rating(self, prompt_id: UUID) -> None:
        """Refresh current collections after a rating change."""
        presenter = self._presenter_supplier()
        if presenter is None:
            return
        search_text = self._current_search_text().strip()
        if search_text:
            self._load_prompts(search_text)
            self._select_prompt(prompt_id)
            return
        presenter.refresh_prompt_after_rating(prompt_id)

    def handle_refresh_scenarios_request(self, detail_widget: PromptDetailWidget) -> None:
        """Delegate scenario refresh requests to the presenter."""
        presenter = self._presenter_supplier()
        if presenter is None:
            return
        presenter.handle_refresh_scenarios(detail_widget)

    # ------------------------------------------------------------------
    # Search orchestration
    # ------------------------------------------------------------------
    def search_changed(self, text: str) -> None:
        """Handle inline search edits triggered by the toolbar."""
        if text.strip():
            return
        panel = self._filter_panel_supplier()
        if not self._search_active and (panel is None or panel.is_sort_enabled()):
            return

        self._search_active = False
        if panel is not None:
            panel.set_sort_enabled(True)
        self._load_prompts("")

    def search_requested(self, text: str | None = None, *, use_indicator: bool) -> None:
        """Trigger a search operation for the provided text or toolbar value."""
        query = text if text is not None else self._current_search_text()
        self._handle_search_request(query, use_indicator=use_indicator)

    def _handle_search_request(self, text: str, *, use_indicator: bool) -> None:
        stripped = text.strip()
        self._search_active = bool(stripped)

        panel = self._filter_panel_supplier()
        if panel is not None:
            panel.set_sort_enabled(not self._search_active)

        if text and len(stripped) < 2:
            return
        self._load_prompts(text, use_indicator=use_indicator)


__all__ = ["PromptSearchController", "LoadPromptsCallable"]
