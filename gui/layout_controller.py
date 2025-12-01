"""Coordinator for splitter geometry and filter persistence.

Updates:
  v0.1.0 - 2025-12-01 - Extracted from MainWindow to handle layout and filter state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QSplitter

from .widgets import PromptFilterPanel, PromptToolbar

if TYPE_CHECKING:
    from .layout_state import WindowStateManager
    from .template_preview import TemplatePreviewWidget
    from .prompt_list_coordinator import PromptSortOrder


class LayoutController:
    """Encapsulate splitter sizing logic and persisted UI preferences."""

    def __init__(self, layout_state: WindowStateManager) -> None:
        self._layout_state = layout_state
        self._main_splitter_left_width: int | None = None
        self._suppress_main_splitter_sync = False
        self._main_splitter: QSplitter | None = None
        self._list_splitter: QSplitter | None = None
        self._workspace_splitter: QSplitter | None = None
        self._template_preview_splitter: QSplitter | None = None
        self._template_preview_list_splitter: QSplitter | None = None
        self._template_preview: TemplatePreviewWidget | None = None
        self._filter_panel: PromptFilterPanel | None = None
        self._toolbar: PromptToolbar | None = None

    def configure(
        self,
        *,
        main_splitter: QSplitter | None,
        list_splitter: QSplitter | None,
        workspace_splitter: QSplitter | None,
        template_preview_splitter: QSplitter | None,
        template_preview_list_splitter: QSplitter | None,
        template_preview: TemplatePreviewWidget | None,
        filter_panel: PromptFilterPanel | None,
        toolbar: PromptToolbar | None,
    ) -> None:
        """Update references after the main view is rebuilt."""
        self._main_splitter = main_splitter
        self._list_splitter = list_splitter
        self._workspace_splitter = workspace_splitter
        self._template_preview_splitter = template_preview_splitter
        self._template_preview_list_splitter = template_preview_list_splitter
        self._template_preview = template_preview
        self._filter_panel = filter_panel
        self._toolbar = toolbar

    def restore_splitter_state(self) -> None:
        """Restore splitter sizes from persisted settings."""
        self._layout_state.restore_splitter_sizes(self._splitter_state_entries())

    def save_splitter_state(self) -> None:
        """Persist splitter sizes for future sessions."""
        self._layout_state.save_splitter_sizes(self._splitter_state_entries())

    def handle_show_event(self) -> None:
        """Ensure the main splitter is initialised after first show."""
        self._capture_main_splitter_left_width()
        self._enforce_main_splitter_left_width()

    def handle_resize_event(self) -> None:
        """Keep the locked left pane width when the window resizes."""
        self._enforce_main_splitter_left_width()

    def handle_main_splitter_moved(self) -> None:
        """Track splitter drags so resize enforcement stays accurate."""
        if self._suppress_main_splitter_sync or self._main_splitter is None:
            return
        sizes = self._main_splitter.sizes()
        if len(sizes) < 2:
            return
        self._main_splitter_left_width = max(sizes[0], 0)

    def persist_filter_preferences(self) -> None:
        """Persist the current category, tag, and quality filters."""
        panel = self._filter_panel
        if panel is None:
            return
        self._layout_state.persist_filter_preferences(
            category_slug=panel.category_slug(),
            tag=panel.tag_value(),
            min_quality=panel.min_quality(),
        )

    def persist_sort_preference(self, sort_order: PromptSortOrder) -> None:
        """Persist the currently selected sort order."""
        self._layout_state.persist_sort_order(sort_order)

    def current_search_text(self) -> str:
        """Return the current text in the toolbar search field."""
        if self._toolbar is None:
            return ""
        return self._toolbar.search_text()

    def _capture_main_splitter_left_width(self) -> None:
        if self._main_splitter is None:
            return
        sizes = self._main_splitter.sizes()
        if len(sizes) < 2:
            return
        self._main_splitter_left_width = max(sizes[0], 0)

    def _enforce_main_splitter_left_width(self) -> None:
        if self._main_splitter is None:
            return
        if self._main_splitter_left_width is None:
            self._capture_main_splitter_left_width()
            return
        sizes = self._main_splitter.sizes()
        if len(sizes) < 2:
            return
        total = sum(sizes)
        if total <= 0:
            return
        minimum_right = 1 if total > 1 else 0
        locked_width = min(self._main_splitter_left_width, total - minimum_right)
        locked_width = max(locked_width, 0)
        right_width = total - locked_width
        if right_width < minimum_right:
            right_width = minimum_right
            locked_width = total - right_width
        if locked_width == sizes[0]:
            self._main_splitter_left_width = locked_width
            return
        self._suppress_main_splitter_sync = True
        try:
            self._main_splitter.setSizes([locked_width, right_width])
        finally:
            self._suppress_main_splitter_sync = False
        self._main_splitter_left_width = locked_width

    def _splitter_state_entries(self) -> list[tuple[str, QSplitter | None]]:
        entries: list[tuple[str, QSplitter | None]] = [
            ("mainSplitter", self._main_splitter),
            ("listSplitter", self._list_splitter),
            ("workspaceSplitter", self._workspace_splitter),
            ("templatePreviewSplitter", self._template_preview_splitter),
            ("templatePreviewListSplitter", self._template_preview_list_splitter),
        ]
        if self._template_preview is not None:
            entries.append(
                ("templatePreviewContentSplitter", self._template_preview.content_splitter)
            )
        return entries


__all__ = ["LayoutController"]
