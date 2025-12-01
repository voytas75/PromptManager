"""Prompt list presenter abstraction for MainWindow.

Updates:
  v0.1.0 - 2025-12-01 - Extract presenter to centralize prompt list filtering logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QListView, QMessageBox, QWidget

from core import (
    PromptManager,
    PromptManagerError,
    RepositoryError,
    ScenarioGenerationError,
)

from .processing_indicator import ProcessingIndicator
from .prompt_list_coordinator import PromptListCoordinator, PromptLoadResult, PromptSortOrder

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Callable, Sequence
    from uuid import UUID

    from models.prompt_model import Prompt

    from .prompt_list_model import PromptListModel
    from .widgets import PromptDetailWidget, PromptFilterPanel, PromptToolbar


@dataclass(slots=True)
class PromptListCallbacks:
    """Callback collection used by :class:`PromptListPresenter`."""
    update_intent_hint: Callable[[Sequence[Prompt]], None]
    select_prompt: Callable[[UUID], None]
    show_error: Callable[[str, str], None]
    show_status: Callable[[str, int], None]
    show_toast: Callable[[str], None]


class PromptListPresenter:
    """Coordinate prompt loading, filtering, and sorting for the list view."""
    def __init__(
        self,
        *,
        manager: PromptManager,
        coordinator: PromptListCoordinator,
        model: PromptListModel,
        detail_widget: PromptDetailWidget,
        list_view: QListView,
        filter_panel: PromptFilterPanel | None,
        toolbar: PromptToolbar | None,
        callbacks: PromptListCallbacks,
        parent: QWidget,
    ) -> None:
        """Store collaborators and initial presenter state."""
        self._manager = manager
        self._coordinator = coordinator
        self._model = model
        self._detail_widget = detail_widget
        self._list_view = list_view
        self._filter_panel = filter_panel
        self._toolbar = toolbar
        self._callbacks = callbacks
        self._parent = parent

        self._all_prompts: list[Prompt] = []
        self._current_prompts: list[Prompt] = []
        self._preserve_search_order = False
        self._pending_category_slug: str | None = None
        self._pending_tag_value: str | None = None
        self._pending_quality_value: float | None = None
        self._sort_order = PromptSortOrder.NAME_ASC
        self._suggestions: PromptManager.IntentSuggestions | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_pending_filter_preferences(
        self,
        *,
        category_slug: str | None,
        tag: str | None,
        min_quality: float | None,
    ) -> None:
        """Cache persisted filter preferences until widgets are ready."""
        self._pending_category_slug = category_slug
        self._pending_tag_value = tag
        self._pending_quality_value = min_quality

    def set_sort_order(self, sort_order: PromptSortOrder) -> None:
        """Synchronize the filter panel with the stored sort order."""
        self._sort_order = sort_order
        panel = self._filter_panel
        if panel is not None:
            panel.set_sort_value(sort_order.value)

    def load_prompts(self, search_text: str = "", *, use_indicator: bool = False) -> None:
        """Fetch prompts and refresh the list view."""
        try:
            if use_indicator:
                result = self._run_with_indicator(
                    "Searching prompts…",
                    self._coordinator.fetch_prompts,
                    search_text,
                )
            else:
                result = self._coordinator.fetch_prompts(search_text)
        except RepositoryError as exc:
            self._callbacks.show_error("Unable to load prompts", str(exc))
            return
        self._apply_prompt_load_result(result)

    def refresh_filtered_view(self, *, preserve_order: bool | None = None) -> None:
        """Re-render the list after filter changes."""
        filtered = self._coordinator.apply_filters(self._filter_panel, self._current_prompts)
        use_preserve_flag = (
            self._preserve_search_order if preserve_order is None else preserve_order
        )
        prompts_to_show = (
            filtered
            if use_preserve_flag
            else self._coordinator.sort_prompts(filtered, self._sort_order)
        )
        self._model.set_prompts(prompts_to_show)
        self._list_view.clearSelection()
        if not prompts_to_show:
            self._detail_widget.clear()
        self._callbacks.update_intent_hint(prompts_to_show)

    def update_sort_order(self, sort_order: PromptSortOrder) -> None:
        """Apply a new sort order and maintain the current selection."""
        if sort_order is self._sort_order:
            return
        selected_prompt = self.current_prompt()
        self._sort_order = sort_order
        panel = self._filter_panel
        if panel is not None:
            panel.set_sort_value(sort_order.value)
        filtered = self._coordinator.apply_filters(panel, self._current_prompts)
        sorted_prompts = self._coordinator.sort_prompts(filtered, sort_order)
        self._model.set_prompts(sorted_prompts)
        if selected_prompt and any(prompt.id == selected_prompt.id for prompt in sorted_prompts):
            self._callbacks.select_prompt(selected_prompt.id)
        elif sorted_prompts:
            self._callbacks.select_prompt(sorted_prompts[0].id)
        else:
            self._detail_widget.clear()
        self._callbacks.update_intent_hint(sorted_prompts)

    def refresh_prompt_after_rating(self, prompt_id: UUID) -> None:
        """Refresh cached collections after a rating changes."""
        try:
            updated_prompt = self._manager.get_prompt(prompt_id)
        except PromptManagerError:
            return

        updated_any = self._replace_prompt_in_collection(self._all_prompts, updated_prompt)
        if self._replace_prompt_in_collection(self._current_prompts, updated_prompt):
            updated_any = True

        if not updated_any:
            self.load_prompts()
            self._callbacks.select_prompt(prompt_id)
            return

        filtered = self._coordinator.apply_filters(self._filter_panel, self._current_prompts)
        sorted_prompts = self._coordinator.sort_prompts(filtered, self._sort_order)
        self._model.set_prompts(sorted_prompts)
        if not sorted_prompts:
            self._detail_widget.clear()
            self._callbacks.update_intent_hint(sorted_prompts)
            return
        self._callbacks.update_intent_hint(sorted_prompts)
        if any(prompt.id == prompt_id for prompt in sorted_prompts):
            self._callbacks.select_prompt(prompt_id)
            self._detail_widget.display_prompt(updated_prompt)

    def apply_suggestions(self, suggestions: PromptManager.IntentSuggestions) -> None:
        """Populate the list with LLM intent suggestions."""
        self._suggestions = suggestions
        self._current_prompts = list(suggestions.prompts)
        filtered = self._coordinator.apply_filters(self._filter_panel, self._current_prompts)
        sorted_prompts = self._coordinator.sort_prompts(filtered, self._sort_order)
        self._model.set_prompts(sorted_prompts)
        self._list_view.clearSelection()
        if sorted_prompts:
            self._callbacks.select_prompt(sorted_prompts[0].id)
        else:
            self._detail_widget.clear()
        self._callbacks.update_intent_hint(sorted_prompts)

    def handle_refresh_scenarios(self, detail_widget: PromptDetailWidget) -> None:
        """Regenerate scenarios for the prompt displayed in *detail_widget*."""
        prompt = detail_widget.current_prompt()
        if prompt is None:
            return
        try:
            indicator = ProcessingIndicator(self._parent, "Refreshing scenarios…")
            updated_prompt = indicator.run(
                self._manager.refresh_prompt_scenarios,
                prompt.id,
            )
        except (ScenarioGenerationError, PromptManagerError) as exc:
            QMessageBox.warning(self._parent, "Scenario refresh failed", str(exc))
            return
        self.refresh_prompt_after_rating(updated_prompt.id)
        self._callbacks.show_toast("Scenarios refreshed.")

    def show_similar_prompts(self, prompt: Prompt) -> None:
        """Replace the list view with embedding-based recommendations."""
        embedding_vector: list[float] | None
        if prompt.ext4 is not None:
            try:
                embedding_vector = [float(value) for value in prompt.ext4]
            except (TypeError, ValueError):
                embedding_vector = None
        else:
            embedding_vector = None

        if embedding_vector is None and not prompt.document.strip():
            self._callbacks.show_status(
                "Selected prompt does not include enough text for similarity search.",
                4000,
            )
            return

        try:
            similar_prompts = self._manager.search_prompts(
                "" if embedding_vector is not None else prompt.document,
                limit=10,
                embedding=embedding_vector,
            )
        except PromptManagerError as exc:
            self._callbacks.show_error("Unable to load similar prompts", str(exc))
            return

        recommendations = [candidate for candidate in similar_prompts if candidate.id != prompt.id]
        if not recommendations:
            self._callbacks.show_status("No similar prompts found.", 4000)
            return

        self._suggestions = None
        self._current_prompts = list(recommendations)
        self._preserve_search_order = True
        filtered = self._coordinator.apply_filters(self._filter_panel, self._current_prompts)
        self._model.set_prompts(filtered)
        self._list_view.clearSelection()
        if filtered:
            self._callbacks.select_prompt(filtered[0].id)
        else:
            self._detail_widget.clear()
        self._callbacks.update_intent_hint(filtered)
        self._callbacks.show_status(
            f"Showing prompts similar to '{prompt.name}'.",
            4000,
        )

    def display_prompt_collection(
        self,
        prompts: Sequence[Prompt],
        *,
        preserve_order: bool,
        selected_prompt_id: UUID | None = None,
    ) -> None:
        """Display *prompts* with optional ordering and selection."""
        self._suggestions = None
        self._all_prompts = list(prompts)
        self._current_prompts = list(prompts)
        self._preserve_search_order = preserve_order
        self._pending_category_slug, self._pending_tag_value = self._coordinator.populate_filters(
            self._filter_panel,
            self._all_prompts,
            pending_category_slug=self._pending_category_slug,
            pending_tag_value=self._pending_tag_value,
        )

        filtered = self._coordinator.apply_filters(self._filter_panel, self._current_prompts)
        prompts_to_show = (
            filtered
            if preserve_order
            else self._coordinator.sort_prompts(filtered, self._sort_order)
        )
        self._model.set_prompts(prompts_to_show)
        self._list_view.clearSelection()
        if selected_prompt_id and any(
            prompt.id == selected_prompt_id for prompt in prompts_to_show
        ):
            self._callbacks.select_prompt(selected_prompt_id)
        elif prompts_to_show:
            self._callbacks.select_prompt(prompts_to_show[0].id)
        else:
            self._detail_widget.clear()
        self._callbacks.update_intent_hint(prompts_to_show)

    def current_prompt(self) -> Prompt | None:
        """Return the prompt currently highlighted in the list view."""
        index = self._list_view.currentIndex()
        if not index.isValid():
            return None
        return self._model.prompt_at(index.row())

    @property
    def current_prompts(self) -> Sequence[Prompt]:
        """Expose the unfiltered prompt collection backing the list view."""
        return tuple(self._current_prompts)

    @property
    def all_prompts(self) -> Sequence[Prompt]:
        """Return all prompts known to the presenter."""
        return tuple(self._all_prompts)

    @property
    def suggestions(self) -> PromptManager.IntentSuggestions | None:
        """Return the active intent suggestions, if any."""
        return self._suggestions

    def clear_suggestions(self) -> None:
        """Remove any stored intent suggestions."""
        self._suggestions = None

    def set_intent_suggestions(self, suggestions: PromptManager.IntentSuggestions) -> None:
        """Persist *suggestions* without altering the list ordering."""
        self._suggestions = suggestions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _apply_prompt_load_result(self, result: PromptLoadResult) -> None:
        self._all_prompts = list(result.all_prompts)
        self._preserve_search_order = result.preserve_search_order
        self._pending_category_slug, self._pending_tag_value = self._coordinator.populate_filters(
            self._filter_panel,
            self._all_prompts,
            pending_category_slug=self._pending_category_slug,
            pending_tag_value=self._pending_tag_value,
        )

        panel = self._filter_panel
        if panel is not None and self._pending_quality_value is not None:
            panel.set_min_quality(self._pending_quality_value)
            self._pending_quality_value = None

        if result.search_error:
            self._callbacks.show_error("Unable to search prompts", result.search_error)

        if result.search_results is not None and result.preserve_search_order:
            self._suggestions = None
            self._current_prompts = list(result.search_results)
            filtered = self._coordinator.apply_filters(panel, self._current_prompts)
            self._model.set_prompts(filtered)
            self._list_view.clearSelection()
            if filtered:
                self._callbacks.select_prompt(filtered[0].id)
            else:
                self._detail_widget.clear()
            self._callbacks.update_intent_hint(filtered)
            return

        self._suggestions = None
        self._current_prompts = list(self._all_prompts)

        filtered = self._coordinator.apply_filters(panel, self._current_prompts)
        sorted_prompts = self._coordinator.sort_prompts(filtered, self._sort_order)
        self._model.set_prompts(sorted_prompts)
        self._list_view.clearSelection()
        if not sorted_prompts:
            self._detail_widget.clear()
        self._callbacks.update_intent_hint(sorted_prompts)

    def _run_with_indicator(
        self,
        message: str,
        func: Callable[..., PromptLoadResult],
        *args,
    ) -> PromptLoadResult:
        toolbar = self._toolbar
        if toolbar is not None:
            toolbar.set_search_enabled(False)
        try:
            indicator = ProcessingIndicator(self._parent, message, title="Searching Prompts")
            return indicator.run(func, *args)
        finally:
            if toolbar is not None:
                toolbar.set_search_enabled(True)

    @staticmethod
    def _replace_prompt_in_collection(collection: list[Prompt], updated: Prompt) -> bool:
        for index, existing in enumerate(collection):
            if existing.id == updated.id:
                collection[index] = updated
                return True
        return False


__all__ = ["PromptListCallbacks", "PromptListPresenter"]
