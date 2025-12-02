"""Helper builders for gui.main_window controller wiring.

Updates:
  v0.16.0 - 2025-12-02 - Extract presenter, quick action, and execution controller setup.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .controllers.execution_controller import ExecutionController
from .prompt_list_coordinator import PromptSortOrder
from .prompt_list_presenter import PromptListCallbacks, PromptListPresenter
from .quick_action_controller import QuickActionController

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from .main_window import MainWindow
    from .main_window_composition import FilterPreferences
else:  # pragma: no cover - runtime placeholders for type-only imports
    MainWindow = object  # type: ignore[assignment]
    FilterPreferences = object  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PromptPresenterAssembly:
    """Bundle describing the prompt list presenter state."""

    presenter: PromptListPresenter
    sort_order: PromptSortOrder


def assemble_prompt_presenter(
    window: MainWindow,
    *,
    filter_preferences: FilterPreferences,
) -> PromptPresenterAssembly:
    """Build the prompt list presenter and restore persisted filters."""
    callbacks = PromptListCallbacks(
        update_intent_hint=window._update_intent_hint,  # type: ignore[attr-defined]
        select_prompt=window._select_prompt,  # type: ignore[attr-defined]
        show_error=window._show_error,  # type: ignore[attr-defined]
        show_status=window._show_status_message,  # type: ignore[attr-defined]
        show_toast=window._show_toast,  # type: ignore[attr-defined]
    )
    presenter = PromptListPresenter(
        manager=window._manager,  # type: ignore[attr-defined]
        coordinator=window._prompt_coordinator,  # type: ignore[attr-defined]
        model=window._model,  # type: ignore[attr-defined]
        detail_widget=window._detail_widget,  # type: ignore[attr-defined]
        list_view=window._list_view,  # type: ignore[attr-defined]
        filter_panel=window._filter_panel,  # type: ignore[attr-defined]
        toolbar=window._toolbar,  # type: ignore[attr-defined]
        callbacks=callbacks,
        parent=window,  # type: ignore[arg-type]
    )
    presenter.set_pending_filter_preferences(
        category_slug=filter_preferences.category_slug,
        tag=filter_preferences.tag,
        min_quality=filter_preferences.min_quality,
    )

    sort_order = PromptSortOrder.NAME_ASC
    if filter_preferences.sort_value:
        try:
            sort_order = PromptSortOrder(filter_preferences.sort_value)
        except ValueError:
            logger.warning("Unknown stored sort order: %s", filter_preferences.sort_value)
    presenter.set_sort_order(sort_order)
    return PromptPresenterAssembly(presenter=presenter, sort_order=sort_order)


def assemble_quick_action_controller(window: MainWindow) -> QuickActionController:
    """Create the quick action controller and synchronise its button state."""
    controller = QuickActionController(
        parent=window,  # type: ignore[arg-type]
        manager=window._manager,  # type: ignore[attr-defined]
        presenter=window._prompt_presenter,  # type: ignore[attr-defined]
        detail_widget=window._detail_widget,  # type: ignore[attr-defined]
        query_input=window._query_input,  # type: ignore[attr-defined]
        quick_actions_button=window._quick_actions_button,  # type: ignore[attr-defined]
        button_default_text=window._quick_actions_button_default_text,  # type: ignore[attr-defined]
        button_default_tooltip=window._quick_actions_button_default_tooltip,  # type: ignore[attr-defined]
        status_callback=window._show_status_message,  # type: ignore[attr-defined]
        error_callback=window._show_error,  # type: ignore[attr-defined]
        exit_callback=window._prompt_actions_bridge.close_application,  # type: ignore[attr-defined]
        quick_actions_config=window._runtime_settings.get("quick_actions"),  # type: ignore[attr-defined]
    )
    controller.sync_button()
    return controller


def assemble_execution_controller(window: MainWindow) -> ExecutionController:
    """Create the execution controller for prompt + chat runs."""
    controller = ExecutionController(
        manager=window._manager,  # type: ignore[attr-defined]
        runtime_settings=window._runtime_settings,  # type: ignore[attr-defined]
        usage_logger=window._usage_logger,  # type: ignore[attr-defined]
        share_controller=window._share_controller,  # type: ignore[attr-defined]
        query_input=window._query_input,  # type: ignore[attr-defined]
        result_label=window._result_label,  # type: ignore[attr-defined]
        result_meta=window._result_meta,  # type: ignore[attr-defined]
        result_tabs=window._result_tabs,  # type: ignore[attr-defined]
        result_text=window._result_text,  # type: ignore[attr-defined]
        chat_history_view=window._chat_history_view,  # type: ignore[attr-defined]
        render_markdown_checkbox=window._render_markdown_checkbox,  # type: ignore[attr-defined]
        copy_result_button=window._copy_result_button,  # type: ignore[attr-defined]
        copy_result_to_text_window_button=window._copy_result_to_text_window_button,  # type: ignore[attr-defined]
        save_button=window._save_button,  # type: ignore[attr-defined]
        share_result_button=window._share_result_button,  # type: ignore[attr-defined]
        continue_chat_button=window._continue_chat_button,  # type: ignore[attr-defined]
        end_chat_button=window._end_chat_button,  # type: ignore[attr-defined]
        status_callback=window._show_status_message,  # type: ignore[attr-defined]
        clear_status_callback=window.statusBar().clearMessage,  # type: ignore[attr-defined]
        error_callback=window._show_error,  # type: ignore[attr-defined]
        toast_callback=window._show_toast,  # type: ignore[attr-defined]
        settings=window._settings,  # type: ignore[attr-defined]
    )
    controller.notify_share_providers_changed()
    return controller


__all__ = [
    "PromptPresenterAssembly",
    "assemble_execution_controller",
    "assemble_prompt_presenter",
    "assemble_quick_action_controller",
]
