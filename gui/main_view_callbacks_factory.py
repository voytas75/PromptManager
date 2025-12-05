"""Factories that build callback bundles for gui.main_window.

Updates:
  v0.16.4 - 2025-12-05 - Remove prompt template callback after toolbar removal.
  v0.16.3 - 2025-12-05 - Remove chains toolbar callback now that the Chain tab is built-in.
  v0.16.2 - 2025-12-05 - Route chains toolbar action to focus the Chain tab.
  v0.16.1 - 2025-12-04 - Add prompt chain dialog callback wiring.
  v0.16.0 - 2025-12-02 - Move MainViewCallbacks assembly out of MainWindow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .main_view_builder import MainViewCallbacks

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from .main_window import MainWindow
else:  # pragma: no cover - runtime placeholders for type-only imports
    MainWindow = object  # type: ignore[assignment]


def build_main_view_callbacks(window: MainWindow) -> MainViewCallbacks:
    """Return callbacks wired to the window's handlers."""
    prompt_search = window._prompt_search_controller  # type: ignore[attr-defined]
    prompt_actions = window._prompt_actions_bridge  # type: ignore[attr-defined]
    workspace_input = window._workspace_input_bridge  # type: ignore[attr-defined]
    template_preview = window._template_preview_bridge  # type: ignore[attr-defined]
    return MainViewCallbacks(
        search_requested=lambda text=None: prompt_search.search_requested(text, use_indicator=True),
        search_text_changed=prompt_search.search_changed,
        refresh_requested=prompt_actions.refresh_prompts,
        add_requested=prompt_actions.add_prompt,
        workbench_requested=prompt_actions.open_workbench,
        import_requested=prompt_actions.import_catalog,
        export_requested=prompt_actions.export_catalog,
        maintenance_requested=prompt_actions.open_maintenance_dialog,
        notifications_requested=window._on_notifications_clicked,  # type: ignore[attr-defined]
        info_requested=prompt_actions.open_info_dialog,
        settings_requested=lambda: prompt_actions.open_settings_dialog(window._settings),  # type: ignore[attr-defined]
        exit_requested=prompt_actions.close_application,
        show_command_palette=window._show_command_palette,  # type: ignore[attr-defined]
        detect_intent_clicked=window._on_detect_intent_clicked,  # type: ignore[attr-defined]
        suggest_prompt_clicked=window._on_suggest_prompt_clicked,  # type: ignore[attr-defined]
        run_prompt_clicked=workspace_input.run_prompt,
        clear_workspace_clicked=workspace_input.clear_workspace,
        continue_chat_clicked=workspace_input.continue_chat,
        end_chat_clicked=workspace_input.end_chat,
        copy_prompt_clicked=prompt_actions.copy_prompt,
        copy_result_clicked=workspace_input.copy_result,
        copy_result_to_text_window_clicked=workspace_input.copy_result_to_workspace,
        save_result_clicked=workspace_input.save_result,
        share_result_clicked=prompt_actions.share_result,
        speak_result_clicked=workspace_input.play_result_audio,
        filters_changed=prompt_search.filters_changed,
        sort_changed=prompt_search.sort_changed,
        manage_categories_clicked=prompt_actions.manage_categories,
        query_text_changed=window._on_query_text_changed,  # type: ignore[attr-defined]
        tab_changed=window._on_tab_changed,  # type: ignore[attr-defined]
        selection_changed=window._on_selection_changed,  # type: ignore[attr-defined]
        prompt_double_clicked=window._on_prompt_double_clicked,  # type: ignore[attr-defined]
        prompt_context_menu=prompt_actions.show_prompt_context_menu,
        render_markdown_toggled=window._on_render_markdown_toggled,  # type: ignore[attr-defined]
        template_preview_run_requested=template_preview.handle_run_requested,
        template_preview_run_state_changed=template_preview.handle_run_state_changed,
        template_tab_run_clicked=template_preview.handle_tab_run_clicked,
    )


__all__ = ["build_main_view_callbacks"]
