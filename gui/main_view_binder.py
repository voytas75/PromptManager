"""Bind :mod:`gui.main_view_builder` components to the :class:`MainWindow`.

Updates:
  v0.15.84 - 2025-12-08 - Bind token usage label for workspace execution summaries.
  v0.15.83 - 2025-12-07 - Store Workbench panel reference for tab activation.
  v0.15.82 - 2025-12-04 - Bind web search checkbox for execution controller.
  v0.15.81 - 2025-12-01 - Moved widget wiring and controller construction out of gui.main_window.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

from .code_highlighter import CodeHighlighter
from .prompt_actions_controller import PromptActionsController
from .template_preview_controller import TemplatePreviewController
from .workspace_actions_controller import WorkspaceActionsController
from .workspace_view_controller import WorkspaceViewController

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Callable
    from uuid import UUID

    from core import PromptManager
    from models.prompt_model import Prompt

    from .controllers.execution_controller import ExecutionController
    from .layout_controller import LayoutController
    from .layout_state import WindowStateManager
    from .main_view_builder import MainViewComponents
    from .prompt_list_model import PromptListModel
    from .quick_action_controller import QuickActionController
    from .usage_logger import IntentUsageLogger
    from .widgets import PromptDetailWidget
else:  # pragma: no cover - runtime placeholders for type-only imports
    from typing import Any as _Any

    Callable = _Any
    PromptManager = _Any
    LayoutController = _Any
    WindowStateManager = _Any
    ExecutionController = _Any
    MainViewComponents = _Any
    PromptListModel = _Any
    IntentUsageLogger = _Any
    PromptDetailWidget = _Any
    Prompt = _Any
    QuickActionController = _Any


@dataclass(slots=True)
class MainViewBinderConfig:
    """Dependencies required to bind main view components."""

    layout_controller: LayoutController
    layout_state: WindowStateManager
    manager: PromptManager
    model: PromptListModel
    detail_widget: PromptDetailWidget
    usage_logger: IntentUsageLogger
    status_callback: Callable[[str, int], None]
    error_callback: Callable[[str, str], None]
    toast_callback: Callable[[str, int], None]
    current_prompt_supplier: Callable[[], Prompt | None]
    execution_controller_supplier: Callable[[], ExecutionController | None]
    quick_action_controller_supplier: Callable[[], QuickActionController | None]
    refresh_prompt_after_rating: Callable[[UUID], None]
    execute_from_prompt_body: Callable[[Prompt], None]


def bind_main_view(
    window: Any,
    *,
    components: MainViewComponents,
    config: MainViewBinderConfig,
) -> None:
    """Populate ``window`` attributes using :class:`MainViewComponents`."""
    window._main_container = components.container  # noqa: SLF001 - assigned during bootstrap
    window._toolbar = components.toolbar
    window._language_label = components.language_label
    window._quick_actions_button = components.quick_actions_button
    window._quick_actions_button_default_text = components.quick_actions_button_default_text
    window._quick_actions_button_default_tooltip = components.quick_actions_button_default_tooltip
    window._quick_actions_button.setToolTip(window._quick_actions_button_default_tooltip)
    window._detect_button = components.detect_button
    window._suggest_button = components.suggest_button
    window._run_button = components.run_button
    window._clear_button = components.clear_button
    window._continue_chat_button = components.continue_chat_button
    window._end_chat_button = components.end_chat_button
    window._copy_button = components.copy_button
    window._web_search_checkbox = components.web_search_checkbox
    window._copy_result_button = components.copy_result_button
    window._copy_result_to_text_window_button = components.copy_result_to_text_window_button
    window._save_button = components.save_button
    window._share_result_button = components.share_result_button
    window._speak_result_button = components.speak_result_button
    window._intent_hint = components.intent_hint
    window._filter_panel = components.filter_panel
    window._query_input = components.query_input
    window._highlighter = CodeHighlighter(window._query_input.document())
    window._tab_widget = components.tab_widget

    main_splitter = components.main_splitter
    if main_splitter is not None:
        main_splitter.splitterMoved.connect(  # type: ignore[arg-type]
            lambda *_: config.layout_controller.handle_main_splitter_moved()
        )
    config.layout_controller.configure(
        main_splitter=main_splitter,
        list_splitter=components.list_splitter,
        workspace_splitter=components.workspace_splitter,
        template_preview_splitter=components.template_preview_splitter,
        template_preview_list_splitter=components.template_preview_list_splitter,
        template_preview=components.template_preview,
        filter_panel=components.filter_panel,
        toolbar=components.toolbar,
    )

    window._list_view = components.list_view
    window._result_label = components.result_label
    window._result_meta = components.result_meta
    window._token_usage_label = components.token_usage_label
    window._result_tabs = components.result_tabs
    window._result_text = components.result_text
    window._result_overlay = components.result_overlay
    window._chat_history_view = components.chat_history_view
    window._render_markdown_checkbox = components.render_markdown_checkbox
    window._history_panel = components.history_panel
    window._notes_panel = components.notes_panel
    window._response_styles_panel = components.response_styles_panel
    window._analytics_panel = components.analytics_panel
    window._chain_panel = components.chain_panel
    window._workbench_window = components.workbench_panel
    window._template_list_view = components.template_list_view
    window._template_detail_widget = components.template_detail_widget
    window._template_preview = components.template_preview
    window._template_run_shortcut_button = components.template_run_shortcut_button

    template_detail = window._template_detail_widget
    template_detail.delete_requested.connect(window._on_delete_clicked)  # type: ignore[arg-type]
    template_detail.edit_requested.connect(window._on_edit_clicked)  # type: ignore[arg-type]
    template_detail.version_history_requested.connect(window._open_version_history_dialog)  # type: ignore[arg-type]
    template_detail.fork_requested.connect(window._on_fork_clicked)  # type: ignore[arg-type]
    template_detail.refresh_scenarios_requested.connect(  # type: ignore[arg-type]
        partial(window._handle_refresh_scenarios_request, template_detail)
    )

    window._workspace_view = WorkspaceViewController(
        window._query_input,
        window._result_tabs,
        window._intent_hint,
        status_callback=config.status_callback,
        execution_controller_supplier=config.execution_controller_supplier,
        quick_action_controller_supplier=config.quick_action_controller_supplier,
    )
    window._prompt_actions_controller = PromptActionsController(
        parent=window,
        model=config.model,
        list_view=window._list_view,
        query_input=window._query_input,
        layout_state=config.layout_state,
        workspace_view=window._workspace_view,
        execution_controller_supplier=config.execution_controller_supplier,
        current_prompt_supplier=config.current_prompt_supplier,
        edit_callback=window._on_edit_clicked,
        duplicate_callback=window._duplicate_prompt,
        fork_callback=window._fork_prompt,
        similar_callback=window._show_similar_prompts,
        status_callback=config.status_callback,
        error_callback=config.error_callback,
        toast_callback=config.toast_callback,
        usage_logger=config.usage_logger,
    )
    window._template_preview_controller = TemplatePreviewController(
        parent=window,
        tab_widget=window._tab_widget,
        template_preview=window._template_preview,
        template_run_button=window._template_run_shortcut_button,
        execution_controller_supplier=config.execution_controller_supplier,
        current_prompt_supplier=config.current_prompt_supplier,
        error_callback=config.error_callback,
        status_callback=config.status_callback,
    )

    window._workspace_actions = WorkspaceActionsController(
        parent=window,
        manager=config.manager,
        query_input=window._query_input,
        execution_controller_supplier=config.execution_controller_supplier,
        current_prompt_supplier=config.current_prompt_supplier,
        prompt_list_supplier=lambda: config.model.prompts(),
        workspace_view=window._workspace_view,
        history_panel=window._history_panel,
        usage_logger=config.usage_logger,
        status_callback=config.status_callback,
        error_callback=config.error_callback,
        disable_save_button=lambda enabled: window._save_button.setEnabled(enabled),
        refresh_prompt_after_rating=config.refresh_prompt_after_rating,
        execute_from_prompt_body=config.execute_from_prompt_body,
    )


__all__ = [
    "MainViewBinderConfig",
    "bind_main_view",
]
