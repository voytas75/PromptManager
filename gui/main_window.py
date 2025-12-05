r"""Main window widgets and models for the Prompt Manager GUI.

Updates:
  v0.16.6 - 2025-12-05 - Normalize module docstring quoting and reorder imports for lint compliance.
  v0.16.5 - 2025-12-05 - Remove prompt chain toolbar shortcut; Chain tab remains embedded.
  v0.16.4 - 2025-12-05 - Embed prompt chain panel and add toolbar shortcut to focus Chain tab.
  v0.16.3 - 2025-12-04 - Track \"Use web search\" checkbox for execution controller.
  v0.16.2 - 2025-12-04 - Wire template preview bridge to handler so tab run works on load.
  v0.16.0 - 2025-12-02 - Delegate controller assembly, view callbacks, and handlers
    to helper modules for leaner MainWindow.
  v0.15.82 - 2025-12-01 - Extract main window composition, workspace insight,
    and action handlers for leaner MainWindow.
  v0.15.81 - 2025-12-01 - Extracted bootstrapper, binder, generation, search,
    and workspace controllers to trim MainWindow.
  v0.15.80 - 2025-12-01 - Modularized layout, workspace, template preview,
    notifications, and prompt actions into dedicated controllers.
  v0.15.79 - 2025-12-01 - Delegated theme/palette, runtime settings, catalog,
    and share workflows to helpers.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QLabel,
    QListView,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QWidget,
)

from models.category_model import PromptCategory, slugify_category

from .dialog_launcher import DialogLauncher
from .main_view_binder import MainViewBinderConfig, bind_main_view
from .main_view_builder import build_main_view
from .main_view_callbacks_factory import build_main_view_callbacks
from .main_window_assembler import (
    assemble_execution_controller,
    assemble_prompt_presenter,
    assemble_quick_action_controller,
)
from .main_window_bootstrapper import DetailWidgetCallbacks
from .main_window_bridges import (
    PromptActionsBridge,
    TemplatePreviewBridge,
    WorkspaceInputBridge,
)
from .main_window_composition import (
    PromptGenerationHooks,
    PromptSearchHooks,
    build_main_window_composition,
)
from .main_window_handlers import (
    PromptActionsHandler,
    TemplatePreviewHandler,
    WorkspaceInputHandler,
)
from .notification_controller import NotificationController
from .prompt_list_coordinator import PromptSortOrder
from .settings_workflow import SettingsWorkflow
from .toast import show_toast
from .workspace_command_router import WorkspaceCommandRouter
from .workspace_history_controller import WorkspaceHistoryController
from .workspace_insight_controller import WorkspaceInsightController

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Sequence
    from uuid import UUID

    from PySide6.QtCore import QModelIndex
    from PySide6.QtGui import QCloseEvent, QResizeEvent, QShowEvent

    from config import PromptManagerSettings
    from core import PromptManager
    from models.prompt_model import Prompt

    from .analytics_panel import AnalyticsDashboardPanel
    from .catalog_workflow_controller import CatalogWorkflowController
    from .code_highlighter import CodeHighlighter
    from .controllers.execution_controller import ExecutionController
    from .dialogs.prompt_chains import PromptChainManagerPanel
    from .history_panel import HistoryPanel
    from .notes_panel import NotesPanel
    from .prompt_actions_controller import PromptActionsController
    from .prompt_editor_flow import PromptDialogFactory, PromptEditorFlow
    from .prompt_list_presenter import PromptListPresenter
    from .quick_action_controller import QuickActionController
    from .response_styles_panel import ResponseStylesPanel
    from .result_overlay import ResultActionsOverlay
    from .template_preview import TemplatePreviewWidget
    from .template_preview_controller import TemplatePreviewController
    from .widgets import PromptDetailWidget, PromptFilterPanel, PromptToolbar
    from .workspace_actions_controller import WorkspaceActionsController
    from .workspace_view_controller import WorkspaceViewController
else:  # pragma: no cover - runtime placeholders for type-only imports
    from typing import Any as _Any

    PromptManagerSettings = _Any
    CatalogWorkflowController = _Any
    HistoryPanel = _Any
    NotesPanel = _Any
    ResponseStylesPanel = _Any
    AnalyticsDashboardPanel = _Any
    PromptChainManagerPanel = _Any
    Prompt = _Any
    PromptDetailWidget = _Any
    PromptFilterPanel = _Any
    PromptToolbar = _Any
    ResultActionsOverlay = _Any
    TemplatePreviewWidget = _Any
    TemplatePreviewController = _Any
    WorkspaceActionsController = _Any
    WorkspaceViewController = _Any
    PromptDialogFactory = _Any
    PromptEditorFlow = _Any
    PromptActionsController = _Any


def _match_category_label(  # pyright: ignore[reportUnusedFunction]
    value: str | None, categories: Sequence[PromptCategory]
) -> str | None:
    """Return the canonical category label matching *value* via exact, slug, or fuzzy match."""
    text = (value or "").strip()
    if not text:
        return None
    lowered = text.lower()
    slug = slugify_category(text)

    for category in categories:
        label_lower = category.label.lower()
        if label_lower == lowered:
            return category.label
        if slug and category.slug == slug:
            return category.label

    if slug:
        for category in categories:
            if slug in category.slug:
                return category.label

    for category in categories:
        if lowered in category.label.lower():
            return category.label

    return None


class MainWindow(QMainWindow):
    """Primary window exposing prompt CRUD operations."""

    _SORT_OPTIONS: Sequence[tuple[str, PromptSortOrder]] = (
        ("Name (A-Z)", PromptSortOrder.NAME_ASC),
        ("Name (Z-A)", PromptSortOrder.NAME_DESC),
        ("Quality (high-low)", PromptSortOrder.QUALITY_DESC),
        ("Last modified (newest)", PromptSortOrder.MODIFIED_DESC),
        ("Created (newest)", PromptSortOrder.CREATED_DESC),
        ("Body size (long-short)", PromptSortOrder.BODY_SIZE_DESC),
        ("Rating (high-low)", PromptSortOrder.RATING_DESC),
        ("Usage count (high-low)", PromptSortOrder.USAGE_DESC),
    )

    def __init__(
        self,
        prompt_manager: PromptManager,
        settings: PromptManagerSettings | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Initialise widgets, state, and helper controllers for the GUI."""
        super().__init__(parent)
        self._manager = prompt_manager
        self._settings = settings
        self._prompt_presenter: PromptListPresenter | None = None
        self._prompt_editor_factory: PromptDialogFactory | None = None
        self._prompt_editor_flow: PromptEditorFlow | None = None
        self._catalog_controller: CatalogWorkflowController | None = None
        self._quick_action_controller: QuickActionController | None = None
        self._workspace_view: WorkspaceViewController | None = None
        self._workspace_actions: WorkspaceActionsController | None = None
        self._workspace_router = WorkspaceCommandRouter(lambda: self._workspace_actions)
        self._workspace_history_controller: WorkspaceHistoryController | None = None
        self._settings_workflow: SettingsWorkflow | None = None
        self._dialog_launcher: DialogLauncher | None = None
        self._prompt_actions_controller: PromptActionsController | None = None
        self._template_preview_controller: TemplatePreviewController | None = None
        self._execution_controller: ExecutionController | None = None
        self._history_limit = 50
        self._sort_order = PromptSortOrder.NAME_ASC
        self._notification_controller: NotificationController | None = None
        self._tab_widget: QTabWidget | None = None
        self._template_preview: TemplatePreviewWidget | None = None
        self._prompt_actions_handler: PromptActionsHandler | None = None
        self._workspace_input_handler: WorkspaceInputHandler | None = None
        self._template_preview_handler: TemplatePreviewHandler | None = None
        self._workspace_insight_controller: WorkspaceInsightController | None = None

        self._prompt_actions_bridge = PromptActionsBridge(
            handler_supplier=lambda: self._prompt_actions_handler,
            close_fallback=self._close_window,
        )
        self._workspace_input_bridge = WorkspaceInputBridge(
            handler_supplier=lambda: self._workspace_input_handler,
        )
        self._template_preview_bridge = TemplatePreviewBridge(
            handler_supplier=lambda: self._template_preview_handler,
        )

        self._list_view: QListView = cast("QListView", None)
        self._toolbar = cast("PromptToolbar", None)
        self._language_label: QLabel = cast("QLabel", None)
        self._quick_actions_button: QPushButton = cast("QPushButton", None)
        self._quick_actions_button_default_text: str = ""
        self._quick_actions_button_default_tooltip: str = ""
        self._detect_button: QPushButton = cast("QPushButton", None)
        self._suggest_button: QPushButton = cast("QPushButton", None)
        self._run_button: QPushButton = cast("QPushButton", None)
        self._clear_button: QPushButton = cast("QPushButton", None)
        self._continue_chat_button: QPushButton = cast("QPushButton", None)
        self._end_chat_button: QPushButton = cast("QPushButton", None)
        self._copy_button: QPushButton = cast("QPushButton", None)
        self._web_search_checkbox: QCheckBox = cast("QCheckBox", None)
        self._copy_result_button: QPushButton = cast("QPushButton", None)
        self._copy_result_to_text_window_button: QPushButton = cast("QPushButton", None)
        self._save_button: QPushButton = cast("QPushButton", None)
        self._share_result_button: QPushButton = cast("QPushButton", None)
        self._intent_hint: QLabel = cast("QLabel", None)
        self._filter_panel = cast("PromptFilterPanel", None)
        self._query_input: QPlainTextEdit = cast("QPlainTextEdit", None)
        self._highlighter: CodeHighlighter = cast("CodeHighlighter", None)
        self._main_container = cast("QFrame", None)
        self._result_label: QLabel = cast("QLabel", None)
        self._result_meta: QLabel = cast("QLabel", None)
        self._result_tabs: QTabWidget = cast("QTabWidget", None)
        self._result_text: QTextEdit = cast("QTextEdit", None)
        self._result_overlay: ResultActionsOverlay = cast("ResultActionsOverlay", None)
        self._chat_history_view: QTextEdit = cast("QTextEdit", None)
        self._render_markdown_checkbox: QCheckBox = cast("QCheckBox", None)
        self._history_panel: HistoryPanel = cast("HistoryPanel", None)
        self._notes_panel: NotesPanel | None = None
        self._response_styles_panel: ResponseStylesPanel | None = None
        self._analytics_panel: AnalyticsDashboardPanel | None = None
        self._chain_panel: PromptChainManagerPanel | None = None
        self._template_list_view: QListView = cast("QListView", None)
        self._template_detail_widget: PromptDetailWidget = cast("PromptDetailWidget", None)
        self._template_run_shortcut_button: QPushButton = cast("QPushButton", None)

        detail_callbacks = DetailWidgetCallbacks(
            delete_requested=self._prompt_actions_bridge.delete_current_prompt,
            edit_requested=self._prompt_actions_bridge.edit_prompt,
            version_history_requested=self._prompt_actions_bridge.open_version_history_dialog,
            fork_requested=self._prompt_actions_bridge.fork_prompt,
            refresh_scenarios_requested=self._prompt_actions_bridge.handle_refresh_scenarios_request,
            share_requested=self._prompt_actions_bridge.share_prompt,
        )

        def _execute_prompt_from_dialog(
            prompt: Prompt,
            dialog: QWidget | None,
            context: str | None,
        ) -> None:
            self._prompt_actions_bridge.execute_prompt_as_context(
                prompt,
                parent=dialog,
                context_override=context,
            )

        composition = build_main_window_composition(
            parent=self,
            manager=self._manager,
            settings=settings,
            detail_callbacks=detail_callbacks,
            toast_callback=self._show_toast,
            status_callback=self._show_status_message,
            error_callback=self._show_error,
            prompt_generation_hooks=PromptGenerationHooks(
                execute_prompt_as_context=self._prompt_actions_bridge.execute_prompt_as_context,
                load_prompts=self._load_prompts,
                current_search_text=self._current_search_text,
                select_prompt=self._select_prompt,
                delete_prompt=self._prompt_actions_bridge.delete_prompt,
                status_callback=self._show_status_message,
                error_callback=self._show_error,
                current_prompt_supplier=self._current_prompt,
                open_version_history_dialog=self._prompt_actions_bridge.open_version_history_dialog,
            ),
            prompt_search_hooks=PromptSearchHooks(
                presenter_supplier=lambda: self._prompt_presenter,
                filter_panel_supplier=lambda: self._filter_panel,
                load_prompts=self._load_prompts,
                current_search_text=self._current_search_text,
                select_prompt=self._select_prompt,
            ),
            share_result_button_supplier=lambda: self._share_result_button,
            execution_controller_supplier=lambda: self._execution_controller,
        )
        bootstrap = composition.bootstrap
        self._model = bootstrap.model
        self._detail_widget = bootstrap.detail_widget
        self._prompt_coordinator = bootstrap.prompt_coordinator
        self._usage_logger = bootstrap.usage_logger
        self._share_controller = bootstrap.share_controller
        self._share_workflow = bootstrap.share_workflow
        self._layout_state = bootstrap.layout_state
        self._layout_controller = bootstrap.layout_controller
        self._runtime_settings_service = bootstrap.runtime_settings_service
        self._runtime_settings = bootstrap.runtime_settings
        self._appearance_controller = bootstrap.appearance_controller
        self._notification_indicator = bootstrap.notification_indicator

        self._prompt_generation_service = composition.prompt_generation_service
        self._prompt_search_controller = composition.prompt_search_controller

        self.setWindowTitle("Prompt Manager")
        self._layout_state.restore_window_geometry(self)
        self._build_ui()
        presenter_assembly = assemble_prompt_presenter(
            self,
            filter_preferences=composition.filter_preferences,
        )
        self._prompt_presenter = presenter_assembly.presenter
        self._sort_order = presenter_assembly.sort_order
        self._quick_action_controller = assemble_quick_action_controller(self)
        self._rebuild_prompt_generation_components()
        self._execution_controller = assemble_execution_controller(self)
        self._appearance_controller.apply_theme()
        self._restore_splitter_state()
        self.statusBar().addPermanentWidget(self._notification_indicator)
        self._initialize_notification_controller()
        self._load_prompts()

    def _on_delete_clicked(self) -> None:
        """Handle delete actions originating from the detail panel."""
        self._prompt_actions_bridge.delete_current_prompt()

    def _on_edit_clicked(self) -> None:
        """Open the editor for the currently selected prompt."""
        self._prompt_actions_bridge.edit_prompt()

    def _on_fork_clicked(self) -> None:
        """Fork the currently selected prompt via the detail panel."""
        self._prompt_actions_bridge.fork_prompt()

    def _duplicate_prompt(self, prompt: Prompt) -> None:
        """Duplicate *prompt* when invoked from the list context menu."""
        self._prompt_actions_bridge.duplicate_prompt(prompt)

    def _fork_prompt(self, prompt: Prompt) -> None:
        """Fork *prompt* when invoked from the list context menu."""
        self._prompt_actions_bridge.fork_prompt_direct(prompt)

    def _open_version_history_dialog(self) -> None:
        """Open the version history dialog for the active prompt."""
        self._prompt_actions_bridge.open_version_history_dialog()

    def _handle_refresh_scenarios_request(self, detail_widget: PromptDetailWidget) -> None:
        """Regenerate scenarios requested by the detail widget."""
        self._prompt_actions_bridge.handle_refresh_scenarios_request(detail_widget)

    def _build_ui(self) -> None:
        """Create the main layout with list/search/detail panes."""
        callbacks = build_main_view_callbacks(self)
        usage_log_path = getattr(self._usage_logger, "log_path", None)
        components = build_main_view(
            parent=self,
            model=self._model,
            detail_widget=self._detail_widget,
            manager=self._manager,
            history_limit=self._history_limit,
            sort_options=[(label, order.value) for label, order in self._SORT_OPTIONS],
            callbacks=callbacks,
            status_callback=self._show_status_message,
            toast_callback=self._show_toast,
            event_filter_target=self,
            usage_log_path=usage_log_path,
            history_note_callback=self._workspace_input_bridge.handle_note_update,
            history_export_callback=self._workspace_input_bridge.handle_history_export,
        )
        binder_config = MainViewBinderConfig(
            layout_controller=self._layout_controller,
            layout_state=self._layout_state,
            manager=self._manager,
            model=self._model,
            detail_widget=self._detail_widget,
            usage_logger=self._usage_logger,
            status_callback=self._show_status_message,
            error_callback=self._show_error,
            toast_callback=self._show_toast,
            current_prompt_supplier=self._current_prompt,
            execution_controller_supplier=lambda: self._execution_controller,
            quick_action_controller_supplier=lambda: self._quick_action_controller,
            refresh_prompt_after_rating=self._prompt_search_controller.refresh_prompt_after_rating,
            execute_from_prompt_body=self._prompt_actions_bridge.execute_prompt_from_body,
        )
        bind_main_view(self, components=components, config=binder_config)
        self._workspace_history_controller = WorkspaceHistoryController(
            manager=self._manager,
            model=self._model,
            detail_widget=self._detail_widget,
            list_view=self._list_view,
            current_prompt_supplier=self._current_prompt,
            template_detail_widget_supplier=lambda: self._template_detail_widget,
            template_preview_controller_supplier=lambda: self._template_preview_controller,
            execution_controller_supplier=lambda: self._execution_controller,
        )
        self._settings_workflow = SettingsWorkflow(
            parent=self,
            manager=self._manager,
            runtime_settings_service=self._runtime_settings_service,
            runtime_settings=self._runtime_settings,
            quick_action_supplier=lambda: self._quick_action_controller,
            prompt_generation_refresher=self._rebuild_prompt_generation_components,
            appearance_controller=self._appearance_controller,
            execution_controller_supplier=lambda: self._execution_controller,
            load_prompts=self._load_prompts,
            current_search_text=self._current_search_text,
            run_button=self._run_button,
            template_preview_supplier=lambda: self._template_preview,
            toast_callback=self._show_toast,
        )
        self._dialog_launcher = DialogLauncher(
            parent=self,
            manager=self._manager,
            current_prompt_supplier=self._current_prompt,
            load_prompts=self._load_prompts,
            current_search_text=self._current_search_text,
            select_prompt=lambda prompt: self._workspace_history_controller.select_prompt(prompt.id)
            if self._workspace_history_controller is not None
            else None,
        )
        self._workspace_router = WorkspaceCommandRouter(lambda: self._workspace_actions)
        self._workspace_input_handler = WorkspaceInputHandler(
            workspace_router=self._workspace_router,
        )
        self._template_preview_handler = TemplatePreviewHandler(
            controller_supplier=lambda: self._template_preview_controller,
        )
        self._prompt_actions_handler = PromptActionsHandler(
            parent=self,
            manager=self._manager,
            model_prompts_supplier=lambda: self._model.prompts(),
            current_prompt_supplier=self._current_prompt,
            detail_widget=self._detail_widget,
            prompt_search_controller=self._prompt_search_controller,
            prompt_actions_controller_supplier=lambda: self._prompt_actions_controller,
            prompt_editor_flow_supplier=lambda: self._prompt_editor_flow,
            catalog_controller_supplier=lambda: self._catalog_controller,
            settings_workflow_supplier=lambda: self._settings_workflow,
            dialog_launcher_supplier=lambda: self._dialog_launcher,
            share_workflow_supplier=lambda: self._share_workflow,
            load_prompts=self._load_prompts,
            current_search_text=self._current_search_text,
            status_callback=self._show_status_message,
            exit_callback=self._close_window,
        )
        self._appearance_controller.set_container(self._main_container)
        self._workspace_insight_controller = WorkspaceInsightController(
            manager=self._manager,
            query_input=self._query_input,
            intent_hint_label=self._intent_hint,
            language_label=self._language_label,
            highlighter=self._highlighter,
            presenter_supplier=lambda: self._prompt_presenter,
            current_prompts_supplier=lambda: self._model.prompts(),
            status_callback=self._show_status_message,
            error_callback=self._show_error,
            usage_logger=self._usage_logger,
            quick_action_controller_supplier=lambda: self._quick_action_controller,
            current_search_text=self._current_search_text,
        )
        self._workspace_insight_controller.initialise_language()
        self.setCentralWidget(self._main_container)

        has_executor = self._manager.executor is not None
        self._run_button.setEnabled(has_executor)
        if self._template_preview is not None:
            self._template_preview.set_run_enabled(has_executor)
        self._copy_result_button.setEnabled(False)
        self._copy_result_to_text_window_button.setEnabled(False)
        self._save_button.setEnabled(False)
        self._share_result_button.setEnabled(False)
        self._continue_chat_button.setEnabled(False)
        self._end_chat_button.setEnabled(False)

    def _initialize_notification_controller(self) -> None:
        """Wire the notification bridge and seed the indicator state."""
        self._notification_controller = NotificationController(
            parent=self,
            indicator=self._notification_indicator,
            notification_center=self._manager.notification_center,
            status_callback=self._show_status_message,
            toast_callback=self._show_toast,
        )
        self._notification_controller.bootstrap_history(self._manager.notification_center.history())

    def _restore_splitter_state(self) -> None:
        """Restore splitter sizes from persisted settings."""
        self._layout_controller.restore_splitter_state()

    def _save_splitter_state(self) -> None:
        """Persist splitter sizes for future sessions."""
        self._layout_controller.save_splitter_state()

    def _current_search_text(self) -> str:
        """Return the current text in the toolbar search field."""
        return self._layout_controller.current_search_text()

    def showEvent(self, event: QShowEvent) -> None:  # type: ignore[override]
        """Ensure splitter state is captured once the window is visible."""
        super().showEvent(event)
        self._layout_controller.handle_show_event()

    def resizeEvent(self, event: QResizeEvent) -> None:  # type: ignore[override]
        """Keep splitter stable when the window resizes."""
        super().resizeEvent(event)
        self._layout_controller.handle_resize_event()

    def _load_prompts(self, search_text: str = "", *, use_indicator: bool = False) -> None:
        """Delegate prompt loading to the presenter."""
        if self._prompt_presenter is None:
            return
        self._prompt_presenter.load_prompts(search_text, use_indicator=use_indicator)

    def _update_intent_hint(self, prompts: Sequence[Prompt]) -> None:
        """Update the hint label with detected intent context and matches."""
        controller = self._workspace_insight_controller
        if controller is None:
            return
        controller.update_intent_hint(prompts)

    def _on_query_text_changed(self) -> None:
        """Update language detection and syntax highlighting as the user types."""
        controller = self._workspace_insight_controller
        if controller is None:
            return
        controller.handle_query_text_changed()

    def _show_command_palette(self) -> None:
        if self._quick_action_controller is None:
            return
        self._quick_action_controller.show_command_palette()

    def _on_detect_intent_clicked(self) -> None:
        """Run intent detection on the free-form query input."""
        controller = self._workspace_insight_controller
        if controller is None:
            return
        controller.detect_intent()

    def _on_suggest_prompt_clicked(self) -> None:
        """Generate prompt suggestions from the free-form query input."""
        controller = self._workspace_insight_controller
        if controller is None:
            return
        controller.suggest_prompt()

    def _show_similar_prompts(self, prompt: Prompt) -> None:
        """Delegate to the presenter to display similar prompts."""
        presenter = self._prompt_presenter
        if presenter is None:
            return
        presenter.show_similar_prompts(prompt)

    def _on_tab_changed(self, index: int) -> None:
        if self._tab_widget is None:
            return
        widget = self._tab_widget.widget(index)
        if widget is self._history_panel:
            self._history_panel.refresh()
            self._usage_logger.log_history_view(total=self._history_panel.row_count())
        if index == 0 and self._template_preview_controller is not None:
            self._template_preview_controller.hide_transition_indicator()

    def _on_render_markdown_toggled(self, _state: int) -> None:
        """Refresh result and chat panes when the markdown toggle changes."""
        if self._execution_controller is not None:
            self._execution_controller.refresh_rendering()

    def _current_prompt(self) -> Prompt | None:
        """Return the prompt currently selected in the list."""
        if self._prompt_presenter is None:
            return None
        return self._prompt_presenter.current_prompt()

    def _on_prompt_double_clicked(self, index: QModelIndex) -> None:
        """Open the edit dialog when a prompt is double-clicked."""
        if not index.isValid():
            return
        self._list_view.setCurrentIndex(index)
        self._prompt_actions_bridge.edit_prompt()

    def _close_window(self) -> None:
        """Close the window without additional handlers."""
        self.close()

    def _rebuild_prompt_generation_components(self) -> None:
        """Refresh prompt dialog flows after configuration or bootstrap changes."""
        components = self._prompt_generation_service.rebuild_components(self)
        self._prompt_editor_factory = components.dialog_factory
        self._prompt_editor_flow = components.editor_flow
        self._catalog_controller = components.catalog_controller

    def _on_selection_changed(self, *_: object) -> None:
        """Update the detail panel to reflect the new selection."""
        controller = self._workspace_history_controller
        if controller is None:
            return
        controller.handle_selection_changed()

    def _select_prompt(self, prompt_id: UUID) -> None:
        """Highlight the given prompt in the list view when present."""
        controller = self._workspace_history_controller
        if controller is None:
            return
        controller.select_prompt(prompt_id)

    def _on_notifications_clicked(self) -> None:
        if self._notification_controller is None:
            return
        self._notification_controller.show_task_center()

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        """Persist layout before closing and dispose transient dialogs."""
        self._layout_state.save_window_geometry(self)
        self._save_splitter_state()
        if self._notification_controller is not None:
            self._notification_controller.close()
        if self._template_preview_controller is not None:
            self._template_preview_controller.hide_transition_indicator()
        super().closeEvent(event)

    @staticmethod
    def _truncate_text(value: str, limit: int = 160) -> str:
        """Return the provided text truncated with an ellipsis if needed."""
        text = value.strip()
        if len(text) <= limit:
            return text
        return f"{text[: limit - 1].rstrip()}…"

    @staticmethod
    def _format_timestamp(value: datetime) -> str:
        """Return a compact UTC timestamp for lineage summaries."""
        return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")

    def _show_error(self, title: str, message: str) -> None:
        """Display an error dialog and log to status bar."""
        QMessageBox.critical(self, title, message)
        self.statusBar().showMessage(message, 5000)

    def _show_status_message(self, message: str, duration_ms: int = 3000) -> None:
        """Show a transient status bar message."""
        self.statusBar().showMessage(message, duration_ms)

    def _show_toast(self, message: str, duration_ms: int = 2500) -> None:
        """Display a toast anchored to the main window."""
        show_toast(self, message, duration_ms)


__all__ = ["MainWindow"]
