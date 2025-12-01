"""Main window widgets and models for the Prompt Manager GUI.

Updates:
  v0.15.81 - 2025-12-01 - Extracted bootstrapper, binder, generation, search,
    and workspace controllers to trim MainWindow.
  v0.15.80 - 2025-12-01 - Modularized layout, workspace, template preview,
    notifications, and prompt actions into dedicated controllers.
  v0.15.79 - 2025-12-01 - Delegated theme/palette, runtime settings, catalog,
    and share workflows to helpers.
  v0.15.78 - 2025-12-01 - Guard tab change handler until widgets are initialised.
  v0.15.77 - 2025-12-01 - Extract prompt list coordinator for loading/filtering/sorting logic.
  v0.15.76 - 2025-12-01 - Delegate layout persistence to WindowStateManager helper.
  v0.15.75 - 2025-11-30 - Extract toolbar and filter panels into reusable widgets.
  v0.15.74 - 2025-11-30 - Extract execution and chat controller plus delegate workspace actions.
  v0.15.73 - 2025-11-30 - Extract widgets, list model, and layout persistence modules.
  v0.15.72 - 2025-11-30 - Extract result overlay and share workflow helpers into modules.
  v0.15.71 - 2025-11-30 - Add workspace result sharing via overlay action button.
  v0.15.70 - 2025-11-29 - Make workspace splitter resize only the prompt input textarea.
  v0.15.69 - 2025-11-29 - Toggle inline markdown rendering with a checkbox for output/chat.
  v0.15.68 - 2025-11-29 - Embed result actions inside the output text area with overlay controls.
  v0.15.67 - 2025-11-29 - Move result action buttons inside the output tab.
  v0.15.66 - 2025-11-29 - Add Enhanced Prompt Workbench launcher and wiring.
  v0.15.65 - 2025-11-29 - Remove QModelIndex default construction for Ruff B008 compliance.
  v0.15.64 - 2025-11-29 - Added prompt template editor dialog shortcut.
  v0.15.63 - 2025-11-28 - Added analytics dashboard tab with CSV export.
  v0.15.62 - 2025-11-28 - Enabled ShareText publishing plus clipboard copy.
  v0.15.61 - 2025-11-28 - Improved prompt list refresh when clearing search.
  v0.15.60 - 2025-11-28 - Introduced background task center with toasts.
  v0.15.59 - 2025-11-28 - Wired Refresh Scenarios action to LiteLLM.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QListView,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QWidget,
)

from config import DEFAULT_THEME_MODE, PromptManagerSettings
from core import (
    IntentLabel,
    NameGenerationError,
    PromptManager,
    PromptManagerError,
    PromptNotFoundError,
    PromptStorageError,
    PromptVersionError,
    RepositoryError,
)
from models.category_model import PromptCategory, slugify_category

from .controllers.execution_controller import ExecutionController
from .dialogs import InfoDialog, PromptVersionHistoryDialog
from .language_tools import DetectedLanguage, detect_language
from .main_view_binder import MainViewBinderConfig, bind_main_view
from .main_view_builder import MainViewCallbacks, build_main_view
from .main_window_bootstrapper import DetailWidgetCallbacks, MainWindowBootstrapper
from .notification_controller import NotificationController
from .prompt_generation_service import PromptGenerationService
from .prompt_list_coordinator import PromptSortOrder
from .prompt_list_presenter import PromptListCallbacks, PromptListPresenter
from .prompt_search_controller import PromptSearchController
from .prompt_templates_dialog import PromptTemplateEditorDialog
from .quick_action_controller import QuickActionController
from .runtime_settings_service import RuntimeSettingsResult
from .settings_dialog import SettingsDialog
from .toast import show_toast
from .workbench.workbench_window import WorkbenchModeDialog, WorkbenchWindow
from .workspace_command_router import WorkspaceCommandRouter

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Sequence
    from uuid import UUID

    from PySide6.QtCore import QModelIndex, QPoint
    from PySide6.QtGui import QResizeEvent, QShowEvent

    from models.prompt_model import Prompt

    from .catalog_workflow_controller import CatalogWorkflowController
    from .prompt_actions_controller import PromptActionsController
    from .prompt_editor_flow import PromptDialogFactory, PromptEditorFlow
    from .result_overlay import ResultActionsOverlay
    from .template_preview import TemplatePreviewWidget
    from .template_preview_controller import TemplatePreviewController
    from .widgets import PromptDetailWidget, PromptFilterPanel, PromptToolbar
    from .workspace_actions_controller import WorkspaceActionsController
    from .workspace_view_controller import WorkspaceViewController

logger = logging.getLogger(__name__)

def _match_category_label(
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
        parent=None,
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
        self._prompt_actions_controller: PromptActionsController | None = None
        self._template_preview_controller: TemplatePreviewController | None = None
        self._execution_controller: ExecutionController | None = None
        self._history_limit = 50
        self._sort_order = PromptSortOrder.NAME_ASC
        self._detected_language: DetectedLanguage = detect_language("")
        self._share_result_button: QPushButton | None = None
        self._query_input: QPlainTextEdit | None = None
        self._render_markdown_checkbox: QCheckBox | None = None
        self._template_preview: TemplatePreviewWidget | None = None
        self._template_list_view: QListView | None = None
        self._template_detail_widget: PromptDetailWidget | None = None
        self._template_run_shortcut_button: QPushButton | None = None
        self._result_overlay: ResultActionsOverlay | None = None
        self._tab_widget: QTabWidget | None = None
        self._toolbar: PromptToolbar | None = None
        self._filter_panel: PromptFilterPanel | None = None
        self._main_container: QFrame | None = None
        self._notification_controller: NotificationController | None = None
        self._workbench_windows: list[WorkbenchWindow] = []

        detail_callbacks = DetailWidgetCallbacks(
            delete_requested=self._on_delete_clicked,
            edit_requested=self._on_edit_clicked,
            version_history_requested=self._open_version_history_dialog,
            fork_requested=self._on_fork_clicked,
            refresh_scenarios_requested=self._handle_refresh_scenarios_request,
            share_requested=self._on_share_prompt_requested,
        )
        bootstrapper = MainWindowBootstrapper(
            parent=self,
            manager=self._manager,
            settings=settings,
            detail_callbacks=detail_callbacks,
            toast_callback=self._show_toast,
            status_callback=self._show_status_message,
            error_callback=self._show_error,
            share_result_button_supplier=lambda: self._share_result_button,
            execution_controller_supplier=lambda: self._execution_controller,
        )
        bootstrap = bootstrapper.bootstrap()
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

        self._prompt_generation_service = PromptGenerationService(
            manager=self._manager,
            execute_context_handler=lambda prompt, dlg, context: self._execute_prompt_as_context(
                prompt,
                parent=dlg,
                context_override=context,
            ),
            load_prompts=self._load_prompts,
            current_search_text=self._current_search_text,
            select_prompt=self._select_prompt,
            delete_prompt=self._delete_prompt,
            status_callback=self._show_status_message,
            error_callback=self._show_error,
            current_prompt_supplier=self._current_prompt,
            open_version_history_dialog=self._open_version_history_dialog,
        )
        self._prompt_search_controller = PromptSearchController(
            parent=self,
            manager=self._manager,
            presenter_supplier=lambda: self._prompt_presenter,
            filter_panel_supplier=lambda: self._filter_panel,
            layout_controller=self._layout_controller,
            load_prompts=self._load_prompts,
            current_search_text=self._current_search_text,
            select_prompt=self._select_prompt,
        )

        filter_prefs = self._layout_state.load_filter_preferences()
        pending_category_slug = filter_prefs.category_slug
        pending_tag_value = filter_prefs.tag
        pending_quality_value = filter_prefs.min_quality
        stored_sort_value = filter_prefs.sort_value

        self.setWindowTitle("Prompt Manager")
        self._layout_state.restore_window_geometry(self)
        self._build_ui()
        presenter_callbacks = PromptListCallbacks(
            update_intent_hint=self._update_intent_hint,
            select_prompt=self._select_prompt,
            show_error=self._show_error,
            show_status=self._show_status_message,
            show_toast=self._show_toast,
        )
        self._prompt_presenter = PromptListPresenter(
            manager=self._manager,
            coordinator=self._prompt_coordinator,
            model=self._model,
            detail_widget=self._detail_widget,
            list_view=self._list_view,
            filter_panel=self._filter_panel,
            toolbar=self._toolbar,
            callbacks=presenter_callbacks,
            parent=self,
        )
        self._prompt_presenter.set_pending_filter_preferences(
            category_slug=pending_category_slug,
            tag=pending_tag_value,
            min_quality=pending_quality_value,
        )
        if stored_sort_value:
            try:
                self._sort_order = PromptSortOrder(stored_sort_value)
            except ValueError:
                logger.warning("Unknown stored sort order: %s", stored_sort_value)
        self._prompt_presenter.set_sort_order(self._sort_order)
        self._quick_action_controller = QuickActionController(
            parent=self,
            manager=self._manager,
            presenter=self._prompt_presenter,
            detail_widget=self._detail_widget,
            query_input=self._query_input,
            quick_actions_button=self._quick_actions_button,
            button_default_text=self._quick_actions_button_default_text,
            button_default_tooltip=self._quick_actions_button_default_tooltip,
            status_callback=self._show_status_message,
            error_callback=self._show_error,
            exit_callback=self._on_exit_clicked,
            quick_actions_config=self._runtime_settings.get("quick_actions"),
        )
        self._quick_action_controller.sync_button()
        self._rebuild_prompt_generation_components()
        self._execution_controller = ExecutionController(
            manager=self._manager,
            runtime_settings=self._runtime_settings,
            usage_logger=self._usage_logger,
            share_controller=self._share_controller,
            query_input=self._query_input,
            result_label=self._result_label,
            result_meta=self._result_meta,
            result_tabs=self._result_tabs,
            result_text=self._result_text,
            chat_history_view=self._chat_history_view,
            render_markdown_checkbox=self._render_markdown_checkbox,
            copy_result_button=self._copy_result_button,
            copy_result_to_text_window_button=self._copy_result_to_text_window_button,
            save_button=self._save_button,
            share_result_button=self._share_result_button,
            continue_chat_button=self._continue_chat_button,
            end_chat_button=self._end_chat_button,
            status_callback=self._show_status_message,
            clear_status_callback=self.statusBar().clearMessage,
            error_callback=self._show_error,
            toast_callback=self._show_toast,
            settings=self._settings,
        )
        self._execution_controller.notify_share_providers_changed()
        self._appearance_controller.apply_theme()
        self._restore_splitter_state()
        self.statusBar().addPermanentWidget(self._notification_indicator)
        self._initialize_notification_controller()
        self._load_prompts()

    def _build_ui(self) -> None:
        """Create the main layout with list/search/detail panes."""
        callbacks = MainViewCallbacks(
            search_requested=lambda text=None: self._prompt_search_controller.search_requested(
                text, use_indicator=True
            ),
            search_text_changed=self._prompt_search_controller.search_changed,
            refresh_requested=self._on_refresh_clicked,
            add_requested=self._on_add_clicked,
            workbench_requested=self._on_workbench_clicked,
            import_requested=self._on_import_clicked,
            export_requested=self._on_export_clicked,
            maintenance_requested=self._on_maintenance_clicked,
            notifications_requested=self._on_notifications_clicked,
            info_requested=self._on_info_clicked,
            templates_requested=self._on_prompt_templates_clicked,
            settings_requested=self._on_settings_clicked,
            exit_requested=self._on_exit_clicked,
            show_command_palette=self._show_command_palette,
            detect_intent_clicked=self._on_detect_intent_clicked,
            suggest_prompt_clicked=self._on_suggest_prompt_clicked,
            run_prompt_clicked=self._on_run_prompt_clicked,
            clear_workspace_clicked=self._on_clear_workspace_clicked,
            continue_chat_clicked=self._on_continue_chat_clicked,
            end_chat_clicked=self._on_end_chat_clicked,
            copy_prompt_clicked=self._on_copy_prompt_clicked,
            copy_result_clicked=self._on_copy_result_clicked,
            copy_result_to_text_window_clicked=self._on_copy_result_to_text_window_clicked,
            save_result_clicked=self._on_save_result_clicked,
            share_result_clicked=self._on_share_result_clicked,
            filters_changed=self._prompt_search_controller.filters_changed,
            sort_changed=self._prompt_search_controller.sort_changed,
            manage_categories_clicked=self._on_manage_categories_clicked,
            query_text_changed=self._on_query_text_changed,
            tab_changed=self._on_tab_changed,
            selection_changed=self._on_selection_changed,
            prompt_double_clicked=self._on_prompt_double_clicked,
            prompt_context_menu=self._on_prompt_context_menu,
            render_markdown_toggled=self._on_render_markdown_toggled,
            template_preview_run_requested=self._on_template_preview_run_requested,
            template_preview_run_state_changed=self._on_template_run_state_changed,
            template_tab_run_clicked=self._on_template_tab_run_clicked,
        )
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
            history_note_callback=self._handle_note_update,
            history_export_callback=self._handle_history_export,
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
            execute_from_prompt_body=self._execute_prompt_from_context_menu,
        )
        bind_main_view(self, components=components, config=binder_config)
        self._workspace_router = WorkspaceCommandRouter(lambda: self._workspace_actions)
        self._appearance_controller.set_container(self._main_container)
        self._update_detected_language(self._query_input.toPlainText(), force=True)
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


    def _on_manage_categories_clicked(self) -> None:
        """Open the category management dialog."""
        self._prompt_search_controller.manage_categories()

    def _handle_refresh_scenarios_request(self, detail_widget: PromptDetailWidget) -> None:
        """Regenerate persisted scenarios for the currently displayed prompt."""
        self._prompt_search_controller.handle_refresh_scenarios_request(detail_widget)

    def _update_intent_hint(self, prompts: Sequence[Prompt]) -> None:
        """Update the hint label with detected intent context and matches."""
        presenter = self._prompt_presenter
        suggestions = presenter.suggestions if presenter is not None else None
        if suggestions is None:
            self._intent_hint.clear()
            self._intent_hint.setVisible(False)
            return

        prompt_list = list(prompts)
        prediction = suggestions.prediction
        label_text = prediction.label.value.replace("_", " ").title()
        confidence_pct = int(round(prediction.confidence * 100))

        summary_parts: list[str] = []
        if prediction.label is not IntentLabel.GENERAL or prediction.category_hints:
            summary_parts.append(f"Detected intent: {label_text} ({confidence_pct}%).")
        if prediction.category_hints:
            summary_parts.append(f"Focus: {', '.join(prediction.category_hints)}")
        if prediction.language_hints:
            summary_parts.append(f"Lang: {', '.join(prediction.language_hints)}")

        top_names = [prompt.name for prompt in prompt_list[:3]]
        if top_names:
            summary_parts.append(f"Top matches: {', '.join(top_names)}")
        if suggestions.fallback_used:
            summary_parts.append("Fallback ranking applied")

        if summary_parts:
            message = " | ".join(summary_parts)
        elif top_names:
            message = "Top matches: " + ", ".join(top_names)
        else:
            self._intent_hint.clear()
            self._intent_hint.setVisible(False)
            return

        self._intent_hint.setText(message)
        self._intent_hint.setVisible(True)

    def _on_query_text_changed(self) -> None:
        """Update language detection and syntax highlighting as the user types."""
        text = self._query_input.toPlainText()
        self._update_detected_language(text)
        controller = self._quick_action_controller
        if controller is not None and controller.is_workspace_signal_suppressed():
            return
        if controller is not None:
            controller.clear_workspace_seed()

    def _update_detected_language(self, text: str, *, force: bool = False) -> None:
        """Detect the language for `text` and refresh UI elements when it changes."""
        detection = detect_language(text)
        if not force and detection.code == self._detected_language.code:
            return
        self._detected_language = detection
        if detection.confidence:
            percentage = int(round(detection.confidence * 100))
            self._language_label.setText(f"Language: {detection.name} ({percentage}%)")
        else:
            self._language_label.setText(f"Language: {detection.name}")
        self._highlighter.set_language(detection.code)
        self.statusBar().showMessage(f"Workspace language: {detection.name}", 3000)

    def _show_command_palette(self) -> None:
        if self._quick_action_controller is None:
            return
        self._quick_action_controller.show_command_palette()

    def _on_detect_intent_clicked(self) -> None:
        """Run intent detection on the free-form query input."""
        text = self._query_input.toPlainText().strip()
        if not text:
            self.statusBar().showMessage("Paste some text or code to analyse.", 4000)
            return
        classifier = self._manager.intent_classifier
        if classifier is None:
            self._show_error("Intent detection unavailable", "No intent classifier is configured.")
            return
        prediction = classifier.classify(text)
        current_prompts = list(self._model.prompts())
        presenter = self._prompt_presenter
        if presenter is not None:
            presenter.set_intent_suggestions(
                PromptManager.IntentSuggestions(
                    prediction=prediction,
                    prompts=list(current_prompts),
                    fallback_used=False,
                )
            )
        self._update_intent_hint(current_prompts)
        label = prediction.label.value.replace("_", " ").title()
        self.statusBar().showMessage(
            f"Detected intent: {label} ({int(prediction.confidence * 100)}%)", 5000
        )
        self._usage_logger.log_detect(prediction=prediction, query_text=text)

    def _on_suggest_prompt_clicked(self) -> None:
        """Generate prompt suggestions from the free-form query input."""
        query = self._query_input.toPlainText().strip() or self._current_search_text().strip()
        if not query:
            self.statusBar().showMessage("Provide text or use search to fetch suggestions.", 4000)
            return
        try:
            suggestions = self._manager.suggest_prompts(query, limit=20)
        except PromptManagerError as exc:
            self._show_error("Unable to suggest prompts", str(exc))
            return
        self._usage_logger.log_suggest(
            prediction=suggestions.prediction,
            query_text=query,
            prompts=suggestions.prompts,
            fallback_used=suggestions.fallback_used,
        )
        if self._prompt_presenter is not None:
            self._prompt_presenter.apply_suggestions(suggestions)
        top_name = suggestions.prompts[0].name if suggestions.prompts else None
        if top_name:
            self.statusBar().showMessage(f"Top suggestion: {top_name}", 5000)

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

    def _on_save_result_clicked(self) -> None:
        """Persist the latest execution result with optional user notes."""
        self._workspace_router.save_result()

    def _on_continue_chat_clicked(self) -> None:
        """Send a follow-up message within the active chat session."""
        self._workspace_router.continue_chat()

    def _on_end_chat_clicked(self) -> None:
        """Terminate the active chat session without clearing the transcript."""
        self._workspace_router.end_chat()

    def _on_run_prompt_clicked(self) -> None:
        """Execute the selected prompt, seeding from the workspace or the prompt body."""
        self._workspace_router.run_prompt()

    def _on_clear_workspace_clicked(self) -> None:
        """Clear the workspace editor, output tab, and chat transcript."""
        self._workspace_router.clear_workspace()

    def _handle_note_update(self, execution_id: UUID, note: str) -> None:
        """Record analytics when execution notes are edited."""
        self._workspace_router.handle_note_update(execution_id, note)

    def _handle_history_export(self, entries: int, path: str) -> None:
        """Record analytics when history is exported."""
        self._workspace_router.handle_history_export(entries, path)

    def _on_copy_prompt_clicked(self) -> None:
        """Copy the currently selected prompt body to the clipboard."""
        prompt = self._current_prompt()
        if prompt is None:
            prompts = self._model.prompts()
            prompt = prompts[0] if prompts else None
        if prompt is None:
            self.statusBar().showMessage("Select a prompt to copy first.", 3000)
            return
        self._copy_prompt_to_clipboard(prompt)

    def _copy_prompt_to_clipboard(self, prompt: Prompt) -> None:
        """Copy a prompt's primary text to the clipboard with status feedback."""
        if self._prompt_actions_controller is None:
            return
        self._prompt_actions_controller.copy_prompt_to_clipboard(prompt)

    def _on_share_prompt_requested(self) -> None:
        """Display provider choices and initiate the share workflow."""
        self._share_workflow.share_prompt()

    def _on_share_result_clicked(self) -> None:
        """Display provider choices for sharing the latest output text."""
        self._share_workflow.share_result()

    def _show_prompt_description(self, prompt: Prompt) -> None:
        """Display the prompt description in a dialog for quick reference."""
        if self._prompt_actions_controller is None:
            return
        self._prompt_actions_controller.show_prompt_description(prompt)

    def _on_render_markdown_toggled(self, _state: int) -> None:
        """Refresh result and chat panes when the markdown toggle changes."""
        if self._execution_controller is not None:
            self._execution_controller.refresh_rendering()

    def _on_copy_result_clicked(self) -> None:
        """Copy the latest execution result to the clipboard."""
        self._workspace_router.copy_result_to_clipboard()

    def _on_copy_result_to_text_window_clicked(self) -> None:
        """Populate the workspace text window with the latest execution result."""
        self._workspace_router.copy_result_to_workspace()

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
        self._on_edit_clicked()

    def _on_prompt_context_menu(self, point: QPoint) -> None:
        """Show a context menu with prompt-specific actions."""
        if self._prompt_actions_controller is None:
            return
        self._prompt_actions_controller.show_context_menu(point)

    def _execute_prompt_from_context_menu(self, prompt: Prompt) -> None:
        """Populate the workspace with the prompt body and execute immediately."""
        if self._prompt_actions_controller is None:
            return
        self._prompt_actions_controller.execute_prompt_from_body(prompt)

    def _execute_prompt_as_context(
        self,
        prompt: Prompt,
        *,
        parent: QWidget | None = None,
        context_override: str | None = None,
    ) -> None:
        """Ask for a task and run the prompt using its body as contextual input."""
        if self._prompt_actions_controller is None:
            return
        self._prompt_actions_controller.execute_prompt_as_context(
            prompt,
            parent=parent,
            context_override=context_override,
        )

    def _on_template_preview_run_requested(
        self,
        rendered_text: str,
        variables: dict[str, str],
    ) -> None:
        """Execute the selected prompt using the rendered template preview text."""
        if self._template_preview_controller is None:
            return
        self._template_preview_controller.handle_run_requested(rendered_text, variables)

    def _duplicate_prompt(self, prompt: Prompt) -> None:
        """Open a creation dialog with the selected prompt pre-filled and persist the copy."""
        if self._prompt_editor_flow is None:
            return
        self._prompt_editor_flow.duplicate_prompt(prompt)

    def _fork_prompt(self, prompt: Prompt) -> None:
        """Fork the selected prompt and allow optional edits before saving."""
        if self._prompt_editor_flow is None:
            return
        self._prompt_editor_flow.fork_prompt(prompt)

    def _on_refresh_clicked(self) -> None:
        """Reload catalogue data, respecting the current search text."""
        self._load_prompts(self._current_search_text())

    def _on_add_clicked(self) -> None:
        """Open the creation dialog and persist a new prompt."""
        if self._prompt_editor_flow is None:
            return
        self._prompt_editor_flow.create_prompt()

    def _on_workbench_clicked(self) -> None:
        """Launch the Enhanced Prompt Workbench with the desired starting mode."""
        try:
            templates = self._manager.repository.list(limit=200)
        except RepositoryError as exc:
            logger.warning("Unable to load templates for workbench: %s", exc)
            templates = []
        dialog = WorkbenchModeDialog(templates, self)
        if dialog.exec() != QDialog.Accepted:
            return
        selection = dialog.result_selection()
        window = WorkbenchWindow(
            self._manager,
            mode=selection.mode,
            template_prompt=selection.template_prompt,
            parent=self,
        )
        self._workbench_windows.append(window)

        window.destroyed.connect(  # type: ignore[arg-type]
            lambda *_: self._remove_workbench_window(window)
        )
        window.show()

    def _remove_workbench_window(self, window: WorkbenchWindow) -> None:
        """Remove closed workbench windows from the tracking list."""
        try:
            self._workbench_windows.remove(window)
        except ValueError:
            return

    def _on_edit_clicked(self) -> None:
        """Open the edit dialog for the selected prompt and persist changes."""
        prompt = self._current_prompt()
        if prompt is None:
            return
        if self._prompt_editor_flow is None:
            return
        self._prompt_editor_flow.edit_prompt(prompt)

    def _on_fork_clicked(self) -> None:
        """Handle toolbar fork button taps."""
        prompt = self._current_prompt()
        if prompt is None:
            return
        self._fork_prompt(prompt)

    def _on_delete_clicked(self) -> None:
        """Remove the selected prompt after confirmation."""
        prompt = self._current_prompt()
        if prompt is None:
            return
        self._delete_prompt(prompt)

    def _open_version_history_dialog(self, prompt: Prompt | None = None) -> None:
        """Open the version history dialog for the requested or selected prompt."""
        if prompt is None:
            prompt = self._current_prompt()
        if prompt is None:
            return
        dialog = PromptVersionHistoryDialog(
            self._manager,
            prompt,
            self,
            status_callback=self._show_status_message,
        )
        dialog.exec()
        if dialog.last_restored_prompt is not None:
            self._load_prompts(self._current_search_text())
            self._select_prompt(dialog.last_restored_prompt.id)
            self.statusBar().showMessage("Prompt restored to selected version.", 4000)
        else:
            self._update_prompt_lineage_summary(prompt)

    def _delete_prompt(self, prompt: Prompt, *, skip_confirmation: bool = False) -> None:
        """Delete the provided prompt and refresh listings."""
        if not skip_confirmation:
            confirmation = QMessageBox.question(
                self,
                "Delete prompt",
                f"Are you sure you want to delete '{prompt.name}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if confirmation != QMessageBox.Yes:
                return
        try:
            self._manager.delete_prompt(prompt.id)
        except PromptNotFoundError:
            self._show_error("Prompt missing", "The prompt was already removed.")
        except PromptStorageError as exc:
            self._show_error("Unable to delete prompt", str(exc))
            return
        self._detail_widget.clear()
        self._load_prompts(self._current_search_text())

    def _on_import_clicked(self) -> None:
        """Delegate to the catalogue workflow controller."""
        self._catalog_controller.open_import_dialog()

    def _on_export_clicked(self) -> None:
        """Export prompts via the catalogue controller."""
        self._catalog_controller.export_catalog()

    def _on_maintenance_clicked(self) -> None:
        """Open maintenance workflows via the controller."""
        self._catalog_controller.open_maintenance_dialog()

    def _on_info_clicked(self) -> None:
        """Display application metadata and system characteristics."""
        dialog = InfoDialog(self)
        dialog.exec()

    def _on_settings_clicked(self) -> None:
        """Open configuration dialog and apply updates."""
        dialog = SettingsDialog(
            self,
            litellm_model=self._runtime_settings.get("litellm_model"),
            litellm_inference_model=self._runtime_settings.get("litellm_inference_model"),
            litellm_api_key=self._runtime_settings.get("litellm_api_key"),
            litellm_api_base=self._runtime_settings.get("litellm_api_base"),
            litellm_api_version=self._runtime_settings.get("litellm_api_version"),
            litellm_drop_params=self._runtime_settings.get("litellm_drop_params"),
            litellm_reasoning_effort=self._runtime_settings.get("litellm_reasoning_effort"),
            litellm_stream=self._runtime_settings.get("litellm_stream"),
            litellm_workflow_models=self._runtime_settings.get("litellm_workflow_models"),
            embedding_model=self._runtime_settings.get("embedding_model"),
            quick_actions=self._runtime_settings.get("quick_actions"),
            chat_user_bubble_color=self._runtime_settings.get("chat_user_bubble_color"),
            theme_mode=self._runtime_settings.get("theme_mode"),
            chat_colors=self._runtime_settings.get("chat_colors"),
            prompt_templates=self._runtime_settings.get("prompt_templates"),
        )
        if dialog.exec() != QDialog.Accepted:
            return
        updates = dialog.result_settings()
        self._apply_settings(updates)

    def _on_prompt_templates_clicked(self) -> None:
        """Open the lightweight prompt template editor dialog."""
        dialog = PromptTemplateEditorDialog(
            self,
            templates=self._runtime_settings.get("prompt_templates"),
        )
        if dialog.exec() != QDialog.Accepted:
            return
        overrides = dialog.result_templates()
        cleaned_overrides: dict[str, str] | None = overrides or None
        current_templates = self._runtime_settings.get("prompt_templates")
        normalised_current = current_templates or None
        if normalised_current == cleaned_overrides:
            self._show_toast("Prompt templates are already up to date.")
            return
        self._apply_settings({"prompt_templates": cleaned_overrides})
        self._show_toast("Prompt templates updated.")

    def _on_exit_clicked(self) -> None:
        """Close the application via the toolbar exit control."""
        logger.info("Exit control activated; closing Prompt Manager.")
        self.statusBar().showMessage("Closing Prompt Manager…", 2000)
        self.close()

    def _rebuild_prompt_generation_components(self) -> None:
        """Refresh prompt dialog flows after configuration or bootstrap changes."""
        components = self._prompt_generation_service.rebuild_components(self)
        self._prompt_editor_factory = components.dialog_factory
        self._prompt_editor_flow = components.editor_flow
        self._catalog_controller = components.catalog_controller


    def _apply_settings(self, updates: dict[str, object | None]) -> None:
        """Persist settings, refresh catalogue, and update dependent controllers."""
        if not updates:
            return

        try:
            result = self._runtime_settings_service.apply_updates(
                self._runtime_settings,
                updates,
            )
        except NameGenerationError as exc:
            QMessageBox.warning(self, "LiteLLM configuration", str(exc))
            result = RuntimeSettingsResult(
                theme_mode=str(self._runtime_settings.get("theme_mode") or DEFAULT_THEME_MODE),
                has_executor=self._manager.executor is not None,
            )

        if self._quick_action_controller is not None:
            self._quick_action_controller.refresh_actions(
                self._runtime_settings.get("quick_actions")
            )
        self._rebuild_prompt_generation_components()
        self._appearance_controller.apply_theme(result.theme_mode)
        if self._execution_controller is not None:
            self._execution_controller.refresh_chat_history_view()

        self._load_prompts(self._current_search_text())
        self._run_button.setEnabled(result.has_executor)
        if self._template_preview is not None:
            self._template_preview.set_run_enabled(result.has_executor)

    def _on_selection_changed(self, *_: object) -> None:
        """Update the detail panel to reflect the new selection."""
        prompt = self._current_prompt()
        if prompt is None:
            self._detail_widget.clear()
            if self._execution_controller is not None:
                self._execution_controller.handle_prompt_selection_change(None)
            self._update_template_preview(None)
            if self._template_detail_widget is not None:
                self._template_detail_widget.clear()
            return
        if self._execution_controller is not None:
            self._execution_controller.handle_prompt_selection_change(prompt.id)
        self._detail_widget.display_prompt(prompt)
        if self._template_detail_widget is not None:
            self._template_detail_widget.display_prompt(prompt)
        self._update_prompt_lineage_summary(prompt)
        self._update_template_preview(prompt)

    def _select_prompt(self, prompt_id: UUID) -> None:
        """Highlight the given prompt in the list view when present."""
        for row, prompt in enumerate(self._model.prompts()):
            if prompt.id == prompt_id:
                index = self._model.index(row, 0)
                self._list_view.setCurrentIndex(index)
                break

    def _update_prompt_lineage_summary(self, prompt: Prompt) -> None:
        """Fetch version/fork metadata for the detail pane."""
        summary_parts: list[str] = []
        try:
            parent_link = self._manager.get_prompt_parent_fork(prompt.id)
        except PromptVersionError:
            parent_link = None
        if parent_link is not None:
            summary_parts.append(f"Forked from {parent_link.source_prompt_id}")

        try:
            children = self._manager.list_prompt_forks(prompt.id)
        except PromptVersionError:
            children = []
        if children:
            child_label = "fork" if len(children) == 1 else "forks"
            summary_parts.append(f"{len(children)} {child_label}")
        summary_text = " | ".join(summary_parts)
        if summary_text:
            self._detail_widget.update_lineage_summary(summary_text)
            if self._template_detail_widget is not None:
                self._template_detail_widget.update_lineage_summary(summary_text)
        else:
            self._detail_widget.update_lineage_summary("No lineage data yet.")
            if self._template_detail_widget is not None:
                self._template_detail_widget.update_lineage_summary("No lineage data yet.")

    def _update_template_preview(self, prompt: Prompt | None) -> None:
        """Refresh the workspace template preview widget for the selected prompt."""
        if self._template_preview_controller is None:
            return
        self._template_preview_controller.update_preview(prompt)

    def _on_template_run_state_changed(self, can_run: bool) -> None:
        """Synchronise the Template tab shortcut button with preview availability."""
        if self._template_preview_controller is None:
            return
        self._template_preview_controller.handle_run_state_changed(can_run)

    def _on_template_tab_run_clicked(self) -> None:
        """Invoke the template preview execution shortcut."""
        if self._template_preview_controller is None:
            return
        self._template_preview_controller.handle_template_tab_run_clicked()

    def _on_notifications_clicked(self) -> None:
        if self._notification_controller is None:
            return
        self._notification_controller.show_task_center()

    def closeEvent(self, event) -> None:  # type: ignore[override]
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
