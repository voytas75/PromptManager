"""Main window widgets and models for the Prompt Manager GUI.

Updates:
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

import json
import logging
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
)

from PySide6.QtCore import QModelIndex, QPoint, QSettings, Qt
from PySide6.QtGui import (
    QColor,
    QGuiApplication,
    QKeySequence,
    QPalette,
    QResizeEvent,
    QShortcut,
    QShowEvent,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QLabel,
    QListView,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QWidget,
)

from config import (
    DEFAULT_CHAT_ASSISTANT_BUBBLE_COLOR,
    DEFAULT_CHAT_USER_BUBBLE_COLOR,
    DEFAULT_EMBEDDING_BACKEND,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_THEME_MODE,
    ChatColors,
    PromptManagerSettings,
    PromptTemplateOverrides,
)
from core import (
    IntentLabel,
    NameGenerationError,
    PromptExecutionUnavailable,
    PromptHistoryError,
    PromptManager,
    PromptManagerError,
    PromptNotFoundError,
    PromptStorageError,
    PromptVersionError,
    RepositoryError,
    ScenarioGenerationError,
    diff_prompt_catalog,
    export_prompt_catalog,
    import_prompt_catalog,
)
from core.notifications import Notification, NotificationStatus
from core.sharing import ShareProvider, ShareTextProvider, format_prompt_for_share
from models.category_model import PromptCategory, slugify_category

from .code_highlighter import CodeHighlighter
from .command_palette import CommandPaletteDialog, QuickAction, rank_prompts_for_action
from .controllers.execution_controller import ExecutionController
from .dialogs import (
    CatalogPreviewDialog,
    CategoryManagerDialog,
    InfoDialog,
    PromptDialog,
    PromptMaintenanceDialog,
    PromptVersionHistoryDialog,
    SaveResultDialog,
)
from .execute_context_dialog import ExecuteContextDialog
from .language_tools import DetectedLanguage, detect_language
from .layout_state import WindowStateManager
from .main_view_builder import MainViewCallbacks, MainViewComponents, build_main_view
from .notifications import BackgroundTaskCenterDialog, QtNotificationBridge
from .processing_indicator import ProcessingIndicator
from .prompt_list_model import PromptListModel
from .prompt_templates_dialog import PromptTemplateEditorDialog
from .settings_dialog import SettingsDialog, persist_settings_to_config
from .share_controller import ShareController
from .toast import show_toast
from .usage_logger import IntentUsageLogger
from .widgets import PromptDetailWidget, PromptFilterPanel, PromptToolbar
from .workbench.workbench_window import WorkbenchModeDialog, WorkbenchWindow

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Iterable, Mapping, Sequence

    from core.prompt_engineering import PromptRefinement
    from models.prompt_model import Prompt

    from .result_overlay import ResultActionsOverlay
    from .template_preview import TemplatePreviewWidget

logger = logging.getLogger(__name__)

_CHAT_PALETTE_KEYS = {"user", "assistant"}


@dataclass(slots=True)
class _PromptLoadResult:
    """Fetched prompt data assembled from repository and optional search."""

    all_prompts: list[Prompt]
    search_results: list[Prompt] | None
    preserve_search_order: bool
    search_error: str | None


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


def _default_chat_palette() -> dict[str, str]:
    return {
        "user": QColor(DEFAULT_CHAT_USER_BUBBLE_COLOR).name().lower(),
        "assistant": QColor(DEFAULT_CHAT_ASSISTANT_BUBBLE_COLOR).name().lower(),
    }



def normalise_chat_palette(palette: Mapping[str, object] | None) -> dict[str, str]:
    """Return a validated chat palette containing user/assistant colours.

    Invalid entries, unsupported roles, and malformed colour values are ignored.
    Hex codes are normalised to lowercase ``#rrggbb`` strings for consistency.
    """
    cleaned: dict[str, str] = {}
    if not palette:
        return cleaned
    for role, value in palette.items():
        if role not in _CHAT_PALETTE_KEYS:
            continue
        text = str(value).strip()
        if not text:
            continue
        candidate = QColor(text)
        if candidate.isValid():
            cleaned[role] = candidate.name().lower()
    return cleaned


def palette_differs_from_defaults(palette: Mapping[str, str] | None) -> bool:
    if not palette:
        return False
    defaults = _default_chat_palette()
    for role, default_hex in defaults.items():
        value = palette.get(role)
        if value is not None and value != default_hex:
            return True
    return False


class PromptSortOrder(Enum):
    """Supported sorting orders for the prompt list view."""

    NAME_ASC = "name_asc"
    NAME_DESC = "name_desc"
    QUALITY_DESC = "quality_desc"
    MODIFIED_DESC = "modified_desc"
    CREATED_DESC = "created_desc"
    BODY_SIZE_DESC = "body_size_desc"
    RATING_DESC = "rating_desc"
    USAGE_DESC = "usage_desc"


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
        self._model = PromptListModel(parent=self)
        self._detail_widget = PromptDetailWidget(self)
        self._detail_widget.delete_requested.connect(self._on_delete_clicked)  # type: ignore[arg-type]
        self._detail_widget.edit_requested.connect(self._on_edit_clicked)  # type: ignore[arg-type]
        self._detail_widget.version_history_requested.connect(self._open_version_history_dialog)  # type: ignore[arg-type]
        self._detail_widget.fork_requested.connect(self._on_fork_clicked)  # type: ignore[arg-type]
        self._detail_widget.refresh_scenarios_requested.connect(  # type: ignore[arg-type]
            partial(self._handle_refresh_scenarios_request, self._detail_widget)
        )
        self._detail_widget.share_requested.connect(self._on_share_prompt_requested)  # type: ignore[arg-type]
        self._all_prompts: list[Prompt] = []
        self._current_prompts: list[Prompt] = []
        self._preserve_search_order: bool = False
        self._search_active: bool = False
        self._suggestions: PromptManager.IntentSuggestions | None = None
        self._toolbar: PromptToolbar | None = None
        self._filter_panel: PromptFilterPanel | None = None
        self._sort_order = PromptSortOrder.NAME_ASC
        self._pending_category_slug: str | None = None
        self._pending_tag_value: str | None = None
        self._pending_quality_value: float | None = None
        self._history_limit = 50
        self._runtime_settings = self._initial_runtime_settings(settings)
        self._usage_logger = IntentUsageLogger()
        self._detected_language: DetectedLanguage = detect_language("")
        self._quick_actions: list[QuickAction] = self._build_quick_actions(
            self._runtime_settings.get("quick_actions")
        )
        self._active_quick_action_id: str | None = None
        self._query_seeded_by_quick_action: str | None = None
        self._execution_controller: ExecutionController | None = None
        self._render_markdown_checkbox: QCheckBox | None = None
        self._query_input: QPlainTextEdit | None = None
        self._suppress_query_signal = False
        self._quick_shortcuts: list[QShortcut] = []
        self._template_preview: TemplatePreviewWidget | None = None
        self._template_list_view: QListView | None = None
        self._template_detail_widget: PromptDetailWidget | None = None
        self._template_run_shortcut_button: QPushButton | None = None
        self._template_transition_indicator: ProcessingIndicator | None = None
        self._result_overlay: ResultActionsOverlay | None = None
        self._share_controller = ShareController(
            self,
            toast_callback=self._show_toast,
            status_callback=self.statusBar().showMessage,
            error_callback=self._show_error,
            usage_logger=self._usage_logger,
        )
        self._share_result_button: QPushButton | None = None
        self._workbench_windows: list[WorkbenchWindow] = []
        self._register_share_provider(ShareTextProvider())
        settings = QSettings("PromptManager", "MainWindow")
        self._layout_state = WindowStateManager(settings)
        filter_prefs = self._layout_state.load_filter_preferences()
        self._pending_category_slug = filter_prefs.category_slug
        self._pending_tag_value = filter_prefs.tag
        self._pending_quality_value = filter_prefs.min_quality
        stored_sort_value = filter_prefs.sort_value
        if stored_sort_value:
            try:
                self._sort_order = PromptSortOrder(stored_sort_value)
            except ValueError:
                logger.warning("Unknown stored sort order: %s", stored_sort_value)
        execute_state = self._layout_state.load_execute_context_state()
        self._last_execute_context_task = execute_state.last_task
        self._execute_context_history_limit = self._layout_state.history_limit
        self._execute_context_history = execute_state.history
        self._main_container: QFrame | None = None
        self._main_splitter: QSplitter | None = None
        self._list_splitter: QSplitter | None = None
        self._workspace_splitter: QSplitter | None = None
        self._template_preview_splitter: QSplitter | None = None
        self._template_preview_list_splitter: QSplitter | None = None
        self._main_splitter_left_width: int | None = None
        self._suppress_main_splitter_sync = False
        self._notification_history: deque[Notification] = deque(maxlen=200)
        self._active_notifications: dict[str, Notification] = {}
        for note in self._manager.notification_center.history():
            self._notification_history.append(note)
            self._update_active_notification(note)
        self._notification_indicator = QLabel("", self)
        self._notification_indicator.setObjectName("notificationIndicator")
        self._notification_indicator.setStyleSheet("color: #2f80ed; font-weight: 500;")
        self._notification_indicator.setVisible(bool(self._active_notifications))
        self._task_center_dialog: BackgroundTaskCenterDialog | None = None
        self._notification_bridge = QtNotificationBridge(self._manager.notification_center, self)
        self._notification_bridge.notification_received.connect(self._handle_notification)
        self.setWindowTitle("Prompt Manager")
        self._layout_state.restore_window_geometry(self)
        self._build_ui()
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
        self._refresh_theme_styles()
        self._apply_theme()
        self._restore_splitter_state()
        self._capture_main_splitter_left_width()
        self.statusBar().addPermanentWidget(self._notification_indicator)
        self._update_notification_indicator()
        self._register_quick_shortcuts()
        self._load_prompts()

    def _build_ui(self) -> None:
        """Create the main layout with list/search/detail panes."""
        callbacks = MainViewCallbacks(
            search_requested=self._on_search_button_clicked,
            search_text_changed=self._on_search_changed,
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
            filters_changed=self._on_filters_changed,
            sort_changed=self._on_sort_changed,
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
        self._assign_main_view_components(components)
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

    def _assign_main_view_components(self, components: MainViewComponents) -> None:
        """Store widget references built by ``build_main_view``."""
        self._main_container = components.container
        self._toolbar = components.toolbar
        self._language_label = components.language_label
        self._quick_actions_button = components.quick_actions_button
        self._quick_actions_button_default_text = (
            components.quick_actions_button_default_text
        )
        self._quick_actions_button_default_tooltip = (
            components.quick_actions_button_default_tooltip
        )
        self._quick_actions_button.setToolTip(self._quick_actions_button_default_tooltip)
        self._detect_button = components.detect_button
        self._suggest_button = components.suggest_button
        self._run_button = components.run_button
        self._clear_button = components.clear_button
        self._continue_chat_button = components.continue_chat_button
        self._end_chat_button = components.end_chat_button
        self._copy_button = components.copy_button
        self._copy_result_button = components.copy_result_button
        self._copy_result_to_text_window_button = (
            components.copy_result_to_text_window_button
        )
        self._save_button = components.save_button
        self._share_result_button = components.share_result_button
        self._intent_hint = components.intent_hint
        self._filter_panel = components.filter_panel
        if self._pending_quality_value is not None:
            self._filter_panel.set_min_quality(self._pending_quality_value)
            self._pending_quality_value = None
        self._filter_panel.set_sort_value(self._sort_order.value)
        self._query_input = components.query_input
        self._highlighter = CodeHighlighter(self._query_input.document())
        self._tab_widget = components.tab_widget
        self._main_splitter = components.main_splitter
        self._main_splitter.splitterMoved.connect(self._on_main_splitter_moved)  # type: ignore[arg-type]
        self._list_splitter = components.list_splitter
        self._list_view = components.list_view
        self._workspace_splitter = components.workspace_splitter
        self._result_label = components.result_label
        self._result_meta = components.result_meta
        self._result_tabs = components.result_tabs
        self._result_text = components.result_text
        self._result_overlay = components.result_overlay
        self._chat_history_view = components.chat_history_view
        self._render_markdown_checkbox = components.render_markdown_checkbox
        self._history_panel = components.history_panel
        self._notes_panel = components.notes_panel
        self._response_styles_panel = components.response_styles_panel
        self._analytics_panel = components.analytics_panel
        self._template_preview_splitter = components.template_preview_splitter
        self._template_preview_list_splitter = components.template_preview_list_splitter
        self._template_list_view = components.template_list_view
        self._template_detail_widget = components.template_detail_widget
        self._template_preview = components.template_preview
        self._template_run_shortcut_button = components.template_run_shortcut_button
        self._template_detail_widget.delete_requested.connect(self._on_delete_clicked)  # type: ignore[arg-type]
        self._template_detail_widget.edit_requested.connect(self._on_edit_clicked)  # type: ignore[arg-type]
        self._template_detail_widget.version_history_requested.connect(
            self._open_version_history_dialog
        )  # type: ignore[arg-type]
        self._template_detail_widget.fork_requested.connect(self._on_fork_clicked)  # type: ignore[arg-type]
        self._template_detail_widget.refresh_scenarios_requested.connect(  # type: ignore[arg-type]
            partial(self._handle_refresh_scenarios_request, self._template_detail_widget)
        )

    def _restore_splitter_state(self) -> None:
        """Restore splitter sizes from persisted settings."""
        self._layout_state.restore_splitter_sizes(self._splitter_state_entries())

    def _capture_main_splitter_left_width(self) -> None:
        """Record the current width of the left pane for resize management."""
        if self._main_splitter is None:
            return
        sizes = self._main_splitter.sizes()
        if len(sizes) < 2:
            return
        self._main_splitter_left_width = max(sizes[0], 0)

    def _enforce_main_splitter_left_width(self) -> None:
        """Ensure left pane width stays constant when the window is resized."""
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

    def _on_main_splitter_moved(self, _position: int, _index: int) -> None:
        """Update stored left pane width when the splitter is dragged."""
        if self._suppress_main_splitter_sync or self._main_splitter is None:
            return
        sizes = self._main_splitter.sizes()
        if len(sizes) < 2:
            return
        self._main_splitter_left_width = max(sizes[0], 0)

    def _save_splitter_state(self) -> None:
        """Persist splitter sizes for future sessions."""
        self._layout_state.save_splitter_sizes(self._splitter_state_entries())

    def _splitter_state_entries(self) -> list[tuple[str, QSplitter | None]]:
        """Return splitter/key mappings used for persistence."""
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

    def _persist_filter_preferences(self) -> None:
        """Persist the current category, tag, and quality filters."""
        panel = self._filter_panel
        if panel is None:
            return
        self._layout_state.persist_filter_preferences(
            category_slug=panel.category_slug(),
            tag=panel.tag_value(),
            min_quality=panel.min_quality(),
        )

    def _persist_sort_preference(self) -> None:
        """Persist the currently selected sort order."""
        self._layout_state.persist_sort_order(self._sort_order)

    def _current_search_text(self) -> str:
        """Return the current text in the toolbar search field."""
        if self._toolbar is None:
            return ""
        return self._toolbar.search_text()

    def showEvent(self, event: QShowEvent) -> None:  # type: ignore[override]
        """Ensure splitter state is captured once the window is visible."""
        super().showEvent(event)
        self._capture_main_splitter_left_width()
        self._enforce_main_splitter_left_width()

    def resizeEvent(self, event: QResizeEvent) -> None:  # type: ignore[override]
        """Keep splitter stable when the window resizes."""
        super().resizeEvent(event)
        self._enforce_main_splitter_left_width()

    def _load_prompts(self, search_text: str = "") -> None:
        """Populate the list model from the repository or semantic search."""
        try:
            result = self._fetch_prompt_load_result(search_text)
        except RepositoryError as exc:
            self._show_error("Unable to load prompts", str(exc))
            return
        self._apply_prompt_load_result(result)

    def _load_prompts_with_indicator(self, search_text: str) -> None:
        """Fetch prompts on a worker thread while displaying a busy dialog."""
        if self._toolbar is not None:
            self._toolbar.set_search_enabled(False)
        try:
            result = ProcessingIndicator(self, "Searching prompts…", title="Searching Prompts").run(
                self._fetch_prompt_load_result,
                search_text,
            )
        except RepositoryError as exc:
            self._show_error("Unable to load prompts", str(exc))
            return
        finally:
            if self._toolbar is not None:
                self._toolbar.set_search_enabled(True)
        self._apply_prompt_load_result(result)

    def _fetch_prompt_load_result(self, search_text: str) -> _PromptLoadResult:
        """Return prompt data queried from the repository and optional search."""
        stripped = search_text.strip()
        all_prompts = list(self._manager.repository.list())
        search_results: list[Prompt] | None = None
        search_error: str | None = None
        preserve_order = False

        if stripped:
            try:
                search_results = list(self._manager.search_prompts(stripped, limit=50))
            except PromptManagerError as exc:
                search_error = str(exc)
            else:
                preserve_order = True

        return _PromptLoadResult(
            all_prompts=all_prompts,
            search_results=search_results,
            preserve_search_order=preserve_order,
            search_error=search_error,
        )

    def _apply_prompt_load_result(self, result: _PromptLoadResult) -> None:
        """Update UI models based on fetched prompt data."""
        self._all_prompts = list(result.all_prompts)
        self._preserve_search_order = result.preserve_search_order
        self._populate_filters(self._all_prompts)

        if result.search_error:
            self._show_error("Unable to search prompts", result.search_error)

        if result.search_results is not None and result.preserve_search_order:
            # Directly display semantic search results; they already include
            # similarity scores and are sorted by best match.
            self._suggestions = None
            self._current_prompts = list(result.search_results)

            filtered = self._apply_filters(self._current_prompts)
            # Do **not** apply additional sorting – relevance order matters.
            self._model.set_prompts(filtered)
            self._list_view.clearSelection()
            if filtered:
                self._select_prompt(filtered[0].id)
            else:
                self._detail_widget.clear()
            self._update_intent_hint(filtered)
            return

        self._suggestions = None
        self._current_prompts = list(self._all_prompts)

        filtered = self._apply_filters(self._current_prompts)
        sorted_prompts = self._sort_prompts(filtered)
        self._model.set_prompts(sorted_prompts)
        self._list_view.clearSelection()
        if not sorted_prompts:
            self._detail_widget.clear()
        self._update_intent_hint(sorted_prompts)

    def _populate_filters(self, prompts: Sequence[Prompt]) -> None:
        """Refresh category and tag filters based on available prompts."""
        self._populate_category_filter()
        self._populate_tag_filter(prompts)

    def _populate_category_filter(self) -> None:
        """Populate the category filter from the registry."""
        panel = self._filter_panel
        if panel is None:
            return
        categories = self._manager.list_categories()
        target_category = panel.category_slug() or self._pending_category_slug
        panel.set_categories(categories, target_category)
        if target_category and panel.category_slug() == target_category:
            self._pending_category_slug = None

    def _populate_tag_filter(self, prompts: Sequence[Prompt]) -> None:
        """Populate the tag filter options."""
        panel = self._filter_panel
        if panel is None:
            return
        tags = sorted({tag for prompt in prompts for tag in prompt.tags})
        target_tag = panel.tag_value() or self._pending_tag_value
        panel.set_tags(tags, target_tag)
        if target_tag and panel.tag_value() == target_tag:
            self._pending_tag_value = None

    def _on_manage_categories_clicked(self) -> None:
        """Open the category management dialog."""
        dialog = CategoryManagerDialog(self._manager, self)
        dialog.exec()
        if dialog.has_changes:
            self._load_prompts(self._current_search_text())
        else:
            self._populate_category_filter()

    @staticmethod
    def _prompt_category_slug(prompt: Prompt) -> str | None:
        """Return the prompt's category slug, deriving it when missing."""
        if prompt.category_slug:
            return prompt.category_slug
        return slugify_category(prompt.category)

    @staticmethod
    def _prompt_body_length(prompt: Prompt) -> int:
        """Return the length of the prompt body used for embedding/search."""
        payload = prompt.context or prompt.description or ""
        return len(payload)

    @staticmethod
    def _prompt_average_rating(prompt: Prompt) -> float:
        """Return the average rating for a prompt, defaulting to zero."""
        if prompt.rating_count and prompt.rating_sum is not None:
            try:
                return float(prompt.rating_sum) / float(prompt.rating_count)
            except ZeroDivisionError:  # pragma: no cover - defensive
                return 0.0
        return 0.0

    def _apply_filters(self, prompts: Sequence[Prompt]) -> list[Prompt]:
        """Apply category, tag, and quality filters to a prompt sequence."""
        panel = self._filter_panel
        selected_category = panel.category_slug() if panel is not None else None
        selected_tag = panel.tag_value() if panel is not None else None
        min_quality = panel.min_quality() if panel is not None else 0.0

        filtered: list[Prompt] = []
        for prompt in prompts:
            if selected_category:
                prompt_slug = self._prompt_category_slug(prompt)
                if prompt_slug != selected_category:
                    continue
            if selected_tag and selected_tag not in prompt.tags:
                continue
            if min_quality > 0.0:
                quality = prompt.quality_score or 0.0
                if quality < min_quality:
                    continue
            filtered.append(prompt)
        return filtered

    def _sort_prompts(self, prompts: Sequence[Prompt]) -> list[Prompt]:
        """Return prompts sorted according to the active sort order."""
        if not prompts:
            return []

        order = self._sort_order
        if order is PromptSortOrder.NAME_ASC:
            return sorted(prompts, key=lambda prompt: (prompt.name.casefold(), str(prompt.id)))
        if order is PromptSortOrder.NAME_DESC:
            return sorted(
                prompts,
                key=lambda prompt: (prompt.name.casefold(), str(prompt.id)),
                reverse=True,
            )
        if order is PromptSortOrder.QUALITY_DESC:

            def quality_key(prompt: Prompt) -> tuple[float, str, str]:
                quality = (
                    prompt.quality_score if prompt.quality_score is not None else float("-inf")
                )
                return (-quality, prompt.name.casefold(), str(prompt.id))

            return sorted(prompts, key=quality_key)
        if order is PromptSortOrder.MODIFIED_DESC:

            def modified_key(prompt: Prompt) -> tuple[float, str, str]:
                timestamp = prompt.last_modified.timestamp() if prompt.last_modified else 0.0
                return (-timestamp, prompt.name.casefold(), str(prompt.id))

            return sorted(prompts, key=modified_key)
        if order is PromptSortOrder.CREATED_DESC:

            def created_key(prompt: Prompt) -> tuple[float, str, str]:
                timestamp = prompt.created_at.timestamp() if prompt.created_at else 0.0
                return (-timestamp, prompt.name.casefold(), str(prompt.id))

            return sorted(prompts, key=created_key)
        if order is PromptSortOrder.BODY_SIZE_DESC:

            def body_size_key(prompt: Prompt) -> tuple[int, str, str]:
                length = self._prompt_body_length(prompt)
                return (-length, prompt.name.casefold(), str(prompt.id))

            return sorted(prompts, key=body_size_key)
        if order is PromptSortOrder.RATING_DESC:

            def rating_key(prompt: Prompt) -> tuple[float, int, str, str]:
                average = self._prompt_average_rating(prompt)
                count = prompt.rating_count
                return (-average, -(count or 0), prompt.name.casefold(), str(prompt.id))

            return sorted(prompts, key=rating_key)
        if order is PromptSortOrder.USAGE_DESC:

            def usage_key(prompt: Prompt) -> tuple[int, float, str, str]:
                usage = prompt.usage_count
                modified = prompt.last_modified.timestamp() if prompt.last_modified else 0.0
                return (-(usage or 0), -modified, prompt.name.casefold(), str(prompt.id))

            return sorted(prompts, key=usage_key)
        return list(prompts)

    def _on_filters_changed(self, *_: object) -> None:
        """Refresh the prompt list when filter widgets change."""
        self._refresh_filtered_view(preserve_order=self._preserve_search_order)
        self._persist_filter_preferences()

    def _on_sort_changed(self, raw_order: str) -> None:
        """Re-sort the prompt list when the sort selection changes."""
        try:
            sort_order = PromptSortOrder(str(raw_order))
        except ValueError:
            logger.warning("Unknown sort order selection: %s", raw_order)
            return
        if sort_order is self._sort_order:
            return
        selected_prompt = self._current_prompt()
        self._sort_order = sort_order
        filtered = self._apply_filters(self._current_prompts)
        sorted_prompts = self._sort_prompts(filtered)
        self._model.set_prompts(sorted_prompts)
        if selected_prompt and any(prompt.id == selected_prompt.id for prompt in sorted_prompts):
            self._select_prompt(selected_prompt.id)
        elif sorted_prompts:
            self._select_prompt(sorted_prompts[0].id)
        else:
            self._detail_widget.clear()
        self._update_intent_hint(sorted_prompts)
        self._persist_sort_preference()

    def _refresh_filtered_view(self, *, preserve_order: bool = False) -> None:
        filtered = self._apply_filters(self._current_prompts)
        prompts_to_show = filtered if preserve_order else self._sort_prompts(filtered)
        self._model.set_prompts(prompts_to_show)
        self._list_view.clearSelection()
        if not prompts_to_show:
            self._detail_widget.clear()
        self._update_intent_hint(prompts_to_show)

    @staticmethod
    def _replace_prompt_in_collection(collection: list[Prompt], updated: Prompt) -> bool:
        """Replace a prompt in the provided collection when present."""
        for index, existing in enumerate(collection):
            if existing.id == updated.id:
                collection[index] = updated
                return True
        return False

    def _refresh_prompt_after_rating(self, prompt_id: uuid.UUID) -> None:
        """Refresh prompt collections and UI after capturing a rating."""
        try:
            updated_prompt = self._manager.get_prompt(prompt_id)
        except PromptManagerError:
            return

        search_text = self._current_search_text().strip()
        if search_text:
            self._load_prompts(search_text)
            self._select_prompt(prompt_id)
            return

        updated_any = self._replace_prompt_in_collection(self._all_prompts, updated_prompt)
        if self._replace_prompt_in_collection(self._current_prompts, updated_prompt):
            updated_any = True

        if not updated_any:
            self._load_prompts()
            self._select_prompt(prompt_id)
            return

        filtered = self._apply_filters(self._current_prompts)
        sorted_prompts = self._sort_prompts(filtered)
        self._model.set_prompts(sorted_prompts)
        if sorted_prompts:
            self._update_intent_hint(sorted_prompts)
        else:
            self._detail_widget.clear()
            self._intent_hint.clear()
            self._intent_hint.setVisible(False)

        if any(prompt.id == prompt_id for prompt in sorted_prompts):
            self._select_prompt(prompt_id)
            self._detail_widget.display_prompt(updated_prompt)

    def _handle_refresh_scenarios_request(self, detail_widget: PromptDetailWidget) -> None:
        """Regenerate persisted scenarios for the currently displayed prompt."""
        prompt = detail_widget.current_prompt()
        if prompt is None:
            return
        try:
            indicator = ProcessingIndicator(self, "Refreshing scenarios…")
            updated_prompt = indicator.run(
                self._manager.refresh_prompt_scenarios,
                prompt.id,
            )
        except (ScenarioGenerationError, PromptManagerError) as exc:
            QMessageBox.warning(self, "Scenario refresh failed", str(exc))
            return
        self._refresh_prompt_after_rating(updated_prompt.id)
        self._show_toast("Scenarios refreshed.")

    def _update_intent_hint(self, prompts: Sequence[Prompt]) -> None:
        """Update the hint label with detected intent context and matches."""
        if self._suggestions is None:
            self._intent_hint.clear()
            self._intent_hint.setVisible(False)
            return

        prompt_list = list(prompts)
        prediction = self._suggestions.prediction
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
        if self._suggestions.fallback_used:
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

    def _apply_theme(self, mode: str | None = None) -> None:
        """Apply the configured theme palette across the application."""
        app = QGuiApplication.instance()
        if app is None:
            return

        theme = (
            (mode or self._runtime_settings.get("theme_mode") or DEFAULT_THEME_MODE).strip().lower()
        )
        if theme not in {"light", "dark"}:
            theme = DEFAULT_THEME_MODE

        if theme == "dark":
            palette = QPalette()
            palette.setColor(QPalette.Window, QColor(31, 41, 51))
            palette.setColor(QPalette.WindowText, Qt.white)
            palette.setColor(QPalette.Base, QColor(24, 31, 41))
            palette.setColor(QPalette.AlternateBase, QColor(37, 46, 59))
            palette.setColor(QPalette.ToolTipBase, QColor(45, 55, 68))
            palette.setColor(QPalette.ToolTipText, Qt.white)
            palette.setColor(QPalette.Text, QColor(232, 235, 244))
            palette.setColor(QPalette.Button, QColor(45, 55, 68))
            palette.setColor(QPalette.ButtonText, Qt.white)
            palette.setColor(QPalette.BrightText, QColor(255, 107, 107))
            palette.setColor(QPalette.Link, QColor(140, 180, 255))
            palette.setColor(QPalette.Highlight, QColor(99, 102, 241))
            palette.setColor(QPalette.HighlightedText, Qt.white)
            palette.setColor(QPalette.PlaceholderText, QColor(156, 163, 175))
            palette.setColor(QPalette.Disabled, QPalette.Text, QColor(110, 115, 125))
            palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(110, 115, 125))
            app.setPalette(palette)
            app.setStyleSheet(
                "QToolTip { color: #f9fafc; background-color: #1f2933; border: 1px solid #3b4252; }"
            )
        else:
            palette = app.style().standardPalette()
            app.setPalette(palette)
            app.setStyleSheet("")
            theme = "light"

        self.setPalette(app.palette())
        self._runtime_settings["theme_mode"] = theme
        self._refresh_theme_styles()

    def _refresh_theme_styles(self) -> None:
        """Update widgets styled using palette-derived colours."""
        container = self._main_container
        if container is None:
            return

        app = QGuiApplication.instance()
        palette = app.palette() if app is not None else self.palette()
        window_color = palette.color(QPalette.Window)
        border_color = QColor(
            255 - window_color.red(),
            255 - window_color.green(),
            255 - window_color.blue(),
        )
        border_color.setAlpha(255)
        container.setStyleSheet(
            "#mainContainer { "
            f"border: 1px solid {border_color.name()}; "
            "border-radius: 6px; background-color: palette(base); }"
        )

    def _on_query_text_changed(self) -> None:
        """Update language detection and syntax highlighting as the user types."""
        text = self._query_input.toPlainText()
        self._update_detected_language(text)
        if self._suppress_query_signal:
            return
        self._query_seeded_by_quick_action = None

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
        dialog = CommandPaletteDialog(self._quick_actions, self)
        if dialog.exec() != QDialog.Accepted:
            return
        action = dialog.selected_action
        if action is not None:
            self._execute_quick_action(action)

    def _execute_quick_action(self, action: QuickAction) -> None:
        try:
            prompts = self._manager.repository.list()
        except RepositoryError as exc:
            self._show_error("Unable to load prompts", str(exc))
            return

        self._all_prompts = prompts
        ranked = rank_prompts_for_action(prompts, action)
        selected_prompt = self._resolve_quick_action_prompt(action, prompts, ranked)

        if selected_prompt is None:
            self.statusBar().showMessage(
                f"No prompts matched quick action '{action.title}'.",
                5000,
            )
            return

        self._suggestions = None
        self._populate_filters(prompts)
        self._current_prompts = list(prompts)
        filtered = self._apply_filters(self._current_prompts)
        sorted_prompts = self._sort_prompts(filtered)
        self._model.set_prompts(sorted_prompts)
        self._select_prompt(selected_prompt.id)
        self._detail_widget.display_prompt(selected_prompt)
        self._apply_quick_action_template(action)
        self._query_input.setFocus(Qt.ShortcutFocusReason)
        self._set_active_quick_action(action)
        self.statusBar().showMessage(f"Quick action applied: {action.title}", 4000)

    def _register_quick_shortcuts(self) -> None:
        for shortcut in self._quick_shortcuts:
            shortcut.setParent(None)
        self._quick_shortcuts.clear()

        palette_shortcuts = ["Ctrl+K", "Ctrl+Shift+P"]
        for seq in palette_shortcuts:
            shortcut = QShortcut(QKeySequence(seq), self)
            shortcut.activated.connect(self._show_command_palette)  # type: ignore[arg-type]
            self._quick_shortcuts.append(shortcut)

        exit_shortcut = QShortcut(QKeySequence("Ctrl+Q"), self)
        exit_shortcut.activated.connect(self._on_exit_clicked)  # type: ignore[arg-type]
        self._quick_shortcuts.append(exit_shortcut)

        for action in self._quick_actions:
            if not action.shortcut:
                continue
            shortcut = QShortcut(QKeySequence(action.shortcut), self)
            shortcut.activated.connect(lambda a=action: self._execute_quick_action(a))  # type: ignore[arg-type]
            self._quick_shortcuts.append(shortcut)

    def _set_active_quick_action(self, action: QuickAction | None) -> None:
        """Update the quick actions button label to mirror the chosen action."""
        if action is None:
            self._active_quick_action_id = None
            self._quick_actions_button.setText(self._quick_actions_button_default_text)
            self._quick_actions_button.setToolTip(self._quick_actions_button_default_tooltip)
            self._query_seeded_by_quick_action = None
            return

        self._active_quick_action_id = action.identifier
        label = action.title
        if action.shortcut:
            label = f"{label} ({action.shortcut})"
        self._quick_actions_button.setText(label)
        self._quick_actions_button.setToolTip(action.description or action.title)

    def _sync_active_quick_action_button(self) -> None:
        """Ensure the button reflects the stored active quick action identifier."""
        if not self._active_quick_action_id:
            self._set_active_quick_action(None)
            return

        for action in self._quick_actions:
            if action.identifier == self._active_quick_action_id:
                self._set_active_quick_action(action)
                return
        self._set_active_quick_action(None)

    def _apply_quick_action_template(self, action: QuickAction) -> None:
        """Populate the workspace with the action template when appropriate."""
        template = action.template
        if not template:
            return

        current_text = self._query_input.toPlainText()
        if current_text.strip() and self._query_seeded_by_quick_action is None:
            return

        self._suppress_query_signal = True
        try:
            self._query_input.setPlainText(template)
            cursor = self._query_input.textCursor()
            cursor.movePosition(QTextCursor.End)
            self._query_input.setTextCursor(cursor)
        finally:
            self._suppress_query_signal = False
        self._query_seeded_by_quick_action = action.identifier

    def _default_quick_actions(self) -> list[QuickAction]:
        return [
            QuickAction(
                identifier="explain",
                title="Explain This Code",
                description="Select analysis prompts to describe behaviour and intent.",
                category_hint="Code Analysis",
                tag_hints=("analysis", "code-review"),
                template="Explain what this code does and highlight any risks:\n",
                shortcut="Ctrl+1",
            ),
            QuickAction(
                identifier="fix-errors",
                title="Fix Errors",
                description="Surface debugging prompts to diagnose and resolve failures.",
                category_hint="Reasoning / Debugging",
                tag_hints=("debugging", "incident-response"),
                template="Identify and fix the issues in this snippet:\n",
                shortcut="Ctrl+2",
            ),
            QuickAction(
                identifier="add-comments",
                title="Add Comments",
                description=(
                    "Jump to documentation prompts that generate docstrings "
                    "and commentary."
                ),
                category_hint="Documentation",
                tag_hints=("documentation", "docstrings"),
                template="Add detailed docstrings and inline comments explaining this code:\n",
                shortcut="Ctrl+3",
            ),
            QuickAction(
                identifier="enhance",
                title="Suggest Improvements",
                description="Open enhancement prompts that brainstorm new ideas or edge cases.",
                category_hint="Enhancement",
                tag_hints=("enhancement", "product"),
                template="Suggest improvements, safeguards, and edge cases for this work:\n",
                shortcut="Ctrl+4",
            ),
        ]

    def _build_quick_actions(self, custom_actions: object | None) -> list[QuickAction]:
        actions_by_id: dict[str, QuickAction] = {
            action.identifier: action for action in self._default_quick_actions()
        }
        if not custom_actions:
            return list(actions_by_id.values())

        data: Iterable[dict[str, Any]]
        if isinstance(custom_actions, list):
            data = [entry for entry in custom_actions if isinstance(entry, dict)]
        else:
            logger.warning("Ignoring invalid quick_actions settings value: %s", custom_actions)
            return list(actions_by_id.values())

        for entry in data:
            try:
                action = QuickAction.from_mapping(entry)
            except ValueError as exc:
                logger.warning("Skipping invalid quick action definition: %s", exc)
                continue
            actions_by_id[action.identifier] = action
        return list(actions_by_id.values())

    def _resolve_quick_action_prompt(
        self,
        action: QuickAction,
        prompts: Iterable[Prompt],
        ranked: list[Prompt],
    ) -> Prompt | None:
        if action.prompt_id:
            prompt_id = action.prompt_id
            try:
                target_uuid = uuid.UUID(prompt_id)
            except ValueError:
                for prompt in prompts:
                    if prompt.name.lower() == prompt_id.lower():
                        return prompt
            else:
                for prompt in prompts:
                    if prompt.id == target_uuid:
                        return prompt
        return ranked[0] if ranked else None

    def _generate_prompt_name(self, context: str) -> str:
        """Delegate name generation to PromptManager, surfacing errors."""
        if not context.strip():
            return ""
        return self._manager.generate_prompt_name(context)

    def _generate_prompt_description(self, context: str) -> str:
        """Delegate description generation to PromptManager."""
        if not context.strip():
            return ""
        return self._manager.generate_prompt_description(context)

    def _generate_prompt_scenarios(self, context: str) -> list[str]:
        """Delegate scenario generation to PromptManager."""
        text = (context or "").strip()
        if not text:
            return []
        return list(self._manager.generate_prompt_scenarios(text))

    def _generate_prompt_category(self, context: str) -> str:
        """Suggest a category using LiteLLM when configured, else fallback heuristics."""
        text = (context or "").strip()
        if not text:
            return ""
        try:
            return self._manager.generate_prompt_category(text)
        except PromptManagerError:
            logger.debug("PromptManager category suggestion failed", exc_info=True)
            return ""

    def _generate_prompt_tags(self, context: str) -> list[str]:
        """Suggest tags using the intent classifier and language heuristics."""
        text = (context or "").strip()
        if not text:
            return []
        tags: list[str] = []
        classifier = self._manager.intent_classifier
        if classifier is not None:
            prediction = classifier.classify(text)
            tags.extend(prediction.tag_hints)
            tags.extend(prediction.language_hints)
            default_tag_map = {
                IntentLabel.ANALYSIS: "analysis",
                IntentLabel.DEBUG: "debugging",
                IntentLabel.REFACTOR: "refactor",
                IntentLabel.ENHANCEMENT: "enhancement",
                IntentLabel.DOCUMENTATION: "documentation",
                IntentLabel.REPORTING: "reporting",
                IntentLabel.GENERAL: "general",
            }
            default_tag = default_tag_map.get(prediction.label)
            if default_tag:
                tags.append(default_tag)
        detected = detect_language(text)
        if detected.code and detected.code not in {"", "plain"}:
            tags.append(detected.code)
        lowered = text.lower()
        keyword_tags = {
            "security": "security",
            "performance": "performance",
            "optimize": "optimization",
            "refactor": "refactor",
            "document": "documentation",
            "explain": "explanation",
            "bug": "bugfix",
        }
        for keyword, tag in keyword_tags.items():
            if keyword in lowered:
                tags.append(tag)
        unique: list[str] = []
        seen: set[str] = set()
        for tag in tags:
            normalized = tag.strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(normalized)
        return unique[:8]

    def _refine_prompt_body(
        self,
        prompt_text: str,
        *,
        name: str | None = None,
        description: str | None = None,
        category: str | None = None,
        tags: Sequence[str] | None = None,
    ) -> PromptRefinement:
        """Delegate prompt refinement to PromptManager."""
        return self._manager.refine_prompt_text(
            prompt_text,
            name=name,
            description=description,
            category=category,
            tags=tags,
        )

    def _refine_prompt_structure(
        self,
        prompt_text: str,
        *,
        name: str | None = None,
        description: str | None = None,
        category: str | None = None,
        tags: Sequence[str] | None = None,
    ) -> PromptRefinement:
        """Delegate structure-only prompt refinement to PromptManager."""
        return self._manager.refine_prompt_structure(
            prompt_text,
            name=name,
            description=description,
            category=category,
            tags=tags,
        )

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
        self._suggestions = PromptManager.IntentSuggestions(
            prediction=prediction,
            prompts=list(current_prompts),
            fallback_used=False,
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
        self._apply_suggestions(suggestions)
        top_name = suggestions.prompts[0].name if suggestions.prompts else None
        if top_name:
            self.statusBar().showMessage(f"Top suggestion: {top_name}", 5000)

    def _on_tab_changed(self, index: int) -> None:
        widget = self._tab_widget.widget(index)
        if widget is self._history_panel:
            self._history_panel.refresh()
            self._usage_logger.log_history_view(total=self._history_panel.row_count())
        if index == 0:
            self._hide_template_transition_indicator()

    def _on_save_result_clicked(self) -> None:
        """Persist the latest execution result with optional user notes."""
        controller = self._execution_controller
        if controller is None:
            self._show_error("Workspace unavailable", "Execution controller is not ready.")
            return
        outcome = controller.last_execution
        if outcome is None:
            self.statusBar().showMessage("Run a prompt before saving the result.", 3000)
            return
        prompt = self._current_prompt()
        if prompt is None and self._current_prompts:
            prompt = self._current_prompts[0]
        if prompt is None:
            self.statusBar().showMessage("Select a prompt to associate with the result.", 3000)
            return
        default_summary = outcome.result.response_text[:200] if outcome.result.response_text else ""
        dialog = SaveResultDialog(
            self,
            prompt_name=prompt.name,
            default_text=default_summary,
            button_text="Save",
            enable_rating=True,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        note = dialog.note
        rating_value = dialog.rating
        metadata: dict[str, Any] = {"source": "manual-save"}
        if note:
            metadata["note"] = note
        if outcome.history_entry is not None:
            metadata["source_execution_id"] = str(outcome.history_entry.id)

        usage_metadata = outcome.result.usage if outcome.result.usage else None
        try:
            saved_entry = self._manager.save_execution_result(
                prompt.id,
                outcome.result.request_text,
                outcome.result.response_text,
                duration_ms=outcome.result.duration_ms,
                usage=usage_metadata,
                metadata=metadata,
                rating=rating_value,
            )
        except PromptExecutionUnavailable as exc:
            self._save_button.setEnabled(False)
            self._show_error("History unavailable", str(exc))
            return
        except PromptHistoryError as exc:
            self._show_error("Unable to save result", str(exc))
            return

        self._usage_logger.log_save(
            prompt_name=prompt.name,
            note_length=len(note),
            rating=rating_value,
        )
        executed_at = saved_entry.executed_at.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        self.statusBar().showMessage(f"Result saved ({executed_at}).", 5000)
        if rating_value is not None:
            self._refresh_prompt_after_rating(prompt.id)
        self._history_panel.refresh()

    def _on_continue_chat_clicked(self) -> None:
        """Send a follow-up message within the active chat session."""
        controller = self._execution_controller
        if controller is None:
            self._show_error("Workspace unavailable", "Execution controller is not ready.")
            return
        controller.continue_chat()

    def _on_end_chat_clicked(self) -> None:
        """Terminate the active chat session without clearing the transcript."""
        controller = self._execution_controller
        if controller is None:
            self._show_error("Workspace unavailable", "Execution controller is not ready.")
            return
        controller.end_chat()

    def _set_workspace_text(self, text: str, *, focus: bool = False) -> None:
        """Populate the paste-text workspace with *text* and optionally focus the field."""
        if self._query_input is None:
            return
        self._query_input.setPlainText(text)
        cursor = self._query_input.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._query_input.setTextCursor(cursor)
        if focus:
            self._query_input.setFocus(Qt.ShortcutFocusReason)

    def _on_run_prompt_clicked(self) -> None:
        """Execute the selected prompt, seeding from the workspace or the prompt body."""
        prompt = self._current_prompt()
        if prompt is None:
            prompts = self._model.prompts()
            prompt = prompts[0] if prompts else None
        if prompt is None:
            self.statusBar().showMessage("Select a prompt to execute first.", 4000)
            return
        request_text = self._query_input.toPlainText()
        if not request_text.strip():
            self._execute_prompt_from_context_menu(prompt)
            return
        controller = self._execution_controller
        if controller is None:
            self._show_error("Workspace unavailable", "Execution controller is not ready.")
            return
        controller.execute_prompt_with_text(
            prompt,
            request_text,
            status_prefix="Executed",
            empty_text_message="Paste some text or code before executing a prompt.",
            keep_text_after=False,
        )

    def _on_clear_workspace_clicked(self) -> None:
        """Clear the workspace editor, output tab, and chat transcript."""
        controller = self._execution_controller
        if controller is not None:
            controller.abort_streaming()
            controller.clear_execution_result()

        self._query_input.clear()
        self._result_tabs.setCurrentIndex(0)
        self._intent_hint.clear()
        self._intent_hint.setVisible(False)
        self.statusBar().showMessage("Workspace cleared.", 3000)

    def _handle_note_update(self, execution_id: uuid.UUID, note: str) -> None:
        """Record analytics when execution notes are edited."""
        self._usage_logger.log_note_edit(note_length=len(note))

    def _handle_history_export(self, entries: int, path: str) -> None:
        """Record analytics when history is exported."""
        self._usage_logger.log_history_export(entries=entries, path=path)

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
        payload = prompt.context or prompt.description
        if not payload:
            self.statusBar().showMessage("Selected prompt does not include a body to copy.", 3000)
            return
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(payload)
        self._show_toast(f"Copied '{prompt.name}' to the clipboard.")
        self._usage_logger.log_copy(prompt_name=prompt.name, prompt_has_body=bool(prompt.context))

    def _register_share_provider(self, provider: ShareProvider) -> None:
        """Store a share provider so it can be shown in the share menu."""
        self._share_controller.register_provider(provider)
        if self._execution_controller is not None:
            self._execution_controller.notify_share_providers_changed()

    def _on_share_prompt_requested(self) -> None:
        """Display provider choices and initiate the share workflow."""
        prompt = self._detail_widget.current_prompt()
        if prompt is None:
            self.statusBar().showMessage("Select a prompt to share first.", 4000)
            return
        provider_name = self._share_controller.choose_provider(self._detail_widget.share_button())
        if not provider_name:
            return
        self._share_prompt(provider_name, prompt)

    def _on_share_result_clicked(self) -> None:
        """Display provider choices for sharing the latest output text."""
        controller = self._execution_controller
        if controller is None:
            self._show_error("Workspace unavailable", "Execution controller is not ready.")
            return
        provider_name = self._share_controller.choose_provider(self._share_result_button)
        if not provider_name:
            return
        controller.share_result_text(provider_name)

    def _share_prompt(self, provider_name: str, prompt: Prompt) -> None:
        """Share *prompt* using the requested provider and copy the share link."""
        payload = self._build_share_payload(prompt)
        if payload is None:
            return
        self._share_controller.share_payload(
            provider_name,
            payload,
            prompt=prompt,
            prompt_name=prompt.name,
            indicator_title="Sharing Prompt",
            error_title="Unable to share prompt",
        )

    def _build_share_payload(self, prompt: Prompt) -> str | None:
        """Return the formatted share payload for the selected mode."""
        mode = self._detail_widget.share_payload_mode()
        include_description = mode in {"body_description", "body_description_scenarios"}
        include_scenarios = mode == "body_description_scenarios"
        if not (prompt.context or "").strip():
            self._show_error("Unable to share prompt", "Prompt body is empty.")
            return None
        payload = format_prompt_for_share(
            prompt,
            include_description=include_description,
            include_scenarios=include_scenarios,
            include_metadata=self._detail_widget.share_include_metadata(),
        )
        if not payload.strip():
            self._show_error("Unable to share prompt", "Share payload is empty.")
            return None
        return payload

    def _show_prompt_description(self, prompt: Prompt) -> None:
        """Display the prompt description in a dialog for quick reference."""
        description = (prompt.description or "").strip()
        if not description:
            QMessageBox.information(
                self,
                "No description available",
                "The selected prompt does not have a description yet.",
            )
            return
        QMessageBox.information(
            self,
            f"{prompt.name} — Description",
            description,
        )

    def _show_similar_prompts(self, prompt: Prompt) -> None:
        """Populate the prompt list with embedding-based recommendations."""
        embedding_vector: list[float] | None
        if prompt.ext4 is not None:
            try:
                embedding_vector = [float(value) for value in prompt.ext4]
            except (TypeError, ValueError):  # pragma: no cover - defensive
                embedding_vector = None
        else:
            embedding_vector = None

        if embedding_vector is None and not prompt.document.strip():
            self.statusBar().showMessage(
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
            self._show_error("Unable to load similar prompts", str(exc))
            return

        recommendations = [candidate for candidate in similar_prompts if candidate.id != prompt.id]
        if not recommendations:
            self.statusBar().showMessage("No similar prompts found.", 4000)
            return

        self._suggestions = None
        self._current_prompts = list(recommendations)
        self._preserve_search_order = True
        filtered = self._apply_filters(self._current_prompts)
        self._model.set_prompts(filtered)
        self._list_view.clearSelection()
        if filtered:
            self._select_prompt(filtered[0].id)
        else:
            self._detail_widget.clear()
        self._update_intent_hint(filtered)
        self.statusBar().showMessage(
            f"Showing prompts similar to '{prompt.name}'.",
            4000,
        )

    def _on_render_markdown_toggled(self, _state: int) -> None:
        """Refresh result and chat panes when the markdown toggle changes."""
        if self._execution_controller is not None:
            self._execution_controller.refresh_rendering()

    def _on_copy_result_clicked(self) -> None:
        """Copy the latest execution result to the clipboard."""
        controller = self._execution_controller
        if controller is None:
            self._show_error("Workspace unavailable", "Execution controller is not ready.")
            return
        controller.copy_result_to_clipboard()

    def _on_copy_result_to_text_window_clicked(self) -> None:
        """Populate the workspace text window with the latest execution result."""
        controller = self._execution_controller
        if controller is None:
            self._show_error("Workspace unavailable", "Execution controller is not ready.")
            return
        controller.copy_result_to_workspace()

    def _apply_suggestions(self, suggestions: PromptManager.IntentSuggestions) -> None:
        """Apply intent suggestions to the list view and update filters."""
        self._suggestions = suggestions
        self._current_prompts = list(suggestions.prompts)
        filtered = self._apply_filters(self._current_prompts)
        sorted_prompts = self._sort_prompts(filtered)
        self._model.set_prompts(sorted_prompts)
        self._list_view.clearSelection()
        if sorted_prompts:
            self._select_prompt(sorted_prompts[0].id)
        else:
            self._detail_widget.clear()
        self._update_intent_hint(sorted_prompts)

    def _current_prompt(self) -> Prompt | None:
        """Return the prompt currently selected in the list."""
        index = self._list_view.currentIndex()
        if not index.isValid():
            return None
        return self._model.prompt_at(index.row())

    def _on_search_changed(self, text: str) -> None:
        """Refresh list + sorting when the inline clear action removes the query."""
        if text.strip():
            return

        if not self._search_active and (
            self._filter_panel is None or self._filter_panel.is_sort_enabled()
        ):
            return

        self._search_active = False
        if self._filter_panel is not None:
            self._filter_panel.set_sort_enabled(True)
        self._load_prompts("")

    def _on_search_button_clicked(self, text: str | None = None) -> None:
        """Run the prompt search explicitly via the toolbar search trigger."""
        query = text if text is not None else self._current_search_text()
        self._handle_search_request(query, use_indicator=True)

    def _handle_search_request(self, text: str, *, use_indicator: bool) -> None:
        """Normalize search input handling and trigger prompt loads."""
        stripped = text.strip()
        self._search_active = bool(stripped)

        # When a search query is active disable manual sorting so the user sees
        # results strictly ordered by relevance. Re-enable it once the query
        # field is cleared so manual sorting becomes available again.
        if self._filter_panel is not None:
            self._filter_panel.set_sort_enabled(not self._search_active)

        if text and len(stripped) < 2:
            return
        if use_indicator:
            self._load_prompts_with_indicator(text)
        else:
            self._load_prompts(text)

    def _on_prompt_double_clicked(self, index: QModelIndex) -> None:
        """Open the edit dialog when a prompt is double-clicked."""
        if not index.isValid():
            return
        self._list_view.setCurrentIndex(index)
        self._on_edit_clicked()

    def _on_prompt_context_menu(self, point: QPoint) -> None:
        """Show a context menu with prompt-specific actions."""
        index = self._list_view.indexAt(point)
        prompt: Prompt | None = None
        if index.isValid():
            self._list_view.setCurrentIndex(index)
            prompt = self._model.prompt_at(index.row())
        else:
            prompt = self._current_prompt()

        menu = QMenu(self)
        edit_action = menu.addAction("Edit Prompt")
        duplicate_action = menu.addAction("Duplicate Prompt")
        fork_action = menu.addAction("Fork Prompt")
        similar_action = menu.addAction("Similar Prompts")
        execute_action = menu.addAction("Execute Prompt")
        execute_context_action = menu.addAction("Execute as Context…")
        copy_action = menu.addAction("Copy Prompt Text")
        description_action = menu.addAction("Show Description")

        if prompt is None:
            edit_action.setEnabled(False)
            duplicate_action.setEnabled(False)
            fork_action.setEnabled(False)
            similar_action.setEnabled(False)
            execute_action.setEnabled(False)
            execute_context_action.setEnabled(False)
            copy_action.setEnabled(False)
            description_action.setEnabled(False)
        else:
            can_execute = bool((prompt.context or prompt.description) and self._manager.executor)
            execute_action.setEnabled(can_execute)
            has_context_body = bool((prompt.context or "").strip())
            execute_context_action.setEnabled(bool(self._manager.executor) and has_context_body)
            fork_action.setEnabled(True)
            similar_action.setEnabled(True)
            if not (prompt.context or prompt.description):
                copy_action.setEnabled(False)
            if not (prompt.description and prompt.description.strip()):
                description_action.setEnabled(False)

        selected_action = menu.exec(self._list_view.viewport().mapToGlobal(point))
        if selected_action is None:
            return
        if selected_action is edit_action:
            self._on_edit_clicked()
        elif selected_action is duplicate_action and prompt is not None:
            self._duplicate_prompt(prompt)
        elif selected_action is fork_action and prompt is not None:
            self._fork_prompt(prompt)
        elif selected_action is similar_action and prompt is not None:
            self._show_similar_prompts(prompt)
        elif selected_action is execute_action and prompt is not None:
            self._execute_prompt_from_context_menu(prompt)
        elif selected_action is execute_context_action and prompt is not None:
            self._execute_prompt_as_context(prompt)
        elif selected_action is copy_action and prompt is not None:
            self._copy_prompt_to_clipboard(prompt)
        elif selected_action is description_action and prompt is not None:
            self._show_prompt_description(prompt)

    def _execute_prompt_from_context_menu(self, prompt: Prompt) -> None:
        """Populate the workspace with the prompt body and execute immediately."""
        raw_payload = prompt.context or prompt.description or ""
        if not raw_payload.strip():
            self.statusBar().showMessage(
                "Selected prompt does not include any text to execute.",
                4000,
            )
            return
        self._query_input.setPlainText(raw_payload)
        cursor = self._query_input.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._query_input.setTextCursor(cursor)
        self._query_input.setFocus(Qt.ShortcutFocusReason)
        controller = self._execution_controller
        if controller is None:
            self._show_error("Workspace unavailable", "Execution controller is not ready.")
            return
        controller.execute_prompt_with_text(
            prompt,
            raw_payload,
            status_prefix="Executed",
            empty_text_message="Selected prompt does not include any text to execute.",
            keep_text_after=True,
        )

    def _execute_prompt_as_context(
        self,
        prompt: Prompt,
        *,
        parent: QWidget | None = None,
        context_override: str | None = None,
    ) -> None:
        """Ask for a task and run the prompt using its body as contextual input."""
        context_text = context_override if context_override is not None else prompt.context or ""
        cleaned_context = context_text.strip()
        if not cleaned_context:
            message = "Selected prompt does not include a prompt body to use as context."
            if parent is not None:
                QMessageBox.information(parent, "Execute as context", message)
            else:
                self._show_status_message(message, 5000)
            return
        parent_widget = parent or self
        self._set_workspace_text(context_text, focus=True)
        dialog = ExecuteContextDialog(
            parent=parent_widget,
            last_task=self._last_execute_context_task,
            history=tuple(self._execute_context_history),
        )
        if dialog.exec() != QDialog.Accepted:
            return
        task_text = dialog.task_text()
        cleaned_task = task_text.strip()
        if not cleaned_task:
            QMessageBox.warning(parent_widget, "Task required", "Enter a task before executing.")
            return
        self._last_execute_context_task = task_text
        self._layout_state.persist_last_execute_task(task_text)
        self._layout_state.record_execute_task(task_text, self._execute_context_history)
        request_payload = (
            "You will receive a task and a context block. "
            "Use the context exclusively when fulfilling the task.\n\n"
            f"Task:\n{cleaned_task}\n\n"
            f"Context:\n{cleaned_context}"
        )
        controller = self._execution_controller
        if controller is None:
            self._show_error("Workspace unavailable", "Execution controller is not ready.")
            return
        try:
            controller.execute_prompt_with_text(
                prompt,
                request_payload,
                status_prefix="Executed context",
                empty_text_message="Provide context text before executing.",
                keep_text_after=False,
            )
        finally:
            self._set_workspace_text(context_text, focus=False)

    def _on_template_preview_run_requested(
        self,
        rendered_text: str,
        variables: dict[str, str],
    ) -> None:
        """Execute the selected prompt using the rendered template preview text."""
        prompt = self._current_prompt()
        if prompt is None:
            self._show_status_message("Select a prompt before running from the preview.", 4000)
            return
        payload = rendered_text.strip()
        if not payload:
            self._show_status_message("Render the template before running it.", 4000)
            return
        if variables:
            self.statusBar().showMessage(
                f"Running template with {len(variables)} variable(s)…",
                2000,
            )
        if self._tab_widget.currentIndex() != 0:
            self._show_template_transition_indicator()
            self._tab_widget.setCurrentIndex(0)
        controller = self._execution_controller
        if controller is None:
            self._show_error("Workspace unavailable", "Execution controller is not ready.")
            return
        controller.execute_prompt_with_text(
            prompt,
            payload,
            status_prefix="Executed preview",
            empty_text_message="Render the template before running it.",
            keep_text_after=False,
        )

    def _duplicate_prompt(self, prompt: Prompt) -> None:
        """Open a creation dialog with the selected prompt pre-filled and persist the copy."""
        dialog = PromptDialog(
            self,
            name_generator=self._generate_prompt_name,
            description_generator=self._generate_prompt_description,
            category_provider=self._manager.list_categories,
            category_generator=self._generate_prompt_category,
            tags_generator=self._generate_prompt_tags,
            scenario_generator=self._generate_prompt_scenarios,
            prompt_engineer=(
                self._refine_prompt_body if self._manager.prompt_engineer is not None else None
            ),
            structure_refiner=(
                self._refine_prompt_structure
                if self._manager.prompt_structure_engineer is not None
                else None
            ),
            version_history_handler=self._open_version_history_dialog,
        )
        self._connect_prompt_dialog_signals(dialog)
        dialog.prefill_from_prompt(prompt)
        if dialog.exec() != QDialog.Accepted:
            return
        duplicate = dialog.result_prompt
        if duplicate is None:
            return
        try:
            created = self._manager.create_prompt(duplicate)
        except PromptStorageError as exc:
            self._show_error("Unable to duplicate prompt", str(exc))
            return
        self._load_prompts(self._current_search_text())
        self._select_prompt(created.id)
        self.statusBar().showMessage("Prompt duplicated.", 4000)

    def _fork_prompt(self, prompt: Prompt) -> None:
        """Fork the selected prompt and allow optional edits before saving."""
        try:
            forked = self._manager.fork_prompt(prompt.id)
        except PromptManagerError as exc:
            self._show_error("Unable to fork prompt", str(exc))
            return

        dialog = PromptDialog(
            self,
            forked,
            category_provider=self._manager.list_categories,
            name_generator=self._generate_prompt_name,
            description_generator=self._generate_prompt_description,
            category_generator=self._generate_prompt_category,
            tags_generator=self._generate_prompt_tags,
            scenario_generator=self._generate_prompt_scenarios,
            prompt_engineer=(
                self._refine_prompt_body if self._manager.prompt_engineer is not None else None
            ),
            structure_refiner=(
                self._refine_prompt_structure
                if self._manager.prompt_structure_engineer is not None
                else None
            ),
            version_history_handler=self._open_version_history_dialog,
        )
        self._connect_prompt_dialog_signals(dialog)
        dialog.setWindowTitle("Edit Forked Prompt")
        if dialog.exec() == QDialog.Accepted and dialog.result_prompt is not None:
            try:
                forked = self._manager.update_prompt(dialog.result_prompt)
            except PromptManagerError as exc:
                self._show_error("Unable to save forked prompt", str(exc))
                return
        self._load_prompts(self._current_search_text())
        self._select_prompt(forked.id)
        self.statusBar().showMessage("Prompt fork created.", 4000)

    def _on_refresh_clicked(self) -> None:
        """Reload catalogue data, respecting the current search text."""
        self._load_prompts(self._current_search_text())

    def _on_add_clicked(self) -> None:
        """Open the creation dialog and persist a new prompt."""
        dialog = PromptDialog(
            self,
            category_provider=self._manager.list_categories,
            name_generator=self._generate_prompt_name,
            description_generator=self._generate_prompt_description,
            category_generator=self._generate_prompt_category,
            tags_generator=self._generate_prompt_tags,
            scenario_generator=self._generate_prompt_scenarios,
            prompt_engineer=(
                self._refine_prompt_body if self._manager.prompt_engineer is not None else None
            ),
            structure_refiner=(
                self._refine_prompt_structure
                if self._manager.prompt_structure_engineer is not None
                else None
            ),
            version_history_handler=self._open_version_history_dialog,
        )
        self._connect_prompt_dialog_signals(dialog)
        if dialog.exec() != QDialog.Accepted:
            return
        prompt = dialog.result_prompt
        if prompt is None:
            return
        try:
            created = self._manager.create_prompt(prompt)
        except PromptStorageError as exc:
            self._show_error("Unable to create prompt", str(exc))
            return
        self._load_prompts()
        self._select_prompt(created.id)

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
        dialog = PromptDialog(
            self,
            prompt,
            category_provider=self._manager.list_categories,
            name_generator=self._generate_prompt_name,
            description_generator=self._generate_prompt_description,
            category_generator=self._generate_prompt_category,
            tags_generator=self._generate_prompt_tags,
            scenario_generator=self._generate_prompt_scenarios,
            prompt_engineer=(
                self._refine_prompt_body if self._manager.prompt_engineer is not None else None
            ),
            structure_refiner=(
                self._refine_prompt_structure
                if self._manager.prompt_structure_engineer is not None
                else None
            ),
            version_history_handler=self._open_version_history_dialog,
        )
        self._connect_prompt_dialog_signals(dialog)
        dialog.applied.connect(  # type: ignore[arg-type]
            lambda updated_prompt: self._on_prompt_applied(updated_prompt, dialog)
        )
        if dialog.exec() != QDialog.Accepted:
            return
        if dialog.delete_requested:
            self._delete_prompt(prompt, skip_confirmation=True)
            return
        updated = dialog.result_prompt
        if updated is None:
            return
        try:
            stored = ProcessingIndicator(self, "Saving prompt changes…").run(
                self._manager.update_prompt,
                updated,
            )
        except PromptNotFoundError:
            self._show_error(
                "Prompt missing", "The prompt cannot be located. Refresh and try again."
            )
            self._load_prompts()
            return
        except PromptStorageError as exc:
            self._show_error("Unable to update prompt", str(exc))
            return
        self._load_prompts(self._current_search_text())
        self._select_prompt(stored.id)

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

    def _on_prompt_applied(self, prompt: Prompt, dialog: PromptDialog) -> None:
        """Persist prompt edits triggered via the Apply button."""
        try:
            stored = ProcessingIndicator(dialog, "Saving prompt changes…").run(
                self._manager.update_prompt,
                prompt,
            )
        except PromptNotFoundError:
            self._show_error(
                "Prompt missing",
                "The prompt cannot be located. Refresh and try again.",
            )
            self._load_prompts()
            dialog.reject()
            return
        except PromptStorageError as exc:
            self._show_error("Unable to update prompt", str(exc))
            return

        dialog.update_source_prompt(stored)
        self._load_prompts(self._current_search_text())
        self._select_prompt(stored.id)
        self.statusBar().showMessage("Prompt changes applied.", 4000)

    def _connect_prompt_dialog_signals(self, dialog: PromptDialog) -> None:
        """Wire shared signal handlers for prompt dialogs."""
        dialog.execute_context_requested.connect(  # type: ignore[arg-type]
            lambda prompt, context_text, dlg=dialog: self._execute_prompt_as_context(
                prompt,
                parent=dlg,
                context_override=context_text,
            )
        )

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
        """Preview catalogue diff and optionally apply updates."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select catalogue file",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        catalog_path: Path | None
        if file_path:
            catalog_path = Path(file_path)
        else:
            directory = QFileDialog.getExistingDirectory(self, "Select catalogue directory", "")
            if not directory:
                return
            catalog_path = Path(directory)

        catalog_path = catalog_path.expanduser()

        try:
            preview = diff_prompt_catalog(self._manager, catalog_path)
        except Exception as exc:
            QMessageBox.warning(self, "Catalogue preview failed", str(exc))
            return

        dialog = CatalogPreviewDialog(preview, self)
        if dialog.exec() != QDialog.Accepted or not dialog.apply_requested:
            return

        try:
            result = import_prompt_catalog(self._manager, catalog_path)
        except Exception as exc:
            QMessageBox.critical(self, "Catalogue import failed", str(exc))
            return

        message = (
            f"Catalogue applied (added {result.added}, updated {result.updated}, "
            f"skipped {result.skipped}, errors {result.errors})"
        )
        if result.errors:
            QMessageBox.warning(self, "Catalogue applied with errors", message)
        else:
            self.statusBar().showMessage(message, 5000)
        self._load_prompts(self._current_search_text())

    def _on_export_clicked(self) -> None:
        """Export current prompts to JSON or YAML."""
        default_path = str(Path.home() / "prompt_catalog.json")
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export prompt catalogue",
            default_path,
            "JSON Files (*.json);;YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if not file_path:
            return
        export_path = Path(file_path)
        lower_suffix = export_path.suffix.lower()
        if selected_filter.startswith("YAML") or lower_suffix in {".yaml", ".yml"}:
            fmt = "yaml"
        else:
            fmt = "json"
        try:
            resolved = export_prompt_catalog(self._manager, export_path, fmt=fmt)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.statusBar().showMessage(f"Catalogue exported to {resolved}", 5000)

    def _on_maintenance_clicked(self) -> None:
        """Open the maintenance dialog for batch metadata helpers."""
        dialog = PromptMaintenanceDialog(
            self,
            self._manager,
            category_generator=self._generate_prompt_category,
            tags_generator=self._generate_prompt_tags,
        )
        dialog.maintenance_applied.connect(self._on_maintenance_applied)  # type: ignore[arg-type]
        dialog.exec()

    def _on_maintenance_applied(self, message: str) -> None:
        """Refresh listings after maintenance tasks run."""
        selected = self._current_prompt()
        selected_id = selected.id if selected else None
        self._load_prompts(self._current_search_text())
        if selected_id is not None:
            self._select_prompt(selected_id)
        if message:
            self.statusBar().showMessage(message, 5000)

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

    def _apply_settings(self, updates: dict[str, object | None]) -> None:
        """Persist settings, refresh catalogue, and update name generator."""
        if not updates:
            return

        runtime = self._runtime_settings

        simple_keys = (
            "litellm_model",
            "litellm_inference_model",
            "litellm_api_key",
            "litellm_api_base",
            "litellm_api_version",
            "litellm_reasoning_effort",
        )
        for key in simple_keys:
            if key in updates:
                runtime[key] = updates.get(key)

        if "embedding_backend" in updates or "embedding_model" in updates:
            backend_value = (
                updates.get("embedding_backend") or runtime.get("embedding_backend")
            ) or DEFAULT_EMBEDDING_BACKEND
            model_value = (
                updates.get("embedding_model") or runtime.get("embedding_model")
            ) or DEFAULT_EMBEDDING_MODEL
            runtime["embedding_backend"] = backend_value
            runtime["embedding_model"] = model_value
        embedding_backend_value = runtime.get("embedding_backend", DEFAULT_EMBEDDING_BACKEND)

        cleaned_drop_params: list[str] | None = runtime.get("litellm_drop_params")  # type: ignore[assignment]
        if "litellm_drop_params" in updates:
            drop_params_value = updates.get("litellm_drop_params")
            if isinstance(drop_params_value, list):
                cleaned_drop_params = [
                    str(item).strip() for item in drop_params_value if str(item).strip()
                ]
            elif isinstance(drop_params_value, str):
                cleaned_drop_params = [
                    part.strip() for part in drop_params_value.split(",") if part.strip()
                ]
            else:
                cleaned_drop_params = None
            runtime["litellm_drop_params"] = cleaned_drop_params

        stream_flag = bool(runtime.get("litellm_stream"))
        if "litellm_stream" in updates:
            stream_flag = bool(updates.get("litellm_stream"))
            runtime["litellm_stream"] = stream_flag

        def _normalise_workflows(value: object | None) -> dict[str, str] | None:
            if not isinstance(value, dict):
                return None
            return {
                str(key): "inference"
                for key, route in value.items()
                if isinstance(route, str) and route.strip().lower() == "inference"
            }

        cleaned_workflow_models = _normalise_workflows(runtime.get("litellm_workflow_models"))
        if "litellm_workflow_models" in updates:
            cleaned_workflow_models = _normalise_workflows(updates.get("litellm_workflow_models"))
            runtime["litellm_workflow_models"] = cleaned_workflow_models

        cleaned_quick_actions: list[dict[str, object]] | None = None
        existing_quick_actions = runtime.get("quick_actions")
        if isinstance(existing_quick_actions, list):
            cleaned_quick_actions = [
                dict(entry) for entry in existing_quick_actions if isinstance(entry, dict)
            ]
        if "quick_actions" in updates:
            quick_actions_value = updates.get("quick_actions")
            if isinstance(quick_actions_value, list):
                cleaned_quick_actions = [
                    dict(entry) for entry in quick_actions_value if isinstance(entry, dict)
                ]
            else:
                cleaned_quick_actions = None
            runtime["quick_actions"] = cleaned_quick_actions

        chat_colour = runtime.get("chat_user_bubble_color", DEFAULT_CHAT_USER_BUBBLE_COLOR)
        if "chat_user_bubble_color" in updates:
            color_value = updates.get("chat_user_bubble_color")
            if isinstance(color_value, str) and QColor(color_value).isValid():
                chat_colour = QColor(color_value).name().lower()
            else:
                chat_colour = DEFAULT_CHAT_USER_BUBBLE_COLOR
            runtime["chat_user_bubble_color"] = chat_colour

        cleaned_palette = normalise_chat_palette(
            runtime.get("chat_colors") if isinstance(runtime.get("chat_colors"), dict) else None
        )
        if "chat_colors" in updates:
            palette_input = (
                updates.get("chat_colors") if isinstance(updates.get("chat_colors"), dict) else None
            )
            cleaned_palette = normalise_chat_palette(palette_input)
            runtime["chat_colors"] = cleaned_palette or None

        theme_choice = runtime.get("theme_mode", DEFAULT_THEME_MODE)
        if "theme_mode" in updates:
            theme_value = updates.get("theme_mode")
            if isinstance(theme_value, str) and theme_value.strip().lower() in {"light", "dark"}:
                theme_choice = theme_value.strip().lower()
            else:
                theme_choice = DEFAULT_THEME_MODE
            runtime["theme_mode"] = theme_choice
        if theme_choice not in {"light", "dark"}:
            theme_choice = DEFAULT_THEME_MODE

        prompt_templates_payload: dict[str, str] | None = runtime.get("prompt_templates")  # type: ignore[assignment]
        if "prompt_templates" in updates:
            prompt_templates_value = updates.get("prompt_templates")
            if isinstance(prompt_templates_value, dict):
                cleaned_prompt_templates = {
                    str(key): value.strip()
                    for key, value in prompt_templates_value.items()
                    if isinstance(value, str) and value.strip()
                }
                prompt_templates_payload = cleaned_prompt_templates or None
            else:
                prompt_templates_payload = None
            runtime["prompt_templates"] = prompt_templates_payload
        persist_settings_to_config(
            {
                "litellm_model": self._runtime_settings.get("litellm_model"),
                "litellm_inference_model": self._runtime_settings.get("litellm_inference_model"),
                "litellm_api_base": self._runtime_settings.get("litellm_api_base"),
                "litellm_api_version": self._runtime_settings.get("litellm_api_version"),
                "litellm_reasoning_effort": self._runtime_settings.get("litellm_reasoning_effort"),
                "litellm_workflow_models": self._runtime_settings.get("litellm_workflow_models"),
                "quick_actions": self._runtime_settings.get("quick_actions"),
                "litellm_drop_params": self._runtime_settings.get("litellm_drop_params"),
                "litellm_stream": self._runtime_settings.get("litellm_stream"),
                "litellm_api_key": self._runtime_settings.get("litellm_api_key"),
                "embedding_backend": self._runtime_settings.get("embedding_backend"),
                "embedding_model": self._runtime_settings.get("embedding_model"),
                "chat_user_bubble_color": self._runtime_settings.get("chat_user_bubble_color"),
                "chat_colors": (
                    self._runtime_settings.get("chat_colors")
                    if palette_differs_from_defaults(self._runtime_settings.get("chat_colors"))
                    else None
                ),
                "theme_mode": self._runtime_settings.get("theme_mode"),
                "prompt_templates": self._runtime_settings.get("prompt_templates"),
            }
        )

        if self._settings is not None:
            self._settings.litellm_model = updates.get("litellm_model")
            self._settings.litellm_inference_model = updates.get("litellm_inference_model")
            self._settings.litellm_api_key = updates.get("litellm_api_key")
            self._settings.litellm_api_base = updates.get("litellm_api_base")
            self._settings.litellm_api_version = updates.get("litellm_api_version")
            self._settings.litellm_reasoning_effort = updates.get("litellm_reasoning_effort")
            self._settings.litellm_workflow_models = cleaned_workflow_models
            self._settings.quick_actions = cleaned_quick_actions
            self._settings.litellm_drop_params = cleaned_drop_params
            self._settings.litellm_stream = stream_flag
            self._settings.embedding_backend = embedding_backend_value
            self._settings.embedding_model = self._runtime_settings.get("embedding_model")
            self._settings.chat_user_bubble_color = chat_colour
            self._settings.theme_mode = theme_choice
            if cleaned_palette:
                palette_model = getattr(self._settings, "chat_colors", None)
                if isinstance(palette_model, ChatColors):
                    self._settings.chat_colors = palette_model.model_copy(update=cleaned_palette)
                else:
                    self._settings.chat_colors = ChatColors(**cleaned_palette)
            else:
                self._settings.chat_colors = ChatColors()
            if prompt_templates_payload:
                overrides_model = getattr(self._settings, "prompt_templates", None)
                if isinstance(overrides_model, PromptTemplateOverrides):
                    self._settings.prompt_templates = overrides_model.model_copy(
                        update=prompt_templates_payload
                    )
                else:
                    self._settings.prompt_templates = PromptTemplateOverrides(
                        **prompt_templates_payload
                    )
            else:
                self._settings.prompt_templates = PromptTemplateOverrides()

        self._quick_actions = self._build_quick_actions(self._runtime_settings.get("quick_actions"))
        self._register_quick_shortcuts()
        self._sync_active_quick_action_button()
        self._apply_theme(theme_choice)
        if self._execution_controller is not None:
            self._execution_controller.refresh_chat_history_view()

        try:
            self._manager.set_name_generator(
                self._runtime_settings.get("litellm_model"),
                self._runtime_settings.get("litellm_api_key"),
                self._runtime_settings.get("litellm_api_base"),
                self._runtime_settings.get("litellm_api_version"),
                inference_model=self._runtime_settings.get("litellm_inference_model"),
                workflow_models=self._runtime_settings.get("litellm_workflow_models"),
                drop_params=self._runtime_settings.get("litellm_drop_params"),
                reasoning_effort=self._runtime_settings.get("litellm_reasoning_effort"),
                stream=self._runtime_settings.get("litellm_stream"),
                prompt_templates=self._runtime_settings.get("prompt_templates"),
            )
        except NameGenerationError as exc:
            QMessageBox.warning(self, "LiteLLM configuration", str(exc))

        self._load_prompts(self._current_search_text())
        has_executor = self._manager.executor is not None
        self._run_button.setEnabled(has_executor)
        if self._template_preview is not None:
            self._template_preview.set_run_enabled(has_executor)

    def _initial_runtime_settings(
        self, settings: PromptManagerSettings | None
    ) -> dict[str, object | None]:
        """Load current settings snapshot from configuration."""
        derived_quick_actions: list[dict[str, object]] | None
        if settings and settings.quick_actions:
            derived_quick_actions = [dict(entry) for entry in settings.quick_actions]
        else:
            derived_quick_actions = None

        runtime = {
            "litellm_model": settings.litellm_model if settings else None,
            "litellm_inference_model": settings.litellm_inference_model if settings else None,
            "litellm_api_key": settings.litellm_api_key if settings else None,
            "litellm_api_base": settings.litellm_api_base if settings else None,
            "litellm_api_version": settings.litellm_api_version if settings else None,
            "litellm_drop_params": list(settings.litellm_drop_params)
            if settings and settings.litellm_drop_params
            else None,
            "litellm_reasoning_effort": settings.litellm_reasoning_effort if settings else None,
            "litellm_stream": settings.litellm_stream if settings is not None else None,
            "litellm_workflow_models": dict(settings.litellm_workflow_models)
            if settings and settings.litellm_workflow_models
            else None,
            "embedding_backend": settings.embedding_backend
            if settings
            else DEFAULT_EMBEDDING_BACKEND,
            "embedding_model": settings.embedding_model if settings else DEFAULT_EMBEDDING_MODEL,
            "quick_actions": derived_quick_actions,
            "chat_user_bubble_color": (
                settings.chat_user_bubble_color if settings else DEFAULT_CHAT_USER_BUBBLE_COLOR
            ),
            "theme_mode": settings.theme_mode if settings else DEFAULT_THEME_MODE,
            "chat_colors": (
                {
                    "user": settings.chat_colors.user,
                    "assistant": settings.chat_colors.assistant,
                }
                if settings
                else None
            ),
            "prompt_templates": (
                settings.prompt_templates.model_dump(exclude_none=True)
                if settings and settings.prompt_templates
                else None
            ),
        }

        config_path = Path("config/config.json")
        if config_path.exists():
            try:
                data = json.loads(config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
            else:
                for key in (
                    "litellm_model",
                    "litellm_inference_model",
                    "litellm_api_key",
                    "litellm_api_base",
                    "litellm_api_version",
                    "litellm_reasoning_effort",
                    "embedding_backend",
                    "embedding_model",
                ):
                    if runtime.get(key) is None and isinstance(data.get(key), str):
                        runtime[key] = data[key]
                if runtime.get("litellm_drop_params") is None:
                    drop_value = data.get("litellm_drop_params")
                    if isinstance(drop_value, list):
                        runtime["litellm_drop_params"] = [
                            str(item).strip() for item in drop_value if str(item).strip()
                        ]
                    elif isinstance(drop_value, str):
                        parsed = [part.strip() for part in drop_value.split(",") if part.strip()]
                        runtime["litellm_drop_params"] = parsed or None
                if runtime.get("litellm_stream") is None:
                    stream_value = data.get("litellm_stream")
                    if isinstance(stream_value, bool):
                        runtime["litellm_stream"] = stream_value
                    elif isinstance(stream_value, str):
                        lowered = stream_value.strip().lower()
                        if lowered in {"true", "1", "yes", "on"}:
                            runtime["litellm_stream"] = True
                        elif lowered in {"false", "0", "no", "off"}:
                            runtime["litellm_stream"] = False
                if runtime.get("litellm_workflow_models") is None:
                    routing_value = data.get("litellm_workflow_models")
                    if isinstance(routing_value, dict):
                        runtime["litellm_workflow_models"] = {
                            str(key): "inference"
                            for key, value in routing_value.items()
                            if isinstance(value, str) and value.strip().lower() == "inference"
                        }
                if runtime["quick_actions"] is None and isinstance(data.get("quick_actions"), list):
                    runtime["quick_actions"] = [
                        dict(entry) for entry in data["quick_actions"] if isinstance(entry, dict)
                    ]
                color_value = data.get("chat_user_bubble_color")
                if isinstance(color_value, str) and color_value.strip():
                    runtime["chat_user_bubble_color"] = color_value.strip()
                palette_value = data.get("chat_colors")
                palette = normalise_chat_palette(
                    palette_value if isinstance(palette_value, dict) else None
                )
                if palette:
                    runtime["chat_colors"] = palette
                theme_value = data.get("theme_mode")
                if isinstance(theme_value, str) and theme_value.strip():
                    runtime["theme_mode"] = theme_value.strip()
        raw_colour = runtime.get("chat_user_bubble_color")
        if isinstance(raw_colour, str):
            candidate_colour = QColor(raw_colour)
            runtime["chat_user_bubble_color"] = (
                candidate_colour.name().lower()
                if candidate_colour.isValid()
                else DEFAULT_CHAT_USER_BUBBLE_COLOR
            )
        else:
            runtime["chat_user_bubble_color"] = DEFAULT_CHAT_USER_BUBBLE_COLOR
        theme_value = runtime.get("theme_mode")
        if isinstance(theme_value, str):
            theme_choice = theme_value.strip().lower()
            runtime["theme_mode"] = (
                theme_choice if theme_choice in {"light", "dark"} else DEFAULT_THEME_MODE
            )
        else:
            runtime["theme_mode"] = DEFAULT_THEME_MODE
        return runtime

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

    def _select_prompt(self, prompt_id: uuid.UUID) -> None:
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
        if self._template_preview is None:
            return
        if prompt is None:
            self._template_preview.clear_template()
            return
        template_text = prompt.context or prompt.description or ""
        self._template_preview.set_template(template_text, str(prompt.id))

    def _on_template_run_state_changed(self, can_run: bool) -> None:
        """Synchronise the Template tab shortcut button with preview availability."""
        if self._template_run_shortcut_button is not None:
            self._template_run_shortcut_button.setEnabled(can_run)

    def _on_template_tab_run_clicked(self) -> None:
        """Invoke the template preview execution shortcut."""
        if self._template_preview is None:
            return
        if not self._template_preview.request_run():
            self._show_status_message("Render the template before running it.", 4000)

    def _show_template_transition_indicator(self) -> None:
        """Display a busy dialog while switching from Template to Prompts."""
        if self._template_transition_indicator is not None:
            return
        indicator = ProcessingIndicator(self, "Opening Prompts tab…", title="Switching Tabs")
        indicator.__enter__()
        self._template_transition_indicator = indicator

    def _hide_template_transition_indicator(self) -> None:
        """Dismiss the template transition indicator if it is visible."""
        indicator = self._template_transition_indicator
        if indicator is None:
            return
        self._template_transition_indicator = None
        indicator.__exit__(None, None, None)

    def _handle_notification(self, notification: Notification) -> None:
        """React to notification updates published by the core manager."""
        self._register_notification(notification, show_status=True)

    def _register_notification(self, notification: Notification, *, show_status: bool) -> None:
        self._notification_history.append(notification)
        self._update_active_notification(notification)
        self._update_notification_indicator()
        if self._task_center_dialog is not None:
            self._task_center_dialog.handle_notification(notification)
        if show_status:
            message = self._format_notification_message(notification)
            duration = 0 if notification.status is NotificationStatus.STARTED else 5000
            self.statusBar().showMessage(message, duration)
        if notification.task_id and notification.status in {
            NotificationStatus.SUCCEEDED,
            NotificationStatus.FAILED,
        }:
            toast_duration = 3500 if notification.status is NotificationStatus.SUCCEEDED else 4500
            self._show_toast(self._format_notification_message(notification), toast_duration)

    def _update_active_notification(self, notification: Notification) -> None:
        task_id = notification.task_id
        if not task_id:
            return
        if notification.status is NotificationStatus.STARTED:
            self._active_notifications[task_id] = notification
        elif notification.status in {NotificationStatus.SUCCEEDED, NotificationStatus.FAILED}:
            self._active_notifications.pop(task_id, None)

    def _update_notification_indicator(self) -> None:
        active = len(self._active_notifications)
        if active:
            self._notification_indicator.setText(f"Tasks: {active}")
            self._notification_indicator.setVisible(True)
        else:
            self._notification_indicator.clear()
            self._notification_indicator.setVisible(False)

    @staticmethod
    def _format_notification_message(notification: Notification) -> str:
        status = notification.status.value.replace("_", " ").title()
        return f"{notification.title}: {status} — {notification.message}"

    def _on_notifications_clicked(self) -> None:
        dialog = self._ensure_task_center_dialog()
        dialog.set_history(tuple(self._notification_history))
        dialog.set_active_notifications(tuple(self._active_notifications.values()))
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _ensure_task_center_dialog(self) -> BackgroundTaskCenterDialog:
        if self._task_center_dialog is None:
            self._task_center_dialog = BackgroundTaskCenterDialog(self)
        return self._task_center_dialog

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """Persist layout before closing and dispose transient dialogs."""
        self._layout_state.save_window_geometry(self)
        self._save_splitter_state()
        if hasattr(self, "_notification_bridge"):
            self._notification_bridge.close()
        if self._task_center_dialog is not None:
            self._task_center_dialog.close()
        self._hide_template_transition_indicator()
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
