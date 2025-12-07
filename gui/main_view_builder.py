r"""Composable builders for the Prompt Manager main window.

Updates:
  v0.2.9 - 2025-12-07 - Embed the Workbench tab alongside Chain for inline sessions.
  v0.2.8 - 2025-12-07 - Convert Run Prompt button into split actions for prompt/text runs.
  v0.2.7 - 2025-12-07 - Provide provider-agnostic default tooltip for web search toggle.
  v0.2.6 - 2025-12-05 - Pass prompt edit callbacks into analytics and chain panels.
  v0.2.5 - 2025-12-05 - Mark module docstring as raw for lint compliance.
  v0.2.4 - 2025-12-05 - Remove prompt template toolbar wiring.
  v0.2.3 - 2025-12-05 - Drop prompt chain toolbar wiring; Chain tab is always visible.
  v0.2.2 - 2025-12-05 - Embed prompt chain manager panel as a Chain tab.
  v0.2.1 - 2025-12-04 - Add \"Use web search\" toggle to the workspace actions row.
  v0.2.0-and-earlier - 2025-11-30 - Extract main window UI assembly into reusable builder helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QPoint, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QListView,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStyle,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .analytics_panel import AnalyticsDashboardPanel
from .dialogs.prompt_chains import PromptChainManagerPanel
from .history_panel import HistoryPanel
from .notes_panel import NotesPanel
from .response_styles_panel import ResponseStylesPanel
from .result_overlay import ResultActionsOverlay
from .template_preview import TemplatePreviewWidget
from .widgets import FlowLayout, PromptDetailWidget, PromptFilterPanel, PromptToolbar
from .workbench import WorkbenchMode, WorkbenchWindow

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Callable, Sequence
    from uuid import UUID

    from core import PromptManager

    from .prompt_list_model import PromptListModel


@dataclass(slots=True)
class MainViewCallbacks:
    """Signal targets required to wire the main view widgets."""

    search_requested: Callable[[str | None], None]
    search_text_changed: Callable[[str], None]
    refresh_requested: Callable[[], None]
    add_requested: Callable[[], None]
    workbench_requested: Callable[[], None]
    import_requested: Callable[[], None]
    export_requested: Callable[[], None]
    maintenance_requested: Callable[[], None]
    notifications_requested: Callable[[], None]
    info_requested: Callable[[], None]
    settings_requested: Callable[[], None]
    exit_requested: Callable[[], None]
    show_command_palette: Callable[[], None]
    detect_intent_clicked: Callable[[], None]
    suggest_prompt_clicked: Callable[[], None]
    run_prompt_clicked: Callable[[], None]
    run_text_only_clicked: Callable[[], None]
    clear_workspace_clicked: Callable[[], None]
    continue_chat_clicked: Callable[[], None]
    end_chat_clicked: Callable[[], None]
    copy_prompt_clicked: Callable[[], None]
    copy_result_clicked: Callable[[], None]
    copy_result_to_text_window_clicked: Callable[[], None]
    save_result_clicked: Callable[[], None]
    share_result_clicked: Callable[[], None]
    speak_result_clicked: Callable[[], None]
    edit_prompt_by_id: Callable[[UUID], None]
    filters_changed: Callable[..., None]
    sort_changed: Callable[[str], None]
    manage_categories_clicked: Callable[[], None]
    query_text_changed: Callable[[], None]
    tab_changed: Callable[[int], None]
    selection_changed: Callable[..., None]
    prompt_double_clicked: Callable[..., None]
    prompt_context_menu: Callable[[QPoint], None]
    render_markdown_toggled: Callable[[int], None]
    template_preview_run_requested: Callable[[str, dict[str, str]], None]
    template_preview_run_state_changed: Callable[[bool], None]
    template_tab_run_clicked: Callable[[], None]


@dataclass(slots=True)
class MainViewComponents:
    """Public widgets built for the main window."""

    container: QFrame
    toolbar: PromptToolbar
    language_label: QLabel
    quick_actions_button: QPushButton
    quick_actions_button_default_text: str
    quick_actions_button_default_tooltip: str
    detect_button: QPushButton
    suggest_button: QPushButton
    run_button: QToolButton
    clear_button: QPushButton
    continue_chat_button: QPushButton
    end_chat_button: QPushButton
    copy_button: QPushButton
    web_search_checkbox: QCheckBox
    copy_result_button: QPushButton
    copy_result_to_text_window_button: QPushButton
    save_button: QPushButton
    share_result_button: QPushButton
    speak_result_button: QToolButton
    intent_hint: QLabel
    filter_panel: PromptFilterPanel
    query_input: QPlainTextEdit
    tab_widget: QTabWidget
    main_splitter: QSplitter
    list_splitter: QSplitter
    list_view: QListView
    workspace_splitter: QSplitter
    result_label: QLabel
    result_meta: QLabel
    result_tabs: QTabWidget
    result_text: QTextEdit
    result_overlay: ResultActionsOverlay
    chat_history_view: QTextEdit
    render_markdown_checkbox: QCheckBox
    history_panel: HistoryPanel
    notes_panel: NotesPanel
    response_styles_panel: ResponseStylesPanel
    analytics_panel: AnalyticsDashboardPanel
    chain_panel: PromptChainManagerPanel
    workbench_panel: WorkbenchWindow
    template_preview_splitter: QSplitter
    template_preview_list_splitter: QSplitter
    template_list_view: QListView
    template_detail_widget: PromptDetailWidget
    template_preview: TemplatePreviewWidget
    template_run_shortcut_button: QPushButton


def build_main_view(
    *,
    parent: QWidget,
    model: PromptListModel,
    detail_widget: PromptDetailWidget,
    manager: PromptManager,
    history_limit: int,
    sort_options: Sequence[tuple[str, str]],
    callbacks: MainViewCallbacks,
    status_callback: Callable[[str, int], None] | None,
    toast_callback: Callable[[str, int], None] | None,
    event_filter_target: QObject,
    usage_log_path: str | None,
    history_note_callback: Callable[[UUID, str], None] | None,
    history_export_callback: Callable[[int, str], None] | None,
) -> MainViewComponents:
    """Build the main Prompt Manager workspace and return widget references."""
    container = QFrame(parent)
    container.setObjectName("mainContainer")
    layout = QVBoxLayout(container)
    layout.setContentsMargins(12, 12, 12, 12)

    toolbar = PromptToolbar(parent)
    toolbar.search_requested.connect(callbacks.search_requested)  # type: ignore[arg-type]
    toolbar.search_text_changed.connect(callbacks.search_text_changed)
    toolbar.refresh_requested.connect(callbacks.refresh_requested)
    toolbar.add_requested.connect(callbacks.add_requested)
    toolbar.workbench_requested.connect(callbacks.workbench_requested)
    toolbar.import_requested.connect(callbacks.import_requested)
    toolbar.export_requested.connect(callbacks.export_requested)
    toolbar.maintenance_requested.connect(callbacks.maintenance_requested)
    toolbar.notifications_requested.connect(callbacks.notifications_requested)
    toolbar.info_requested.connect(callbacks.info_requested)
    toolbar.settings_requested.connect(callbacks.settings_requested)
    toolbar.exit_requested.connect(callbacks.exit_requested)
    layout.addWidget(toolbar)

    language_layout = QHBoxLayout()
    language_layout.setContentsMargins(0, 0, 0, 0)
    language_label = QLabel(parent)
    language_label.setObjectName("detectedLanguageLabel")
    language_layout.addWidget(language_label)
    language_layout.addStretch(1)

    actions_layout = FlowLayout(spacing=8)

    quick_actions_button = QPushButton("Quick Actions", parent)
    quick_actions_button.clicked.connect(callbacks.show_command_palette)  # type: ignore[arg-type]
    quick_actions_button_default_text = quick_actions_button.text()
    quick_actions_button_default_tooltip = "Open the command palette for quick actions."
    quick_actions_button.setToolTip(quick_actions_button_default_tooltip)
    actions_layout.addWidget(quick_actions_button)

    detect_button = QPushButton("Detect Need", parent)
    detect_button.clicked.connect(callbacks.detect_intent_clicked)  # type: ignore[arg-type]
    actions_layout.addWidget(detect_button)

    suggest_button = QPushButton("Suggest Prompt", parent)
    suggest_button.clicked.connect(callbacks.suggest_prompt_clicked)  # type: ignore[arg-type]
    actions_layout.addWidget(suggest_button)

    run_button = QToolButton(parent)
    run_button.setText("Run Prompt")
    run_button.setPopupMode(QToolButton.MenuButtonPopup)
    run_button.clicked.connect(callbacks.run_prompt_clicked)  # type: ignore[arg-type]
    run_menu = QMenu(run_button)
    run_selected_action = run_menu.addAction("Run selected prompt")
    run_selected_action.triggered.connect(callbacks.run_prompt_clicked)  # type: ignore[arg-type]
    run_text_only_action = run_menu.addAction("Run provided text only")
    run_text_only_action.triggered.connect(callbacks.run_text_only_clicked)  # type: ignore[arg-type]
    run_button.setMenu(run_menu)
    actions_layout.addWidget(run_button)

    clear_button = QPushButton("Clear", parent)
    clear_button.setToolTip("Clear the workspace input, output, and chat panes.")
    clear_button.clicked.connect(callbacks.clear_workspace_clicked)  # type: ignore[arg-type]
    actions_layout.addWidget(clear_button)

    continue_chat_button = QPushButton("Continue Chat", parent)
    continue_chat_button.clicked.connect(callbacks.continue_chat_clicked)  # type: ignore[arg-type]
    actions_layout.addWidget(continue_chat_button)

    end_chat_button = QPushButton("End Chat", parent)
    end_chat_button.clicked.connect(callbacks.end_chat_clicked)  # type: ignore[arg-type]
    actions_layout.addWidget(end_chat_button)

    copy_button = QPushButton("Copy Prompt", parent)
    copy_button.clicked.connect(callbacks.copy_prompt_clicked)  # type: ignore[arg-type]
    actions_layout.addWidget(copy_button)

    web_search_checkbox = QCheckBox("Use web search", parent)
    web_search_checkbox.setChecked(True)
    web_search_checkbox.setToolTip("Include live web search context before executing prompts.")
    actions_layout.addWidget(web_search_checkbox)

    copy_result_button = QPushButton("Copy Result", parent)
    copy_result_button.clicked.connect(callbacks.copy_result_clicked)  # type: ignore[arg-type]

    copy_result_to_text_window_button = QPushButton("Copy to Text Window", parent)
    copy_result_to_text_window_button.clicked.connect(callbacks.copy_result_to_text_window_clicked)  # type: ignore[arg-type]

    save_button = QPushButton("Save Result", parent)
    save_button.clicked.connect(callbacks.save_result_clicked)  # type: ignore[arg-type]

    share_result_button = QPushButton("Share Result", parent)
    share_result_button.clicked.connect(callbacks.share_result_clicked)  # type: ignore[arg-type]

    speak_result_button = QToolButton(parent)
    speak_result_button.setToolTip("Read the latest result aloud.")
    speak_result_button.setIcon(parent.style().standardIcon(QStyle.SP_MediaPlay))
    speak_result_button.setCheckable(True)
    speak_result_button.setEnabled(False)
    speak_result_button.clicked.connect(callbacks.speak_result_clicked)  # type: ignore[arg-type]

    intent_hint = QLabel("", parent)
    intent_hint.setObjectName("intentHintLabel")
    intent_hint.setStyleSheet("color: #5b5b5b; font-style: italic;")
    intent_hint.setWordWrap(True)
    intent_hint.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
    intent_hint.setVisible(False)

    filter_panel = PromptFilterPanel(
        sort_options=list(sort_options),
        parent=parent,
    )
    filter_panel.filters_changed.connect(callbacks.filters_changed)
    filter_panel.sort_changed.connect(callbacks.sort_changed)
    filter_panel.manage_categories_requested.connect(callbacks.manage_categories_clicked)
    layout.addWidget(filter_panel)

    query_input = QPlainTextEdit(parent)
    query_input.setPlaceholderText("Paste code or text to analyse and suggest prompts…")
    query_input.setMinimumHeight(120)
    query_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    query_input.textChanged.connect(callbacks.query_text_changed)

    tab_widget = QTabWidget(parent)
    tab_widget.currentChanged.connect(callbacks.tab_changed)  # type: ignore[arg-type]

    result_tab = QWidget(parent)
    result_tab_layout = QVBoxLayout(result_tab)
    result_tab_layout.setContentsMargins(0, 0, 0, 0)

    main_splitter = QSplitter(Qt.Horizontal, result_tab)

    list_splitter = QSplitter(Qt.Vertical, main_splitter)

    list_view = QListView(list_splitter)
    list_view.setModel(model)
    list_view.setSelectionMode(QAbstractItemView.SingleSelection)
    list_view.selectionModel().selectionChanged.connect(callbacks.selection_changed)
    list_view.doubleClicked.connect(callbacks.prompt_double_clicked)  # type: ignore[arg-type]
    list_view.setContextMenuPolicy(Qt.CustomContextMenu)
    list_view.customContextMenuRequested.connect(callbacks.prompt_context_menu)
    list_splitter.addWidget(list_view)

    list_splitter.addWidget(detail_widget)
    list_splitter.setStretchFactor(0, 5)
    list_splitter.setStretchFactor(1, 2)
    main_splitter.addWidget(list_splitter)

    workspace_panel = QWidget(main_splitter)
    workspace_layout = QVBoxLayout(workspace_panel)
    workspace_layout.setContentsMargins(0, 0, 0, 0)
    workspace_layout.setSpacing(0)

    workspace_splitter = QSplitter(Qt.Vertical, workspace_panel)
    workspace_splitter.setChildrenCollapsible(False)
    workspace_layout.addWidget(workspace_splitter, 1)

    query_panel = QWidget(workspace_splitter)
    query_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    query_panel_layout = QVBoxLayout(query_panel)
    query_panel_layout.setSizeConstraint(QLayout.SetMinimumSize)
    query_panel_layout.setContentsMargins(0, 0, 0, 0)
    query_panel_layout.setSpacing(8)
    query_panel_layout.addWidget(query_input, 1)
    query_panel_layout.addLayout(actions_layout)
    query_panel_layout.addLayout(language_layout)
    query_panel_layout.addWidget(intent_hint)

    output_panel = QWidget(workspace_splitter)
    output_layout = QVBoxLayout(output_panel)
    output_layout.setContentsMargins(0, 0, 0, 0)
    output_layout.setSpacing(8)

    result_label = QLabel("No prompt executed yet", parent)
    result_label.setObjectName("resultTitle")
    result_meta = QLabel("", parent)
    result_meta.setStyleSheet("color: #5b5b5b; font-style: italic;")

    result_tabs = QTabWidget(parent)
    output_tab = QWidget(parent)
    output_tab_layout = QVBoxLayout(output_tab)
    output_tab_layout.setContentsMargins(0, 0, 0, 0)
    output_tab_layout.setSpacing(8)

    result_text = QTextEdit(parent)
    result_text.setReadOnly(True)
    result_text.setAcceptRichText(True)
    result_text.setPlaceholderText("Run a prompt to see output here.")
    result_text.viewport().installEventFilter(event_filter_target)
    result_text.installEventFilter(event_filter_target)
    output_tab_layout.addWidget(result_text, 1)

    result_overlay = ResultActionsOverlay(result_text)
    result_overlay.rebuild(
        [
            save_button,
            copy_result_button,
            copy_result_to_text_window_button,
            share_result_button,
            speak_result_button,
        ]
    )

    result_tabs.addTab(output_tab, "Output")

    chat_history_view = QTextEdit(parent)
    chat_history_view.setReadOnly(True)
    chat_history_view.setAcceptRichText(True)
    chat_history_view.setPlaceholderText("Run a prompt to start chatting.")
    result_tabs.addTab(chat_history_view, "Chat")

    output_layout.addWidget(result_label)
    output_layout.addWidget(result_meta)
    output_layout.addWidget(result_tabs, 1)

    render_actions_layout = QHBoxLayout()
    render_actions_layout.setContentsMargins(0, 0, 0, 0)
    render_actions_layout.setSpacing(8)
    render_markdown_checkbox = QCheckBox("Render Markdown", parent)
    render_markdown_checkbox.setChecked(True)
    render_markdown_checkbox.stateChanged.connect(callbacks.render_markdown_toggled)  # type: ignore[arg-type]
    render_actions_layout.addWidget(render_markdown_checkbox)
    render_actions_layout.addStretch(1)
    output_layout.addLayout(render_actions_layout)

    workspace_splitter.addWidget(output_panel)
    workspace_splitter.addWidget(query_panel)
    workspace_splitter.setStretchFactor(0, 5)
    workspace_splitter.setStretchFactor(1, 3)
    main_splitter.addWidget(workspace_panel)
    main_splitter.setStretchFactor(0, 0)
    main_splitter.setStretchFactor(1, 1)

    result_tab_layout.addWidget(main_splitter)

    history_panel = HistoryPanel(
        manager,
        parent,
        limit=history_limit,
        on_note_updated=history_note_callback,
        on_export=history_export_callback,
    )

    notes_panel = NotesPanel(
        manager,
        parent,
        status_callback=status_callback,
        toast_callback=toast_callback,
    )

    response_styles_panel = ResponseStylesPanel(
        manager,
        parent,
        status_callback=status_callback,
        toast_callback=toast_callback,
    )

    analytics_panel = AnalyticsDashboardPanel(
        manager,
        parent,
        usage_log_path=usage_log_path,
        prompt_edit_callback=callbacks.edit_prompt_by_id,
    )

    tab_widget.addTab(result_tab, "Prompts")
    tab_widget.addTab(history_panel, "History")
    tab_widget.addTab(notes_panel, "Notes")
    tab_widget.addTab(response_styles_panel, "Prompt Parts")
    tab_widget.addTab(analytics_panel, "Analytics")

    preview_tab = QWidget(parent)
    preview_layout = QVBoxLayout(preview_tab)
    preview_layout.setContentsMargins(12, 12, 12, 12)
    preview_layout.setSpacing(8)
    template_preview_splitter = QSplitter(Qt.Horizontal, preview_tab)
    template_preview_splitter.setChildrenCollapsible(False)

    preview_list_panel = QWidget(template_preview_splitter)
    preview_list_layout = QVBoxLayout(preview_list_panel)
    preview_list_layout.setContentsMargins(0, 0, 0, 0)
    preview_list_layout.setSpacing(6)
    preview_hint = QLabel(
        (
            "Select a prompt to inspect variables and render it as a Jinja2 template. "
            "Selections stay in sync with the main Prompts tab."
        ),
        preview_list_panel,
    )
    preview_hint.setWordWrap(True)
    preview_hint.setStyleSheet("color: #4b5563;")
    preview_list_layout.addWidget(preview_hint)
    template_preview_list_splitter = QSplitter(Qt.Vertical, preview_list_panel)
    template_preview_list_splitter.setChildrenCollapsible(False)
    template_list_view = QListView(template_preview_list_splitter)
    template_list_view.setModel(model)
    template_list_view.setSelectionMode(QAbstractItemView.SingleSelection)
    template_list_view.setSelectionModel(list_view.selectionModel())
    template_list_view.doubleClicked.connect(callbacks.prompt_double_clicked)  # type: ignore[arg-type]
    template_preview_list_splitter.addWidget(template_list_view)

    template_detail_widget = PromptDetailWidget(template_preview_list_splitter)
    template_preview_list_splitter.addWidget(template_detail_widget)
    template_preview_list_splitter.setStretchFactor(0, 1)
    template_preview_list_splitter.setStretchFactor(1, 2)
    preview_list_layout.addWidget(template_preview_list_splitter, 1)

    preview_render_panel = QWidget(template_preview_splitter)
    preview_render_layout = QVBoxLayout(preview_render_panel)
    preview_render_layout.setContentsMargins(0, 0, 0, 0)
    preview_render_layout.setSpacing(8)
    template_preview = TemplatePreviewWidget(preview_render_panel)
    template_preview.clear_template()
    template_preview.run_requested.connect(callbacks.template_preview_run_requested)
    template_preview.run_state_changed.connect(callbacks.template_preview_run_state_changed)
    preview_render_layout.addWidget(template_preview)

    template_preview_splitter.addWidget(preview_list_panel)
    template_preview_splitter.addWidget(preview_render_panel)
    template_preview_splitter.setStretchFactor(0, 1)
    template_preview_splitter.setStretchFactor(1, 2)
    preview_layout.addWidget(template_preview_splitter, 1)

    template_actions = QHBoxLayout()
    template_actions.setContentsMargins(0, 0, 0, 0)
    template_actions.setSpacing(8)
    template_actions.addStretch(1)
    template_run_shortcut_button = QPushButton("Run Rendered Template", preview_tab)
    template_run_shortcut_button.setEnabled(False)
    template_run_shortcut_button.setToolTip(
        "Execute the rendered template using the selected prompt."
    )
    template_run_shortcut_button.clicked.connect(callbacks.template_tab_run_clicked)  # type: ignore[arg-type]
    template_actions.addWidget(template_run_shortcut_button)
    preview_layout.addLayout(template_actions)
    tab_widget.addTab(preview_tab, "Template")

    chain_panel = PromptChainManagerPanel(
        manager,
        parent,
        prompt_edit_callback=callbacks.edit_prompt_by_id,
    )
    tab_widget.addTab(chain_panel, "Chain")

    workbench_panel = WorkbenchWindow(
        manager,
        mode=WorkbenchMode.BLANK,
        parent=parent,
    )
    tab_widget.addTab(workbench_panel, "Workbench")

    layout.addWidget(tab_widget, stretch=1)

    return MainViewComponents(
        container=container,
        toolbar=toolbar,
        language_label=language_label,
        quick_actions_button=quick_actions_button,
        quick_actions_button_default_text=quick_actions_button_default_text,
        quick_actions_button_default_tooltip=quick_actions_button_default_tooltip,
        detect_button=detect_button,
        suggest_button=suggest_button,
        run_button=run_button,
        clear_button=clear_button,
        continue_chat_button=continue_chat_button,
        end_chat_button=end_chat_button,
        copy_button=copy_button,
        web_search_checkbox=web_search_checkbox,
        copy_result_button=copy_result_button,
        copy_result_to_text_window_button=copy_result_to_text_window_button,
        save_button=save_button,
        share_result_button=share_result_button,
        speak_result_button=speak_result_button,
        intent_hint=intent_hint,
        filter_panel=filter_panel,
        query_input=query_input,
        tab_widget=tab_widget,
        main_splitter=main_splitter,
        list_splitter=list_splitter,
        list_view=list_view,
        workspace_splitter=workspace_splitter,
        result_label=result_label,
        result_meta=result_meta,
        result_tabs=result_tabs,
        result_text=result_text,
        result_overlay=result_overlay,
        chat_history_view=chat_history_view,
        render_markdown_checkbox=render_markdown_checkbox,
        history_panel=history_panel,
        notes_panel=notes_panel,
        response_styles_panel=response_styles_panel,
        analytics_panel=analytics_panel,
        chain_panel=chain_panel,
        workbench_panel=workbench_panel,
        template_preview_splitter=template_preview_splitter,
        template_preview_list_splitter=template_preview_list_splitter,
        template_list_view=template_list_view,
        template_detail_widget=template_detail_widget,
        template_preview=template_preview,
        template_run_shortcut_button=template_run_shortcut_button,
    )
