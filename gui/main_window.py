"""Main window widgets and models for the Prompt Manager GUI.

Updates: v0.14.6 - 2025-11-16 - Stack workspace vertically so result tabs consume remaining height.
Updates: v0.14.5 - 2025-11-02 - Persist main window geometry across sessions.
Updates: v0.14.4 - 2025-11-02 - Enable double-click editing for prompts in the list view.
Updates: v0.14.3 - 2025-11-02 - Stabilise query workspace layout by fixing editor height and wrapping detection hints.
Updates: v0.14.2 - 2025-11-02 - Persist splitter sizes so panel layout is restored between sessions.
Updates: v0.14.1 - 2025-11-02 - Allow action toolbar buttons to wrap on resize for responsive layouts.
Updates: v0.14.0 - 2025-11-02 - Rework layout to align query editor with output and expand prompt browsing area.
Updates: v0.13.1 - 2025-11-16 - Add palette-aware border styling to the primary window frame.
Updates: v0.13.0 - 2025-11-16 - Add rendered markdown preview for prompt execution output.
Updates: v0.12.0 - 2025-11-15 - Wire prompt engineering refinement into prompt editor dialogs.
Updates: v0.11.0 - 2025-11-12 - Add multi-turn chat controls and conversation history display.
Updates: v0.10.0 - 2025-11-11 - Surface notification centre with task status indicator and history dialog.
Updates: v0.9.0 - 2025-11-10 - Introduce command palette quick actions with keyboard shortcuts.
Updates: v0.8.0 - 2025-11-10 - Add language detection and syntax highlighting to the query workspace.
Updates: v0.7.0 - 2025-11-10 - Add diff preview tab alongside generated output for prompt executions.
Updates: v0.6.0 - 2025-11-09 - Add execution rating workflow and display aggregated prompt scores.
Updates: v0.5.1 - 2025-11-09 - Wrap prompt details in a scroll area to avoid Wayland resize crashes.
Updates: v0.5.0 - 2025-11-08 - Add prompt execution workflow with result pane.
Updates: v0.4.0 - 2025-11-07 - Add intent workspace with detect/suggest/copy actions and usage logging.
Updates: v0.3.0 - 2025-11-06 - Surface intent-aware search hints and recommendations.
Updates: v0.2.0 - 2025-11-05 - Add catalogue filters, LiteLLM name generation, and settings UI.
Updates: v0.1.0 - 2025-11-04 - Provide list/search/detail panes with CRUD controls.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence

from PySide6.QtCore import (
    QAbstractListModel,
    QByteArray,
    QModelIndex,
    QPoint,
    QRect,
    QSize,
    Qt,
    QSettings,
)
from PySide6.QtGui import QColor, QGuiApplication, QKeySequence, QPalette, QTextCursor, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QLayout,
    QHBoxLayout,
    QLayoutItem,
    QLabel,
    QLineEdit,
    QListView,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QScrollArea,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QWidgetItem,
    QSizePolicy,
)

from config import PromptManagerSettings
from core import (
    IntentLabel,
    NameGenerationError,
    PromptManager,
    PromptManagerError,
    PromptExecutionError,
    PromptExecutionUnavailable,
    PromptHistoryError,
    PromptNotFoundError,
    PromptStorageError,
    RepositoryError,
    diff_prompt_catalog,
    export_prompt_catalog,
    import_prompt_catalog,
)
from core.prompt_engineering import PromptRefinement
from core.notifications import Notification, NotificationStatus
from models.prompt_model import Prompt

from .command_palette import CommandPaletteDialog, QuickAction, rank_prompts_for_action
from .code_highlighter import CodeHighlighter
from .dialogs import CatalogPreviewDialog, MarkdownPreviewDialog, PromptDialog, SaveResultDialog
from .history_panel import HistoryPanel
from .settings_dialog import SettingsDialog, persist_settings_to_config
from .diff_utils import build_diff_preview
from .language_tools import DetectedLanguage, detect_language
from .usage_logger import IntentUsageLogger
from .notifications import QtNotificationBridge, NotificationHistoryDialog


logger = logging.getLogger(__name__)


class PromptListModel(QAbstractListModel):
    """List model providing prompt summaries for the QListView."""

    def __init__(self, prompts: Optional[Sequence[Prompt]] = None, parent=None) -> None:
        super().__init__(parent)
        self._prompts: List[Prompt] = list(prompts or [])

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802 - Qt API
        if parent.isValid():
            return 0
        return len(self._prompts)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Optional[str]:  # noqa: N802
        if not index.isValid() or index.row() >= len(self._prompts):
            return None
        prompt = self._prompts[index.row()]
        if role in {Qt.DisplayRole, Qt.EditRole}:
            category = f" ({prompt.category})" if prompt.category else ""
            return f"{prompt.name}{category}"
        return None

    def prompt_at(self, row: int) -> Optional[Prompt]:
        """Return the prompt at the given list index."""

        if 0 <= row < len(self._prompts):
            return self._prompts[row]
        return None

    def set_prompts(self, prompts: Iterable[Prompt]) -> None:
        """Replace the backing prompt list and notify listeners."""

        self.beginResetModel()
        self._prompts = list(prompts)
        self.endResetModel()

    def prompts(self) -> Sequence[Prompt]:
        """Expose the underlying prompts for selection helpers."""

        return tuple(self._prompts)


class PromptDetailWidget(QWidget):
    """Panel that summarises the currently selected prompt."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._scroll_area = QScrollArea(self)
        self._scroll_area.setWidgetResizable(True)
        layout.addWidget(self._scroll_area)

        content = QWidget(self._scroll_area)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(12, 12, 12, 12)

        self._title = QLabel("Select a prompt to view details", content)
        self._title.setObjectName("promptTitle")
        self._rating_label = QLabel("Rating: n/a", content)
        self._description = QLabel("", content)
        self._description.setWordWrap(True)

        self._context = QLabel("", content)
        self._context.setWordWrap(True)
        self._examples = QLabel("", content)
        self._examples.setWordWrap(True)

        content_layout.addWidget(self._title)
        content_layout.addSpacing(8)
        content_layout.addWidget(self._rating_label)
        content_layout.addSpacing(4)
        content_layout.addWidget(self._description)
        content_layout.addSpacing(8)
        content_layout.addWidget(self._context)
        content_layout.addSpacing(8)
        content_layout.addWidget(self._examples)
        content_layout.addStretch(1)

        self._scroll_area.setWidget(content)

    def display_prompt(self, prompt: Prompt) -> None:
        """Populate labels using the provided prompt."""

        tags = ", ".join(prompt.tags) if prompt.tags else "No tags"
        language = prompt.language or "en"
        header = f"{prompt.name} — {prompt.category or 'Uncategorised'}\nLanguage: {language}\nTags: {tags}"
        self._title.setText(header)
        if prompt.rating_count > 0 and prompt.quality_score is not None:
            rating_text = f"Rating: {prompt.quality_score:.1f}/10 ({prompt.rating_count} ratings)"
        else:
            rating_text = "Rating: not yet rated"
        self._rating_label.setText(rating_text)
        self._description.setText(prompt.description)
        context_text = prompt.context or "No prompt text provided."
        self._context.setText(f"Prompt Body:\n{context_text}")
        example_lines = []
        if prompt.example_input:
            example_lines.append(f"Example input:\n{prompt.example_input}")
        if prompt.example_output:
            example_lines.append(f"Example output:\n{prompt.example_output}")
        self._examples.setText("\n\n".join(example_lines) or "")

    def clear(self) -> None:
        """Reset the panel to its empty state."""

        self._title.setText("Select a prompt to view details")
        self._rating_label.setText("Rating: n/a")
        self._description.clear()
        self._context.clear()
        self._examples.clear()


class FlowLayout(QLayout):
    """Layout that arranges widgets left-to-right and wraps on overflow."""

    def __init__(self, parent: Optional[QWidget] = None, *, margin: int = 0, spacing: int = -1) -> None:
        super().__init__(parent)
        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)
        self._item_list: List[QLayoutItem] = []
        default_spacing = spacing if spacing >= 0 else self.spacing()
        self.setSpacing(default_spacing if default_spacing >= 0 else 0)

    def addItem(self, item: QLayoutItem) -> None:
        self._item_list.append(item)

    def addWidget(self, widget: QWidget) -> None:
        self.addChildWidget(widget)
        self.addItem(QWidgetItem(widget))

    def count(self) -> int:
        return len(self._item_list)

    def itemAt(self, index: int) -> Optional[QLayoutItem]:
        if 0 <= index < len(self._item_list):
            return self._item_list[index]
        return None

    def takeAt(self, index: int) -> Optional[QLayoutItem]:
        if 0 <= index < len(self._item_list):
            return self._item_list.pop(index)
        return None

    def expandingDirections(self) -> Qt.Orientations:
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        size = QSize()
        for item in self._item_list:
            size = size.expandedTo(item.minimumSize())
        left, top, right, bottom = self.getContentsMargins()
        size += QSize(left + right, top + bottom)
        return size

    def _do_layout(self, rect: QRect, *, test_only: bool) -> int:
        left, top, right, bottom = self.getContentsMargins()
        effective_rect = rect.adjusted(left, top, -right, -bottom)
        x = effective_rect.x()
        y = effective_rect.y()
        line_height = 0
        space_x = self.spacing()
        space_y = self.spacing()

        for item in self._item_list:
            widget = item.widget()
            if widget is None or not widget.isVisible():
                continue
            next_x = x + item.sizeHint().width() + space_x
            if next_x - space_x > effective_rect.right() and line_height > 0:
                x = effective_rect.x()
                y = y + line_height + space_y
                next_x = x + item.sizeHint().width() + space_x
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))
            x = next_x
            line_height = max(line_height, item.sizeHint().height())

        return y + line_height - rect.y() + bottom


class MainWindow(QMainWindow):
    """Primary window exposing prompt CRUD operations."""

    def __init__(
        self,
        prompt_manager: PromptManager,
        settings: Optional[PromptManagerSettings] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._manager = prompt_manager
        self._settings = settings
        self._model = PromptListModel(parent=self)
        self._detail_widget = PromptDetailWidget(self)
        self._all_prompts: List[Prompt] = []
        self._current_prompts: List[Prompt] = []
        self._suggestions: Optional[PromptManager.IntentSuggestions] = None
        self._last_execution: Optional[PromptManager.ExecutionOutcome] = None
        self._last_prompt_name: Optional[str] = None
        self._chat_conversation: List[Dict[str, str]] = []
        self._chat_prompt_id: Optional[uuid.UUID] = None
        self._history_limit = 50
        self._runtime_settings = self._initial_runtime_settings(settings)
        self._usage_logger = IntentUsageLogger()
        self._detected_language: DetectedLanguage = detect_language("")
        self._quick_actions: List[QuickAction] = self._build_quick_actions(
            self._runtime_settings.get("quick_actions")
        )
        self._quick_shortcuts: List[QShortcut] = []
        self._layout_settings = QSettings("PromptManager", "MainWindow")
        self._main_splitter: Optional[QSplitter] = None
        self._list_splitter: Optional[QSplitter] = None
        self._workspace_splitter: Optional[QSplitter] = None
        self._notification_history: Deque[Notification] = deque(maxlen=200)
        self._active_notifications: Dict[str, Notification] = {}
        for note in self._manager.notification_center.history():
            self._notification_history.append(note)
            self._update_active_notification(note)
        self._notification_indicator = QLabel("", self)
        self._notification_indicator.setObjectName("notificationIndicator")
        self._notification_indicator.setStyleSheet("color: #2f80ed; font-weight: 500;")
        self._notification_indicator.setVisible(bool(self._active_notifications))
        self._notification_bridge = QtNotificationBridge(self._manager.notification_center, self)
        self._notification_bridge.notification_received.connect(self._handle_notification)
        self.setWindowTitle("Prompt Manager")
        self._restore_window_geometry()
        self._build_ui()
        self._restore_splitter_state()
        self.statusBar().addPermanentWidget(self._notification_indicator)
        self._update_notification_indicator()
        self._register_quick_shortcuts()
        self._load_prompts()

    def _build_ui(self) -> None:
        """Create the main layout with list/search/detail panes."""

        palette = self.palette()
        window_color = palette.color(QPalette.Window)
        border_color = QColor(
            255 - window_color.red(),
            255 - window_color.green(),
            255 - window_color.blue(),
        )
        border_color.setAlpha(255)

        container = QFrame(self)
        container.setObjectName("mainContainer")
        container.setStyleSheet(
            "#mainContainer { "
            f"border: 1px solid {border_color.name()}; "
            "border-radius: 6px; background-color: palette(base); }"
        )
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)

        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(8)

        self._search_input = QLineEdit(self)
        self._search_input.setPlaceholderText("Search prompts…")
        self._search_input.textChanged.connect(self._on_search_changed)  # type: ignore[arg-type]
        controls_layout.addWidget(self._search_input)

        self._refresh_button = QPushButton("Refresh", self)
        self._refresh_button.clicked.connect(self._on_refresh_clicked)  # type: ignore[arg-type]
        controls_layout.addWidget(self._refresh_button)

        self._add_button = QPushButton("Add", self)
        self._add_button.clicked.connect(self._on_add_clicked)  # type: ignore[arg-type]
        controls_layout.addWidget(self._add_button)

        self._edit_button = QPushButton("Edit", self)
        self._edit_button.clicked.connect(self._on_edit_clicked)  # type: ignore[arg-type]
        controls_layout.addWidget(self._edit_button)

        self._delete_button = QPushButton("Delete", self)
        self._delete_button.clicked.connect(self._on_delete_clicked)  # type: ignore[arg-type]
        controls_layout.addWidget(self._delete_button)

        self._import_button = QPushButton("Import", self)
        self._import_button.clicked.connect(self._on_import_clicked)  # type: ignore[arg-type]
        controls_layout.addWidget(self._import_button)

        self._export_button = QPushButton("Export", self)
        self._export_button.clicked.connect(self._on_export_clicked)  # type: ignore[arg-type]
        controls_layout.addWidget(self._export_button)

        self._history_button = QPushButton("History", self)
        self._history_button.clicked.connect(self._on_history_clicked)  # type: ignore[arg-type]
        controls_layout.addWidget(self._history_button)

        self._notifications_button = QPushButton("Notifications", self)
        self._notifications_button.clicked.connect(self._on_notifications_clicked)  # type: ignore[arg-type]
        controls_layout.addWidget(self._notifications_button)

        self._settings_button = QPushButton("Settings", self)
        self._settings_button.clicked.connect(self._on_settings_clicked)  # type: ignore[arg-type]
        controls_layout.addWidget(self._settings_button)

        layout.addLayout(controls_layout)

        language_layout = QHBoxLayout()
        language_layout.setContentsMargins(0, 0, 0, 0)
        self._language_label = QLabel(self)
        self._language_label.setObjectName("detectedLanguageLabel")
        language_layout.addWidget(self._language_label)
        language_layout.addStretch(1)

        actions_layout = FlowLayout(spacing=8)

        self._quick_actions_button = QPushButton("Quick Actions", self)
        self._quick_actions_button.clicked.connect(self._show_command_palette)  # type: ignore[arg-type]
        actions_layout.addWidget(self._quick_actions_button)

        self._detect_button = QPushButton("Detect Need", self)
        self._detect_button.clicked.connect(self._on_detect_intent_clicked)  # type: ignore[arg-type]
        actions_layout.addWidget(self._detect_button)

        self._suggest_button = QPushButton("Suggest Prompt", self)
        self._suggest_button.clicked.connect(self._on_suggest_prompt_clicked)  # type: ignore[arg-type]
        actions_layout.addWidget(self._suggest_button)

        self._run_button = QPushButton("Run Prompt", self)
        self._run_button.clicked.connect(self._on_run_prompt_clicked)  # type: ignore[arg-type]
        actions_layout.addWidget(self._run_button)

        self._continue_chat_button = QPushButton("Continue Chat", self)
        self._continue_chat_button.clicked.connect(self._on_continue_chat_clicked)  # type: ignore[arg-type]
        actions_layout.addWidget(self._continue_chat_button)

        self._end_chat_button = QPushButton("End Chat", self)
        self._end_chat_button.clicked.connect(self._on_end_chat_clicked)  # type: ignore[arg-type]
        actions_layout.addWidget(self._end_chat_button)

        self._save_button = QPushButton("Save Result", self)
        self._save_button.clicked.connect(self._on_save_result_clicked)  # type: ignore[arg-type]
        actions_layout.addWidget(self._save_button)

        self._copy_button = QPushButton("Copy Prompt", self)
        self._copy_button.clicked.connect(self._on_copy_prompt_clicked)  # type: ignore[arg-type]
        actions_layout.addWidget(self._copy_button)

        self._copy_result_button = QPushButton("Copy Result", self)
        self._copy_result_button.clicked.connect(self._on_copy_result_clicked)  # type: ignore[arg-type]
        actions_layout.addWidget(self._copy_result_button)

        self._render_markdown_button = QPushButton("Render Output", self)
        self._render_markdown_button.setEnabled(False)
        self._render_markdown_button.clicked.connect(self._on_render_markdown_clicked)  # type: ignore[arg-type]
        actions_layout.addWidget(self._render_markdown_button)

        self._intent_hint = QLabel("", self)
        self._intent_hint.setObjectName("intentHintLabel")
        self._intent_hint.setStyleSheet("color: #5b5b5b; font-style: italic;")
        self._intent_hint.setWordWrap(True)
        self._intent_hint.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self._intent_hint.setVisible(False)

        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(8)

        self._category_filter = QComboBox(self)
        self._category_filter.addItem("All categories", None)
        self._category_filter.currentIndexChanged.connect(self._on_filters_changed)

        self._tag_filter = QComboBox(self)
        self._tag_filter.addItem("All tags", None)
        self._tag_filter.currentIndexChanged.connect(self._on_filters_changed)

        self._quality_filter = QDoubleSpinBox(self)
        self._quality_filter.setRange(0.0, 10.0)
        self._quality_filter.setDecimals(1)
        self._quality_filter.setSingleStep(0.1)
        self._quality_filter.valueChanged.connect(self._on_filters_changed)

        filter_layout.addWidget(QLabel("Category:", self))
        filter_layout.addWidget(self._category_filter)
        filter_layout.addWidget(QLabel("Tag:", self))
        filter_layout.addWidget(self._tag_filter)
        filter_layout.addWidget(QLabel("Quality ≥", self))
        filter_layout.addWidget(self._quality_filter)
        filter_layout.addStretch(1)

        layout.addLayout(filter_layout)

        self._query_input = QPlainTextEdit(self)
        self._query_input.setPlaceholderText("Paste code or text to analyse and suggest prompts…")
        fixed_query_height = 160
        self._query_input.setMinimumHeight(fixed_query_height)
        self._query_input.setMaximumHeight(fixed_query_height)
        self._query_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._query_input.textChanged.connect(self._on_query_text_changed)
        self._highlighter = CodeHighlighter(self._query_input.document())

        self._tab_widget = QTabWidget(self)
        self._tab_widget.currentChanged.connect(self._on_tab_changed)  # type: ignore[arg-type]

        result_tab = QWidget(self)
        result_tab_layout = QVBoxLayout(result_tab)
        result_tab_layout.setContentsMargins(0, 0, 0, 0)

        self._main_splitter = QSplitter(Qt.Horizontal, result_tab)

        self._list_splitter = QSplitter(Qt.Vertical, self._main_splitter)

        self._list_view = QListView(self._list_splitter)
        self._list_view.setModel(self._model)
        self._list_view.setSelectionMode(QAbstractItemView.SingleSelection)
        self._list_view.selectionModel().selectionChanged.connect(self._on_selection_changed)  # type: ignore[arg-type]
        self._list_view.doubleClicked.connect(self._on_prompt_double_clicked)  # type: ignore[arg-type]
        self._list_splitter.addWidget(self._list_view)

        self._list_splitter.addWidget(self._detail_widget)
        self._list_splitter.setStretchFactor(0, 5)
        self._list_splitter.setStretchFactor(1, 2)
        self._main_splitter.addWidget(self._list_splitter)

        workspace_panel = QWidget(self._main_splitter)
        workspace_layout = QVBoxLayout(workspace_panel)
        workspace_layout.setContentsMargins(0, 0, 0, 0)

        self._workspace_splitter = None

        query_panel = QWidget(workspace_panel)
        query_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        query_panel_layout = QVBoxLayout(query_panel)
        query_panel_layout.setSizeConstraint(QLayout.SetMinimumSize)
        query_panel_layout.setContentsMargins(0, 0, 0, 0)
        query_panel_layout.setSpacing(8)
        query_panel_layout.addWidget(self._query_input)

        query_panel_layout.addLayout(actions_layout)
        query_panel_layout.addLayout(language_layout)

        query_panel_layout.addWidget(self._intent_hint)
        workspace_layout.addWidget(query_panel)

        output_panel = QWidget(workspace_panel)
        output_layout = QVBoxLayout(output_panel)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(8)

        self._result_label = QLabel("No prompt executed yet", self)
        self._result_label.setObjectName("resultTitle")
        self._result_meta = QLabel("", self)
        self._result_meta.setStyleSheet("color: #5b5b5b; font-style: italic;")

        self._result_tabs = QTabWidget(self)
        self._result_text = QPlainTextEdit(self)
        self._result_text.setReadOnly(True)
        self._result_text.setPlaceholderText("Run a prompt to see output here.")
        self._result_tabs.addTab(self._result_text, "Output")

        self._diff_view = QPlainTextEdit(self)
        self._diff_view.setReadOnly(True)
        self._diff_view.setPlaceholderText("Run a prompt to compare input and output.")
        self._result_tabs.addTab(self._diff_view, "Diff")

        self._chat_history_view = QPlainTextEdit(self)
        self._chat_history_view.setReadOnly(True)
        self._chat_history_view.setPlaceholderText("Run a prompt to start chatting.")
        self._result_tabs.addTab(self._chat_history_view, "Chat")

        output_layout.addWidget(self._result_label)
        output_layout.addWidget(self._result_meta)
        output_layout.addWidget(self._result_tabs, 1)
        workspace_layout.addWidget(output_panel, 1)
        self._main_splitter.addWidget(workspace_panel)
        self._main_splitter.setStretchFactor(0, 3)
        self._main_splitter.setStretchFactor(1, 2)

        result_tab_layout.addWidget(self._main_splitter)

        self._history_panel = HistoryPanel(
            self._manager,
            self,
            limit=self._history_limit,
            on_note_updated=self._handle_note_update,
            on_export=self._handle_history_export,
        )

        self._tab_widget.addTab(result_tab, "Result")
        self._tab_widget.addTab(self._history_panel, "History")

        layout.addWidget(self._tab_widget, stretch=1)

        self._update_detected_language(self._query_input.toPlainText(), force=True)

        self.setCentralWidget(container)

        has_executor = self._manager.executor is not None
        self._run_button.setEnabled(has_executor)
        self._copy_result_button.setEnabled(False)
        self._save_button.setEnabled(False)
        self._continue_chat_button.setEnabled(False)
        self._end_chat_button.setEnabled(False)
        self._history_button.setEnabled(True)

    def _restore_window_geometry(self) -> None:
        """Restore window size/position from persistent settings."""

        stored = self._layout_settings.value("windowGeometry")
        if isinstance(stored, QByteArray):
            if not self.restoreGeometry(stored):
                self.resize(1024, 640)
            return
        if isinstance(stored, (bytes, bytearray)):
            if not self.restoreGeometry(QByteArray(stored)):
                self.resize(1024, 640)
            return
        if isinstance(stored, str):
            try:
                width_str, height_str, *_ = stored.split(",")
                width = int(width_str)
                height = int(height_str)
            except (ValueError, TypeError):
                self.resize(1024, 640)
            else:
                if width > 0 and height > 0:
                    self.resize(width, height)
                else:
                    self.resize(1024, 640)
            return
        self.resize(1024, 640)

    def _restore_splitter_state(self) -> None:
        """Restore splitter sizes from persisted settings."""

        entries = (
            ("mainSplitter", self._main_splitter),
            ("listSplitter", self._list_splitter),
            ("workspaceSplitter", self._workspace_splitter),
        )
        for key, splitter in entries:
            if splitter is None:
                continue
            stored = self._layout_settings.value(key)
            if stored is None:
                continue
            if isinstance(stored, str):
                parts = [segment for segment in stored.split(",") if segment]
            else:
                parts = list(stored) if isinstance(stored, (list, tuple)) else []
            try:
                sizes = [int(part) for part in parts]
            except (TypeError, ValueError):
                continue
            if len(sizes) != splitter.count() or not sizes or sum(sizes) <= 0:
                continue
            splitter.setSizes(sizes)

    def _save_splitter_state(self) -> None:
        """Persist splitter sizes for future sessions."""

        entries = (
            ("mainSplitter", self._main_splitter),
            ("listSplitter", self._list_splitter),
            ("workspaceSplitter", self._workspace_splitter),
        )
        for key, splitter in entries:
            if splitter is None:
                continue
            self._layout_settings.setValue(key, splitter.sizes())
        self._layout_settings.sync()

    def _save_window_geometry(self) -> None:
        """Persist window size/position to settings."""

        geometry = self.saveGeometry()
        self._layout_settings.setValue("windowGeometry", geometry)
        self._layout_settings.sync()

    def _load_prompts(self, search_text: str = "") -> None:
        """Populate the list model from the repository or semantic search."""

        try:
            self._all_prompts = self._manager.repository.list()
        except RepositoryError as exc:
            self._show_error("Unable to load prompts", str(exc))
            return

        self._populate_filters(self._all_prompts)

        stripped = search_text.strip()
        if stripped:
            try:
                suggestions = self._manager.suggest_prompts(stripped, limit=50)
            except PromptManagerError as exc:
                self._show_error("Unable to search prompts", str(exc))
            else:
                self._apply_suggestions(suggestions)
                return

        self._suggestions = None
        self._current_prompts = list(self._all_prompts)

        filtered = self._apply_filters(self._current_prompts)
        self._model.set_prompts(filtered)
        self._list_view.clearSelection()
        if not filtered:
            self._detail_widget.clear()
        self._update_intent_hint(filtered)

    def _populate_filters(self, prompts: Sequence[Prompt]) -> None:
        """Refresh category and tag filters based on available prompts."""

        categories = sorted({prompt.category for prompt in prompts if prompt.category})
        current_category = self._category_filter.currentData()
        self._category_filter.blockSignals(True)
        self._category_filter.clear()
        self._category_filter.addItem("All categories", None)
        for category in categories:
            self._category_filter.addItem(category, category)
        category_index = self._category_filter.findData(current_category)
        self._category_filter.setCurrentIndex(category_index if category_index != -1 else 0)
        self._category_filter.blockSignals(False)

        tags = sorted({tag for prompt in prompts for tag in prompt.tags})
        current_tag = self._tag_filter.currentData()
        self._tag_filter.blockSignals(True)
        self._tag_filter.clear()
        self._tag_filter.addItem("All tags", None)
        for tag in tags:
            self._tag_filter.addItem(tag, tag)
        tag_index = self._tag_filter.findData(current_tag)
        self._tag_filter.setCurrentIndex(tag_index if tag_index != -1 else 0)
        self._tag_filter.blockSignals(False)

    def _apply_filters(self, prompts: Sequence[Prompt]) -> List[Prompt]:
        """Apply category, tag, and quality filters to a prompt sequence."""

        selected_category = self._category_filter.currentData()
        selected_tag = self._tag_filter.currentData()
        min_quality = self._quality_filter.value()

        filtered: List[Prompt] = []
        for prompt in prompts:
            if selected_category and prompt.category != selected_category:
                continue
            if selected_tag and selected_tag not in prompt.tags:
                continue
            if min_quality > 0.0:
                quality = prompt.quality_score or 0.0
                if quality < min_quality:
                    continue
            filtered.append(prompt)
        return filtered

    def _on_filters_changed(self, *_: object) -> None:
        """Refresh the prompt list when filter widgets change."""

        self._refresh_filtered_view()

    def _refresh_filtered_view(self) -> None:
        filtered = self._apply_filters(self._current_prompts)
        self._model.set_prompts(filtered)
        self._list_view.clearSelection()
        if not filtered:
            self._detail_widget.clear()
        self._update_intent_hint(filtered)

    @staticmethod
    def _replace_prompt_in_collection(collection: List[Prompt], updated: Prompt) -> bool:
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

        search_text = self._search_input.text().strip()
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
        self._model.set_prompts(filtered)
        if filtered:
            self._update_intent_hint(filtered)
        else:
            self._detail_widget.clear()
            self._intent_hint.clear()
            self._intent_hint.setVisible(False)

        if any(prompt.id == prompt_id for prompt in filtered):
            self._select_prompt(prompt_id)
            self._detail_widget.display_prompt(updated_prompt)

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

        summary_parts: List[str] = []
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

    def _display_execution_result(
        self,
        prompt: Prompt,
        outcome: PromptManager.ExecutionOutcome,
    ) -> None:
        """Render the most recent execution result in the result pane."""

        self._last_execution = outcome
        self._last_prompt_name = prompt.name
        self._chat_prompt_id = prompt.id
        self._chat_conversation = [dict(message) for message in outcome.conversation]
        self._result_label.setText(f"Last Result — {prompt.name}")
        meta_parts: List[str] = [f"Duration: {outcome.result.duration_ms} ms"]
        history_entry = outcome.history_entry
        if history_entry is not None:
            executed_at = history_entry.executed_at.astimezone()
            meta_parts.append(f"Logged: {executed_at.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        self._result_meta.setText(" | ".join(meta_parts))
        response_text = outcome.result.response_text or ""
        request_text = outcome.result.request_text or ""
        self._result_text.setPlainText(response_text)
        self._update_diff_view(request_text, response_text)
        self._render_markdown_button.setEnabled(bool(response_text.strip()))
        self._result_tabs.setCurrentIndex(0)
        self._copy_result_button.setEnabled(bool(outcome.result.response_text))
        self._save_button.setEnabled(True)
        self._continue_chat_button.setEnabled(True)
        self._end_chat_button.setEnabled(True)
        self._refresh_chat_history_view()
        self._query_input.clear()
        self._query_input.setFocus(Qt.ShortcutFocusReason)

    def _clear_execution_result(self) -> None:
        """Reset the result pane to its default state."""

        self._last_execution = None
        self._last_prompt_name = None
        self._result_label.setText("No prompt executed yet")
        self._result_meta.clear()
        self._result_text.clear()
        self._diff_view.clear()
        self._diff_view.setPlaceholderText("Run a prompt to compare input and output.")
        self._render_markdown_button.setEnabled(False)
        self._copy_result_button.setEnabled(False)
        self._save_button.setEnabled(False)
        self._reset_chat_session()

    def _reset_chat_session(self, *, clear_view: bool = True) -> None:
        """Disable chat continuation controls and optionally clear the transcript."""

        self._chat_prompt_id = None
        self._chat_conversation = []
        self._continue_chat_button.setEnabled(False)
        self._end_chat_button.setEnabled(False)
        if clear_view:
            self._chat_history_view.clear()
            self._chat_history_view.setPlaceholderText("Run a prompt to start chatting.")

    def _refresh_chat_history_view(self) -> None:
        """Render the active conversation into the chat tab."""

        if not self._chat_conversation:
            self._chat_history_view.clear()
            self._chat_history_view.setPlaceholderText("Run a prompt to start chatting.")
            return
        formatted = self._format_chat_history(self._chat_conversation)
        self._chat_history_view.setPlainText(formatted)

    @staticmethod
    def _format_chat_history(conversation: Sequence[Dict[str, str]]) -> str:
        """Return a readable transcript for the chat history tab."""

        lines: List[str] = []
        for message in conversation:
            role = message.get("role", "").strip().lower()
            content = message.get("content", "")
            if role == "user":
                speaker = "You"
            elif role == "assistant":
                speaker = "Assistant"
            elif role == "system":
                speaker = "System"
            else:
                speaker = message.get("role", "Message")
            lines.append(f"{speaker}:\n{content}")
        return "\n\n".join(lines)

    def _update_diff_view(self, original: str, generated: str) -> None:
        """Render a unified diff comparing the request text with the response."""

        diff_text = build_diff_preview(original, generated)
        self._diff_view.setPlainText(diff_text)

    def _on_query_text_changed(self) -> None:
        """Update language detection and syntax highlighting as the user types."""

        text = self._query_input.toPlainText()
        self._update_detected_language(text)

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
        self._model.set_prompts(filtered)
        self._select_prompt(selected_prompt.id)
        self._detail_widget.display_prompt(selected_prompt)
        if action.template and not self._query_input.toPlainText().strip():
            self._query_input.setPlainText(action.template)
            cursor = self._query_input.textCursor()
            cursor.movePosition(QTextCursor.End)
            self._query_input.setTextCursor(cursor)
        self._query_input.setFocus(Qt.ShortcutFocusReason)
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

        for action in self._quick_actions:
            if not action.shortcut:
                continue
            shortcut = QShortcut(QKeySequence(action.shortcut), self)
            shortcut.activated.connect(lambda a=action: self._execute_quick_action(a))  # type: ignore[arg-type]
            self._quick_shortcuts.append(shortcut)

    def _default_quick_actions(self) -> List[QuickAction]:
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
                description="Jump to documentation prompts that generate docstrings and commentary.",
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

    def _build_quick_actions(
        self, custom_actions: Optional[object]
    ) -> List[QuickAction]:
        actions_by_id: Dict[str, QuickAction] = {
            action.identifier: action for action in self._default_quick_actions()
        }
        if not custom_actions:
            return list(actions_by_id.values())

        data: Iterable[dict[str, Any]]
        if isinstance(custom_actions, list):
            data = [
                entry
                for entry in custom_actions
                if isinstance(entry, dict)
            ]
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
        ranked: List[Prompt],
    ) -> Optional[Prompt]:
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

    def _refine_prompt_body(
        self,
        prompt_text: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> PromptRefinement:
        """Delegate prompt refinement to PromptManager."""

        return self._manager.refine_prompt_text(
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
        self.statusBar().showMessage(f"Detected intent: {label} ({int(prediction.confidence * 100)}%)", 5000)
        self._usage_logger.log_detect(prediction=prediction, query_text=text)

    def _on_suggest_prompt_clicked(self) -> None:
        """Generate prompt suggestions from the free-form query input."""

        query = self._query_input.toPlainText().strip() or self._search_input.text().strip()
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

    def _on_history_clicked(self) -> None:
        """Switch to the history tab."""

        self._tab_widget.setCurrentWidget(self._history_panel)

    def _on_tab_changed(self, index: int) -> None:
        widget = self._tab_widget.widget(index)
        if widget is self._history_panel:
            self._history_panel.refresh()
            self._usage_logger.log_history_view(total=self._history_panel.row_count())

    def _on_save_result_clicked(self) -> None:
        """Persist the latest execution result with optional user notes."""

        if self._last_execution is None:
            self.statusBar().showMessage("Run a prompt before saving the result.", 3000)
            return
        prompt = self._current_prompt()
        if prompt is None and self._current_prompts:
            prompt = self._current_prompts[0]
        if prompt is None:
            self.statusBar().showMessage("Select a prompt to associate with the result.", 3000)
            return

        outcome = self._last_execution
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
        metadata: Dict[str, Any] = {"source": "manual-save"}
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

        if not self._chat_conversation or self._chat_prompt_id is None:
            self.statusBar().showMessage("Run a prompt to start a chat before continuing.", 4000)
            return

        follow_up = self._query_input.toPlainText().strip()
        if not follow_up:
            self.statusBar().showMessage("Type a follow-up message before continuing the chat.", 4000)
            return

        prompt_id = self._chat_prompt_id
        try:
            prompt = self._manager.get_prompt(prompt_id)
        except (PromptNotFoundError, PromptManagerError) as exc:
            self._show_error("Prompt unavailable", str(exc))
            self._reset_chat_session()
            return

        try:
            outcome = self._manager.execute_prompt(
                prompt.id,
                follow_up,
                conversation=self._chat_conversation,
            )
        except PromptExecutionUnavailable as exc:
            self._continue_chat_button.setEnabled(False)
            self._usage_logger.log_execute(
                prompt_name=prompt.name,
                success=False,
                duration_ms=None,
                error=str(exc),
            )
            self._show_error("Prompt execution unavailable", str(exc))
            return
        except (PromptExecutionError, PromptManagerError) as exc:
            self._usage_logger.log_execute(
                prompt_name=prompt.name,
                success=False,
                duration_ms=None,
                error=str(exc),
            )
            self._show_error("Continue chat failed", str(exc))
            return

        self._display_execution_result(prompt, outcome)
        self._usage_logger.log_execute(
            prompt_name=prompt.name,
            success=True,
            duration_ms=outcome.result.duration_ms,
        )
        self.statusBar().showMessage(
            f"Continued chat with '{prompt.name}' in {outcome.result.duration_ms} ms.",
            5000,
        )

    def _on_end_chat_clicked(self) -> None:
        """Terminate the active chat session without clearing the transcript."""

        if not self._chat_conversation:
            self.statusBar().showMessage("There is no active chat session to end.", 4000)
            return
        self._chat_prompt_id = None
        self._continue_chat_button.setEnabled(False)
        self._end_chat_button.setEnabled(False)
        self.statusBar().showMessage("Chat session ended. Conversation preserved in history.", 5000)

    def _on_run_prompt_clicked(self) -> None:
        """Execute the selected prompt via the manager."""

        request_text = self._query_input.toPlainText().strip()
        if not request_text:
            self.statusBar().showMessage("Paste some text or code before executing a prompt.", 4000)
            return
        prompt = self._current_prompt()
        if prompt is None:
            prompts = self._model.prompts()
            prompt = prompts[0] if prompts else None
        if prompt is None:
            self.statusBar().showMessage("Select a prompt to execute first.", 4000)
            return

        try:
            outcome = self._manager.execute_prompt(prompt.id, request_text)
        except PromptExecutionUnavailable as exc:
            self._run_button.setEnabled(False)
            self._usage_logger.log_execute(
                prompt_name=prompt.name,
                success=False,
                duration_ms=None,
                error=str(exc),
            )
            self._show_error("Prompt execution unavailable", str(exc))
            return
        except (PromptExecutionError, PromptManagerError) as exc:
            self._usage_logger.log_execute(
                prompt_name=prompt.name,
                success=False,
                duration_ms=None,
                error=str(exc),
            )
            self._show_error("Prompt execution failed", str(exc))
            return

        self._display_execution_result(prompt, outcome)
        self._usage_logger.log_execute(
            prompt_name=prompt.name,
            success=True,
            duration_ms=outcome.result.duration_ms,
        )
        self.statusBar().showMessage(
            f"Executed '{prompt.name}' in {outcome.result.duration_ms} ms.",
            5000,
        )

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
        payload = prompt.context or prompt.description
        if not payload:
            self.statusBar().showMessage("Selected prompt does not include a body to copy.", 3000)
            return
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(payload)
        self.statusBar().showMessage(f"Copied '{prompt.name}' to the clipboard.", 4000)
        self._usage_logger.log_copy(prompt_name=prompt.name, prompt_has_body=bool(prompt.context))

    def _on_render_markdown_clicked(self) -> None:
        """Open a rendered markdown preview for the latest execution result."""

        if self._last_execution is None:
            QMessageBox.information(
                self,
                "No output available",
                "Run a prompt to view rendered output.",
            )
            return
        content = (self._last_execution.result.response_text or "").strip()
        if not content:
            QMessageBox.information(
                self,
                "No output available",
                "The latest result did not include any text to render.",
            )
            return
        prompt_name = self._last_prompt_name or "Prompt Output"
        dialog = MarkdownPreviewDialog(
            content,
            self,
            title=f"Rendered Output — {prompt_name}",
        )
        dialog.exec()

    def _on_copy_result_clicked(self) -> None:
        """Copy the latest execution result to the clipboard."""

        if self._last_execution is None:
            self.statusBar().showMessage("Run a prompt to generate a result first.", 3000)
            return
        response = self._last_execution.result.response_text
        if not response:
            self.statusBar().showMessage("Latest result is empty; nothing to copy.", 3000)
            return
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(response)
        self.statusBar().showMessage("Prompt result copied to the clipboard.", 4000)

    def _apply_suggestions(self, suggestions: PromptManager.IntentSuggestions) -> None:
        """Apply intent suggestions to the list view and update filters."""

        self._suggestions = suggestions
        self._current_prompts = list(suggestions.prompts)
        filtered = self._apply_filters(self._current_prompts)
        self._model.set_prompts(filtered)
        self._list_view.clearSelection()
        if filtered:
            self._select_prompt(filtered[0].id)
        else:
            self._detail_widget.clear()
        self._update_intent_hint(filtered)

    def _current_prompt(self) -> Optional[Prompt]:
        """Return the prompt currently selected in the list."""

        index = self._list_view.currentIndex()
        if not index.isValid():
            return None
        return self._model.prompt_at(index.row())

    def _on_search_changed(self, text: str) -> None:
        """Trigger prompt search with debounce-friendly minimum length."""

        if not text or len(text.strip()) >= 2:
            self._load_prompts(text)

    def _on_prompt_double_clicked(self, index: QModelIndex) -> None:
        """Open the edit dialog when a prompt is double-clicked."""

        if not index.isValid():
            return
        self._list_view.setCurrentIndex(index)
        self._on_edit_clicked()

    def _on_refresh_clicked(self) -> None:
        """Reload prompts from storage, respecting current search text."""

        self._load_prompts(self._search_input.text())

    def _on_add_clicked(self) -> None:
        """Open the creation dialog and persist a new prompt."""

        dialog = PromptDialog(
            self,
            name_generator=self._generate_prompt_name,
            description_generator=self._generate_prompt_description,
            prompt_engineer=(
                self._refine_prompt_body if self._manager.prompt_engineer is not None else None
            ),
        )
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

    def _on_edit_clicked(self) -> None:
        """Open the edit dialog for the selected prompt and persist changes."""

        prompt = self._current_prompt()
        if prompt is None:
            return
        dialog = PromptDialog(
            self,
            prompt,
            name_generator=self._generate_prompt_name,
            description_generator=self._generate_prompt_description,
            prompt_engineer=(
                self._refine_prompt_body if self._manager.prompt_engineer is not None else None
            ),
        )
        if dialog.exec() != QDialog.Accepted:
            return
        updated = dialog.result_prompt
        if updated is None:
            return
        try:
            stored = self._manager.update_prompt(updated)
        except PromptNotFoundError:
            self._show_error("Prompt missing", "The prompt cannot be located. Refresh and try again.")
            self._load_prompts()
            return
        except PromptStorageError as exc:
            self._show_error("Unable to update prompt", str(exc))
            return
        self._load_prompts(self._search_input.text())
        self._select_prompt(stored.id)

    def _on_delete_clicked(self) -> None:
        """Remove the selected prompt after confirmation."""

        prompt = self._current_prompt()
        if prompt is None:
            return
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
        self._load_prompts(self._search_input.text())

    def _on_import_clicked(self) -> None:
        """Preview catalogue diff and optionally apply updates."""

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select catalogue file",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        catalog_path: Optional[Path]
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
        self._load_prompts(self._search_input.text())

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
    def _on_settings_clicked(self) -> None:
        """Open configuration dialog and apply updates."""

        dialog = SettingsDialog(
            self,
            catalog_path=self._runtime_settings.get("catalog_path"),
            litellm_model=self._runtime_settings.get("litellm_model"),
            litellm_api_key=self._runtime_settings.get("litellm_api_key"),
            litellm_api_base=self._runtime_settings.get("litellm_api_base"),
            litellm_api_version=self._runtime_settings.get("litellm_api_version"),
            quick_actions=self._runtime_settings.get("quick_actions"),
        )
        if dialog.exec() != QDialog.Accepted:
            return
        updates = dialog.result_settings()
        self._apply_settings(updates)

    def _apply_settings(self, updates: dict[str, Optional[str | list[dict[str, object]]]]) -> None:
        """Persist settings, refresh catalogue, and update name generator."""

        catalog_path = updates.get("catalog_path")
        normalized_catalog = (
            str(Path(catalog_path).expanduser()) if catalog_path else None
        )
        self._runtime_settings["catalog_path"] = normalized_catalog
        self._runtime_settings["litellm_model"] = updates.get("litellm_model")
        self._runtime_settings["litellm_api_key"] = updates.get("litellm_api_key")
        self._runtime_settings["litellm_api_base"] = updates.get("litellm_api_base")
        self._runtime_settings["litellm_api_version"] = updates.get("litellm_api_version")
        quick_actions_value = updates.get("quick_actions")
        if isinstance(quick_actions_value, list):
            cleaned_quick_actions = [
                dict(entry) for entry in quick_actions_value if isinstance(entry, dict)
            ]
        else:
            cleaned_quick_actions = None
        self._runtime_settings["quick_actions"] = cleaned_quick_actions
        persist_settings_to_config(self._runtime_settings)

        if self._settings is not None:
            self._settings.catalog_path = (
                Path(normalized_catalog).expanduser()
                if normalized_catalog
                else None
            )
            self._settings.litellm_model = updates.get("litellm_model")
            self._settings.litellm_api_key = updates.get("litellm_api_key")
            self._settings.litellm_api_base = updates.get("litellm_api_base")
            self._settings.litellm_api_version = updates.get("litellm_api_version")
            self._settings.quick_actions = cleaned_quick_actions

        self._quick_actions = self._build_quick_actions(self._runtime_settings.get("quick_actions"))
        self._register_quick_shortcuts()

        try:
            self._manager.set_name_generator(
                self._runtime_settings.get("litellm_model"),
                self._runtime_settings.get("litellm_api_key"),
                self._runtime_settings.get("litellm_api_base"),
                self._runtime_settings.get("litellm_api_version"),
            )
        except NameGenerationError as exc:
            QMessageBox.warning(self, "LiteLLM configuration", str(exc))

        try:
            result = import_prompt_catalog(
                self._manager,
                Path(normalized_catalog).expanduser() if normalized_catalog else None,
            )
            self.statusBar().showMessage(
                f"Catalogue synced (added {result.added}, updated {result.updated}, skipped {result.skipped})",
                5000,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Catalogue import failed", str(exc))

        self._load_prompts(self._search_input.text())

    def _initial_runtime_settings(
        self, settings: Optional[PromptManagerSettings]
    ) -> dict[str, Optional[str | list[dict[str, object]]]]:
        """Load current settings snapshot from configuration."""

        derived_quick_actions: Optional[list[dict[str, object]]]
        if settings and settings.quick_actions:
            derived_quick_actions = [dict(entry) for entry in settings.quick_actions]
        else:
            derived_quick_actions = None

        runtime = {
            "catalog_path": str(settings.catalog_path)
            if settings and settings.catalog_path is not None
            else None,
            "litellm_model": settings.litellm_model if settings else None,
            "litellm_api_key": settings.litellm_api_key if settings else None,
            "litellm_api_base": settings.litellm_api_base if settings else None,
            "litellm_api_version": settings.litellm_api_version if settings else None,
            "quick_actions": derived_quick_actions,
        }

        config_path = Path("config/config.json")
        if config_path.exists():
            try:
                data = json.loads(config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
            else:
                for key in (
                    "catalog_path",
                    "litellm_model",
                    "litellm_api_key",
                    "litellm_api_base",
                    "litellm_api_version",
                ):
                    if isinstance(data.get(key), str):
                        runtime[key] = data[key]
                if isinstance(data.get("quick_actions"), list):
                    runtime["quick_actions"] = [dict(entry) for entry in data["quick_actions"] if isinstance(entry, dict)]
        return runtime

    def _on_selection_changed(self, *_: object) -> None:
        """Update the detail panel to reflect the new selection."""

        prompt = self._current_prompt()
        if prompt is None:
            self._detail_widget.clear()
            self._reset_chat_session()
            return
        if self._chat_prompt_id and prompt.id != self._chat_prompt_id:
            self._reset_chat_session()
        self._detail_widget.display_prompt(prompt)

    def _select_prompt(self, prompt_id: uuid.UUID) -> None:
        """Highlight the given prompt in the list view when present."""

        for row, prompt in enumerate(self._model.prompts()):
            if prompt.id == prompt_id:
                index = self._model.index(row, 0)
                self._list_view.setCurrentIndex(index)
                break

    def _handle_notification(self, notification: Notification) -> None:
        """React to notification updates published by the core manager."""

        self._register_notification(notification, show_status=True)

    def _register_notification(self, notification: Notification, *, show_status: bool) -> None:
        self._notification_history.append(notification)
        self._update_active_notification(notification)
        self._update_notification_indicator()
        if show_status:
            message = self._format_notification_message(notification)
            duration = 0 if notification.status is NotificationStatus.STARTED else 5000
            self.statusBar().showMessage(message, duration)

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
        dialog = NotificationHistoryDialog(tuple(self._notification_history), self)
        dialog.exec()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._save_window_geometry()
        self._save_splitter_state()
        if hasattr(self, "_notification_bridge"):
            self._notification_bridge.close()
        super().closeEvent(event)

    def _show_error(self, title: str, message: str) -> None:
        """Display an error dialog and log to status bar."""

        QMessageBox.critical(self, title, message)
        self.statusBar().showMessage(message, 5000)


__all__ = ["MainWindow", "PromptListModel", "PromptDetailWidget"]
