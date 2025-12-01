"""Dialog widgets used by the Prompt Manager GUI.

Updates:
  v0.11.14 - 2025-11-29 - Shorten update summaries and wrap maintenance tooltips.
  v0.11.13 - 2025-11-28 - Keep maintenance button pinned below scroll surface.
  v0.11.12 - 2025-11-28 - Add backup snapshot action to the maintenance dialog.
  v0.11.11 - 2025-11-28 - Make maintenance dialog scrollable with shorter default height.
  v0.11.10 - 2025-11-28 - Add prompt body diff tab for comparing selected versions.
  v0.11.9 - 2025-11-28 - Add category health analytics table to maintenance dialog.
  v0.11.8 - 2025-11-27 - Show toast confirmations for dialog copy actions.
  v0.11.7 - 2025-11-27 - Add copy prompt body control to version history dialog.
  v0.11.6 - 2025-11-27 - Add prompt part selector to response style dialog and rename copy action.
  v0.11.5 - 2025-11-27 - Add import fallback for processing indicator in test harnesses.
  v0.11.4 - 2025-11-27 - Ensure clearing scenarios removes persisted metadata.
  v0.11.3 - 2025-11-27 - Add busy indicators to prompt metadata generators.
  v0.11.2 - 2025-11-22 - Allow selecting prompt categories from the registry list.
  v0.11.1 - 2025-11-22 - Add execute-as-context shortcut to the prompt dialog.
  v0.11.0 - 2025-11-22 - Add prompt category management dialog with create/edit workflows.
  v0.10.8 - 2025-11-22 - Surface prompt versions and quick history access inside edit dialog.
  v0.10.7 - 2025-11-22 - Add structure-only refinement action to the prompt dialog.
  v0.10.6 - 2025-11-22 - Add prompt body tab to version history dialog.
  v0.10.5 - 2025-11-22 - Redesign prompt dialog layout for taller editing surface.
  v0.10.4 - 2025-11-22 - Align example section toggles with the active theme palette.
  v0.10.3 - 2025-11-22 - Add author input to prompt create/edit workflows.
  v0.10.2 - 2025-11-22 - Default prompt versions to integer labels for new entries.
  v0.10.1 - 2025-11-22 - Replace prompt refined alert with resizable, scrollable dialog.
  v0.10.0 - 2025-11-22 - Add prompt version history dialog for diffing or restoring snapshots.
  v0.9.2 - 2025-12-08 - Resolve info dialog version label from metadata or pyproject.
  v0.9.1 - 2025-12-08 - Remove task template dialog and references.
  v0.9.0 - 2025-12-06 - Add PromptNote dialog and auto-generated note metadata.
  v0.8.15 - 2025-12-06 - Add PromptNote dialog and auto-generated note metadata.
  v0.8.14 - 2025-12-06 - Auto-generate required response style fields from pasted text.
  v0.8.13 - 2025-12-05 - Add ResponseStyle editor dialog for CRUD workflows.
  v0.8.12 - 2025-11-05 - Display application icon and credit source in info dialog.
  v0.8.11 - 2025-11-05 - Add SQLite maintenance actions to the maintenance dialog.
  v0.8.10 - 2025-11-05 - Add ChromaDB integrity verification action to maintenance dialog.
  v0.8.9 - 2025-11-05 - Add ChromaDB maintenance actions to the maintenance dialog.
  v0.8.8 - 2025-11-30 - Restore catalogue preview dialog for GUI import workflows.
  v0.8.7 - 2025-11-30 - Remove catalogue preview dialog after retiring import workflow.
  v0.8.6 - 2025-11-27 - Support pre-filling prompts when duplicating entries.
  v0.8.5 - 2025-11-26 - Add application info dialog.
  v0.8.4 - 2025-11-25 - Add prompt maintenance overview stats panel.
  v0.8.3 - 2025-11-19 - Add scenario generation controls and persistence to prompt dialog.
  v0.8.2 - 2025-11-17 - Add Apply workflow so prompt edits persist without closing dialog.
  v0.8.1 - 2025-11-16 - Add destructive delete control and metadata suggestion helpers.
  v0.8.0 - 2025-11-16 - Add task template editor dialog.
  v0.7.2 - 2025-11-02 - Collapse example sections when empty and expand on demand.
  v0.7.1 - 2025-11-02 - Increase default prompt dialog size for edit and creation workflows.
  v0.7.0 - 2025-11-16 - Add markdown preview dialog for rendered execution output.
  v0.6.0 - 2025-11-15 - Add prompt engineering refinement button to prompt dialog.
  v0.5.0 - 2025-11-09 - Capture execution ratings alongside optional notes.
  v0.4.0 - 2025-11-08 - Add execution Save Result dialog with optional notes.
  v0.3.0 - 2025-11-06 - Add catalogue preview dialog with diff summary output.
  v0.2.0 - 2025-11-05 - Add prompt name suggestion based on context.
  v0.1.0 - 2025-11-04 - Implement create/edit prompt dialog backed by Prompt dataclass.
"""
from __future__ import annotations

import difflib
import json
import logging
import os
import platform
import re
import textwrap
import uuid
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    TypeVar,
)

from PySide6.QtCore import QEvent, QSettings, Qt, Signal
from PySide6.QtGui import QGuiApplication, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

# When imported via `importlib` as a stand‑alone module (common in tests), the
# usual *package* context is absent, causing the standard relative import to
# fail.  Fallback to an absolute import so the module remains usable both as a
# package sub‑module (``prompt_manager.gui.dialogs``) and as an ad‑hoc loaded
# file (``dialogs``).

try:
    from .resources import load_application_icon  # type: ignore
except ImportError:  # pragma: no cover – fallback for direct execution
    from gui.resources import load_application_icon  # type: ignore

try:
    from .processing_indicator import ProcessingIndicator
except ImportError:  # pragma: no cover – fallback when loaded outside package
    from gui.processing_indicator import ProcessingIndicator
try:
    from .toast import show_toast
except ImportError:  # pragma: no cover - fallback when loaded outside package
    from gui.toast import show_toast
from core import (
    CatalogDiff,
    CatalogDiffEntry,
    CategoryNotFoundError,
    CategoryStorageError,
    DescriptionGenerationError,
    NameGenerationError,
    PromptEngineeringUnavailable,
    PromptManager,
    PromptManagerError,
    RepositoryError,
    ScenarioGenerationError,
)
from core.prompt_engineering import PromptEngineeringError, PromptRefinement
from models.category_model import PromptCategory, slugify_category
from models.prompt_model import Prompt, PromptVersion
from models.prompt_note import PromptNote
from models.response_style import ResponseStyle

if TYPE_CHECKING:
    from collections.abc import Callable, MutableMapping, Sequence

logger = logging.getLogger("prompt_manager.gui.dialogs")

_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+")
_TaskResult = TypeVar("_TaskResult")


class SystemInfo(NamedTuple):
    """Container describing runtime platform characteristics for display."""
    cpu: str
    architecture: str
    platform_family: str
    os_label: str


def _classify_platform_family(system_name: str) -> str:
    """Return a high-level platform family string based on the system identifier."""
    name = system_name.lower()
    if name.startswith("win"):
        return "Windows"
    if name.startswith(("darwin", "mac")):
        return "macOS"
    if name.startswith(("linux", "freebsd", "openbsd", "netbsd", "aix", "hp-ux", "sunos")):
        return "Unix-like"
    if not system_name:
        return "Unknown"
    return system_name


def _collect_system_info() -> SystemInfo:
    """Gather CPU and platform metadata for the info dialog."""
    uname = platform.uname()

    raw_cpu = platform.processor() or uname.processor
    if not raw_cpu:
        cpu_count = os.cpu_count()
        raw_cpu = f"{cpu_count} logical cores" if cpu_count else "Unknown"

    architecture = platform.machine() or uname.machine or "Unknown"

    system_name = platform.system() or uname.system or "Unknown"
    release = platform.release() or uname.release
    os_label = f"{system_name} {release}".strip() if release else system_name

    return SystemInfo(
        cpu=raw_cpu,
        architecture=architecture,
        platform_family=_classify_platform_family(system_name),
        os_label=os_label or "Unknown",
    )


def _strip_scenarios_metadata(
    metadata: MutableMapping[str, Any] | None,
) -> MutableMapping[str, Any] | None:
    """Return a deep copy of metadata without stored usage scenarios."""
    if metadata is None:
        return None
    cleaned = deepcopy(metadata)
    cleaned.pop("scenarios", None)
    return cleaned or None


class CollapsibleTextSection(QWidget):
    """Wrapper providing an expandable/collapsible plain text editor."""
    textChanged = Signal()

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = title
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)

        self._toggle = QToolButton(self)
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(False)
        self._toggle.setArrowType(Qt.RightArrow)
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.clicked.connect(self._on_toggle_clicked)  # type: ignore[arg-type]
        self._layout.addWidget(self._toggle)
        self._apply_toggle_style()

        self._editor = QPlainTextEdit(self)
        self._editor.setVisible(False)
        self._layout.addWidget(self._editor)

        self._editor.textChanged.connect(self._on_text_changed)  # type: ignore[arg-type]
        self._editor.installEventFilter(self)
        self._collapsed_height = 0
        self._expanded_height = 100

    def event(self, event: QEvent) -> bool:  # noqa: D401 - Qt override, documentation inherited
        if event.type() in {QEvent.PaletteChange, QEvent.StyleChange}:
            self._apply_toggle_style()
        return super().event(event)

    def setPlaceholderText(self, text: str) -> None:
        self._editor.setPlaceholderText(text)

    def setPlainText(self, text: str) -> None:
        self._editor.setPlainText(text)
        stripped = text.strip()
        expanded = bool(stripped)
        self._set_expanded(expanded, focus=False)

    def toPlainText(self) -> str:
        return self._editor.toPlainText()

    def editor(self) -> QPlainTextEdit:
        return self._editor

    def focusEditor(self) -> None:
        self._set_expanded(True, focus=True)

    def isExpanded(self) -> bool:
        return self._toggle.isChecked()

    def setExpanded(self, expanded: bool) -> None:
        self._set_expanded(expanded, focus=expanded)

    def eventFilter(self, obj, event):  # type: ignore[override]
        if obj is self._editor and event.type() == QEvent.FocusOut:
            if not self._editor.toPlainText().strip():
                self._set_expanded(False, focus=False)
        return super().eventFilter(obj, event)

    def _apply_toggle_style(self) -> None:
        """Align the toggle button background with the active theme palette."""
        palette = self._toggle.palette()
        button_color = palette.color(QPalette.Button).name()
        text_color = palette.color(QPalette.ButtonText).name()
        border_color = palette.color(QPalette.Mid).name()
        checked_color = palette.color(QPalette.Highlight).name()
        checked_text = palette.color(QPalette.HighlightedText).name()
        self._toggle.setStyleSheet(
            "QToolButton {"
            f"background-color: {button_color};"
            f"color: {text_color};"
            f"border: 1px solid {border_color};"
            "border-radius: 4px;"
            "padding: 4px 8px;"
            "text-align: left;"
            "}"
            "QToolButton:checked {"
            f"background-color: {checked_color};"
            f"color: {checked_text};"
            "}"
        )

    def _set_expanded(self, expanded: bool, *, focus: bool) -> None:
        if self._toggle.isChecked() == expanded and self._editor.isVisible() == expanded:
            return
        self._toggle.blockSignals(True)
        self._toggle.setChecked(expanded)
        self._toggle.blockSignals(False)
        self._toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._editor.setVisible(expanded)
        if expanded:
            self._editor.setFixedHeight(self._expanded_height)
        else:
            self._editor.setFixedHeight(self._collapsed_height)
        if expanded and focus:
            self._editor.setFocus(Qt.OtherFocusReason)

    def _on_toggle_clicked(self, checked: bool) -> None:
        self._set_expanded(checked, focus=checked)

    def _on_text_changed(self) -> None:
        if self._editor.toPlainText().strip():
            if not self._toggle.isChecked():
                self._set_expanded(True, focus=False)
        self.textChanged.emit()


class PromptRefinedDialog(QDialog):
    """Modal dialog presenting prompt refinement output in a resizable view."""
    def __init__(
        self,
        content: str,
        parent: QWidget | None = None,
        *,
        title: str = "Prompt refined",
    ) -> None:
        """Initialize the dialog with the refinement summary content.

        Args:
            content: Text produced by the refinement workflow.
            parent: Optional parent widget that owns the dialog.
            title: Window title describing the refinement action.
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setSizeGripEnabled(True)

        layout = QVBoxLayout(self)
        header = QLabel("Review the refinement details below.", self)
        header.setWordWrap(True)
        layout.addWidget(header)

        self._body = QPlainTextEdit(self)
        self._body.setReadOnly(True)
        self._body.setPlainText(content)
        self._body.setMinimumHeight(200)
        layout.addWidget(self._body)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok, Qt.Horizontal, self)
        button_box.accepted.connect(self.accept)  # type: ignore[arg-type]
        layout.addWidget(button_box)

        self._apply_initial_size()

    def _apply_initial_size(self) -> None:
        """Resize the dialog to fit comfortably under the active screen size."""
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(720, 480)
            return

        geometry = screen.availableGeometry()
        screen_height = geometry.height()
        screen_width = geometry.width()

        height_buffer = 60
        max_height = max(screen_height - height_buffer, int(screen_height * 0.8))
        preferred_height = max(320, int(screen_height * 0.6))
        height = min(preferred_height, max_height)

        width_buffer = 120
        max_width = max(screen_width - width_buffer, int(screen_width * 0.7))
        preferred_width = max(560, int(screen_width * 0.45))
        width = min(preferred_width, max_width)

        self.resize(width, height)


def fallback_suggest_prompt_name(context: str, *, max_words: int = 5) -> str:
    """Generate a concise prompt name from free-form context text."""
    text = context.strip()
    if not text:
        return ""
    first_line = text.splitlines()[0]
    tokens = _WORD_PATTERN.findall(first_line)
    if not tokens:
        return ""
    words = tokens[:max_words]
    name = " ".join(word.capitalize() for word in words)
    if len(tokens) > max_words:
        name += "…"
    return name


def fallback_generate_description(context: str, *, max_length: int = 240) -> str:
    """Create a lightweight summary from the prompt body when LLMs are unavailable."""
    stripped = " ".join(context.split())
    if not stripped:
        return ""
    if len(stripped) <= max_length:
        return stripped
    trimmed = stripped[:max_length]
    last_space = trimmed.rfind(" ")
    if last_space > 0:
        trimmed = trimmed[:last_space]
    return trimmed.rstrip(".") + "…"


def fallback_generate_scenarios(context: str, *, max_items: int = 3) -> list[str]:
    """Provide heuristic usage scenarios when LLM support is unavailable."""
    cleaned = context.strip()
    if not cleaned:
        return []

    # Prefer full sentences; fall back to newline segments when punctuation is sparse.
    sentence_pattern = re.compile(r"(?<=[.!?])\s+|\n")
    segments = [segment.strip() for segment in sentence_pattern.split(cleaned) if segment.strip()]
    if not segments:
        segments = [cleaned]

    scenarios: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        clause = segment.rstrip(".!?")
        if not clause:
            continue
        lowered = clause[0].lower() + clause[1:] if len(clause) > 1 else clause.lower()
        scenario = f"Use when you need to {lowered}"
        if not scenario.endswith('.'):
            scenario += "."
        normalised = textwrap.shorten(scenario, width=140, placeholder="…")
        key = normalised.lower()
        if key in seen:
            continue
        seen.add(key)
        scenarios.append(normalised)
        if len(scenarios) >= max_items:
            break
    return scenarios


class PromptDialog(QDialog):
    """Modal dialog used for creating or editing prompt records."""
    applied = Signal(Prompt)
    execute_context_requested = Signal(Prompt, str)

    def __init__(
        self,
        parent=None,
        prompt: Prompt | None = None,
        category_provider: Callable[[], Sequence[PromptCategory]] | None = None,
        name_generator: Callable[[str], str] | None = None,
        description_generator: Callable[[str], str] | None = None,
        category_generator: Callable[[str], str] | None = None,
        tags_generator: Callable[[str], Sequence[str]] | None = None,
        scenario_generator: Callable[[str], Sequence[str]] | None = None,
        prompt_engineer: Callable[..., PromptRefinement] | None = None,
        structure_refiner: Callable[..., PromptRefinement] | None = None,
        version_history_handler: Callable[[Prompt], None] | None = None,
    ) -> None:
        """Configure the prompt editor dialog, injecting optional helpers."""
        super().__init__(parent)
        self._source_prompt = prompt
        self._result_prompt: Prompt | None = None
        self._category_provider = category_provider
        self._name_generator = name_generator
        self._description_generator = description_generator
        self._prompt_engineer = prompt_engineer
        self._structure_refiner = structure_refiner
        self._category_generator = category_generator
        self._tags_generator = tags_generator
        self._scenario_generator = scenario_generator
        self._categories: list[PromptCategory] = []
        self._delete_requested = False
        self._delete_button: QPushButton | None = None
        self._apply_button: QPushButton | None = None
        self._version_label: QLabel | None = None
        self._version_history_button: QPushButton | None = None
        self._version_history_handler = version_history_handler
        self._generate_category_button: QPushButton | None = None
        self._generate_tags_button: QPushButton | None = None
        self._generate_scenarios_button: QPushButton | None = None
        self._execute_context_button: QPushButton | None = None
        self.setWindowTitle("Create Prompt" if prompt is None else "Edit Prompt")
        self.setMinimumSize(900, 780)
        self.resize(1100, 880)
        self._build_ui()
        if prompt is not None:
            self._populate(prompt)

    @property
    def result_prompt(self) -> Prompt | None:
        """Return the prompt produced by the dialog after acceptance."""
        return self._result_prompt

    @property
    def delete_requested(self) -> bool:
        """Return True when the user chose to delete the prompt instead of saving."""
        return self._delete_requested

    @property
    def source_prompt(self) -> Prompt | None:
        """Expose the prompt supplied to the dialog for convenience."""
        return self._source_prompt

    def _build_ui(self) -> None:
        """Construct the dialog layout and wire interactions."""
        main_layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignTop)
        form_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self._name_input = QLineEdit(self)
        self._generate_name_button = QPushButton("Generate", self)
        self._generate_name_button.setToolTip("Suggest a name based on the context field.")
        self._generate_name_button.clicked.connect(self._on_generate_name_clicked)  # type: ignore[arg-type]
        name_container = QWidget(self)
        name_container_layout = QHBoxLayout(name_container)
        name_container_layout.setContentsMargins(0, 0, 0, 0)
        name_container_layout.setSpacing(6)
        name_container_layout.addWidget(self._name_input)
        name_container_layout.addWidget(self._generate_name_button)
        form_layout.addRow("Name", name_container)

        version_container = QWidget(self)
        version_container_layout = QHBoxLayout(version_container)
        version_container_layout.setContentsMargins(0, 0, 0, 0)
        version_container_layout.setSpacing(8)
        self._version_label = QLabel("Not yet saved", self)
        self._version_label.setObjectName("promptDialogVersion")
        version_container_layout.addWidget(self._version_label)
        self._version_history_button = QPushButton("Version History", self)
        self._version_history_button.setObjectName("promptDialogHistoryButton")
        self._version_history_button.setCursor(Qt.PointingHandCursor)
        self._version_history_button.clicked.connect(self._on_version_history_clicked)  # type: ignore[arg-type]
        version_container_layout.addWidget(self._version_history_button)
        version_container_layout.addStretch(1)
        form_layout.addRow("Version", version_container)

        self._category_input = QComboBox(self)
        self._category_input.setEditable(True)
        self._category_input.setInsertPolicy(QComboBox.NoInsert)
        self._category_input.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._category_input.setMinimumContentsLength(12)
        line_edit = self._category_input.lineEdit()
        if line_edit is not None:
            line_edit.setPlaceholderText("General")
        self._language_input = QLineEdit(self)
        self._author_input = QLineEdit(self)
        self._tags_input = QLineEdit(self)
        self._context_input = QPlainTextEdit(self)
        self._context_input.setPlaceholderText("Paste the full prompt text here…")
        self._context_input.setMinimumHeight(320)
        context_policy = self._context_input.sizePolicy()
        context_policy.setVerticalPolicy(QSizePolicy.Expanding)
        self._context_input.setSizePolicy(context_policy)
        self._description_input = QPlainTextEdit(self)
        self._description_input.setFixedHeight(60)
        self._description_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._example_input = CollapsibleTextSection("Example Input", self)
        self._example_input.setPlaceholderText("Optional example input…")
        self._example_output = CollapsibleTextSection("Example Output", self)
        self._example_output.setPlaceholderText("Optional example output…")

        self._language_input.setPlaceholderText("en")
        self._author_input.setPlaceholderText("Optional author name…")
        self._author_input.setMaximumWidth(260)
        self._author_input.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._tags_input.setPlaceholderText("tag-a, tag-b")

        metadata_container = QWidget(self)
        metadata_layout = QGridLayout(metadata_container)
        metadata_layout.setContentsMargins(0, 0, 0, 0)
        metadata_layout.setHorizontalSpacing(12)
        metadata_layout.setVerticalSpacing(4)

        author_label = QLabel("Author", metadata_container)
        metadata_layout.addWidget(author_label, 0, 0)
        metadata_layout.addWidget(self._author_input, 1, 0)

        category_label = QLabel("Category", metadata_container)
        metadata_layout.addWidget(category_label, 0, 1)

        category_container = QWidget(metadata_container)
        category_container_layout = QHBoxLayout(category_container)
        category_container_layout.setContentsMargins(0, 0, 0, 0)
        category_container_layout.setSpacing(6)
        category_container_layout.addWidget(self._category_input)
        self._populate_category_options()
        self._generate_category_button = QPushButton("Suggest", self)
        self._generate_category_button.setToolTip("Suggest a category based on the prompt body.")
        self._generate_category_button.clicked.connect(self._on_generate_category_clicked)  # type: ignore[arg-type]
        if self._category_generator is None:
            self._generate_category_button.setEnabled(False)
            self._generate_category_button.setToolTip(
                "Category suggestions require the main app context."
            )
        category_container_layout.addWidget(self._generate_category_button)
        metadata_layout.addWidget(category_container, 1, 1)

        tags_label = QLabel("Tags", metadata_container)
        metadata_layout.addWidget(tags_label, 0, 2)

        tags_container = QWidget(metadata_container)
        tags_container_layout = QHBoxLayout(tags_container)
        tags_container_layout.setContentsMargins(0, 0, 0, 0)
        tags_container_layout.setSpacing(6)
        tags_container_layout.addWidget(self._tags_input)
        self._generate_tags_button = QPushButton("Suggest", self)
        self._generate_tags_button.setToolTip("Suggest tags based on the prompt body.")
        self._generate_tags_button.clicked.connect(self._on_generate_tags_clicked)  # type: ignore[arg-type]
        if self._tags_generator is None:
            self._generate_tags_button.setEnabled(False)
            self._generate_tags_button.setToolTip(
                "Tag suggestions require the main app context."
            )
        tags_container_layout.addWidget(self._generate_tags_button)
        metadata_layout.addWidget(tags_container, 1, 2)
        metadata_layout.setColumnStretch(0, 1)
        metadata_layout.setColumnStretch(1, 2)
        metadata_layout.setColumnStretch(2, 2)

        form_layout.addRow("Language", self._language_input)
        form_layout.addRow("", metadata_container)
        form_layout.addRow("Prompt Body", self._context_input)
        self._refine_button = QPushButton("Refine", self)
        self._refine_button.setToolTip(
            "Analyse and refine the prompt using LiteLLM when configured."
        )
        self._refine_button.clicked.connect(self._on_refine_clicked)  # type: ignore[arg-type]
        if self._prompt_engineer is None:
            try:
                self._refine_button.setEnabled(False)
            except AttributeError:  # pragma: no cover - stubbed widgets during tests
                pass
            self._refine_button.setToolTip(
                "Configure LiteLLM in Settings to enable prompt engineering."
            )
        self._structure_refine_button = QPushButton("Refine Structure", self)
        self._structure_refine_button.setToolTip(
            "Improve formatting and section layout without altering the prompt's intent."
        )
        self._structure_refine_button.clicked.connect(self._on_refine_structure_clicked)  # type: ignore[arg-type]
        if self._structure_refiner is None:
            try:
                self._structure_refine_button.setEnabled(False)
            except AttributeError:  # pragma: no cover - stubbed widgets during tests
                pass
            self._structure_refine_button.setToolTip(
                "Configure LiteLLM in Settings to enable prompt engineering."
            )
        refine_row = QWidget(self)
        refine_row_layout = QHBoxLayout(refine_row)
        refine_row_layout.setContentsMargins(0, 0, 0, 0)
        refine_row_layout.setSpacing(8)
        refine_row_layout.addWidget(self._refine_button)
        refine_row_layout.addWidget(self._structure_refine_button)
        self._execute_context_button = QPushButton("Execute as Context", self)
        self._execute_context_button.setToolTip(
            "Run the current prompt body as context for an ad-hoc execution."
        )
        self._execute_context_button.clicked.connect(self._on_execute_context_clicked)  # type: ignore[arg-type]
        refine_row_layout.addWidget(self._execute_context_button)
        form_layout.addRow("", refine_row)
        form_layout.addRow("Description", self._description_input)
        self._scenarios_input = QPlainTextEdit(self)
        self._scenarios_input.setPlaceholderText("One scenario per line…")
        self._scenarios_input.setFixedHeight(90)
        scenarios_container = QWidget(self)
        scenarios_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        scenarios_layout = QVBoxLayout(scenarios_container)
        scenarios_layout.setContentsMargins(0, 0, 0, 0)
        scenarios_layout.setSpacing(4)
        scenarios_layout.addWidget(self._scenarios_input)
        self._generate_scenarios_button = QPushButton("Generate Scenarios", self)
        self._generate_scenarios_button.setToolTip(
            "Analyse the prompt body to suggest practical usage scenarios."
        )
        self._generate_scenarios_button.clicked.connect(self._on_generate_scenarios_clicked)  # type: ignore[arg-type]
        if self._scenario_generator is None:
            self._generate_scenarios_button.setToolTip(
                "Generate heuristic scenarios from the prompt body. Configure LiteLLM for AI help."
            )
        scenarios_layout.addWidget(self._generate_scenarios_button, alignment=Qt.AlignRight)
        form_layout.addRow("Scenarios", scenarios_container)

        examples_container = QWidget(self)
        examples_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        examples_layout = QHBoxLayout(examples_container)
        examples_layout.setContentsMargins(0, 0, 0, 0)
        examples_layout.setSpacing(12)
        examples_layout.addWidget(self._example_input, 1)
        examples_layout.addWidget(self._example_output, 1)
        form_layout.addRow("Examples", examples_container)

        main_layout.addLayout(form_layout)

        self._buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        self._buttons.accepted.connect(self._on_accept)  # type: ignore[arg-type]
        self._buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        if self._source_prompt is not None:
            self._apply_button = self._buttons.addButton("Apply", QDialogButtonBox.ApplyRole)
            self._apply_button.setToolTip("Save changes without closing the dialog.")
            self._apply_button.clicked.connect(self._on_apply_clicked)  # type: ignore[arg-type]
            self._delete_button = self._buttons.addButton(
                "Delete",
                QDialogButtonBox.DestructiveRole,
            )
            self._delete_button.setToolTip("Delete this prompt from the catalogue.")
            self._delete_button.clicked.connect(self._on_delete_clicked)  # type: ignore[arg-type]
        main_layout.addWidget(self._buttons)
        self._context_input.textChanged.connect(self._on_context_changed)  # type: ignore[arg-type]
        self._update_version_controls(self._source_prompt.version if self._source_prompt else None)
        self._refresh_execute_context_button()

    def prefill_from_prompt(self, prompt: Prompt) -> None:
        """Populate inputs from an existing prompt while staying in creation mode."""
        self._populate(prompt)
        self._result_prompt = None
        self._delete_requested = False

    def _populate(self, prompt: Prompt) -> None:
        """Fill inputs with the existing prompt values for editing."""
        self._name_input.setText(prompt.name)
        self._set_category_value(prompt.category)
        self._language_input.setText(prompt.language)
        self._author_input.setText(prompt.author or "")
        self._tags_input.setText(", ".join(prompt.tags))
        self._context_input.setPlainText(prompt.context or "")
        self._description_input.setPlainText(prompt.description)
        self._example_input.setPlainText(prompt.example_input or "")
        self._example_output.setPlainText(prompt.example_output or "")
        self._scenarios_input.setPlainText("\n".join(prompt.scenarios))
        self._update_version_controls(prompt.version)

    def _on_accept(self) -> None:
        """Validate inputs, build the prompt, and close the dialog."""
        prompt = self._build_prompt()
        if prompt is None:
            return
        self._result_prompt = prompt
        self.accept()

    def _on_apply_clicked(self) -> None:
        """Persist prompt edits while keeping the dialog open."""
        prompt = self._build_prompt()
        if prompt is None:
            return
        self._result_prompt = prompt
        self.applied.emit(prompt)

    def _on_delete_clicked(self) -> None:
        """Handle delete requests issued from the dialog."""
        if self._source_prompt is None:
            return
        current_name = self._name_input.text().strip()
        name = current_name or self._source_prompt.name or "this prompt"
        confirmation = QMessageBox.question(
            self,
            "Delete prompt",
            f"Are you sure you want to delete '{name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirmation != QMessageBox.Yes:
            return
        self._delete_requested = True
        self._result_prompt = None
        self.accept()

    def _run_with_indicator(
        self,
        message: str,
        func: Callable[..., _TaskResult],
        *args,
        **kwargs,
    ) -> _TaskResult:
        """Execute *func* on a worker thread while showing a busy indicator."""
        indicator = ProcessingIndicator(self, message)
        return indicator.run(func, *args, **kwargs)

    def _on_generate_name_clicked(self) -> None:
        """Generate a prompt name from the context field."""
        try:
            suggestion = self._run_with_indicator(
                "Generating prompt name…",
                self._generate_name,
                self._context_input.toPlainText(),
            )
        except NameGenerationError as exc:
            QMessageBox.warning(self, "Name generation failed", str(exc))
            return
        if suggestion:
            self._name_input.setText(suggestion)

    def _on_generate_category_clicked(self) -> None:
        """Generate a category suggestion based on the prompt body."""
        if self._category_generator is None:
            return
        context = self._context_input.toPlainText()
        try:
            suggestion = (
                self._run_with_indicator(
                    "Generating category suggestion…",
                    self._category_generator,
                    context,
                )
                or ""
            ).strip()
        except Exception as exc:  # noqa: BLE001 - surface generator failures to the user
            QMessageBox.warning(self, "Category suggestion failed", str(exc))
            return
        if suggestion:
            self._set_category_value(suggestion)
        else:
            QMessageBox.information(
                self,
                "No suggestion available",
                "The assistant could not determine a suitable category.",
            )

    def _on_generate_tags_clicked(self) -> None:
        """Generate tag suggestions based on the prompt body."""
        if self._tags_generator is None:
            return
        context = self._context_input.toPlainText()
        try:
            suggestions = self._run_with_indicator(
                "Generating tags…",
                self._tags_generator,
                context,
            ) or []
        except Exception as exc:  # noqa: BLE001 - surface generator failures to the user
            QMessageBox.warning(self, "Tag suggestion failed", str(exc))
            return
        tags = [tag.strip() for tag in suggestions if str(tag).strip()]
        if tags:
            self._tags_input.setText(", ".join(tags))
        else:
            QMessageBox.information(
                self,
                "No suggestion available",
                "The assistant could not determine relevant tags.",
            )

    def _collect_scenarios(self) -> list[str]:
        """Return the current scenarios listed in the dialog."""
        scenarios: list[str] = []
        seen: set[str] = set()
        for line in self._scenarios_input.toPlainText().splitlines():
            text = line.strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            scenarios.append(text)
        return scenarios

    def _set_scenarios(self, scenarios: Sequence[str]) -> None:
        """Populate the scenarios editor with the provided entries."""
        sanitized = [str(item).strip() for item in scenarios if str(item).strip()]
        unique: list[str] = []
        seen: set[str] = set()
        for scenario in sanitized:
            key = scenario.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(scenario)
        self._scenarios_input.setPlainText("\n".join(unique))

    def _populate_category_options(self) -> None:
        """Populate the category selector from the registry provider."""
        if self._category_input is None:
            return
        categories = self._load_categories()
        current_text = self._category_input.currentText().strip()
        self._category_input.blockSignals(True)
        self._category_input.clear()
        for category in categories:
            self._category_input.addItem(category.label)
        self._category_input.blockSignals(False)
        if current_text:
            self._set_category_value(current_text)

    def _load_categories(self) -> list[PromptCategory]:
        """Return available categories, refreshing from the provider when possible."""
        if self._category_provider is None:
            return self._categories
        try:
            categories = list(self._category_provider())
        except Exception as exc:  # pragma: no cover - provider errors surface in GUI
            logger.warning("Unable to load categories for prompt dialog: %s", exc, exc_info=True)
            return self._categories
        categories.sort(key=lambda category: category.label.lower())
        self._categories = categories
        return self._categories

    def _set_category_value(self, value: str | None) -> None:
        """Set the category selector text while aligning with registry labels."""
        if self._category_input is None:
            return
        resolved = self._resolve_category_label(value)
        if not resolved:
            self._category_input.setEditText("")
            self._category_input.setCurrentIndex(-1)
            return
        index = self._category_input.findText(resolved, Qt.MatchFixedString)
        if index >= 0:
            self._category_input.setCurrentIndex(index)
            return
        self._category_input.setEditText(resolved)

    def _current_category_value(self) -> str:
        """Return the canonical category text from the selector."""
        if self._category_input is None:
            return ""
        text = self._category_input.currentText().strip()
        if not text:
            return ""
        resolved = self._resolve_category_label(text)
        if resolved != text:
            self._set_category_value(resolved)
        return resolved

    def _resolve_category_label(self, value: str | None) -> str:
        """Return the stored category label when the value matches an entry."""
        text = (value or "").strip()
        if not text:
            return ""
        slug = slugify_category(text)
        lowered = text.lower()
        for category in self._categories:
            if category.label.lower() == lowered:
                return category.label
            if slug and category.slug == slug:
                return category.label
        return text

    def _generate_scenarios(self, context: str) -> list[str]:
        """Generate scenarios using configured helpers with heuristic fallback."""
        context_text = context.strip()
        if not context_text:
            return []

        if self._scenario_generator is not None:
            try:
                generated = self._scenario_generator(context_text) or []
            except ScenarioGenerationError as exc:
                logger.warning("Scenario generation failed: %s", exc, exc_info=True)
                fallback = fallback_generate_scenarios(context_text)
                if fallback:
                    return fallback
                raise
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Scenario generator raised unexpected error: %s", exc, exc_info=True)
                generated = []
            else:
                scenarios = [str(item).strip() for item in generated if str(item).strip()]
                if scenarios:
                    return scenarios
        return fallback_generate_scenarios(context_text)

    def _on_generate_scenarios_clicked(self) -> None:
        """Populate the scenarios field using analysis of the prompt body."""
        context = self._context_input.toPlainText()
        if not context.strip():
            QMessageBox.information(
                self,
                "Prompt required",
                "Provide a prompt body before generating scenarios.",
            )
            return
        try:
            scenarios = self._run_with_indicator(
                "Generating example scenarios…",
                self._generate_scenarios,
                context,
            )
        except ScenarioGenerationError as exc:
            QMessageBox.warning(self, "Scenario generation failed", str(exc))
            return
        if not scenarios:
            QMessageBox.information(
                self,
                "No scenarios available",
                "The assistant could not derive example scenarios for this prompt.",
            )
            return
        self._set_scenarios(scenarios)

    def _on_context_changed(self) -> None:
        """Auto-suggest a prompt name when none has been supplied."""
        if self._source_prompt is not None:
            return
        if self._name_generator is None:
            return
        current_name = self._name_input.text().strip()
        if current_name:
            return
        try:
            suggestion = self._generate_name(self._context_input.toPlainText())
        except NameGenerationError:
            return
        if suggestion:
            self._name_input.setText(suggestion)

    def _generate_name(self, context: str) -> str:
        """Generate name using LiteLLM when configured."""
        context = context.strip()
        if not context:
            return ""
        if self._name_generator is None:
            logger.info(
                "LiteLLM disabled (model not configured); using fallback name suggestion"
            )
            return fallback_suggest_prompt_name(context)
        try:
            return self._name_generator(context)
        except NameGenerationError as exc:
            message = str(exc).strip() or "unknown reason"
            if "not configured" in message.lower():
                logger.info(
                    "LiteLLM disabled (%s); using fallback name suggestion",
                    message,
                )
            else:
                logger.warning(
                    "Name generation failed (%s); using fallback suggestion",
                    message,
                    exc_info=exc,
                )
            return fallback_suggest_prompt_name(context)

    def _generate_description(self, context: str) -> str:
        """Generate description using LiteLLM when configured."""
        context = context.strip()
        if not context:
            return ""
        if self._description_generator is None:
            logger.info(
                "LiteLLM disabled (model not configured); using fallback description summary"
            )
            return fallback_generate_description(context)
        try:
            return self._description_generator(context)
        except DescriptionGenerationError as exc:
            message = str(exc).strip() or "unknown reason"
            if "not configured" in message.lower():
                logger.info(
                    "LiteLLM disabled (%s); using fallback description summary",
                    message,
                )
            else:
                logger.warning(
                    "Description generation failed (%s); using fallback summary",
                    message,
                    exc_info=exc,
                )
            return fallback_generate_description(context)

    def _on_refine_clicked(self) -> None:
        """Invoke the general prompt refinement workflow."""
        self._run_refinement(
            self._prompt_engineer,
            unavailable_title="Prompt refinement unavailable",
            unavailable_message="Configure LiteLLM in Settings to enable prompt engineering.",
            result_title="Prompt refined",
            indicator_message="Refining prompt…",
        )

    def _on_refine_structure_clicked(self) -> None:
        """Invoke the structure-only refinement workflow."""
        self._run_refinement(
            self._structure_refiner,
            unavailable_title="Prompt refinement unavailable",
            unavailable_message="Configure LiteLLM in Settings to enable prompt engineering.",
            result_title="Prompt structure refined",
            indicator_message="Improving prompt structure…",
        )

    def _on_execute_context_clicked(self) -> None:
        """Trigger the execute-as-context workflow from the dialog."""
        if self._source_prompt is None:
            QMessageBox.information(
                self,
                "Execute as context",
                "Save the prompt before executing it as context.",
            )
            return
        context_text = self._context_input.toPlainText().strip()
        if not context_text:
            QMessageBox.warning(
                self,
                "Prompt body required",
                "Enter prompt text before executing as context.",
            )
            return
        self.execute_context_requested.emit(self._source_prompt, context_text)

    def _run_refinement(
        self,
        handler: Callable[..., PromptRefinement] | None,
        *,
        unavailable_title: str,
        unavailable_message: str,
        result_title: str,
        indicator_message: str,
    ) -> None:
        """Execute a refinement handler and surface the summary to the user."""
        if handler is None:
            QMessageBox.information(self, unavailable_title, unavailable_message)
            return

        prompt_text = self._context_input.toPlainText()
        if not prompt_text.strip():
            QMessageBox.information(
                self,
                "Prompt required",
                "Enter prompt text before running refinement.",
            )
            return

        name = self._name_input.text().strip() or None
        description = self._description_input.toPlainText().strip() or None
        category = self._current_category_value() or None
        tags = [tag.strip() for tag in self._tags_input.text().split(",") if tag.strip()]

        try:
            result = self._run_with_indicator(
                indicator_message,
                handler,
                prompt_text,
                name=name,
                description=description,
                category=category,
                tags=tags,
            )
        except PromptEngineeringError as exc:
            QMessageBox.warning(self, "Prompt refinement failed", str(exc))
            return
        except PromptEngineeringUnavailable as exc:
            QMessageBox.warning(self, "Prompt refinement unavailable", str(exc))
            return
        except Exception as exc:  # pragma: no cover - defensive
            QMessageBox.warning(
                self,
                "Prompt refinement failed",
                f"Unexpected error: {exc}",
            )
            return

        self._context_input.setPlainText(result.improved_prompt)
        summary = self._format_refinement_summary(result)
        result_dialog = PromptRefinedDialog(summary, self, title=result_title)
        result_dialog.exec()

    @staticmethod
    def _format_refinement_summary(result: PromptRefinement) -> str:
        """Compose a human-readable summary of the refinement output."""
        summary_parts: list[str] = []
        if result.analysis:
            summary_parts.append(result.analysis)
        if result.checklist:
            checklist = "\n".join(f"• {item}" for item in result.checklist)
            summary_parts.append(f"Checklist:\n{checklist}")
        if result.warnings:
            warnings = "\n".join(f"• {item}" for item in result.warnings)
            summary_parts.append(f"Warnings:\n{warnings}")
        if summary_parts:
            return "\n\n".join(summary_parts)
        return "The prompt has been updated with the refined version."

    def _build_prompt(self) -> Prompt | None:
        """Construct a Prompt object from the dialog inputs."""
        context_text = self._context_input.toPlainText().strip()
        name = self._name_input.text().strip()
        description = self._description_input.toPlainText().strip()

        if not name and context_text:
            generated_name = self._generate_name(context_text)
            if generated_name:
                name = generated_name
                self._name_input.setText(generated_name)

        if not description and context_text:
            generated_description = self._generate_description(context_text)
            if generated_description:
                description = generated_description
                self._description_input.setPlainText(generated_description)

        category = self._current_category_value()
        author = self._author_input.text().strip() or None
        if not name or not description:
            QMessageBox.warning(self, "Missing fields", "Name and description are required.")
            return None

        language = self._language_input.text().strip() or "en"
        tags = [tag.strip() for tag in self._tags_input.text().split(",") if tag.strip()]
        context = context_text or None
        example_input = self._example_input.toPlainText().strip() or None
        example_output = self._example_output.toPlainText().strip() or None
        scenarios = self._collect_scenarios()

        if self._source_prompt is None:
            now = datetime.now(UTC)
            return Prompt(
                id=uuid.uuid4(),
                name=name,
                description=description,
                category=category or "General",
                tags=tags,
                language=language,
                context=context,
                example_input=example_input,
                example_output=example_output,
                scenarios=scenarios,
                created_at=now,
                last_modified=now,
                version="1",
                author=author,
            )

        base = self._source_prompt
        ext2_copy = deepcopy(base.ext2) if base.ext2 is not None else None
        ext5_copy = _strip_scenarios_metadata(base.ext5)
        return Prompt(
            id=base.id,
            name=name,
            description=description,
            category=category or base.category,
            tags=tags,
            language=language,
            context=context,
            example_input=example_input,
            example_output=example_output,
            scenarios=scenarios,
            last_modified=datetime.now(UTC),
            version=base.version,
            author=author,
            quality_score=base.quality_score,
            usage_count=base.usage_count,
            related_prompts=base.related_prompts,
            created_at=base.created_at,
            modified_by=base.modified_by,
            is_active=base.is_active,
            source=base.source,
            checksum=base.checksum,
            ext1=base.ext1,
            ext2=ext2_copy,
            ext3=base.ext3,
            ext4=list(base.ext4) if base.ext4 is not None else None,
            ext5=ext5_copy,
        )

    def update_source_prompt(self, prompt: Prompt) -> None:
        """Refresh the backing prompt after an in-place update."""
        self._source_prompt = prompt
        self._populate(prompt)
        self._refresh_execute_context_button()

    def _update_version_controls(self, version: str | None) -> None:
        """Refresh the version label and history button state."""
        if self._version_label is None or self._version_history_button is None:
            return
        label_text = (version or "").strip() or "Not yet saved"
        self._version_label.setText(label_text)
        history_available = (
            self._source_prompt is not None and self._version_history_handler is not None
        )
        self._version_history_button.setVisible(history_available)
        self._version_history_button.setEnabled(history_available)

    def _refresh_execute_context_button(self) -> None:
        """Toggle execute-context control availability based on dialog state."""
        if self._execute_context_button is None:
            return
        has_source_prompt = self._source_prompt is not None
        self._execute_context_button.setEnabled(has_source_prompt)
        if has_source_prompt:
            self._execute_context_button.setToolTip(
                "Run the current prompt body as context without closing the editor."
            )
        else:
            self._execute_context_button.setToolTip(
                "Available when editing a saved prompt with a prompt body."
            )

    def _on_version_history_clicked(self) -> None:
        """Open the version history dialog via the provided handler."""
        if self._version_history_handler is None or self._source_prompt is None:
            return
        self._version_history_handler(self._source_prompt)


class PromptMaintenanceDialog(QDialog):
    """Expose bulk metadata maintenance utilities."""
    maintenance_applied = Signal(str)

    def __init__(
        self,
        parent: QWidget,
        manager: PromptManager,
        *,
        category_generator: Callable[[str], str] | None = None,
        tags_generator: Callable[[str], Sequence[str]] | None = None,
    ) -> None:
        """Create the maintenance dialog with optional LiteLLM helpers."""
        super().__init__(parent)
        self._manager = manager
        self._category_generator = category_generator
        self._tags_generator = tags_generator
        self._stats_labels: dict[str, QLabel] = {}
        self._log_view: QPlainTextEdit
        self._redis_status_label: QLabel
        self._redis_connection_label: QLabel
        self._redis_stats_view: QPlainTextEdit
        self._redis_refresh_button: QPushButton
        self._tab_widget: QTabWidget
        self._chroma_status_label: QLabel
        self._chroma_path_label: QLabel
        self._chroma_stats_view: QPlainTextEdit
        self._chroma_refresh_button: QPushButton
        self._chroma_compact_button: QPushButton
        self._chroma_optimize_button: QPushButton
        self._chroma_verify_button: QPushButton
        self._storage_status_label: QLabel
        self._storage_path_label: QLabel
        self._storage_stats_view: QPlainTextEdit
        self._storage_refresh_button: QPushButton
        self._sqlite_compact_button: QPushButton
        self._sqlite_optimize_button: QPushButton
        self._sqlite_verify_button: QPushButton
        self._stats_refresh_button: QPushButton
        self._category_table: QTableWidget
        self._category_refresh_button: QPushButton
        self._reset_log_view: QPlainTextEdit
        self._settings = QSettings("PromptManager", "PromptMaintenanceDialog")
        self.setWindowTitle("Prompt Maintenance")
        self.resize(780, 360)
        self._restore_window_size()
        self._build_ui()
        self._refresh_catalogue_stats()
        self._refresh_category_health()
        self._refresh_redis_info()
        self._refresh_chroma_info()
        self._refresh_storage_info()

    def _restore_window_size(self) -> None:
        """Resize the dialog using the last persisted geometry if available."""
        width = self._settings.value("width", type=int)
        height = self._settings.value("height", type=int)
        if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
            self.resize(width, height)

    def closeEvent(self, event: QEvent) -> None:  # type: ignore[override]
        """Persist the current window size before closing."""
        self._settings.setValue("width", self.width())
        self._settings.setValue("height", self.height())
        super().closeEvent(event)

    def _build_ui(self) -> None:
        outer_layout = QVBoxLayout(self)

        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer_layout.addWidget(scroll_area, stretch=1)

        scroll_contents = QWidget(self)
        scroll_area.setWidget(scroll_contents)
        layout = QVBoxLayout(scroll_contents)

        self._tab_widget = QTabWidget(scroll_contents)
        layout.addWidget(self._tab_widget, stretch=1)

        metadata_tab = QWidget(self)
        metadata_layout = QVBoxLayout(metadata_tab)

        stats_group = QGroupBox("Prompt Catalogue Overview", metadata_tab)
        stats_layout = QGridLayout(stats_group)
        stats_layout.setContentsMargins(12, 12, 12, 12)
        stats_layout.setHorizontalSpacing(16)
        stats_layout.setVerticalSpacing(6)

        stat_rows = [
            ("total_prompts", "Total prompts"),
            ("active_prompts", "Active prompts"),
            ("inactive_prompts", "Inactive prompts"),
            ("distinct_categories", "Distinct categories"),
            ("prompts_without_category", "Prompts without category"),
            ("distinct_tags", "Distinct tags"),
            ("prompts_without_tags", "Prompts without tags"),
            ("average_tags_per_prompt", "Average tags per prompt"),
            ("stale_prompts", "Stale prompts (> 4 weeks)"),
            ("last_modified_at", "Last prompt update"),
        ]

        for row_index, (key, label_text) in enumerate(stat_rows):
            label_widget = QLabel(label_text, stats_group)
            value_widget = QLabel("—", stats_group)
            value_widget.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            stats_layout.addWidget(label_widget, row_index, 0)
            stats_layout.addWidget(value_widget, row_index, 1)
            self._stats_labels[key] = value_widget

        stats_layout.setColumnStretch(0, 1)
        stats_layout.setColumnStretch(1, 0)

        self._stats_refresh_button = QPushButton("Refresh Overview", stats_group)
        self._stats_refresh_button.clicked.connect(self._refresh_catalogue_stats)  # type: ignore[arg-type]
        stats_layout.addWidget(
            self._stats_refresh_button,
            len(stat_rows),
            0,
            1,
            2,
            alignment=Qt.AlignRight,
        )

        helper_label = QLabel(
            "Stale prompts have not been updated in the last 30 days.",
            stats_group,
        )
        helper_label.setWordWrap(True)
        stats_layout.addWidget(helper_label, len(stat_rows) + 1, 0, 1, 2)

        metadata_layout.addWidget(stats_group)

        health_group = QGroupBox("Category Health", metadata_tab)
        health_layout = QVBoxLayout(health_group)
        health_layout.setContentsMargins(12, 12, 12, 12)
        health_layout.setSpacing(8)

        self._category_table = QTableWidget(0, 5, health_group)
        self._category_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._category_table.setSelectionMode(QTableWidget.NoSelection)
        self._category_table.setHorizontalHeaderLabels(
            [
                "Category",
                "Prompts",
                "Active",
                "Success rate",
                "Last executed",
            ]
        )
        self._category_table.horizontalHeader().setStretchLastSection(True)
        self._category_table.verticalHeader().setVisible(False)
        health_layout.addWidget(self._category_table)

        self._category_refresh_button = QPushButton("Refresh Category Health", health_group)
        self._category_refresh_button.clicked.connect(self._refresh_category_health)  # type: ignore[arg-type]
        health_layout.addWidget(self._category_refresh_button, alignment=Qt.AlignRight)

        metadata_layout.addWidget(health_group)

        description = QLabel(
            "Run maintenance tasks to enrich prompt metadata. Only prompts missing the "
            "selected metadata are updated.",
            metadata_tab,
        )
        description.setWordWrap(True)
        metadata_layout.addWidget(description)

        button_container = QWidget(metadata_tab)
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(12)

        self._categories_button = QPushButton("Generate Missing Categories", metadata_tab)
        self._categories_button.clicked.connect(self._on_generate_categories_clicked)  # type: ignore[arg-type]
        self._categories_button.setEnabled(self._category_generator is not None)
        if self._category_generator is None:
            self._categories_button.setToolTip("Category suggestions are unavailable.")
        button_layout.addWidget(self._categories_button)

        self._tags_button = QPushButton("Generate Missing Tags", metadata_tab)
        self._tags_button.clicked.connect(self._on_generate_tags_clicked)  # type: ignore[arg-type]
        self._tags_button.setEnabled(self._tags_generator is not None)
        if self._tags_generator is None:
            self._tags_button.setToolTip("Tag suggestions are unavailable.")
        button_layout.addWidget(self._tags_button)

        button_layout.addStretch(1)
        metadata_layout.addWidget(button_container)

        self._log_view = QPlainTextEdit(metadata_tab)
        self._log_view.setReadOnly(True)
        self._log_view.setMinimumHeight(120)
        self._log_view.setMaximumHeight(200)
        metadata_layout.addWidget(self._log_view)

        self._tab_widget.addTab(metadata_tab, "Metadata")

        redis_tab = QWidget(self)
        redis_layout = QVBoxLayout(redis_tab)

        redis_description = QLabel(
            "Inspect the Redis cache used for prompt caching.", redis_tab
        )
        redis_description.setWordWrap(True)
        redis_layout.addWidget(redis_description)

        status_container = QWidget(redis_tab)
        status_layout = QHBoxLayout(status_container)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(12)

        self._redis_status_label = QLabel("Checking…", status_container)
        status_layout.addWidget(self._redis_status_label)

        self._redis_connection_label = QLabel("", status_container)
        self._redis_connection_label.setWordWrap(True)
        status_layout.addWidget(self._redis_connection_label, stretch=1)

        self._redis_refresh_button = QPushButton("Refresh", status_container)
        self._redis_refresh_button.clicked.connect(self._refresh_redis_info)  # type: ignore[arg-type]
        status_layout.addWidget(self._redis_refresh_button)

        redis_layout.addWidget(status_container)

        self._redis_stats_view = QPlainTextEdit(redis_tab)
        self._redis_stats_view.setReadOnly(True)
        redis_layout.addWidget(self._redis_stats_view, stretch=1)

        self._tab_widget.addTab(redis_tab, "Redis")

        chroma_tab = QWidget(self)
        chroma_layout = QVBoxLayout(chroma_tab)

        chroma_description = QLabel(
            "Review the ChromaDB vector store used for semantic search.", chroma_tab
        )
        chroma_description.setWordWrap(True)
        chroma_layout.addWidget(chroma_description)

        chroma_status_container = QWidget(chroma_tab)
        chroma_status_layout = QHBoxLayout(chroma_status_container)
        chroma_status_layout.setContentsMargins(0, 0, 0, 0)
        chroma_status_layout.setSpacing(12)

        self._chroma_status_label = QLabel("Checking…", chroma_status_container)
        chroma_status_layout.addWidget(self._chroma_status_label)

        self._chroma_path_label = QLabel("", chroma_status_container)
        self._chroma_path_label.setWordWrap(True)
        chroma_status_layout.addWidget(self._chroma_path_label, stretch=1)

        self._chroma_refresh_button = QPushButton("Refresh", chroma_status_container)
        self._chroma_refresh_button.clicked.connect(self._refresh_chroma_info)  # type: ignore[arg-type]
        chroma_status_layout.addWidget(self._chroma_refresh_button)

        chroma_layout.addWidget(chroma_status_container)

        self._chroma_stats_view = QPlainTextEdit(chroma_tab)
        self._chroma_stats_view.setReadOnly(True)
        chroma_layout.addWidget(self._chroma_stats_view, stretch=1)

        chroma_actions_container = QWidget(chroma_tab)
        chroma_actions_layout = QHBoxLayout(chroma_actions_container)
        chroma_actions_layout.setContentsMargins(0, 0, 0, 0)
        chroma_actions_layout.setSpacing(12)

        self._chroma_compact_button = QPushButton(
            "Compact Persistent Store",
            chroma_actions_container,
        )
        self._chroma_compact_button.setToolTip(
            "Reclaim disk space by vacuuming the Chroma SQLite store."
        )
        self._chroma_compact_button.clicked.connect(self._on_chroma_compact_clicked)  # type: ignore[arg-type]
        chroma_actions_layout.addWidget(self._chroma_compact_button)

        self._chroma_optimize_button = QPushButton(
            "Optimize Persistent Store",
            chroma_actions_container,
        )
        self._chroma_optimize_button.setToolTip(
            "Refresh query statistics to improve Chroma performance."
        )
        self._chroma_optimize_button.clicked.connect(self._on_chroma_optimize_clicked)  # type: ignore[arg-type]
        chroma_actions_layout.addWidget(self._chroma_optimize_button)

        self._chroma_verify_button = QPushButton(
            "Verify Index Integrity",
            chroma_actions_container,
        )
        self._chroma_verify_button.setToolTip(
            "Run integrity checks against the Chroma index files."
        )
        self._chroma_verify_button.clicked.connect(self._on_chroma_verify_clicked)  # type: ignore[arg-type]
        chroma_actions_layout.addWidget(self._chroma_verify_button)

        chroma_actions_layout.addStretch(1)

        chroma_layout.addWidget(chroma_actions_container)

        self._tab_widget.addTab(chroma_tab, "ChromaDB")

        storage_tab = QWidget(self)
        storage_layout = QVBoxLayout(storage_tab)

        storage_description = QLabel(
            "Inspect the SQLite repository backing prompt storage.", storage_tab
        )
        storage_description.setWordWrap(True)
        storage_layout.addWidget(storage_description)

        storage_status_container = QWidget(storage_tab)
        storage_status_layout = QHBoxLayout(storage_status_container)
        storage_status_layout.setContentsMargins(0, 0, 0, 0)
        storage_status_layout.setSpacing(12)

        self._storage_status_label = QLabel("Checking…", storage_status_container)
        storage_status_layout.addWidget(self._storage_status_label)

        self._storage_path_label = QLabel("", storage_status_container)
        self._storage_path_label.setWordWrap(True)
        storage_status_layout.addWidget(self._storage_path_label, stretch=1)

        self._storage_refresh_button = QPushButton("Refresh", storage_status_container)
        self._storage_refresh_button.clicked.connect(self._refresh_storage_info)  # type: ignore[arg-type]
        storage_status_layout.addWidget(self._storage_refresh_button)

        storage_layout.addWidget(storage_status_container)

        self._storage_stats_view = QPlainTextEdit(storage_tab)
        self._storage_stats_view.setReadOnly(True)
        storage_layout.addWidget(self._storage_stats_view, stretch=1)

        storage_actions_container = QWidget(storage_tab)
        storage_actions_layout = QHBoxLayout(storage_actions_container)
        storage_actions_layout.setContentsMargins(0, 0, 0, 0)
        storage_actions_layout.setSpacing(12)

        self._sqlite_compact_button = QPushButton(
            "Compact Database",
            storage_actions_container,
        )
        self._sqlite_compact_button.setToolTip(
            "Run VACUUM on the prompt database to reclaim space."
        )
        self._sqlite_compact_button.clicked.connect(self._on_sqlite_compact_clicked)  # type: ignore[arg-type]
        storage_actions_layout.addWidget(self._sqlite_compact_button)

        self._sqlite_optimize_button = QPushButton("Optimize Database", storage_actions_container)
        self._sqlite_optimize_button.setToolTip("Refresh SQLite statistics for prompt lookups.")
        self._sqlite_optimize_button.clicked.connect(self._on_sqlite_optimize_clicked)  # type: ignore[arg-type]
        storage_actions_layout.addWidget(self._sqlite_optimize_button)

        self._sqlite_verify_button = QPushButton(
            "Verify Index Integrity",
            storage_actions_container,
        )
        self._sqlite_verify_button.setToolTip(
            "Run integrity checks against the prompt database indexes."
        )
        self._sqlite_verify_button.clicked.connect(self._on_sqlite_verify_clicked)  # type: ignore[arg-type]
        storage_actions_layout.addWidget(self._sqlite_verify_button)

        storage_actions_layout.addStretch(1)

        storage_layout.addWidget(storage_actions_container)

        self._tab_widget.addTab(storage_tab, "SQLite")

        reset_tab = QWidget(self)
        reset_layout = QVBoxLayout(reset_tab)

        reset_intro = QLabel(
            (
                "Use these actions to clear application data while leaving configuration "
                "and settings untouched."
            ),
            reset_tab,
        )
        reset_intro.setWordWrap(True)
        reset_layout.addWidget(reset_intro)

        reset_warning = QLabel(
            (
                "<b>Warning:</b> these operations permanently delete existing prompts, "
                "histories, and embeddings."
            ),
            reset_tab,
        )
        reset_warning.setWordWrap(True)
        reset_layout.addWidget(reset_warning)

        reset_buttons_container = QWidget(reset_tab)
        reset_buttons_layout = QVBoxLayout(reset_buttons_container)
        reset_buttons_layout.setContentsMargins(0, 0, 0, 0)
        reset_buttons_layout.setSpacing(8)

        snapshot_button = QPushButton("Create Backup Snapshot", reset_buttons_container)
        snapshot_button.setToolTip(
            "Zip the SQLite database, Chroma store, and manifest before running resets."
        )
        snapshot_button.clicked.connect(self._on_snapshot_clicked)  # type: ignore[arg-type]
        reset_buttons_layout.addWidget(snapshot_button)

        reset_sqlite_button = QPushButton("Clear Prompt Database", reset_buttons_container)
        reset_sqlite_button.setToolTip("Delete all prompts and execution history from SQLite.")
        reset_sqlite_button.clicked.connect(self._on_reset_prompts_clicked)  # type: ignore[arg-type]
        reset_buttons_layout.addWidget(reset_sqlite_button)

        reset_chroma_button = QPushButton("Clear Embedding Store", reset_buttons_container)
        reset_chroma_button.setToolTip(
            "Remove all vectors from the ChromaDB collection used for semantic search."
        )
        reset_chroma_button.clicked.connect(self._on_reset_chroma_clicked)  # type: ignore[arg-type]
        reset_buttons_layout.addWidget(reset_chroma_button)

        reset_all_button = QPushButton("Reset Application Data", reset_buttons_container)
        reset_all_button.setToolTip(
            "Clear prompts, histories, embeddings, and usage logs in one step. "
            "Settings remain unchanged."
        )
        reset_all_button.clicked.connect(self._on_reset_application_clicked)  # type: ignore[arg-type]
        reset_buttons_layout.addWidget(reset_all_button)

        reset_layout.addWidget(reset_buttons_container)

        self._reset_log_view = QPlainTextEdit(reset_tab)
        self._reset_log_view.setReadOnly(True)
        reset_layout.addWidget(self._reset_log_view, stretch=1)

        self._tab_widget.addTab(reset_tab, "Data Reset")

        self._buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
        self._buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        outer_layout.addWidget(self._buttons)

    def _set_stat_value(self, key: str, value: str) -> None:
        label = self._stats_labels.get(key)
        if label is not None:
            label.setText(value)

    @staticmethod
    def _format_timestamp(value: datetime | None) -> str:
        if value is None:
            return "—"
        return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")

    def _refresh_catalogue_stats(self) -> None:
        try:
            stats = self._manager.get_prompt_catalogue_stats()
        except PromptManagerError as exc:
            logger.warning("Prompt catalogue stats refresh failed", exc_info=True)
            self._append_log(f"Failed to load prompt statistics: {exc}")
            for label in self._stats_labels.values():
                label.setText("—")
            return

        self._set_stat_value("total_prompts", str(stats.total_prompts))
        self._set_stat_value("active_prompts", str(stats.active_prompts))
        self._set_stat_value("inactive_prompts", str(stats.inactive_prompts))
        self._set_stat_value("distinct_categories", str(stats.distinct_categories))
        self._set_stat_value("prompts_without_category", str(stats.prompts_without_category))
        self._set_stat_value("distinct_tags", str(stats.distinct_tags))
        self._set_stat_value("prompts_without_tags", str(stats.prompts_without_tags))
        self._set_stat_value(
            "average_tags_per_prompt",
            f"{stats.average_tags_per_prompt:.2f}",
        )
        self._set_stat_value("stale_prompts", str(stats.stale_prompts))
        self._set_stat_value(
            "last_modified_at",
            self._format_timestamp(stats.last_modified_at),
        )

    def _refresh_category_health(self) -> None:
        try:
            health_entries = self._manager.get_category_health()
        except PromptManagerError as exc:
            QMessageBox.warning(self, "Category health", str(exc))
            self._category_table.setRowCount(0)
            return

        self._category_table.setRowCount(len(health_entries))
        for row, entry in enumerate(health_entries):
            success_text = "—"
            if entry.success_rate is not None:
                success_text = f"{entry.success_rate * 100:.1f}%"
            executed_text = self._format_timestamp(entry.last_executed_at)
            self._category_table.setItem(row, 0, QTableWidgetItem(entry.label))
            self._category_table.setItem(row, 1, QTableWidgetItem(str(entry.total_prompts)))
            self._category_table.setItem(row, 2, QTableWidgetItem(str(entry.active_prompts)))
            self._category_table.setItem(row, 3, QTableWidgetItem(success_text))
            self._category_table.setItem(row, 4, QTableWidgetItem(executed_text or "—"))

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now(UTC).strftime("%H:%M:%S")
        self._log_view.appendPlainText(f"[{timestamp}] {message}")

    def _append_reset_log(self, message: str) -> None:
        timestamp = datetime.now(UTC).strftime("%H:%M:%S")
        self._reset_log_view.appendPlainText(f"[{timestamp}] {message}")

    def _confirm_destructive_action(self, prompt: str) -> bool:
        result = QMessageBox.question(
            self,
            "Confirm Data Reset",
            f"{prompt}\n\nThis action cannot be undone. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return result == QMessageBox.Yes

    def _collect_prompts(self) -> list[Prompt]:
        try:
            return self._manager.repository.list()
        except RepositoryError as exc:
            self._append_log(f"Unable to load prompts: {exc}")
            return []

    @staticmethod
    def _prompt_context(prompt: Prompt) -> str:
        return (prompt.context or prompt.description or prompt.example_input or "").strip()

    def _on_generate_categories_clicked(self) -> None:
        if self._category_generator is None:
            self._append_log("Category generator is unavailable.")
            return
        prompts = self._collect_prompts()
        if not prompts:
            return
        updated = 0
        for prompt in prompts:
            category_value = (prompt.category or "").strip()
            if category_value and category_value.lower() != "general":
                continue
            context = self._prompt_context(prompt)
            if not context:
                continue
            try:
                suggestion = (self._category_generator(context) or "").strip()
            except Exception as exc:  # noqa: BLE001
                self._append_log(f"Failed to suggest category for '{prompt.name}': {exc}")
                continue
            if not suggestion:
                continue
            updated_prompt = replace(
                prompt,
                category=suggestion,
                last_modified=datetime.now(UTC),
            )
            try:
                self._manager.update_prompt(updated_prompt)
            except PromptManagerError as exc:
                self._append_log(f"Unable to update '{prompt.name}': {exc}")
                continue
            updated += 1
            self._append_log(f"Set category '{suggestion}' for '{prompt.name}'.")
        self._append_log(f"Category task completed. Updated {updated} prompt(s).")
        if updated:
            self.maintenance_applied.emit(f"Generated categories for {updated} prompt(s).")

    def _on_generate_tags_clicked(self) -> None:
        if self._tags_generator is None:
            self._append_log("Tag generator is unavailable.")
            return
        prompts = self._collect_prompts()
        if not prompts:
            return
        updated = 0
        for prompt in prompts:
            existing_tags = [tag.strip() for tag in (prompt.tags or []) if tag.strip()]
            if existing_tags:
                continue
            context = self._prompt_context(prompt)
            if not context:
                continue
            try:
                suggestions = self._tags_generator(context) or []
            except Exception as exc:  # noqa: BLE001
                self._append_log(f"Failed to suggest tags for '{prompt.name}': {exc}")
                continue
            tags = [tag.strip() for tag in suggestions if str(tag).strip()]
            if not tags:
                continue
            updated_prompt = replace(
                prompt,
                tags=tags,
                last_modified=datetime.now(UTC),
            )
            try:
                self._manager.update_prompt(updated_prompt)
            except PromptManagerError as exc:
                self._append_log(f"Unable to update '{prompt.name}': {exc}")
                continue
            updated += 1
            self._append_log(f"Assigned tags {tags} to '{prompt.name}'.")
        self._append_log(f"Tag task completed. Updated {updated} prompt(s).")
        if updated:
            self.maintenance_applied.emit(f"Generated tags for {updated} prompt(s).")

    def _on_snapshot_clicked(self) -> None:
        """Prompt for a destination and create a maintenance snapshot."""
        default_name = datetime.now(UTC).strftime("prompt-manager-snapshot-%Y%m%d-%H%M%S.zip")
        default_path = str(Path.home() / default_name)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Maintenance Snapshot",
            default_path,
            "Zip Archives (*.zip)",
        )
        if not path:
            return

        self._append_reset_log("Creating maintenance snapshot…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            archive_path = self._manager.create_data_snapshot(path)
        except PromptManagerError as exc:
            QMessageBox.critical(self, "Snapshot failed", str(exc))
            self._append_reset_log(f"Snapshot failed: {exc}")
        else:
            self._append_reset_log(f"Snapshot saved to {archive_path}")
            QMessageBox.information(
                self,
                "Snapshot created",
                f"Backup stored at:\n{archive_path}",
            )
        finally:
            QApplication.restoreOverrideCursor()

    def _on_reset_prompts_clicked(self) -> None:
        """Clear the SQLite prompt repository."""
        if not self._confirm_destructive_action(
            "Clear the prompt database and execution history?"
        ):
            return
        try:
            self._manager.reset_prompt_repository()
        except PromptManagerError as exc:
            QMessageBox.critical(self, "Reset failed", str(exc))
            self._append_reset_log(f"Prompt database reset failed: {exc}")
            return
        self._append_reset_log("Prompt database cleared.")
        QMessageBox.information(
            self,
            "Prompt database cleared",
            "All prompts and execution history have been removed.",
        )
        self._refresh_catalogue_stats()
        self._refresh_storage_info()
        self.maintenance_applied.emit("Prompt database cleared.")

    def _on_reset_chroma_clicked(self) -> None:
        """Clear the ChromaDB vector store."""
        if not self._confirm_destructive_action(
            "Remove all embeddings from the ChromaDB vector store?"
        ):
            return
        try:
            self._manager.reset_vector_store()
        except PromptManagerError as exc:
            QMessageBox.critical(self, "Reset failed", str(exc))
            self._append_reset_log(f"Embedding store reset failed: {exc}")
            return
        self._append_reset_log("Chroma vector store cleared.")
        QMessageBox.information(
            self,
            "Embedding store cleared",
            "All stored embeddings have been removed.",
        )
        self._refresh_chroma_info()
        self.maintenance_applied.emit("Embedding store cleared.")

    def _set_chroma_actions_busy(self, busy: bool) -> None:
        """Disable Chroma maintenance buttons while a task is running."""
        if not busy:
            return
        for button in (
            self._chroma_refresh_button,
            self._chroma_compact_button,
            self._chroma_optimize_button,
            self._chroma_verify_button,
        ):
            button.setEnabled(False)

    def _on_chroma_compact_clicked(self) -> None:
        """Run VACUUM maintenance on the Chroma persistent store."""
        self._set_chroma_actions_busy(True)
        try:
            self._manager.compact_vector_store()
        except PromptManagerError as exc:
            QMessageBox.critical(self, "Compaction failed", str(exc))
            return
        else:
            QMessageBox.information(
                self,
                "Chroma store compacted",
                "The persistent Chroma store has been vacuumed and reclaimed space.",
            )
        finally:
            self._refresh_chroma_info()

    def _on_chroma_optimize_clicked(self) -> None:
        """Refresh query statistics for the Chroma persistent store."""
        self._set_chroma_actions_busy(True)
        try:
            self._manager.optimize_vector_store()
        except PromptManagerError as exc:
            QMessageBox.critical(self, "Optimization failed", str(exc))
            return
        else:
            QMessageBox.information(
                self,
                "Chroma store optimized",
                "Chroma query statistics have been refreshed for better performance.",
            )
        finally:
            self._refresh_chroma_info()

    def _on_chroma_verify_clicked(self) -> None:
        """Verify integrity of the Chroma persistent store."""
        self._set_chroma_actions_busy(True)
        try:
            summary = self._manager.verify_vector_store()
        except PromptManagerError as exc:
            QMessageBox.critical(self, "Verification failed", str(exc))
            return
        else:
            message = summary or "Chroma store integrity verified successfully."
            QMessageBox.information(self, "Chroma store verified", message)
        finally:
            self._refresh_chroma_info()

    def _set_storage_actions_busy(self, busy: bool) -> None:
        """Disable SQLite maintenance buttons while a task is running."""
        if not busy:
            return
        for button in (
            self._storage_refresh_button,
            self._sqlite_compact_button,
            self._sqlite_optimize_button,
            self._sqlite_verify_button,
        ):
            button.setEnabled(False)

    def _on_sqlite_compact_clicked(self) -> None:
        """Run VACUUM on the prompt repository."""
        self._set_storage_actions_busy(True)
        try:
            self._manager.compact_repository()
        except PromptManagerError as exc:
            QMessageBox.critical(self, "Compaction failed", str(exc))
            return
        else:
            QMessageBox.information(
                self,
                "Prompt database compacted",
                "The SQLite repository has been vacuumed and reclaimed space.",
            )
        finally:
            self._refresh_storage_info()

    def _on_sqlite_optimize_clicked(self) -> None:
        """Refresh SQLite statistics for the prompt repository."""
        self._set_storage_actions_busy(True)
        try:
            self._manager.optimize_repository()
        except PromptManagerError as exc:
            QMessageBox.critical(self, "Optimization failed", str(exc))
            return
        else:
            QMessageBox.information(
                self,
                "Prompt database optimized",
                "SQLite statistics have been refreshed for prompt lookups.",
            )
        finally:
            self._refresh_storage_info()

    def _on_sqlite_verify_clicked(self) -> None:
        """Verify integrity of the prompt repository."""
        self._set_storage_actions_busy(True)
        try:
            summary = self._manager.verify_repository()
        except PromptManagerError as exc:
            QMessageBox.critical(self, "Verification failed", str(exc))
            return
        else:
            message = summary or "SQLite repository integrity verified successfully."
            QMessageBox.information(self, "Prompt database verified", message)
        finally:
            self._refresh_storage_info()

    def _on_reset_application_clicked(self) -> None:
        """Clear prompts, embeddings, and usage logs."""
        if not self._confirm_destructive_action(
            "Reset all application data (prompts, history, embeddings, and logs)?"
        ):
            return
        try:
            self._manager.reset_application_data(clear_logs=True)
        except PromptManagerError as exc:
            QMessageBox.critical(self, "Reset failed", str(exc))
            self._append_reset_log(f"Application reset failed: {exc}")
            return
        self._append_reset_log("Application data reset completed.")
        QMessageBox.information(
            self,
            "Application data reset",
            "Prompt data, embeddings, and usage logs have been cleared.",
        )
        self._refresh_catalogue_stats()
        self._refresh_storage_info()
        self._refresh_chroma_info()
        self.maintenance_applied.emit("Application data reset.")

    def _refresh_redis_info(self) -> None:
        """Update the Redis tab with the latest cache status."""
        details = self._manager.get_redis_details()
        enabled = details.get("enabled", False)
        if not enabled:
            self._redis_status_label.setText("Redis caching is disabled.")
            self._redis_connection_label.setText("")
            self._redis_stats_view.setPlainText("")
            self._redis_refresh_button.setEnabled(False)
            return

        self._redis_refresh_button.setEnabled(True)

        status = details.get("status", "unknown").capitalize()
        if details.get("error"):
            status = f"Error: {details['error']}"
        self._redis_status_label.setText(f"Status: {status}")

        connection = details.get("connection", {})
        connection_parts = []
        if connection.get("host"):
            host = connection["host"]
            port = connection.get("port")
            if port is not None:
                connection_parts.append(f"{host}:{port}")
            else:
                connection_parts.append(str(host))
        if connection.get("database") is not None:
            connection_parts.append(f"DB {connection['database']}")
        if connection.get("ssl"):
            connection_parts.append("SSL")
        if not connection_parts:
            self._redis_connection_label.setText("")
        else:
            self._redis_connection_label.setText("Connection: " + ", ".join(connection_parts))

        stats = details.get("stats", {})
        lines: list[str] = []
        for key, label in (
            ("keys", "Keys"),
            ("used_memory_human", "Used memory"),
            ("used_memory_peak_human", "Peak memory"),
            ("maxmemory_human", "Configured max memory"),
            ("hits", "Keyspace hits"),
            ("misses", "Keyspace misses"),
            ("hit_rate", "Hit rate (%)"),
        ):
            if stats.get(key) is not None:
                lines.append(f"{label}: {stats[key]}")
        if not lines and details.get("error"):
            lines.append(details["error"])
        elif not lines and stats.get("info_error"):
            lines.append(f"Unable to fetch stats: {stats['info_error']}")
        redis_text = "\n".join(lines) if lines else "No Redis statistics available."
        self._redis_stats_view.setPlainText(redis_text)

    def _refresh_chroma_info(self) -> None:
        """Update the ChromaDB tab with vector store information."""
        details = self._manager.get_chroma_details()
        enabled = details.get("enabled", False)
        path = details.get("path") or ""
        collection = details.get("collection") or ""
        if not enabled:
            self._chroma_status_label.setText("ChromaDB is not initialised.")
            self._chroma_path_label.setText(f"Path: {path}" if path else "")
            self._chroma_stats_view.setPlainText("")
            self._chroma_refresh_button.setEnabled(False)
            self._chroma_compact_button.setEnabled(False)
            self._chroma_optimize_button.setEnabled(False)
            self._chroma_verify_button.setEnabled(False)
            return

        self._chroma_refresh_button.setEnabled(True)

        status = details.get("status", "unknown").capitalize()
        if details.get("error"):
            status = f"Error: {details['error']}"
        self._chroma_status_label.setText(f"Status: {status}")

        has_error = bool(details.get("error"))
        self._chroma_compact_button.setEnabled(not has_error)
        self._chroma_optimize_button.setEnabled(not has_error)
        self._chroma_verify_button.setEnabled(not has_error)

        path_parts = []
        if path:
            path_parts.append(f"Path: {path}")
        if collection:
            path_parts.append(f"Collection: {collection}")
        self._chroma_path_label.setText(" | ".join(path_parts))

        stats = details.get("stats", {})
        lines: list[str] = []
        for key, label in (
            ("documents", "Documents"),
            ("disk_usage_bytes", "Disk usage (bytes)"),
        ):
            value = stats.get(key)
            if value is not None:
                lines.append(f"{label}: {value}")
        chroma_text = "\n".join(lines) if lines else "No ChromaDB statistics available."
        self._chroma_stats_view.setPlainText(chroma_text)

    def _refresh_storage_info(self) -> None:
        """Update the SQLite tab with repository information."""
        repository = self._manager.repository
        db_path_obj = getattr(repository, "_db_path", None)
        if isinstance(db_path_obj, Path):
            db_path = str(db_path_obj)
        else:
            db_path = str(db_path_obj) if db_path_obj is not None else ""

        self._storage_path_label.setText(f"Path: {db_path}" if db_path else "Path: unknown")
        self._storage_refresh_button.setEnabled(True)

        stats_lines: list[str] = []
        healthy = True

        size_bytes = None
        if db_path:
            try:
                path_obj = Path(db_path)
                if path_obj.exists():
                    size_bytes = path_obj.stat().st_size
                else:
                    healthy = False
                    stats_lines.append("Database file not found.")
            except OSError as exc:
                healthy = False
                stats_lines.append(f"File size: error ({exc})")
        else:
            healthy = False

        if size_bytes is not None:
            stats_lines.append(f"File size: {size_bytes} bytes")

        try:
            prompt_count = len(repository.list())
            stats_lines.append(f"Prompts: {prompt_count}")
        except RepositoryError as exc:
            healthy = False
            stats_lines.append(f"Prompts: error ({exc})")

        try:
            execution_count = len(repository.list_executions())
            stats_lines.append(f"Executions: {execution_count}")
        except RepositoryError as exc:
            healthy = False
            stats_lines.append(f"Executions: error ({exc})")

        storage_text = "\n".join(stats_lines) if stats_lines else "No SQLite statistics available."
        self._storage_stats_view.setPlainText(storage_text)

        if healthy:
            self._storage_status_label.setText("Status: ready")
        else:
            self._storage_status_label.setText("Status: unavailable")

        self._sqlite_compact_button.setEnabled(healthy)
        self._sqlite_optimize_button.setEnabled(healthy)
        self._sqlite_verify_button.setEnabled(healthy)


class CategoryEditorDialog(QDialog):
    """Collect category details for creation or editing workflows."""
    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        category: PromptCategory | None = None,
    ) -> None:
        super().__init__(parent)
        self._source = category
        self._payload: dict[str, object] = {}
        self.setWindowTitle("Add Category" if category is None else "Edit Category")
        self.resize(460, 360)
        self._build_ui()
        if category is not None:
            self._populate(category)

    @property
    def payload(self) -> dict[str, object]:
        """Return the collected form data."""
        return dict(self._payload)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._label_input = QLineEdit(self)
        self._label_input.setPlaceholderText("Documentation, Refactoring…")
        form.addRow("Label*", self._label_input)

        self._slug_input = QLineEdit(self)
        self._slug_input.setPlaceholderText("documentation, refactoring…")
        form.addRow("Slug*", self._slug_input)

        self._description_input = QLineEdit(self)
        self._description_input.setPlaceholderText("Short sentence describing the scope.")
        form.addRow("Description", self._description_input)

        self._parent_input = QLineEdit(self)
        self._parent_input.setPlaceholderText("Parent category slug (optional)")
        form.addRow("Parent slug", self._parent_input)

        self._color_input = QLineEdit(self)
        self._color_input.setPlaceholderText("#RRGGBB")
        form.addRow("Colour", self._color_input)

        self._icon_input = QLineEdit(self)
        self._icon_input.setPlaceholderText("mdi-icon-name")
        form.addRow("Icon", self._icon_input)

        self._min_quality_input = QLineEdit(self)
        self._min_quality_input.setPlaceholderText("Minimum quality score (optional)")
        form.addRow("Min quality", self._min_quality_input)

        self._tags_input = QLineEdit(self)
        self._tags_input.setPlaceholderText("Comma-separated default tags")
        form.addRow("Default tags", self._tags_input)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self._on_accept)  # type: ignore[arg-type]
        buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        layout.addWidget(buttons)

    def _populate(self, category: PromptCategory) -> None:
        """Populate the form with an existing category."""
        self._label_input.setText(category.label)
        self._slug_input.setText(category.slug)
        self._slug_input.setReadOnly(True)
        self._description_input.setText(category.description)
        self._parent_input.setText(category.parent_slug or "")
        self._color_input.setText(category.color or "")
        self._icon_input.setText(category.icon or "")
        min_quality = "" if category.min_quality is None else str(category.min_quality)
        self._min_quality_input.setText(min_quality)
        self._tags_input.setText(", ".join(category.default_tags))

    def _on_accept(self) -> None:
        """Validate inputs and persist them in payload."""
        label = self._label_input.text().strip()
        if not label:
            QMessageBox.warning(self, "Invalid category", "Label is required.")
            return
        slug_source = self._slug_input.text().strip() or label
        slug = slugify_category(slug_source)
        if not slug:
            QMessageBox.warning(self, "Invalid category", "Slug is required.")
            return
        description = self._description_input.text().strip() or label
        min_quality_text = self._min_quality_input.text().strip()
        min_quality: float | None = None
        if min_quality_text:
            try:
                min_quality = float(min_quality_text)
            except ValueError:
                QMessageBox.warning(self, "Invalid category", "Minimum quality must be a number.")
                return
        tags = [
            tag.strip()
            for tag in self._tags_input.text().split(",")
            if tag.strip()
        ]
        self._payload = {
            "label": label,
            "slug": slug,
            "description": description,
            "parent_slug": self._parent_input.text().strip() or None,
            "color": self._color_input.text().strip() or None,
            "icon": self._icon_input.text().strip() or None,
            "min_quality": min_quality,
            "default_tags": tags,
        }
        self.accept()


class CategoryManagerDialog(QDialog):
    """Provide CRUD workflows for prompt categories."""
    def __init__(self, manager: PromptManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._manager = manager
        self._has_changes = False
        self._categories: list[PromptCategory] = []
        self._settings = QSettings("PromptManager", "CategoryManagerDialog")
        self.setWindowTitle("Manage Categories")
        self.resize(760, 520)
        self._build_ui()
        self._restore_window_size()
        self._load_categories()

    @property
    def has_changes(self) -> bool:
        """Return True when categories were created or updated."""
        return self._has_changes

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        helper = QLabel(
            "Create, edit, and archive prompt categories. Changes apply immediately "
            "and update linked prompts.",
            self,
        )
        helper.setWordWrap(True)
        layout.addWidget(helper)

        self._table = QTableWidget(self)
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels(
            ["Label", "Slug", "Description", "Parent", "Min quality", "Status"]
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.itemSelectionChanged.connect(self._update_button_states)  # type: ignore[arg-type]
        layout.addWidget(self._table, stretch=1)

        button_row = QHBoxLayout()
        self._add_button = QPushButton("Add", self)
        self._add_button.clicked.connect(self._on_add_category)  # type: ignore[arg-type]
        button_row.addWidget(self._add_button)

        self._edit_button = QPushButton("Edit", self)
        self._edit_button.clicked.connect(self._on_edit_category)  # type: ignore[arg-type]
        button_row.addWidget(self._edit_button)

        self._toggle_button = QPushButton("Archive", self)
        self._toggle_button.clicked.connect(self._on_toggle_category)  # type: ignore[arg-type]
        button_row.addWidget(self._toggle_button)

        button_row.addStretch(1)

        self._refresh_button = QPushButton("Refresh", self)
        self._refresh_button.clicked.connect(self._on_refresh_clicked)  # type: ignore[arg-type]
        button_row.addWidget(self._refresh_button)
        layout.addLayout(button_row)

        close_buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
        close_buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        layout.addWidget(close_buttons)

        self._update_button_states()

    def _restore_window_size(self) -> None:
        width = self._settings.value("width", type=int)
        height = self._settings.value("height", type=int)
        if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
            self.resize(width, height)

    def closeEvent(self, event: QEvent) -> None:  # type: ignore[override]
        self._settings.setValue("width", self.width())
        self._settings.setValue("height", self.height())
        super().closeEvent(event)

    def _load_categories(self) -> None:
        """Populate table with repository categories."""
        try:
            categories = self._manager.list_categories(include_archived=True)
        except CategoryStorageError as exc:
            QMessageBox.critical(self, "Unable to load categories", str(exc))
            categories = []
        self._categories = categories
        self._table.setRowCount(len(categories))
        for row, category in enumerate(categories):
            self._table.setItem(row, 0, QTableWidgetItem(category.label))
            self._table.setItem(row, 1, QTableWidgetItem(category.slug))
            self._table.setItem(row, 2, QTableWidgetItem(category.description))
            self._table.setItem(row, 3, QTableWidgetItem(category.parent_slug or "—"))
            min_quality = "—" if category.min_quality is None else f"{category.min_quality:.1f}"
            self._table.setItem(row, 4, QTableWidgetItem(min_quality))
            status = "Active" if category.is_active else "Archived"
            self._table.setItem(row, 5, QTableWidgetItem(status))
        self._table.resizeColumnsToContents()
        self._update_button_states()

    def _selected_category(self) -> PromptCategory | None:
        """Return the currently selected category."""
        selected_rows = self._table.selectionModel().selectedRows()
        if not selected_rows:
            return None
        index = selected_rows[0].row()
        if index < 0 or index >= len(self._categories):
            return None
        return self._categories[index]

    def _update_button_states(self) -> None:
        """Enable/disable actions based on selection."""
        category = self._selected_category()
        has_selection = category is not None
        self._edit_button.setEnabled(has_selection)
        self._toggle_button.setEnabled(has_selection)
        if category is None:
            self._toggle_button.setText("Archive")
        else:
            self._toggle_button.setText("Archive" if category.is_active else "Activate")

    def _on_refresh_clicked(self) -> None:
        """Reload categories from the registry."""
        try:
            self._manager.refresh_categories()
        except CategoryStorageError as exc:
            QMessageBox.warning(self, "Refresh failed", str(exc))
        self._load_categories()

    def _on_add_category(self) -> None:
        """Open the category editor and persist a new category."""
        dialog = CategoryEditorDialog(self)
        if not dialog.exec():
            return
        data = dialog.payload
        try:
            self._manager.create_category(**data)
        except CategoryStorageError as exc:
            QMessageBox.critical(self, "Create failed", str(exc))
            return
        self._has_changes = True
        self._load_categories()

    def _on_edit_category(self) -> None:
        """Edit the selected category."""
        category = self._selected_category()
        if category is None:
            return
        dialog = CategoryEditorDialog(self, category=category)
        if not dialog.exec():
            return
        data = dialog.payload
        try:
            self._manager.update_category(
                category.slug,
                label=data["label"],
                description=data["description"],
                parent_slug=data["parent_slug"],
                color=data["color"],
                icon=data["icon"],
                min_quality=data["min_quality"],
                default_tags=data["default_tags"],
            )
        except CategoryNotFoundError as exc:
            QMessageBox.warning(self, "Edit failed", str(exc))
            return
        except CategoryStorageError as exc:
            QMessageBox.critical(self, "Edit failed", str(exc))
            return
        self._has_changes = True
        self._load_categories()

    def _on_toggle_category(self) -> None:
        """Archive or activate the selected category."""
        category = self._selected_category()
        if category is None:
            return
        try:
            self._manager.set_category_active(category.slug, not category.is_active)
        except CategoryNotFoundError as exc:
            QMessageBox.warning(self, "Update failed", str(exc))
            return
        except CategoryStorageError as exc:
            QMessageBox.critical(self, "Update failed", str(exc))
            return
        self._has_changes = True
        self._load_categories()


class SaveResultDialog(QDialog):
    """Collect optional notes before persisting or updating a prompt execution."""
    def __init__(
        self,
        parent: QWidget | None,
        *,
        prompt_name: str,
        default_text: str = "",
        max_chars: int = 400,
        button_text: str = "Save",
        enable_rating: bool = True,
        initial_rating: float | None = None,
    ) -> None:
        """Configure the result dialog with labels, defaults, and rating controls."""
        super().__init__(parent)
        self._max_chars = max_chars
        self._summary = ""
        self._rating: int | None = (
            int(initial_rating) if initial_rating is not None else None
        )
        self._enable_rating = enable_rating
        self.setWindowTitle(f"{button_text} Result — {prompt_name}")
        self._build_ui(default_text, button_text)

    @property
    def note(self) -> str:
        """Return the trimmed note content."""
        return self._summary

    @property
    def rating(self) -> int | None:
        """Return the selected rating, if any."""
        return self._rating

    def _build_ui(self, default_text: str, button_text: str) -> None:
        layout = QVBoxLayout(self)

        message = QLabel(
            (
                "Add an optional summary or notes for this execution. "
                "The note will be stored with the history entry."
            ),
            self,
        )
        message.setWordWrap(True)
        layout.addWidget(message)

        self._note_input = QPlainTextEdit(self)
        self._note_input.setPlaceholderText("Optional summary / notes…")
        if default_text:
            snippet = default_text.strip()
            if len(snippet) > self._max_chars:
                snippet = snippet[: self._max_chars - 3].rstrip() + "..."
            self._note_input.setPlainText(snippet)
        layout.addWidget(self._note_input)

        if self._enable_rating:
            rating_layout = QHBoxLayout()
            rating_label = QLabel("Rating (1-10):", self)
            self._rating_input = QComboBox(self)
            self._rating_input.addItem("No rating", None)
            for value in range(1, 11):
                self._rating_input.addItem(str(value), value)
            if self._rating is not None:
                index = self._rating_input.findData(self._rating)
                if index >= 0:
                    self._rating_input.setCurrentIndex(index)
            rating_layout.addWidget(rating_label)
            rating_layout.addWidget(self._rating_input)
            layout.addLayout(rating_layout)
        else:
            self._rating_input = None

        buttons = QDialogButtonBox(self)
        self._save_button = buttons.addButton(button_text, QDialogButtonBox.AcceptRole)
        buttons.addButton(QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)  # type: ignore[arg-type]
        buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        summary = self._note_input.toPlainText().strip()
        if summary and len(summary) > self._max_chars:
            summary = summary[: self._max_chars].rstrip()
        self._summary = summary
        if self._enable_rating and self._rating_input is not None:
            selected = self._rating_input.currentData()
            self._rating = int(selected) if selected is not None else None
        self.accept()


class ResponseStyleDialog(QDialog):
    """Modal dialog for creating or editing prompt parts such as response styles."""
    _PROMPT_PART_PRESETS: Sequence[str] = (
        "Response Style",
        "System Instruction",
        "Task Context",
        "Input Formatter",
        "Output Formatter",
        "Reference Section",
    )

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        style: ResponseStyle | None = None,
    ) -> None:
        """Initialise the response style editor with optional existing data."""
        super().__init__(parent)
        self._source_style = style
        self._result_style: ResponseStyle | None = None
        self.setWindowTitle("New Prompt Part" if style is None else "Edit Prompt Part")
        self.resize(640, 620)
        self._build_ui()
        if style is not None:
            self._populate(style)

    @property
    def result_style(self) -> ResponseStyle | None:
        """Return the resulting prompt part entry."""
        return self._result_style

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._name_input = QLineEdit(self)
        form.addRow("Name*", self._name_input)

        self._prompt_part_input = QComboBox(self)
        self._prompt_part_input.setEditable(True)
        self._prompt_part_input.addItems(self._PROMPT_PART_PRESETS)
        self._prompt_part_input.setCurrentText("Response Style")
        form.addRow("Prompt part*", self._prompt_part_input)

        self._tone_input = QLineEdit(self)
        self._tone_input.setPlaceholderText("Friendly, formal, analytical…")
        form.addRow("Tone", self._tone_input)

        self._voice_input = QLineEdit(self)
        self._voice_input.setPlaceholderText("Mentor, reviewer, analyst…")
        form.addRow("Voice", self._voice_input)

        self._tags_input = QLineEdit(self)
        self._tags_input.setPlaceholderText("Comma-separated tags (concise, markdown, …)")
        form.addRow("Tags", self._tags_input)

        self._version_input = QLineEdit(self)
        self._version_input.setPlaceholderText("1.0")
        form.addRow("Version", self._version_input)

        self._is_active_checkbox = QCheckBox("Prompt part is active", self)
        self._is_active_checkbox.setChecked(True)
        form.addRow("", self._is_active_checkbox)

        layout.addLayout(form)

        layout.addWidget(QLabel("Prompt snippet*", self))
        self._phrase_input = QPlainTextEdit(self)
        self._phrase_input.setPlaceholderText(
            "Paste the prompt fragment, response style, or reusable instructions "
            "you want to capture…"
        )
        self._phrase_input.setFixedHeight(100)
        layout.addWidget(self._phrase_input)

        layout.addWidget(QLabel("Description*", self))
        self._description_input = QPlainTextEdit(self)
        self._description_input.setPlaceholderText(
            "Explain the target tone, audience, or behaviour."
        )
        self._description_input.setFixedHeight(80)
        layout.addWidget(self._description_input)

        layout.addWidget(QLabel("Format instructions", self))
        self._format_input = QPlainTextEdit(self)
        self._format_input.setPlaceholderText(
            "Outline formatting requirements such as bullet lists or tables."
        )
        self._format_input.setFixedHeight(80)
        layout.addWidget(self._format_input)

        layout.addWidget(QLabel("Guidelines", self))
        self._guidelines_input = QPlainTextEdit(self)
        self._guidelines_input.setPlaceholderText("Document any do/don't lists or review steps.")
        self._guidelines_input.setFixedHeight(80)
        layout.addWidget(self._guidelines_input)

        layout.addWidget(QLabel("Examples (one per line)", self))
        self._examples_input = QPlainTextEdit(self)
        self._examples_input.setPlaceholderText("Provide sample outputs or outline templates.")
        self._examples_input.setFixedHeight(80)
        layout.addWidget(self._examples_input)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self._on_accept)  # type: ignore[arg-type]
        buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        layout.addWidget(buttons)

    def _populate(self, style: ResponseStyle) -> None:
        """Populate dialog fields from an existing prompt part."""
        self._name_input.setText(style.name)
        self._description_input.setPlainText(style.description)
        self._phrase_input.setPlainText(style.format_instructions or style.description)
        self._prompt_part_input.setCurrentText(style.prompt_part)
        self._tone_input.setText(style.tone or "")
        self._voice_input.setText(style.voice or "")
        self._format_input.setPlainText(style.format_instructions or "")
        self._guidelines_input.setPlainText(style.guidelines or "")
        self._tags_input.setText(", ".join(style.tags))
        self._examples_input.setPlainText("\n".join(style.examples))
        self._version_input.setText(style.version)
        self._is_active_checkbox.setChecked(style.is_active)

    def _on_accept(self) -> None:
        """Validate user input and produce a ResponseStyle instance."""
        phrase = self._phrase_input.toPlainText().strip()
        if not phrase:
            QMessageBox.warning(
                self,
                "Missing snippet",
                "Paste a prompt snippet before saving the entry.",
            )
            return

        prompt_part = self._prompt_part_input.currentText().strip() or "Response Style"
        tone = self._tone_input.text().strip() or None
        voice = self._voice_input.text().strip() or None
        format_instructions = self._format_input.toPlainText().strip() or phrase
        guidelines = self._guidelines_input.toPlainText().strip() or None
        description = self._description_input.toPlainText().strip() or phrase

        tags = [tag.strip() for tag in self._tags_input.text().split(",") if tag.strip()]
        examples = [
            line.strip()
            for line in self._examples_input.toPlainText().splitlines()
            if line.strip()
        ]
        if not examples:
            examples = [phrase]
        version = self._version_input.text().strip() or "1.0"
        is_active = self._is_active_checkbox.isChecked()
        name = self._name_input.text().strip() or self._auto_generate_name(phrase)

        payload = {
            "name": name,
            "description": description,
            "prompt_part": prompt_part,
            "tone": tone,
            "voice": voice,
            "format_instructions": format_instructions,
            "guidelines": guidelines,
            "tags": tags,
            "examples": examples,
            "version": version,
            "is_active": is_active,
        }

        if self._source_style is None:
            style = ResponseStyle(id=uuid.uuid4(), metadata=None, **payload)
        else:
            style = replace(self._source_style, **payload)

        self._result_style = style
        self.accept()

    @staticmethod
    def _auto_generate_name(phrase: str, *, max_words: int = 3) -> str:
        """Derive a friendly style name from the pasted phrase."""
        tokens = [token.strip(".,!?") for token in phrase.split() if token.strip(".,!?")]
        if tokens:
            snippet = " ".join(tokens[:max_words]).title()
            if snippet:
                return f"{snippet} Part"
        timestamp = datetime.now(UTC).strftime("%H%M%S")
        return f"Prompt Part {timestamp}"


class PromptNoteDialog(QDialog):
    """Modal dialog for creating or editing prompt notes."""
    def __init__(self, parent: QWidget | None = None, *, note: PromptNote | None = None) -> None:
        """Initialise the note editor and optionally preload existing text."""
        super().__init__(parent)
        self._source_note = note
        self._result_note: PromptNote | None = None
        self.setWindowTitle("New Note" if note is None else "Edit Note")
        self.resize(520, 320)
        self._build_ui()
        if note is not None:
            self._note_input.setPlainText(note.note)

    @property
    def result_note(self) -> PromptNote | None:
        """Return the resulting note after dialog acceptance."""
        return self._result_note

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self._note_input = QPlainTextEdit(self)
        self._note_input.setPlaceholderText("Write your prompt note here…")
        self._note_input.setMinimumHeight(200)
        layout.addWidget(self._note_input)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self._on_accept)  # type: ignore[arg-type]
        buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        text = self._note_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Missing note", "Enter note text before saving.")
            return
        if self._source_note is None:
            self._result_note = PromptNote(id=uuid.uuid4(), note=text)
        else:
            updated = replace(self._source_note, note=text)
            updated.touch()
            self._result_note = updated
        self.accept()


class MarkdownPreviewDialog(QDialog):
    """Display markdown content rendered in a read-only viewer."""
    def __init__(
        self,
        markdown_text: str,
        parent: QWidget | None,
        *,
        title: str = "Rendered Output",
    ) -> None:
        """Render ``markdown_text`` inside a read-only dialog."""
        super().__init__(parent)
        self._markdown_text = markdown_text
        self.setWindowTitle(title)
        self.resize(720, 540)
        self._build_ui()

    def _build_ui(self) -> None:
        """Construct the markdown preview layout and wire controls."""
        layout = QVBoxLayout(self)
        viewer = QTextBrowser(self)
        viewer.setOpenExternalLinks(True)
        content = self._markdown_text.strip()
        if content:
            viewer.setMarkdown(content)
        else:
            viewer.setMarkdown("*No content available.*")
        layout.addWidget(viewer, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close, parent=self)
        buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        buttons.accepted.connect(self.accept)  # type: ignore[arg-type]
        layout.addWidget(buttons)


class InfoDialog(QDialog):
    """Dialog summarising application metadata and runtime system details."""
    _TAGLINE = "Catalog, execute, and track AI prompts from a single desktop workspace."

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialise the informational dialog and load system metadata."""
        super().__init__(parent)
        self.setWindowTitle("About Prompt Manager")
        self.setModal(True)
        self.setMinimumWidth(420)
        icon = load_application_icon()
        if icon is not None:
            self.setWindowIcon(icon)

        layout = QVBoxLayout(self)

        if icon is not None:
            pixmap = icon.pixmap(96, 96)
            if not pixmap.isNull():
                icon_label = QLabel(self)
                icon_label.setAlignment(Qt.AlignHCenter)
                icon_label.setPixmap(pixmap)
                layout.addWidget(icon_label)

        title_label = QLabel("<b>Prompt Manager</b>", self)
        title_label.setTextFormat(Qt.RichText)
        layout.addWidget(title_label)

        info = _collect_system_info()

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        author_label = QLabel('<a href="https://github.com/voytas75">voytas75</a>', self)
        author_label.setTextFormat(Qt.RichText)
        author_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        author_label.setOpenExternalLinks(True)
        form.addRow("Author:", author_label)

        tagline_label = QLabel(self._TAGLINE, self)
        tagline_label.setWordWrap(True)
        tagline_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("Tagline:", tagline_label)

        app_version = self._resolve_app_version()
        version_label = QLabel(app_version, self)
        version_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("Version:", version_label)

        cpu_label = QLabel(info.cpu, self)
        cpu_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("CPU:", cpu_label)

        platform_label = QLabel(f"{info.platform_family} ({info.os_label})", self)
        platform_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("Platform:", platform_label)

        architecture_label = QLabel(info.architecture, self)
        architecture_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("Architecture:", architecture_label)

        license_label = QLabel("opensource", self)
        license_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("License:", license_label)

        icon_source_label = QLabel('<a href="https://icons8.com/">Icons8</a>', self)
        icon_source_label.setTextFormat(Qt.RichText)
        icon_source_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        icon_source_label.setOpenExternalLinks(True)
        form.addRow("Icon source:", icon_source_label)

        layout.addLayout(form)

        buttons = QDialogButtonBox(parent=self)
        close_button = buttons.addButton("Close", QDialogButtonBox.AcceptRole)
        close_button.clicked.connect(self.accept)  # type: ignore[arg-type]
        layout.addWidget(buttons)

    @staticmethod
    def _resolve_app_version() -> str:
        """Return the application version preferring the local pyproject when available."""
        project_version = InfoDialog._version_from_pyproject()
        if project_version:
            return project_version

        metadata_version = InfoDialog._version_from_metadata()
        if metadata_version:
            return metadata_version

        module_version = InfoDialog._version_from_module()
        if module_version:
            return module_version

        return "dev"

    @staticmethod
    def _version_from_pyproject() -> str | None:
        try:
            import tomllib  # type: ignore[attr-defined]

            project_root = Path(__file__).resolve().parents[1]
            pyproject = project_root / "pyproject.toml"
            if not pyproject.exists():
                return None
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            project_section = data.get("project") or {}
            resolved = project_section.get("version")
            if resolved:
                return str(resolved)
        except Exception:
            return None
        return None

    @staticmethod
    def _version_from_metadata() -> str | None:
        try:
            from importlib.metadata import (  # type: ignore
                PackageNotFoundError,
                version as pkg_version,
            )

            try:
                resolved = pkg_version("prompt-manager")
                if resolved:
                    return resolved
            except PackageNotFoundError:
                return None
        except Exception:
            return None
        return None

    @staticmethod
    def _version_from_module() -> str | None:
        try:
            from importlib import import_module

            module = import_module("core")
            module_version = getattr(module, "__version__", None)
            if module_version:
                return str(module_version)
        except Exception:
            return None
        return None


def _diff_entry_to_text(entry: CatalogDiffEntry) -> str:
    header = f"[{entry.change_type.value.upper()}] {entry.name} ({entry.prompt_id})"
    if entry.diff:
        body = textwrap.indent(entry.diff, "  ")
        return f"{header}\n{body}"
    return f"{header}\n  (no diff)"


class PromptVersionHistoryDialog(QDialog):
    """Display committed prompt versions with diff/restore controls."""
    _BODY_PLACEHOLDER = "Select a version to view the prompt body."
    _EMPTY_BODY_TEXT = "Prompt body is empty."

    def __init__(
        self,
        manager: PromptManager,
        prompt: Prompt,
        parent: QWidget | None = None,
        *,
        status_callback: Callable[[str, int], None] | None = None,
        limit: int = 200,
    ) -> None:
        """Create the history dialog for *prompt* with optional status callbacks."""
        super().__init__(parent)
        self._manager = manager
        self._prompt = prompt
        self._limit = max(1, limit)
        self._status_callback = status_callback or (lambda _msg, _duration=0: None)
        self._versions: list[PromptVersion] = []
        self.last_restored_prompt: Prompt | None = None
        self.setWindowTitle(f"{prompt.name} – Version History")
        self.resize(820, 520)
        self._build_ui()
        self._load_versions()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        header = QLabel(
            (
                "Every save creates a version. Select an entry to inspect the snapshot, "
                "diff against the prior version, or restore it."
            ),
            self,
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        self._table = QTableWidget(self)
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["Version", "Created", "Message"])
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        header_view = self._table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header_view.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header_view.setSectionResizeMode(2, QHeaderView.Stretch)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)  # type: ignore[arg-type]
        layout.addWidget(self._table, stretch=1)

        self._tab_widget = QTabWidget(self)
        self._body_view = QPlainTextEdit(self)
        self._body_view.setReadOnly(True)
        self._tab_widget.addTab(self._body_view, "Prompt body")
        self._diff_view = QPlainTextEdit(self)
        self._diff_view.setReadOnly(True)
        self._tab_widget.addTab(self._diff_view, "Diff vs previous")
        self._current_diff_view = QPlainTextEdit(self)
        self._current_diff_view.setReadOnly(True)
        self._tab_widget.addTab(self._current_diff_view, "Diff vs current")
        self._snapshot_view = QPlainTextEdit(self)
        self._snapshot_view.setReadOnly(True)
        self._tab_widget.addTab(self._snapshot_view, "Snapshot JSON")
        self._tab_widget.setCurrentIndex(0)
        layout.addWidget(self._tab_widget, stretch=1)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        refresh_button = QPushButton("Refresh", self)
        refresh_button.clicked.connect(self._load_versions)  # type: ignore[arg-type]
        button_row.addWidget(refresh_button)

        copy_snapshot_button = QPushButton("Copy Snapshot", self)
        copy_snapshot_button.clicked.connect(self._copy_snapshot_to_clipboard)  # type: ignore[arg-type]
        button_row.addWidget(copy_snapshot_button)

        copy_body_button = QPushButton("Copy Prompt Body", self)
        copy_body_button.clicked.connect(self._copy_body_to_clipboard)  # type: ignore[arg-type]
        button_row.addWidget(copy_body_button)

        restore_button = QPushButton("Restore Version", self)
        restore_button.clicked.connect(self._on_restore_clicked)  # type: ignore[arg-type]
        button_row.addWidget(restore_button)

        close_button = QPushButton("Close", self)
        close_button.clicked.connect(self.reject)  # type: ignore[arg-type]
        button_row.addWidget(close_button)
        layout.addLayout(button_row)

    def _load_versions(self) -> None:
        try:
            versions = self._manager.list_prompt_versions(self._prompt.id, limit=self._limit)
        except PromptManagerError as exc:
            QMessageBox.critical(self, "Unable to load versions", str(exc))
            versions = []
        self._versions = versions
        self._table.setRowCount(len(versions))
        for row, version in enumerate(versions):
            timestamp = self._format_timestamp(version.created_at)
            self._table.setItem(row, 0, QTableWidgetItem(f"v{version.version_number}"))
            self._table.setItem(row, 1, QTableWidgetItem(timestamp))
            message = version.commit_message or "Auto-snapshot"
            self._table.setItem(row, 2, QTableWidgetItem(message))
        if versions:
            self._table.selectRow(0)
        else:
            self._body_view.setPlainText(self._BODY_PLACEHOLDER)
            empty_message = "No versions have been recorded for this prompt yet."
            self._diff_view.setPlainText(empty_message)
            self._current_diff_view.setPlainText(empty_message)
            self._snapshot_view.clear()

    def _on_selection_changed(self) -> None:
        version = self._selected_version()
        if version is None:
            self._body_view.setPlainText(self._BODY_PLACEHOLDER)
            self._diff_view.clear()
            self._current_diff_view.clear()
            self._snapshot_view.clear()
            return
        snapshot_text = json.dumps(version.snapshot, ensure_ascii=False, indent=2)
        self._snapshot_view.setPlainText(snapshot_text)

        body_text = self._body_text_for_version(version)
        self._body_view.setPlainText(body_text)
        self._current_diff_view.setPlainText(self._format_diff_against_current(version, body_text))

        previous_version = self._previous_version(version)
        if previous_version is None:
            self._diff_view.setPlainText("No previous version available for diffing.")
            return
        try:
            diff = self._manager.diff_prompt_versions(previous_version.id, version.id)
        except PromptManagerError as exc:
            self._diff_view.setPlainText(f"Unable to compute diff: {exc}")
            return
        diff_text = diff.body_diff or json.dumps(diff.changed_fields, ensure_ascii=False, indent=2)
        if not diff_text.strip():
            diff_text = "Snapshots are identical."
        self._diff_view.setPlainText(diff_text)

    def _selected_version(self) -> PromptVersion | None:
        selection = self._table.selectionModel()
        if selection is None:
            return None
        indexes = selection.selectedRows()
        if not indexes:
            return None
        row = indexes[0].row()
        if 0 <= row < len(self._versions):
            return self._versions[row]
        return None

    def _previous_version(self, version: PromptVersion) -> PromptVersion | None:
        try:
            current_index = self._versions.index(version)
        except ValueError:
            return None
        next_index = current_index + 1
        if next_index < len(self._versions):
            return self._versions[next_index]
        return None

    def _body_text_for_version(self, version: PromptVersion) -> str:
        """Return the prompt body stored in the snapshot or a placeholder."""
        raw_body = version.snapshot.get("context")
        if isinstance(raw_body, str) and raw_body.strip():
            return raw_body
        if raw_body:
            return str(raw_body)
        return self._EMPTY_BODY_TEXT

    def _current_prompt_body(self) -> str:
        raw_body = getattr(self._prompt, "context", None)
        if isinstance(raw_body, str):
            return raw_body
        if raw_body:
            return str(raw_body)
        return ""

    def _format_diff_against_current(self, version: PromptVersion, version_body: str) -> str:
        current_body = self._current_prompt_body()
        current_text = current_body.strip()
        version_text = version_body.strip()
        if not current_text and not version_text:
            return "Current prompt and selected version bodies are empty."
        if current_body == version_body:
            return "Version body matches the current prompt."
        current_lines = current_body.splitlines()
        version_lines = version_body.splitlines()
        diff = difflib.unified_diff(
            current_lines,
            version_lines,
            fromfile="Current prompt",
            tofile=f"Version v{version.version_number}",
            lineterm="",
        )
        diff_text = "\n".join(diff)
        if not diff_text.strip():
            return "Version body matches the current prompt."
        return diff_text

    def _copy_snapshot_to_clipboard(self) -> None:
        version = self._selected_version()
        if version is None:
            return
        snapshot_text = json.dumps(version.snapshot, ensure_ascii=False, indent=2)
        QGuiApplication.clipboard().setText(snapshot_text)
        self._status_callback("Snapshot copied to clipboard", 2000)
        show_toast(self, "Snapshot copied to clipboard.")

    def _copy_body_to_clipboard(self) -> None:
        """Copy the selected prompt version body to the clipboard."""
        version = self._selected_version()
        if version is None:
            return
        body_text = self._body_text_for_version(version)
        QGuiApplication.clipboard().setText(body_text)
        self._status_callback("Prompt body copied to clipboard", 2000)
        show_toast(self, "Prompt body copied to clipboard.")

    def _on_restore_clicked(self) -> None:
        version = self._selected_version()
        if version is None:
            return
        confirm = QMessageBox.question(
            self,
            "Restore Version",
            "This will replace the current prompt contents with the selected snapshot. Continue?",
        )
        if confirm != QMessageBox.Yes:
            return
        try:
            restored = self._manager.restore_prompt_version(version.id)
        except PromptManagerError as exc:
            QMessageBox.critical(self, "Restore failed", str(exc))
            return
        self.last_restored_prompt = restored
        self._status_callback("Prompt restored to selected version", 4000)
        self._load_versions()

    @staticmethod
    def _format_timestamp(value: datetime) -> str:
        timestamp = value.astimezone(UTC)
        return timestamp.strftime("%Y-%m-%d %H:%M UTC")


class CatalogPreviewDialog(QDialog):
    """Show a diff preview before applying catalogue changes."""
    def __init__(self, diff: CatalogDiff, parent=None) -> None:
        """Display the provided catalogue diff and capture user intent."""
        super().__init__(parent)
        self._diff = diff
        self._apply = False
        self.setWindowTitle("Catalogue Preview")
        self.resize(720, 480)
        self._build_ui()

    @property
    def apply_requested(self) -> bool:
        """Return True when the user confirmed the import."""
        return self._apply

    def _build_ui(self) -> None:
        summary_text = (
            f"Added: {self._diff.added} | Updated: {self._diff.updated} | "
            f"Skipped: {self._diff.skipped} | Unchanged: {self._diff.unchanged}"
        )
        layout = QVBoxLayout(self)
        summary_label = QLabel(summary_text, self)
        summary_label.setWordWrap(True)
        layout.addWidget(summary_label)

        diff_view = QPlainTextEdit(self)
        diff_view.setReadOnly(True)
        diff_view.setPlainText(self._format_diff(self._diff))
        layout.addWidget(diff_view, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        buttons.accepted.connect(self._on_accept)  # type: ignore[arg-type]
        buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        layout.addWidget(buttons)

    @staticmethod
    def _format_diff(diff: CatalogDiff) -> str:
        if not diff.entries:
            return "No changes detected."
        entries = [
            f"Source: {diff.source or 'manual selection'}",
            "",
        ]
        entries.extend(_diff_entry_to_text(entry) for entry in diff.entries)
        return "\n".join(entries)

    def _on_accept(self) -> None:
        self._apply = True
        self.accept()


__all__ = [
    "CatalogPreviewDialog",
    "InfoDialog",
    "MarkdownPreviewDialog",
    "PromptVersionHistoryDialog",
    "PromptDialog",
    "PromptMaintenanceDialog",
    "ResponseStyleDialog",
    "PromptNoteDialog",
    "SaveResultDialog",
    "fallback_suggest_prompt_name",
    "fallback_generate_description",
    "fallback_generate_scenarios",
]
