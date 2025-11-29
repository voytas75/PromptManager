"""Qt widgets for the Enhanced Prompt Workbench experience.

Updates:
  v0.1.11 - 2025-11-29 - Log resolved wizard palette roles for troubleshooting theme differences.
  v0.1.10 - 2025-11-29 - Style wizard footer container explicitly to match host palette.
  v0.1.9 - 2025-11-29 - Apply palette colors to wizard button boxes and buttons.
  v0.1.8 - 2025-11-29 - Enforce palette-colored wizard backgrounds via style attributes.
  v0.1.7 - 2025-11-29 - Force wizard/page styled backgrounds so palette colors render on Windows.
  v0.1.6 - 2025-11-29 - Apply palette snapshots to wizard styling for consistent themes.
  v0.1.5 - 2025-11-29 - Rely on native palette for wizard styling to match host themes.
  v0.1.4 - 2025-11-29 - Apply palette-aware stylesheet for portable wizard colors.
  v0.1.3 - 2025-11-29 - Prevent guided wizard palette updates from re-triggering change events.
  v0.1.2 - 2025-11-29 - Keep guided wizard colors in sync with the current theme palette.
  v0.1.1 - 2025-11-29 - Align guided wizard palette with the active application theme.
  v0.1.0 - 2025-11-29 - Introduce guided Workbench window, mode selector, and export dialog.
"""

from __future__ import annotations

import logging
import re
import textwrap
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QPoint, Qt, Signal, QEvent
from PySide6.QtGui import QFont, QGuiApplication, QPalette, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStatusBar,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)

from core import PromptManager, PromptManagerError
from core.execution import CodexExecutionResult, CodexExecutor, ExecutionError
from core.history_tracker import HistoryTracker, HistoryTrackerError
from models.prompt_model import Prompt

from ..processing_indicator import ProcessingIndicator
from ..template_preview import TemplatePreviewWidget
from ..toast import show_toast
from .session import WorkbenchExecutionRecord, WorkbenchSession, WorkbenchVariable

logger = logging.getLogger("prompt_manager.gui.workbench")


def _inherit_palette(widget: QWidget) -> None:
    parent = widget.parent()
    palette = parent.palette() if isinstance(parent, QWidget) else None
    if palette is None:
        app = QGuiApplication.instance()
        palette = app.palette() if app is not None else None
    if palette is None:
        return
    widget.setPalette(palette)
    widget.setAutoFillBackground(True)


_BLOCK_SNIPPETS: Mapping[str, str] = {
    "System Role": textwrap.dedent(
        """### System Role\nYou are a meticulous assistant that follows instructions exactly."""
    ),
    "Context": textwrap.dedent(
        """### Context\nDescribe the task, background knowledge, and any important caveats here."""
    ),
    "Constraints": textwrap.dedent(
        """### Constraints\n- Limit answers to 200 words.\n- Use professional tone."""
    ),
    "Examples": textwrap.dedent(
        """### Examples\n**Input**\n<example input>\n\n**Expected Output**\n<example output>"""
    ),
    "JSON Response": textwrap.dedent(
        """### Output Format\nReturn a JSON object with these keys:\n- `summary`: One sentence overview.\n- `steps`: Array of ordered actions."""
    ),
}


_JINJA_PATTERN = re.compile(r"{{\s*(?P<name>[A-Za-z0-9_\.]+)\s*(?:\|[^}]*)?}}")


def _variable_at_cursor(cursor: QTextCursor) -> str | None:
    block_text = cursor.block().text()
    column = cursor.positionInBlock()
    for match in _JINJA_PATTERN.finditer(block_text):
        start, end = match.span()
        if start <= column <= end:
            return match.group("name")
    return None


def _normalise_variable_token(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    match = _JINJA_PATTERN.search(stripped)
    if match:
        return match.group("name")
    if stripped.isidentifier():
        return stripped
    return None


class WorkbenchPromptEditor(QPlainTextEdit):
    """Custom prompt editor that surfaces variable tokens on double-click."""

    variableActivated = Signal(str)

    def mouseDoubleClickEvent(self, event) -> None:  # type: ignore[override]
        super().mouseDoubleClickEvent(event)
        name = _variable_at_cursor(self.cursorForPosition(event.position().toPoint()))
        if name:
            self.variableActivated.emit(name)


class WorkbenchMode:
    """Enumeration representing the Workbench launch path."""

    GUIDED = "guided"
    BLANK = "blank"
    TEMPLATE = "template"


@dataclass(slots=True)
class ModeSelection:
    """Result payload returned by WorkbenchModeDialog."""

    mode: str
    template_prompt: Prompt | None


class WorkbenchModeDialog(QDialog):
    """Dialog that lets the user choose between guided, blank, or template modes."""

    def __init__(self, prompts: Sequence[Prompt], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        _inherit_palette(self)
        self._prompts = list(prompts)
        self._selection = ModeSelection(WorkbenchMode.GUIDED, None)
        self.setWindowTitle("Start a New Prompt")
        self._build_ui()

    def result_selection(self) -> ModeSelection:
        """Return the chosen mode and, when relevant, the template prompt."""

        return self._selection

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        description = QLabel(
            "Select how you would like to begin. Guided mode walks through suggested "
            "sections, blank mode opens the editor immediately, and template mode "
            "loads an existing prompt for remixing.",
            self,
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        self._buttons = QButtonGroup(self)
        self._buttons.setExclusive(True)
        guided = self._add_mode_option(
            layout,
            "Guided wizard",
            "Answer a few questions to scaffold the prompt automatically.",
            WorkbenchMode.GUIDED,
        )
        guided.setChecked(True)
        self._add_mode_option(
            layout,
            "Blank editor",
            "Start from an empty canvas with all tooling enabled.",
            WorkbenchMode.BLANK,
        )
        self._template_button = self._add_mode_option(
            layout,
            "Load template",
            "Pick an existing prompt to refine inside the Workbench.",
            WorkbenchMode.TEMPLATE,
        )

        self._template_list = QListWidget(self)
        self._template_list.setEnabled(False)
        for prompt in self._prompts:
            item = QListWidgetItem(prompt.name)
            item.setData(Qt.UserRole, prompt)
            self._template_list.addItem(item)
        layout.addWidget(self._template_list)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self._on_accept)  # type: ignore[arg-type]
        buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        layout.addWidget(buttons)

        self._buttons.idToggled.connect(self._on_mode_toggled)  # type: ignore[arg-type]

    def _add_mode_option(
        self,
        layout: QVBoxLayout,
        label: str,
        description: str,
        mode_value: str,
    ) -> QToolButton:
        button = QToolButton(self)
        button.setCheckable(True)
        button.setText(label)
        button.setToolTip(description)
        button.setStyleSheet("text-align: left; padding: 4px;")
        layout.addWidget(button)
        self._buttons.addButton(button)
        self._buttons.setId(button, len(self._buttons.buttons()))
        button._workbench_mode = mode_value  # type: ignore[attr-defined]
        return button

    def _on_mode_toggled(self, button_id: int, checked: bool) -> None:
        button = self._buttons.button(button_id)
        if not isinstance(button, QToolButton) or not checked:
            return
        mode = getattr(button, "_workbench_mode", WorkbenchMode.GUIDED)
        self._selection.mode = mode
        self._template_list.setEnabled(mode == WorkbenchMode.TEMPLATE)

    def _on_accept(self) -> None:
        if self._selection.mode == WorkbenchMode.TEMPLATE:
            item = self._template_list.currentItem()
            prompt = item.data(Qt.UserRole) if item else None
            if prompt is None:
                QMessageBox.warning(self, "Select template", "Choose a template first.")
                return
            self._selection = ModeSelection(WorkbenchMode.TEMPLATE, prompt)
        self.accept()


class VariableCaptureDialog(QDialog):
    """Dialog for defining placeholder metadata and sample values."""

    def __init__(
        self,
        name: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        _inherit_palette(self)
        self.setWindowTitle("Configure Variable")
        self._build_ui(name or "")

    def result_variable(self) -> WorkbenchVariable | None:
        """Return a WorkbenchVariable when the dialog is accepted."""

        if self.result() != QDialog.Accepted:
            return None
        name = self._name_input.text().strip()
        if not name:
            return None
        description = self._description_input.text().strip()
        value = self._sample_input.toPlainText().strip()
        return WorkbenchVariable(
            name=name,
            description=description or None,
            sample_value=value or None,
        )

    def _build_ui(self, initial_name: str) -> None:
        layout = QFormLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        self._name_input = QLineEdit(self)
        self._name_input.setText(initial_name)
        layout.addRow("Variable name", self._name_input)
        self._description_input = QLineEdit(self)
        layout.addRow("Description", self._description_input)
        self._sample_input = QPlainTextEdit(self)
        self._sample_input.setPlaceholderText("Sample value used for preview")
        self._sample_input.setFixedHeight(80)
        layout.addRow("Sample value", self._sample_input)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)  # type: ignore[arg-type]
        buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        layout.addWidget(buttons)


class _GoalWizardPage(QWizardPage):
    def __init__(self, session: WorkbenchSession) -> None:
        super().__init__()
        self.setTitle("Goal and Audience")
        layout = QFormLayout(self)
        self.name_input = QLineEdit(self)
        self.name_input.setPlaceholderText("Friendly prompt name…")
        self.name_input.setText(session.prompt_name)
        layout.addRow("Prompt name", self.name_input)
        self.goal_input = QPlainTextEdit(self)
        self.goal_input.setPlainText(session.goal_statement)
        self.goal_input.setPlaceholderText("Describe what the prompt must achieve…")
        self.goal_input.setFixedHeight(100)
        layout.addRow("Goal", self.goal_input)
        self.audience_input = QLineEdit(self)
        self.audience_input.setPlaceholderText("Target audience (optional)…")
        self.audience_input.setText(session.audience)
        layout.addRow("Audience", self.audience_input)


class _ContextWizardPage(QWizardPage):
    def __init__(self, session: WorkbenchSession) -> None:
        super().__init__()
        self.setTitle("System Role and Context")
        layout = QFormLayout(self)
        self.role_input = QPlainTextEdit(self)
        self.role_input.setPlainText(session.system_role)
        self.role_input.setPlaceholderText("Describe the assistant persona…")
        self.role_input.setFixedHeight(100)
        layout.addRow("System role", self.role_input)
        self.context_input = QPlainTextEdit(self)
        self.context_input.setPlainText(session.context)
        self.context_input.setPlaceholderText("Describe background info, resources, etc.")
        self.context_input.setFixedHeight(140)
        layout.addRow("Context", self.context_input)


class _DetailWizardPage(QWizardPage):
    def __init__(self, session: WorkbenchSession) -> None:
        super().__init__()
        self.setTitle("Variables and Constraints")
        layout = QGridLayout(self)
        constraint_label = QLabel("One constraint per line", self)
        layout.addWidget(constraint_label, 0, 0)
        self.constraint_input = QPlainTextEdit(self)
        self.constraint_input.setPlaceholderText("Limit to 200 words\nRespond in JSON")
        self.constraint_input.setPlainText("\n".join(session.constraints))
        layout.addWidget(self.constraint_input, 1, 0)
        variables_label = QLabel(
            "Define variables as `name | description | sample value` per line.", self
        )
        layout.addWidget(variables_label, 0, 1)
        self.variables_input = QPlainTextEdit(self)
        preset_lines: list[str] = []
        for variable in session.variables.values():
            line = variable.name
            if variable.description:
                line += f" | {variable.description}"
            if variable.sample_value:
                line += f" | {variable.sample_value}"
            preset_lines.append(line)
        self.variables_input.setPlainText("\n".join(preset_lines))
        layout.addWidget(self.variables_input, 1, 1)


class GuidedPromptWizard(QWizard):
    """Multi-step wizard that emits updates whenever fields change."""

    updated = Signal(dict)

    def __init__(self, session: WorkbenchSession, parent: QWidget | None = None) -> None:
        self._palette_updating: bool = False
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._session = session
        self.setWindowTitle("Guided Prompt Wizard")
        palette = self._resolve_theme_palette(parent)
        if palette is not None:
            logger.debug(
                "GUIDED_WIZARD palette=%s",
                {
                    "window": palette.color(QPalette.Window).name(),
                    "window_text": palette.color(QPalette.WindowText).name(),
                    "base": palette.color(QPalette.Base).name(),
                    "alternate": palette.color(QPalette.AlternateBase).name(),
                    "button": palette.color(QPalette.Button).name(),
                    "button_text": palette.color(QPalette.ButtonText).name(),
                    "mid": palette.color(QPalette.Mid).name(),
                    "midlight": palette.color(QPalette.Midlight).name(),
                    "highlight": palette.color(QPalette.Highlight).name(),
                    "highlighted_text": palette.color(QPalette.HighlightedText).name(),
                },
            )
        self._goal_page = _GoalWizardPage(session)
        self._context_page = _ContextWizardPage(session)
        self._detail_page = _DetailWizardPage(session)
        self.addPage(self._goal_page)
        self.addPage(self._context_page)
        self.addPage(self._detail_page)
        if palette is not None:
            self._apply_palette(palette)
        for widget in (
            self._goal_page.name_input,
            self._goal_page.goal_input,
            self._goal_page.audience_input,
            self._context_page.role_input,
            self._context_page.context_input,
            self._detail_page.constraint_input,
            self._detail_page.variables_input,
        ):
            widget.textChanged.connect(self._emit_update)  # type: ignore[arg-type]
        self.currentIdChanged.connect(lambda _: self._emit_update())  # type: ignore[arg-type]
        self._emit_update()

    def changeEvent(self, event: QEvent) -> None:  # type: ignore[override]
        super().changeEvent(event)
        if self._palette_updating:
            return
        if event.type() in {QEvent.PaletteChange, QEvent.ApplicationPaletteChange}:
            palette = self._resolve_theme_palette(self.parentWidget())
            if palette is not None:
                self._apply_palette(palette)

    def _resolve_theme_palette(self, parent: QWidget | None) -> QPalette | None:
        app = QGuiApplication.instance()
        if app is not None:
            return QPalette(app.palette())
        if parent is not None:
            return QPalette(parent.palette())
        return None

    def _apply_palette(self, palette: QPalette) -> None:
        if self._palette_updating:
            return
        self._palette_updating = True
        try:
            self.setPalette(palette)
            self.setAutoFillBackground(True)
            window_color = palette.color(QPalette.Window).name()
            window_text = palette.color(QPalette.WindowText).name()
            base_color = palette.color(QPalette.Base).name()
            button_color = palette.color(QPalette.Button).name()
            button_text = palette.color(QPalette.ButtonText).name()
            mid_color = palette.color(QPalette.Mid).name()
            highlight_color = palette.color(QPalette.Highlight).name()
            palette_snapshot = {
                "window": window_color,
                "window_text": window_text,
                "base": base_color,
                "button": button_color,
                "mid": mid_color,
                "highlight": highlight_color,
            }
            logger.debug("GUIDED_PALETTE snapshot=%s", palette_snapshot)
            stylesheet = textwrap.dedent(
                f"""
                QWizard,
                QWizardPage {{
                    background-color: {window_color} !important;
                    color: {window_text};
                }}
                QWizard QWidget {{
                    background-color: {window_color};
                    color: {window_text};
                }}
                QWizard::header {{
                    background-color: {window_color} !important;
                    border-bottom: 1px solid {mid_color};
                }}
                QWizard::sidepanel {{
                    background-color: {window_color};
                }}
                QLabel {{
                    color: {window_text};
                }}
                QStackedWidget {{
                    background-color: {base_color};
                }}
                QWizard QDialogButtonBox {{
                    background-color: {window_color} !important;
                    border-top: 1px solid {mid_color};
                }}
                QWizard QDialogButtonBox QWidget {{
                    background-color: {window_color} !important;
                    color: {window_text};
                }}
                QLineEdit,
                QPlainTextEdit,
                QComboBox,
                QTextEdit {{
                    background-color: {base_color};
                    color: {window_text};
                    border: 1px solid {mid_color};
                }}
                QDialogButtonBox QPushButton {{
                    background-color: {button_color};
                    color: {button_text};
                    border: 1px solid {mid_color};
                    border-radius: 4px;
                    padding: 4px 10px;
                }}
                QDialogButtonBox QPushButton:hover {{
                    background-color: {highlight_color};
                    color: {window_text};
                }}
                QDialogButtonBox QPushButton:pressed {{
                    background-color: {mid_color};
                    color: {window_text};
                }}
                """
            ).strip()
            self.setStyleSheet(stylesheet)
            styled_pages = (self._goal_page, self._context_page, self._detail_page)
            for page in styled_pages:
                page.setPalette(palette)
                page.setAutoFillBackground(True)
                page.setAttribute(Qt.WA_StyledBackground, True)
            stack = self.findChild(QStackedWidget)
            if stack is not None:
                stack.setPalette(palette)
                stack.setAutoFillBackground(True)
                stack.setAttribute(Qt.WA_StyledBackground, True)
            button_box = self.findChild(QDialogButtonBox)
            if button_box is not None:
                button_box.setPalette(palette)
                button_box.setAutoFillBackground(True)
                button_box.setAttribute(Qt.WA_StyledBackground, True)
                for button in button_box.findChildren(QPushButton):
                    button.setPalette(palette)
                    button.setAutoFillBackground(True)
                footer_panel = button_box.parentWidget()
                if footer_panel is not None:
                    footer_panel.setPalette(palette)
                    footer_panel.setAutoFillBackground(True)
                    footer_panel.setAttribute(Qt.WA_StyledBackground, True)
                    footer_panel.setStyleSheet(
                        f"background-color: {window_color}; border-top: 1px solid {mid_color};"
                    )
            footer_container = self.findChild(QWidget, "qt_wizard_button_widget")
            if footer_container is not None:
                footer_container.setPalette(palette)
                footer_container.setAutoFillBackground(True)
                footer_container.setAttribute(Qt.WA_StyledBackground, True)
                footer_container.setStyleSheet(
                    f"#qt_wizard_button_widget {{ background-color: {window_color}; border-top: 1px solid {mid_color}; }}"
                )
        finally:
            self._palette_updating = False

    def _emit_update(self) -> None:
        variables: dict[str, WorkbenchVariable] = {}
        for line in self._detail_page.variables_input.toPlainText().splitlines():
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split("|")]
            name = parts[0]
            if not name:
                continue
            description = parts[1] if len(parts) > 1 else None
            sample = parts[2] if len(parts) > 2 else None
            variables[name] = WorkbenchVariable(name=name, description=description, sample_value=sample)
        payload = {
            "prompt_name": self._goal_page.name_input.text(),
            "goal": self._goal_page.goal_input.toPlainText(),
            "audience": self._goal_page.audience_input.text(),
            "system_role": self._context_page.role_input.toPlainText(),
            "context": self._context_page.context_input.toPlainText(),
            "constraints": self._detail_page.constraint_input.toPlainText().splitlines(),
            "variables": variables,
        }
        self.updated.emit(payload)


class WorkbenchExportDialog(QDialog):
    """Collect final metadata before persisting a prompt draft."""

    def __init__(self, session: WorkbenchSession, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        _inherit_palette(self)
        self.setWindowTitle("Export Prompt")
        self._build_ui(session)

    def prompt_kwargs(self) -> dict[str, Any] | None:
        if self.result() != QDialog.Accepted:
            return None
        tags = [tag.strip() for tag in self._tags_input.text().split(",") if tag.strip()]
        return {
            "name": self._name_input.text().strip(),
            "category": self._category_input.text().strip() or "Workbench",
            "language": self._language_input.text().strip() or "en",
            "tags": tags,
            "author": self._author_input.text().strip() or None,
        }

    def _build_ui(self, session: WorkbenchSession) -> None:
        layout = QFormLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        self._name_input = QLineEdit(self)
        self._name_input.setText(session.prompt_name or session.goal_statement or "Workbench Draft")
        layout.addRow("Name", self._name_input)
        self._category_input = QLineEdit(self)
        self._category_input.setText("Workbench")
        layout.addRow("Category", self._category_input)
        self._language_input = QLineEdit(self)
        self._language_input.setText("en")
        layout.addRow("Language", self._language_input)
        self._tags_input = QLineEdit(self)
        layout.addRow("Tags", self._tags_input)
        self._author_input = QLineEdit(self)
        layout.addRow("Author", self._author_input)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)  # type: ignore[arg-type]
        buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        layout.addRow(buttons)


class WorkbenchWindow(QMainWindow):
    """Modal workspace that guides users through crafting prompts iteratively."""

    def __init__(
        self,
        prompt_manager: PromptManager,
        *,
        mode: str = WorkbenchMode.GUIDED,
        template_prompt: Prompt | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._manager = prompt_manager
        self._session = WorkbenchSession()
        self._executor: CodexExecutor | None = prompt_manager.executor
        self._history_tracker: HistoryTracker | None = prompt_manager.history_tracker
        self._suppress_editor_signal = False
        self._active_refinement_target: str | None = None
        self._build_ui()
        self._load_initial_state(mode, template_prompt)

    def _build_ui(self) -> None:
        self.setWindowTitle("Enhanced Prompt Workbench")
        self.setWindowModality(Qt.ApplicationModal)
        self.resize(1280, 860)

        toolbar = QToolBar("Workbench Controls", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        wizard_action = toolbar.addAction("🧙 Wizard", self._launch_wizard)
        wizard_action.setToolTip("Restart the guided wizard")
        link_action = toolbar.addAction("🔗 Link Variable", self._link_variable)
        link_action.setToolTip("Capture metadata and sample values for the selected variable.")
        self._brainstorm_action = toolbar.addAction("🤔 Brainstorm", self._run_brainstorm)
        self._brainstorm_action.setToolTip("Ask CodexExecutor to propose alternative phrasing.")
        self._peek_action = toolbar.addAction("👀 AI Peek", self._run_peek)
        self._peek_action.setToolTip("Request a quick summary of the prompt's behaviour.")
        validate_action = toolbar.addAction("✅ Validate", self._validate_template)
        validate_action.setToolTip("Re-run template preview validation and report status.")
        run_action = toolbar.addAction("▶️ Run Once", self._trigger_run)
        run_action.setToolTip("Render the prompt with sample data and execute it via CodexExecutor.")
        export_action = toolbar.addAction("💾 Export", self._export_prompt)
        export_action.setToolTip("Persist this prompt into the repository.")
        executor_ready = self._executor is not None
        self._brainstorm_action.setEnabled(executor_ready)
        self._peek_action.setEnabled(executor_ready)
        run_action.setEnabled(executor_ready)

        container = QWidget(self)
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(8, 8, 8, 8)
        container_layout.setSpacing(8)
        self.setCentralWidget(container)

        self._summary_label = QLabel("", container)
        self._summary_label.setWordWrap(True)
        self._summary_label.setStyleSheet("font-weight: 500;")
        container_layout.addWidget(self._summary_label)

        splitter = QSplitter(Qt.Horizontal, container)
        container_layout.addWidget(splitter, 1)

        palette_frame = QFrame(splitter)
        palette_layout = QVBoxLayout(palette_frame)
        palette_layout.setContentsMargins(8, 8, 8, 8)
        palette_layout.addWidget(QLabel("Guidance blocks", palette_frame))
        self._palette_list = QListWidget(palette_frame)
        for label, snippet in _BLOCK_SNIPPETS.items():
            item = QListWidgetItem(label)
            item.setToolTip(snippet.splitlines()[0])
            self._palette_list.addItem(item)
        self._palette_list.itemDoubleClicked.connect(self._insert_block)  # type: ignore[arg-type]
        palette_layout.addWidget(self._palette_list, 1)

        editor_frame = QFrame(splitter)
        editor_layout = QVBoxLayout(editor_frame)
        editor_layout.setContentsMargins(8, 8, 8, 8)
        self._editor = WorkbenchPromptEditor(editor_frame)
        self._editor.setPlaceholderText("Use the wizard or block palette to start…")
        self._editor.textChanged.connect(self._on_editor_changed)  # type: ignore[arg-type]
        self._editor.variableActivated.connect(self._open_variable_editor)
        editor_layout.addWidget(self._editor, 1)

        right_splitter = QSplitter(Qt.Vertical, splitter)

        preview_container = QWidget(right_splitter)
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        self._preview = TemplatePreviewWidget(preview_container)
        self._preview.set_run_enabled(self._executor is not None)
        self._preview.run_requested.connect(self._handle_preview_run)  # type: ignore[arg-type]
        preview_layout.addWidget(self._preview, 1)
        preview_layout.addWidget(QLabel("Test input", preview_container))
        self._test_input = QPlainTextEdit(preview_container)
        self._test_input.setPlaceholderText("Provide user input to test this prompt…")
        self._test_input.setFixedHeight(120)
        preview_layout.addWidget(self._test_input)

        output_container = QWidget(right_splitter)
        output_layout = QVBoxLayout(output_container)
        output_layout.setContentsMargins(8, 8, 8, 8)
        self._output_tabs = QTabWidget(output_container)
        self._output_view = QTextEdit(output_container)
        self._output_view.setReadOnly(True)
        self._output_tabs.addTab(self._output_view, "Run Output")
        self._history_list = QListWidget(output_container)
        self._output_tabs.addTab(self._history_list, "History")
        output_layout.addWidget(self._output_tabs, 1)

        feedback_row = QHBoxLayout()
        feedback_row.setSpacing(6)
        feedback_row.addWidget(QLabel("Was the last run helpful?", output_container))
        self._thumbs_up = QToolButton(output_container)
        self._thumbs_up.setText("👍")
        self._thumbs_up.clicked.connect(lambda: self._record_rating(1.0))
        feedback_row.addWidget(self._thumbs_up)
        self._thumbs_down = QToolButton(output_container)
        self._thumbs_down.setText("👎")
        self._thumbs_down.clicked.connect(lambda: self._record_rating(0.0))
        feedback_row.addWidget(self._thumbs_down)
        feedback_row.addWidget(QLabel("Feedback", output_container))
        self._feedback_input = QLineEdit(output_container)
        feedback_row.addWidget(self._feedback_input, 1)
        apply_feedback = QPushButton("Save", output_container)
        apply_feedback.clicked.connect(self._apply_feedback)
        feedback_row.addWidget(apply_feedback)
        output_layout.addLayout(feedback_row)

        self._status = QStatusBar(self)
        self.setStatusBar(self._status)

    def _load_initial_state(self, mode: str, template_prompt: Prompt | None) -> None:
        if mode == WorkbenchMode.TEMPLATE and template_prompt is not None:
            self._session.prompt_name = template_prompt.name
            self._session.goal_statement = template_prompt.description
            self._session.system_role = template_prompt.context or ""
            self._session.context = template_prompt.context or ""
            self._session.template_text = template_prompt.context or ""
            self._apply_editor_text(template_prompt.context or "", from_wizard=True)
        elif mode == WorkbenchMode.BLANK:
            self._apply_editor_text("", from_wizard=True)
        else:
            self._launch_wizard(initial=True)
        self._refresh_summary()
        self._sync_preview()

    def _refresh_summary(self) -> None:
        summary = self._session.goal_statement or "No goal defined yet."
        if self._session.audience:
            summary += f" (Audience: {self._session.audience})"
        self._summary_label.setText(summary)

    def _apply_editor_text(self, text: str, *, from_wizard: bool) -> None:
        self._suppress_editor_signal = True
        self._editor.setPlainText(text)
        self._suppress_editor_signal = False
        self._session.set_template_text(text, source="wizard" if from_wizard else "editor")
        self._sync_preview()

    def _sync_preview(self) -> None:
        prompt_id = self._session.prompt_name or "workbench"
        self._preview.set_template(self._session.template_text, prompt_id)
        self._preview.apply_variable_values(self._session.variable_payload())
        self._preview.refresh_preview()

    def _on_editor_changed(self) -> None:
        if self._suppress_editor_signal:
            return
        self._session.set_template_text(self._editor.toPlainText(), source="editor")
        self._sync_preview()

    def _insert_block(self, item: QListWidgetItem) -> None:
        snippet = _BLOCK_SNIPPETS.get(item.text())
        if not snippet:
            return
        cursor = self._editor.textCursor()
        cursor.beginEditBlock()
        cursor.insertText(snippet + "\n\n")
        cursor.endEditBlock()
        self._on_editor_changed()

    def _launch_wizard(self, *, initial: bool = False) -> None:
        wizard = GuidedPromptWizard(self._session, self)
        wizard.updated.connect(self._handle_wizard_update)
        if wizard.exec() == QDialog.Accepted and not initial:
            show_toast("Wizard applied to prompt.", parent=self)

    def _handle_wizard_update(self, payload: Mapping[str, Any]) -> None:
        constraints = payload.get("constraints") or []
        variables = payload.get("variables") or {}
        self._session.update_from_wizard(
            prompt_name=str(payload.get("prompt_name") or ""),
            goal=str(payload.get("goal") or ""),
            system_role=str(payload.get("system_role") or ""),
            context=str(payload.get("context") or ""),
            audience=str(payload.get("audience") or ""),
            constraints=constraints,
            variables=variables,
        )
        self._apply_editor_text(self._session.template_text, from_wizard=True)
        self._refresh_summary()

    def _open_variable_editor(self, name: str) -> None:
        dialog = VariableCaptureDialog(name, self)
        if dialog.exec() != QDialog.Accepted:
            return
        variable = dialog.result_variable()
        if variable is None:
            return
        self._session.link_variable(variable.name, sample_value=variable.sample_value, description=variable.description)
        self._preview.apply_variable_values({variable.name: variable.sample_value or ""})
        show_toast(f"Variable '{variable.name}' updated.", parent=self)

    def _link_variable(self) -> None:
        cursor = self._editor.textCursor()
        if cursor.hasSelection():
            candidate = _normalise_variable_token(cursor.selectedText())
        else:
            candidate = _variable_at_cursor(cursor)
        dialog = VariableCaptureDialog(candidate, self)
        if dialog.exec() != QDialog.Accepted:
            return
        variable = dialog.result_variable()
        if variable is None:
            return
        stored = self._session.link_variable(
            variable.name,
            sample_value=variable.sample_value,
            description=variable.description,
        )
        self._preview.apply_variable_values({stored.name: stored.sample_value or ""})
        show_toast(f"Variable '{stored.name}' saved.", parent=self)

    def _validate_template(self) -> None:
        self._preview.refresh_preview()
        self._status.showMessage("Template validation refreshed.", 4000)

    def _trigger_run(self) -> None:
        if not self._preview.request_run():
            self._status.showMessage("Preview not ready for execution.", 5000)

    def _handle_preview_run(self, rendered_text: str, variables: Mapping[str, str]) -> None:
        if self._executor is None:
            self._status.showMessage("CodexExecutor is not configured.", 6000)
            return
        raw_request = self._test_input.toPlainText().strip()
        fallback_message = None
        if raw_request:
            request_text = raw_request
        else:
            fallback_value: tuple[str, str] | None = None
            for name, value in variables.items():
                trimmed = value.strip()
                if trimmed:
                    fallback_value = (name, trimmed)
                    break
            if fallback_value is not None:
                request_text = fallback_value[1]
                fallback_message = (
                    f"No test input supplied; using the '{fallback_value[0]}' variable value."
                )
            else:
                request_text = (
                    self._session.goal_statement.strip()
                    or "Run a preview based on the current prompt."
                )
                fallback_message = "No test input supplied; using the prompt goal instead."
        prompt = self._session.build_prompt()
        prompt.context = rendered_text
        indicator = ProcessingIndicator(self, "Running prompt…")
        try:
            result = indicator.run(
                self._executor.execute,
                prompt,
                request_text,
                conversation=None,
                stream=False,
            )
        except (ExecutionError, RuntimeError) as exc:
            logger.exception("Run once failed")
            self._status.showMessage(str(exc), 8000)
            record = WorkbenchExecutionRecord(
                request_text=request_text,
                response_text=str(exc),
                success=False,
                variables=dict(variables),
            )
            self._session.record_execution(record)
            self._update_history()
            return
        self._render_execution(result, request_text, variables)
        if fallback_message:
            self._status.showMessage(fallback_message, 5000)

    def _render_execution(
        self,
        result: CodexExecutionResult,
        request_text: str,
        variables: Mapping[str, str],
    ) -> None:
        self._output_tabs.setCurrentWidget(self._output_view)
        meta = textwrap.dedent(
            f"""Model: {self._executor.model if self._executor else 'n/a'}\nDuration: {result.duration_ms} ms\nTokens: {result.usage.get('total_tokens', 'n/a')}"""
        )
        self._output_view.setPlainText(result.response_text)
        record = WorkbenchExecutionRecord(
            request_text=request_text,
            response_text=result.response_text,
            duration_ms=result.duration_ms,
            success=True,
            variables=dict(variables),
        )
        record.suggested_focus = self._session.suggest_refinement_target(result.response_text)
        self._session.record_execution(record)
        self._update_history()
        self._apply_highlight(record.suggested_focus)
        self._status.showMessage("Execution complete.", 5000)

    def _apply_highlight(self, target: str | None) -> None:
        if target is None:
            self._editor.setExtraSelections([])
            return
        header = {
            "context": "### Context",
            "system": "### System Role",
            "constraints": "### Constraints",
            "output": "### Output Format",
        }.get(target)
        if not header:
            return
        document = self._editor.document()
        cursor = document.find(header)
        if cursor.isNull():
            return
        selection = QTextEdit.ExtraSelection()
        fmt = QTextCharFormat()
        fmt.setBackground(Qt.yellow)
        selection.cursor = cursor
        selection.cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
        selection.format = fmt
        self._editor.setExtraSelections([selection])

    def _run_brainstorm(self) -> None:
        self._invoke_helper(
            "brainstorm",
            "Provide three alternative phrasings that could strengthen this prompt."
        )

    def _run_peek(self) -> None:
        self._invoke_helper(
            "peek",
            "Summarise this prompt in two sentences and point out obvious gaps."
        )

    def _invoke_helper(self, label: str, instruction: str) -> None:
        if self._executor is None:
            self._status.showMessage("CodexExecutor unavailable.", 6000)
            return
        prompt = self._session.build_prompt()
        request_text = f"{instruction}\n---\n{self._session.template_text.strip()}"
        indicator = ProcessingIndicator(self, f"Running {label}…")
        try:
            result = indicator.run(
                self._executor.execute,
                prompt,
                request_text,
                conversation=None,
                stream=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s action failed", label)
            self._status.showMessage(str(exc), 8000)
            return
        self._output_view.setPlainText(result.response_text)
        self._status.showMessage(f"{label.title()} suggestions ready.", 6000)

    def _update_history(self) -> None:
        self._history_list.clear()
        for record in self._session.execution_history[-20:]:
            status = "✅" if record.success else "⚠️"
            summary = record.response_text.splitlines()[0] if record.response_text else "(empty)"
            item = QListWidgetItem(f"{status} {summary}")
            self._history_list.addItem(item)

    def _record_rating(self, rating: float) -> None:
        if not self._session.execution_history:
            return
        record = self._session.execution_history[-1]
        record.rating = rating
        record.feedback = self._feedback_input.text().strip() or None
        self._status.showMessage("Feedback saved for last run.", 4000)

    def _apply_feedback(self) -> None:
        if not self._session.execution_history:
            return
        record = self._session.execution_history[-1]
        baseline = record.rating if record.rating is not None else 1.0
        self._record_rating(baseline)

    def _export_prompt(self) -> None:
        dialog = WorkbenchExportDialog(self._session, self)
        if dialog.exec() != QDialog.Accepted:
            return
        kwargs = dialog.prompt_kwargs()
        if kwargs is None:
            return
        prompt = self._session.build_prompt(
            category=kwargs["category"],
            language=kwargs["language"],
            tags=kwargs["tags"],
            author=kwargs["author"],
        )
        prompt.name = kwargs["name"] or prompt.name
        try:
            created = self._manager.create_prompt(prompt)
        except PromptManagerError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self._persist_history(created)
        show_toast(f"Prompt '{created.name}' saved.", parent=self)
        self._session.clear_history()
        self._update_history()

    def _persist_history(self, prompt: Prompt) -> None:
        tracker = self._history_tracker
        if tracker is None:
            return
        for record in self._session.execution_history:
            try:
                if record.success:
                    tracker.record_success(
                        prompt.id,
                        record.request_text,
                        record.response_text,
                        duration_ms=record.duration_ms,
                        metadata={"variables": dict(record.variables), "source": "workbench"},
                        rating=record.rating,
                    )
                else:
                    tracker.record_failure(
                        prompt.id,
                        record.request_text,
                        record.response_text,
                        duration_ms=record.duration_ms,
                        metadata={"variables": dict(record.variables), "source": "workbench"},
                    )
            except HistoryTrackerError as exc:
                logger.warning("Unable to persist history entry: %s", exc)
