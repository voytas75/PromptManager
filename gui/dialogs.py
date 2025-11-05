"""Dialog widgets used by the Prompt Manager GUI.

Updates: v0.8.10 - 2025-11-05 - Add ChromaDB integrity verification action to maintenance dialog.
Updates: v0.8.9 - 2025-11-05 - Add ChromaDB maintenance actions to the maintenance dialog.
Updates: v0.8.8 - 2025-11-30 - Restore catalogue preview dialog for GUI import workflows.
Updates: v0.8.7 - 2025-11-30 - Remove catalogue preview dialog after retiring import workflow.
Updates: v0.8.6 - 2025-11-27 - Support pre-filling prompts when duplicating entries.
Updates: v0.8.5 - 2025-11-26 - Add application info dialog.
Updates: v0.8.4 - 2025-11-25 - Add prompt maintenance overview stats panel.
Updates: v0.8.3 - 2025-11-19 - Add scenario generation controls and persistence to the prompt dialog.
Updates: v0.8.2 - 2025-11-17 - Add Apply workflow so prompt edits can be persisted without closing the dialog.
Updates: v0.8.1 - 2025-11-16 - Add destructive delete control and metadata suggestion helpers to the prompt dialog.
Updates: v0.8.0 - 2025-11-16 - Add task template editor dialog.
Updates: v0.7.2 - 2025-11-02 - Collapse example sections when empty and expand on demand.
Updates: v0.7.1 - 2025-11-02 - Increase default prompt dialog size for edit and creation workflows.
Updates: v0.7.0 - 2025-11-16 - Add markdown preview dialog for rendered execution output.
Updates: v0.6.0 - 2025-11-15 - Add prompt engineering refinement button to prompt dialog.
Updates: v0.5.0 - 2025-11-09 - Capture execution ratings alongside optional notes.
Updates: v0.4.0 - 2025-11-08 - Add execution Save Result dialog with optional notes.
Updates: v0.3.0 - 2025-11-06 - Add catalogue preview dialog with diff summary output.
Updates: v0.2.0 - 2025-11-05 - Add prompt name suggestion based on context.
Updates: v0.1.0 - 2025-11-04 - Implement create/edit prompt dialog backed by Prompt dataclass.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import textwrap
import uuid
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence

from PySide6.QtCore import Qt, QEvent, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QFormLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QToolButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from core import (
    CatalogDiff,
    CatalogDiffEntry,
    NameGenerationError,
    DescriptionGenerationError,
    ScenarioGenerationError,
    PromptEngineeringUnavailable,
    PromptManager,
    PromptManagerError,
    RepositoryError,
)
from core.prompt_engineering import PromptEngineeringError, PromptRefinement
from models.prompt_model import Prompt, TaskTemplate


logger = logging.getLogger("prompt_manager.gui.dialogs")

_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+")


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


class CollapsibleTextSection(QWidget):
    """Wrapper providing an expandable/collapsible plain text editor."""

    textChanged = Signal()

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._title = title
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

        self._editor = QPlainTextEdit(self)
        self._editor.setVisible(False)
        self._layout.addWidget(self._editor)

        self._editor.textChanged.connect(self._on_text_changed)  # type: ignore[arg-type]
        self._editor.installEventFilter(self)
        self._collapsed_height = 0
        self._expanded_height = 100

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


def fallback_generate_scenarios(context: str, *, max_items: int = 3) -> List[str]:
    """Provide heuristic usage scenarios when LLM support is unavailable."""

    cleaned = context.strip()
    if not cleaned:
        return []

    # Prefer full sentences; fall back to newline segments when punctuation is sparse.
    sentence_pattern = re.compile(r"(?<=[.!?])\s+|\n")
    segments = [segment.strip() for segment in sentence_pattern.split(cleaned) if segment.strip()]
    if not segments:
        segments = [cleaned]

    scenarios: List[str] = []
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

    def __init__(
        self,
        parent=None,
        prompt: Optional[Prompt] = None,
        name_generator: Optional[Callable[[str], str]] = None,
        description_generator: Optional[Callable[[str], str]] = None,
        category_generator: Optional[Callable[[str], str]] = None,
        tags_generator: Optional[Callable[[str], Sequence[str]]] = None,
        scenario_generator: Optional[Callable[[str], Sequence[str]]] = None,
        prompt_engineer: Optional[Callable[..., PromptRefinement]] = None,
    ) -> None:
        super().__init__(parent)
        self._source_prompt = prompt
        self._result_prompt: Optional[Prompt] = None
        self._name_generator = name_generator
        self._description_generator = description_generator
        self._prompt_engineer = prompt_engineer
        self._category_generator = category_generator
        self._tags_generator = tags_generator
        self._scenario_generator = scenario_generator
        self._delete_requested = False
        self._delete_button: Optional[QPushButton] = None
        self._apply_button: Optional[QPushButton] = None
        self._generate_category_button: Optional[QPushButton] = None
        self._generate_tags_button: Optional[QPushButton] = None
        self._generate_scenarios_button: Optional[QPushButton] = None
        self.setWindowTitle("Create Prompt" if prompt is None else "Edit Prompt")
        self.setMinimumWidth(760)
        self.resize(960, 700)
        self._build_ui()
        if prompt is not None:
            self._populate(prompt)

    @property
    def result_prompt(self) -> Optional[Prompt]:
        """Return the prompt produced by the dialog after acceptance."""

        return self._result_prompt

    @property
    def delete_requested(self) -> bool:
        """Return True when the user chose to delete the prompt instead of saving."""

        return self._delete_requested

    @property
    def source_prompt(self) -> Optional[Prompt]:
        """Expose the prompt supplied to the dialog for convenience."""

        return self._source_prompt

    def _build_ui(self) -> None:
        """Construct the dialog layout and wire interactions."""

        main_layout = QVBoxLayout(self)
        form_layout = QFormLayout()

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

        self._category_input = QLineEdit(self)
        self._language_input = QLineEdit(self)
        self._tags_input = QLineEdit(self)
        self._context_input = QPlainTextEdit(self)
        self._context_input.setPlaceholderText("Paste the full prompt text here…")
        self._description_input = QPlainTextEdit(self)
        self._description_input.setFixedHeight(60)
        self._example_input = CollapsibleTextSection("Example Input", self)
        self._example_input.setPlaceholderText("Optional example input…")
        self._example_output = CollapsibleTextSection("Example Output", self)
        self._example_output.setPlaceholderText("Optional example output…")

        self._language_input.setPlaceholderText("en")
        self._tags_input.setPlaceholderText("tag-a, tag-b")

        category_container = QWidget(self)
        category_container_layout = QHBoxLayout(category_container)
        category_container_layout.setContentsMargins(0, 0, 0, 0)
        category_container_layout.setSpacing(6)
        category_container_layout.addWidget(self._category_input)
        self._generate_category_button = QPushButton("Suggest", self)
        self._generate_category_button.setToolTip("Suggest a category based on the prompt body.")
        self._generate_category_button.clicked.connect(self._on_generate_category_clicked)  # type: ignore[arg-type]
        if self._category_generator is None:
            self._generate_category_button.setEnabled(False)
            self._generate_category_button.setToolTip("Category suggestions require the main application context.")
        category_container_layout.addWidget(self._generate_category_button)
        form_layout.addRow("Category", category_container)
        form_layout.addRow("Language", self._language_input)
        tags_container = QWidget(self)
        tags_container_layout = QHBoxLayout(tags_container)
        tags_container_layout.setContentsMargins(0, 0, 0, 0)
        tags_container_layout.setSpacing(6)
        tags_container_layout.addWidget(self._tags_input)
        self._generate_tags_button = QPushButton("Suggest", self)
        self._generate_tags_button.setToolTip("Suggest tags based on the prompt body.")
        self._generate_tags_button.clicked.connect(self._on_generate_tags_clicked)  # type: ignore[arg-type]
        if self._tags_generator is None:
            self._generate_tags_button.setEnabled(False)
            self._generate_tags_button.setToolTip("Tag suggestions require the main application context.")
        tags_container_layout.addWidget(self._generate_tags_button)
        form_layout.addRow("Tags", tags_container)
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
        form_layout.addRow("", self._refine_button)
        form_layout.addRow("Description", self._description_input)
        self._scenarios_input = QPlainTextEdit(self)
        self._scenarios_input.setPlaceholderText("One scenario per line…")
        self._scenarios_input.setFixedHeight(90)
        scenarios_container = QWidget(self)
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
                "Generate heuristic scenarios based on the prompt body. Configure LiteLLM for AI assistance."
            )
        scenarios_layout.addWidget(self._generate_scenarios_button, alignment=Qt.AlignRight)
        form_layout.addRow("Scenarios", scenarios_container)
        form_layout.addRow(self._example_input)
        form_layout.addRow(self._example_output)

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

    def prefill_from_prompt(self, prompt: Prompt) -> None:
        """Populate inputs from an existing prompt while staying in creation mode."""

        self._populate(prompt)
        self._result_prompt = None
        self._delete_requested = False

    def _populate(self, prompt: Prompt) -> None:
        """Fill inputs with the existing prompt values for editing."""

        self._name_input.setText(prompt.name)
        self._category_input.setText(prompt.category)
        self._language_input.setText(prompt.language)
        self._tags_input.setText(", ".join(prompt.tags))
        self._context_input.setPlainText(prompt.context or "")
        self._description_input.setPlainText(prompt.description)
        self._example_input.setPlainText(prompt.example_input or "")
        self._example_output.setPlainText(prompt.example_output or "")
        self._scenarios_input.setPlainText("\n".join(prompt.scenarios))

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

    def _on_generate_name_clicked(self) -> None:
        """Generate a prompt name from the context field."""

        try:
            suggestion = self._generate_name(self._context_input.toPlainText())
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
            suggestion = (self._category_generator(context) or "").strip()
        except Exception as exc:  # noqa: BLE001 - surface generator failures to the user
            QMessageBox.warning(self, "Category suggestion failed", str(exc))
            return
        if suggestion:
            self._category_input.setText(suggestion)
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
            suggestions = self._tags_generator(context) or []
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

    def _collect_scenarios(self) -> List[str]:
        """Return the current scenarios listed in the dialog."""

        scenarios: List[str] = []
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
        unique: List[str] = []
        seen: set[str] = set()
        for scenario in sanitized:
            key = scenario.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(scenario)
        self._scenarios_input.setPlainText("\n".join(unique))

    def _generate_scenarios(self, context: str) -> List[str]:
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
            scenarios = self._generate_scenarios(context)
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
        """Invoke the prompt engineer to refine the prompt body when available."""

        if self._prompt_engineer is None:
            QMessageBox.information(
                self,
                "Prompt refinement unavailable",
                "Configure LiteLLM in Settings to enable prompt engineering.",
            )
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
        category = self._category_input.text().strip() or None
        tags = [tag.strip() for tag in self._tags_input.text().split(",") if tag.strip()]

        try:
            result = self._prompt_engineer(
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

        summary_parts: list[str] = []
        if result.analysis:
            summary_parts.append(result.analysis)
        if result.checklist:
            checklist = "\n".join(f"• {item}" for item in result.checklist)
            summary_parts.append(f"Checklist:\n{checklist}")
        if result.warnings:
            warnings = "\n".join(f"• {item}" for item in result.warnings)
            summary_parts.append(f"Warnings:\n{warnings}")

        message = (
            "\n\n".join(summary_parts)
            if summary_parts
            else "The prompt has been updated with the refined version."
        )
        QMessageBox.information(self, "Prompt refined", message)

    def _build_prompt(self) -> Optional[Prompt]:
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

        category = self._category_input.text().strip()
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
            now = datetime.now(timezone.utc)
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
                version="1.0",
            )

        base = self._source_prompt
        ext2_copy = deepcopy(base.ext2) if base.ext2 is not None else None
        ext5_copy = deepcopy(base.ext5) if base.ext5 is not None else None
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
            last_modified=datetime.now(timezone.utc),
            version=base.version,
            author=base.author,
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


class PromptMaintenanceDialog(QDialog):
    """Expose bulk metadata maintenance utilities."""

    maintenance_applied = Signal(str)

    def __init__(
        self,
        parent: QWidget,
        manager: PromptManager,
        *,
        category_generator: Optional[Callable[[str], str]] = None,
        tags_generator: Optional[Callable[[str], Sequence[str]]] = None,
    ) -> None:
        super().__init__(parent)
        self._manager = manager
        self._category_generator = category_generator
        self._tags_generator = tags_generator
        self._stats_labels: Dict[str, QLabel] = {}
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
        self._stats_refresh_button: QPushButton
        self._reset_log_view: QPlainTextEdit
        self.setWindowTitle("Prompt Maintenance")
        self.resize(640, 420)
        self._build_ui()
        self._refresh_catalogue_stats()
        self._refresh_redis_info()
        self._refresh_chroma_info()
        self._refresh_storage_info()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._tab_widget = QTabWidget(self)
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
        metadata_layout.addWidget(self._log_view, stretch=1)

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

        self._chroma_compact_button = QPushButton("Compact Persistent Store", chroma_actions_container)
        self._chroma_compact_button.setToolTip("Reclaim disk space by vacuuming the Chroma SQLite store.")
        self._chroma_compact_button.clicked.connect(self._on_chroma_compact_clicked)  # type: ignore[arg-type]
        chroma_actions_layout.addWidget(self._chroma_compact_button)

        self._chroma_optimize_button = QPushButton("Optimize Persistent Store", chroma_actions_container)
        self._chroma_optimize_button.setToolTip("Refresh query statistics to improve Chroma performance.")
        self._chroma_optimize_button.clicked.connect(self._on_chroma_optimize_clicked)  # type: ignore[arg-type]
        chroma_actions_layout.addWidget(self._chroma_optimize_button)

        self._chroma_verify_button = QPushButton("Verify Index Integrity", chroma_actions_container)
        self._chroma_verify_button.setToolTip("Run integrity checks against the Chroma index files.")
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

        self._tab_widget.addTab(storage_tab, "SQLite")

        reset_tab = QWidget(self)
        reset_layout = QVBoxLayout(reset_tab)

        reset_intro = QLabel(
            "Use these actions to clear application data while leaving configuration and settings untouched.",
            reset_tab,
        )
        reset_intro.setWordWrap(True)
        reset_layout.addWidget(reset_intro)

        reset_warning = QLabel(
            "<b>Warning:</b> these operations permanently delete existing prompts, histories, and embeddings.",
            reset_tab,
        )
        reset_warning.setWordWrap(True)
        reset_layout.addWidget(reset_warning)

        reset_buttons_container = QWidget(reset_tab)
        reset_buttons_layout = QVBoxLayout(reset_buttons_container)
        reset_buttons_layout.setContentsMargins(0, 0, 0, 0)
        reset_buttons_layout.setSpacing(8)

        reset_sqlite_button = QPushButton("Clear Prompt Database", reset_buttons_container)
        reset_sqlite_button.setToolTip("Delete all prompts, templates, and execution history from SQLite.")
        reset_sqlite_button.clicked.connect(self._on_reset_prompts_clicked)  # type: ignore[arg-type]
        reset_buttons_layout.addWidget(reset_sqlite_button)

        reset_chroma_button = QPushButton("Clear Embedding Store", reset_buttons_container)
        reset_chroma_button.setToolTip("Remove all vectors from the ChromaDB collection used for semantic search.")
        reset_chroma_button.clicked.connect(self._on_reset_chroma_clicked)  # type: ignore[arg-type]
        reset_buttons_layout.addWidget(reset_chroma_button)

        reset_all_button = QPushButton("Reset Application Data", reset_buttons_container)
        reset_all_button.setToolTip(
            "Clear prompts, histories, embeddings, and usage logs in one step. Settings remain unchanged."
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
        layout.addWidget(self._buttons)

    def _set_stat_value(self, key: str, value: str) -> None:
        label = self._stats_labels.get(key)
        if label is not None:
            label.setText(value)

    @staticmethod
    def _format_timestamp(value: Optional[datetime]) -> str:
        if value is None:
            return "—"
        return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

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

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self._log_view.appendPlainText(f"[{timestamp}] {message}")

    def _append_reset_log(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
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

    def _collect_prompts(self) -> List[Prompt]:
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
                last_modified=datetime.now(timezone.utc),
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
                last_modified=datetime.now(timezone.utc),
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

    def _on_reset_prompts_clicked(self) -> None:
        """Clear the SQLite prompt repository."""

        if not self._confirm_destructive_action(
            "Clear the prompt database, templates, and execution history?"
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
            "All prompts, templates, and execution history have been removed.",
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
        lines: List[str] = []
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
        self._redis_stats_view.setPlainText("\n".join(lines) if lines else "No Redis statistics available.")

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
        lines: List[str] = []
        for key, label in (
            ("documents", "Documents"),
            ("disk_usage_bytes", "Disk usage (bytes)"),
        ):
            value = stats.get(key)
            if value is not None:
                lines.append(f"{label}: {value}")
        self._chroma_stats_view.setPlainText("\n".join(lines) if lines else "No ChromaDB statistics available.")

    def _refresh_storage_info(self) -> None:
        """Update the SQLite tab with repository information."""

        repository = self._manager.repository
        db_path_obj = getattr(repository, "_db_path", None)
        if isinstance(db_path_obj, Path):
            db_path = str(db_path_obj)
        else:
            db_path = str(db_path_obj) if db_path_obj is not None else ""

        self._storage_path_label.setText(f"Path: {db_path}" if db_path else "Path: unknown")

        size_bytes = None
        if db_path:
            try:
                path_obj = Path(db_path)
                if path_obj.exists():
                    size_bytes = path_obj.stat().st_size
            except OSError:
                size_bytes = None

        stats_lines: List[str] = []
        if size_bytes is not None:
            stats_lines.append(f"File size: {size_bytes} bytes")

        try:
            prompt_count = len(repository.list())
            stats_lines.append(f"Prompts: {prompt_count}")
        except RepositoryError as exc:
            stats_lines.append(f"Prompts: error ({exc})")

        try:
            template_count = len(repository.list_templates())
            stats_lines.append(f"Templates: {template_count}")
        except RepositoryError as exc:
            stats_lines.append(f"Templates: error ({exc})")

        try:
            execution_count = len(repository.list_executions())
            stats_lines.append(f"Executions: {execution_count}")
        except RepositoryError as exc:
            stats_lines.append(f"Executions: error ({exc})")

        self._storage_stats_view.setPlainText("\n".join(stats_lines))
        self._storage_status_label.setText("Status: ready")

class TemplateDialog(QDialog):
    """Modal dialog for creating or editing task templates."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        template: Optional[TaskTemplate] = None,
        prompts: Sequence[Prompt] = (),
    ) -> None:
        super().__init__(parent)
        self._source_template = template
        self._result_template: Optional[TaskTemplate] = None
        self._prompts = list(prompts)
        self._missing_prompt_ids: list[str] = []
        self.setWindowTitle("New Task Template" if template is None else "Edit Task Template")
        self.resize(720, 640)
        self._build_ui()
        if template is not None:
            self._populate(template)

    @property
    def result_template(self) -> Optional[TaskTemplate]:
        """Return the resulting template after dialog acceptance."""

        return self._result_template

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()

        self._name_input = QLineEdit(self)
        form.addRow("Name*", self._name_input)

        self._description_input = QPlainTextEdit(self)
        self._description_input.setPlaceholderText("Short description shown alongside the template…")
        self._description_input.setFixedHeight(80)
        form.addRow("Description*", self._description_input)

        self._category_input = QLineEdit(self)
        self._category_input.setPlaceholderText("Optional category label (e.g. Code Analysis)")
        form.addRow("Category", self._category_input)

        self._tags_input = QLineEdit(self)
        self._tags_input.setPlaceholderText("Comma-separated tags (analysis, refactor, …)")
        form.addRow("Tags", self._tags_input)

        self._version_input = QLineEdit(self)
        self._version_input.setPlaceholderText("1.0")
        form.addRow("Version", self._version_input)

        self._is_active_checkbox = QCheckBox("Template is active", self)
        self._is_active_checkbox.setChecked(True)
        form.addRow("", self._is_active_checkbox)

        layout.addLayout(form)

        layout.addWidget(QLabel("Default input", self))
        self._default_input_editor = QPlainTextEdit(self)
        self._default_input_editor.setPlaceholderText(
            "Optional starter input pasted into the workspace when the template is applied…"
        )
        self._default_input_editor.setFixedHeight(100)
        layout.addWidget(self._default_input_editor)

        layout.addWidget(QLabel("Notes", self))
        self._notes_input = QPlainTextEdit(self)
        self._notes_input.setPlaceholderText("Optional guidance or checklist shown after applying the template…")
        self._notes_input.setFixedHeight(80)
        layout.addWidget(self._notes_input)

        prompt_label = QLabel("Select prompts to include*", self)
        layout.addWidget(prompt_label)

        self._prompt_list = QListWidget(self)
        self._prompt_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self._prompt_list.setAlternatingRowColors(True)
        for prompt in self._prompts:
            label = prompt.name
            if prompt.category:
                label += f" — {prompt.category}"
            item = QListWidgetItem(label, self._prompt_list)
            item.setData(Qt.UserRole, str(prompt.id))
            item.setToolTip(prompt.description)
        layout.addWidget(self._prompt_list, stretch=1)

        self._missing_label = QLabel("", self)
        self._missing_label.setWordWrap(True)
        self._missing_label.setStyleSheet("color: #d97706;")
        self._missing_label.setVisible(False)
        layout.addWidget(self._missing_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self._on_accept)  # type: ignore[arg-type]
        buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        layout.addWidget(buttons)

    def _populate(self, template: TaskTemplate) -> None:
        self._name_input.setText(template.name)
        self._description_input.setPlainText(template.description)
        self._category_input.setText(template.category or "")
        self._tags_input.setText(", ".join(template.tags))
        self._version_input.setText(template.version)
        self._is_active_checkbox.setChecked(template.is_active)
        if template.default_input:
            self._default_input_editor.setPlainText(template.default_input)
        if template.notes:
            self._notes_input.setPlainText(template.notes)

        selected_ids = {str(pid) for pid in template.prompt_ids}
        available_ids = set()
        for index in range(self._prompt_list.count()):
            item = self._prompt_list.item(index)
            prompt_id = item.data(Qt.UserRole)
            available_ids.add(prompt_id)
            if prompt_id in selected_ids:
                item.setSelected(True)
        missing_ids = sorted(selected_ids - available_ids)
        if missing_ids:
            self._missing_prompt_ids = missing_ids
            self._missing_label.setText(
                "Some prompts referenced by this template are unavailable: "
                + ", ".join(missing_ids)
            )
            self._missing_label.setVisible(True)
        else:
            self._missing_label.setVisible(False)

    def _on_accept(self) -> None:
        name = self._name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Invalid template", "Provide a template name.")
            return
        description = self._description_input.toPlainText().strip()
        if not description:
            QMessageBox.warning(self, "Invalid template", "Provide a description for the template.")
            return

        selected_ids: list[uuid.UUID] = []
        for item in self._prompt_list.selectedItems():
            raw = item.data(Qt.UserRole)
            try:
                selected_ids.append(uuid.UUID(str(raw)))
            except (TypeError, ValueError):
                continue
        if not selected_ids and not self._missing_prompt_ids:
            QMessageBox.warning(
                self,
                "Invalid template",
                "Select at least one prompt to include in the template.",
            )
            return

        category = self._category_input.text().strip() or None
        tags = [tag.strip() for tag in self._tags_input.text().split(",") if tag.strip()]
        default_input = self._default_input_editor.toPlainText().strip() or None
        notes = self._notes_input.toPlainText().strip() or None
        version = self._version_input.text().strip() or (
            self._source_template.version if self._source_template else "1.0"
        )
        is_active = self._is_active_checkbox.isChecked()

        template_id = self._source_template.id if self._source_template else uuid.uuid4()
        created_at = self._source_template.created_at if self._source_template else datetime.now(timezone.utc)
        ext1 = self._source_template.ext1 if self._source_template else None
        ext2 = self._source_template.ext2 if self._source_template else None
        ext3 = self._source_template.ext3 if self._source_template else None

        prompt_ids = selected_ids or [uuid.UUID(value) for value in self._missing_prompt_ids]

        self._result_template = TaskTemplate(
            id=template_id,
            name=name,
            description=description,
            prompt_ids=prompt_ids,
            default_input=default_input,
            category=category,
            tags=tags,
            notes=notes,
            is_active=is_active,
            version=version,
            created_at=created_at,
            last_modified=datetime.now(timezone.utc),
            ext1=ext1,
            ext2=ext2,
            ext3=ext3,
        )
        self.accept()


class SaveResultDialog(QDialog):
    """Collect optional notes before persisting or updating a prompt execution."""

    def __init__(
        self,
        parent: Optional[QWidget],
        *,
        prompt_name: str,
        default_text: str = "",
        max_chars: int = 400,
        button_text: str = "Save",
        enable_rating: bool = True,
        initial_rating: Optional[float] = None,
    ) -> None:
        super().__init__(parent)
        self._max_chars = max_chars
        self._summary = ""
        self._rating: Optional[int] = (
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
    def rating(self) -> Optional[int]:
        """Return the selected rating, if any."""

        return self._rating

    def _build_ui(self, default_text: str, button_text: str) -> None:
        layout = QVBoxLayout(self)

        message = QLabel(
            "Add an optional summary or notes for this execution. The note will be stored with the history entry.",
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


class MarkdownPreviewDialog(QDialog):
    """Display markdown content rendered in a read-only viewer."""

    def __init__(self, markdown_text: str, parent: Optional[QWidget], *, title: str = "Rendered Output") -> None:
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

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About Prompt Manager")
        self.setModal(True)
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)

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

        layout.addLayout(form)

        buttons = QDialogButtonBox(parent=self)
        close_button = buttons.addButton("Close", QDialogButtonBox.AcceptRole)
        close_button.clicked.connect(self.accept)  # type: ignore[arg-type]
        layout.addWidget(buttons)


def _diff_entry_to_text(entry: CatalogDiffEntry) -> str:
    header = f"[{entry.change_type.value.upper()}] {entry.name} ({entry.prompt_id})"
    if entry.diff:
        body = textwrap.indent(entry.diff, "  ")
        return f"{header}\n{body}"
    return f"{header}\n  (no diff)"


class CatalogPreviewDialog(QDialog):
    """Show a diff preview before applying catalogue changes."""

    def __init__(self, diff: CatalogDiff, parent=None) -> None:
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
    "PromptDialog",
    "PromptMaintenanceDialog",
    "SaveResultDialog",
    "TemplateDialog",
    "fallback_suggest_prompt_name",
    "fallback_generate_description",
    "fallback_generate_scenarios",
]
