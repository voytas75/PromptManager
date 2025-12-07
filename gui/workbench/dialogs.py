"""Dialog helpers for starting and exporting Enhanced Prompt Workbench sessions.

Updates:
  v0.1.1 - 2025-12-07 - Introduce typed payload for export dialog results.
  v0.1.0 - 2025-12-04 - Extract mode picker, variable capture, and export dialogs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .session import WorkbenchSession, WorkbenchVariable
from .utils import inherit_palette

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Sequence

    from models.prompt_model import Prompt
else:  # pragma: no cover - runtime placeholders for type-only imports
    from typing import Any as _Any

    Sequence = _Any
    Prompt = _Any


__all__ = [
    "ModeSelection",
    "VariableCaptureDialog",
    "WorkbenchExportDialog",
    "WorkbenchMode",
    "WorkbenchModeDialog",
]


class WorkbenchExportPayload(TypedDict):
    """Structured payload returned when exporting a Workbench draft."""

    name: str
    category: str
    language: str
    tags: list[str]
    author: str | None


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
        """Initialise the dialog and populate the template list."""
        super().__init__(parent)
        inherit_palette(self)
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
        """Configure form fields for editing a single variable."""
        super().__init__(parent)
        inherit_palette(self)
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


class WorkbenchExportDialog(QDialog):
    """Collect final metadata before persisting a prompt draft."""

    def __init__(self, session: WorkbenchSession, parent: QWidget | None = None) -> None:
        """Populate export fields using the provided session defaults."""
        super().__init__(parent)
        inherit_palette(self)
        self.setWindowTitle("Export Prompt")
        self._build_ui(session)

    def prompt_kwargs(self) -> WorkbenchExportPayload | None:
        """Return keyword arguments for prompt creation when accepted."""
        if self.result() != QDialog.Accepted:
            return None
        tags = [tag.strip() for tag in self._tags_input.text().split(",") if tag.strip()]
        return WorkbenchExportPayload(
            name=self._name_input.text().strip(),
            category=self._category_input.text().strip() or "Workbench",
            language=self._language_input.text().strip() or "en",
            tags=tags,
            author=self._author_input.text().strip() or None,
        )

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
