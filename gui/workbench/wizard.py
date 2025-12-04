"""Guided wizard dialog for scaffolding Enhanced Prompt Workbench prompts.

Updates:
  v0.1.0 - 2025-12-04 - Extract guided wizard pages and dialog from workbench_window.
"""
from __future__ import annotations

import logging

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QPainter, QPaintEvent, QPalette
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .session import WorkbenchSession, WorkbenchVariable

logger = logging.getLogger("prompt_manager.gui.workbench")

__all__ = ["GuidedPromptWizard"]


class _GoalWizardPage(QWidget):
    def __init__(self, session: WorkbenchSession) -> None:
        super().__init__()
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


class _ContextWizardPage(QWidget):
    def __init__(self, session: WorkbenchSession) -> None:
        super().__init__()
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


class _DetailWizardPage(QWidget):
    def __init__(self, session: WorkbenchSession) -> None:
        super().__init__()
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


class GuidedPromptWizard(QDialog):
    """Custom-styled wizard dialog that emits updates whenever fields change."""

    updated = Signal(dict)

    def __init__(self, session: WorkbenchSession, parent: QWidget | None = None) -> None:
        """Build the wizard pages and sync palette styling."""
        super().__init__(parent)
        self._palette_updating = False
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._session = session
        self.setWindowTitle("Guided Prompt Wizard")
        self.setModal(True)
        self.resize(840, 640)
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
        self._pages: list[tuple[str, QWidget]] = [
            ("Goal and Audience", self._goal_page),
            ("System Role and Context", self._context_page),
            ("Variables and Constraints", self._detail_page),
        ]
        self._build_ui()
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
        self._stack.currentChanged.connect(self._on_page_changed)
        self._on_page_changed()
        self._emit_update()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        header = QVBoxLayout()
        header.setSpacing(4)
        self._title_label = QLabel("", self)
        self._title_label.setStyleSheet("font-size: 20px; font-weight: 600;")
        self._step_label = QLabel("", self)
        self._step_label.setStyleSheet("color: rgba(255, 255, 255, 0.85);")
        header.addWidget(self._title_label)
        header.addWidget(self._step_label)
        layout.addLayout(header)
        self._stack = QStackedWidget(self)
        self._stack.setObjectName("guidedWizardStack")
        for _, widget in self._pages:
            self._stack.addWidget(widget)
        layout.addWidget(self._stack, 1)
        self._footer_widget = QWidget(self)
        button_row = QHBoxLayout(self._footer_widget)
        button_row.setContentsMargins(0, 12, 0, 0)
        button_row.addStretch(1)
        self._back_button = QPushButton("Back", self)
        self._next_button = QPushButton("Next", self)
        self._finish_button = QPushButton("Finish", self)
        self._cancel_button = QPushButton("Cancel", self)
        for button in (
            self._back_button,
            self._next_button,
            self._finish_button,
            self._cancel_button,
        ):
            button.setMinimumWidth(110)
        self._back_button.clicked.connect(self._go_back)
        self._next_button.clicked.connect(self._go_next)
        self._finish_button.clicked.connect(self._finish)
        self._cancel_button.clicked.connect(self.reject)
        button_row.addWidget(self._back_button)
        button_row.addWidget(self._next_button)
        button_row.addWidget(self._finish_button)
        button_row.addWidget(self._cancel_button)
        layout.addWidget(self._footer_widget)

    def changeEvent(self, event: QEvent) -> None:  # type: ignore[override]
        """Reapply palette styling when the application theme changes."""
        super().changeEvent(event)
        if self._palette_updating:
            return
        if event.type() in {QEvent.PaletteChange, QEvent.ApplicationPaletteChange}:
            palette = self._resolve_theme_palette(self.parentWidget())
            if palette is not None:
                self._apply_palette(palette)

    def paintEvent(self, event: QPaintEvent) -> None:  # type: ignore[override]
        """Draw a solid background so palette colors are respected on Windows."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().color(QPalette.Window))
        super().paintEvent(event)

    def _resolve_theme_palette(self, parent: QWidget | None) -> QPalette | None:
        widget = parent or self.parentWidget()
        if widget is None:
            return None
        return widget.palette()

    def _apply_palette(self, palette: QPalette) -> None:
        self._palette_updating = True
        try:
            self.setPalette(palette)
            self.setAutoFillBackground(True)
            self._footer_widget.setPalette(palette)
            self._footer_widget.setAutoFillBackground(True)
            self._stack.setPalette(palette)
            self._stack.setAutoFillBackground(True)
        finally:
            self._palette_updating = False

    def _on_page_changed(self) -> None:
        index = self._stack.currentIndex()
        total = self._stack.count()
        self._title_label.setText(self._pages[index][0])
        self._step_label.setText(f"Step {index + 1} of {total}")
        self._back_button.setEnabled(index > 0)
        self._next_button.setVisible(index < total - 1)
        self._finish_button.setVisible(index == total - 1)
        self._emit_update()

    def _go_next(self) -> None:
        index = self._stack.currentIndex()
        if index < self._stack.count() - 1:
            self._stack.setCurrentIndex(index + 1)

    def _go_back(self) -> None:
        index = self._stack.currentIndex()
        if index > 0:
            self._stack.setCurrentIndex(index - 1)

    def _finish(self) -> None:
        self.accept()

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
            variables[name] = WorkbenchVariable(
                name=name,
                description=description,
                sample_value=sample,
            )
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
