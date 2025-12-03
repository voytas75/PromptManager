"""Dialogs for saving execution results and editing response styles.

Updates:
  v0.1.0 - 2025-12-03 - Extract save-result and response-style dialogs.
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypedDict

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from models.response_style import ResponseStyle

if TYPE_CHECKING:
    from collections.abc import Sequence


class ResponseStylePayload(TypedDict):
    name: str
    description: str
    prompt_part: str
    tone: str | None
    voice: str | None
    format_instructions: str
    guidelines: str | None
    tags: list[str]
    examples: list[str]
    version: str
    is_active: bool


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
        self._rating: int | None = int(initial_rating) if initial_rating is not None else None
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
            line.strip() for line in self._examples_input.toPlainText().splitlines() if line.strip()
        ]
        if not examples:
            examples = [phrase]
        version = self._version_input.text().strip() or "1.0"
        is_active = self._is_active_checkbox.isChecked()
        name = self._name_input.text().strip() or self._auto_generate_name(phrase)

        payload: ResponseStylePayload = {
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


__all__ = ["SaveResultDialog", "ResponseStyleDialog"]
