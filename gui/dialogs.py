"""Dialog widgets used by the Prompt Manager GUI.

Updates: v0.5.0 - 2025-11-09 - Capture execution ratings alongside optional notes.
Updates: v0.4.0 - 2025-11-08 - Add execution Save Result dialog with optional notes.
Updates: v0.3.0 - 2025-11-06 - Add catalogue preview dialog with diff summary output.
Updates: v0.2.0 - 2025-11-05 - Add prompt name suggestion based on context.
Updates: v0.1.0 - 2025-11-04 - Implement create/edit prompt dialog backed by Prompt dataclass.
"""

from __future__ import annotations

import logging
import re
import textwrap
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Callable, Optional

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from core import (
    CatalogDiff,
    CatalogDiffEntry,
    NameGenerationError,
    DescriptionGenerationError,
)
from models.prompt_model import Prompt


logger = logging.getLogger("prompt_manager.gui.dialogs")

_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+")


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


class PromptDialog(QDialog):
    """Modal dialog used for creating or editing prompt records."""

    def __init__(
        self,
        parent=None,
        prompt: Optional[Prompt] = None,
        name_generator: Optional[Callable[[str], str]] = None,
        description_generator: Optional[Callable[[str], str]] = None,
    ) -> None:
        super().__init__(parent)
        self._source_prompt = prompt
        self._result_prompt: Optional[Prompt] = None
        self._name_generator = name_generator
        self._description_generator = description_generator
        self.setWindowTitle("Create Prompt" if prompt is None else "Edit Prompt")
        self._build_ui()
        if prompt is not None:
            self._populate(prompt)

    @property
    def result_prompt(self) -> Optional[Prompt]:
        """Return the prompt produced by the dialog after acceptance."""

        return self._result_prompt

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
        self._example_input = QPlainTextEdit(self)
        self._example_output = QPlainTextEdit(self)

        self._language_input.setPlaceholderText("en")
        self._tags_input.setPlaceholderText("tag-a, tag-b")

        form_layout.addRow("Category", self._category_input)
        form_layout.addRow("Language", self._language_input)
        form_layout.addRow("Tags", self._tags_input)
        form_layout.addRow("Prompt Body", self._context_input)
        form_layout.addRow("Description", self._description_input)
        form_layout.addRow("Example Input", self._example_input)
        form_layout.addRow("Example Output", self._example_output)

        main_layout.addLayout(form_layout)

        self._buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        self._buttons.accepted.connect(self._on_accept)  # type: ignore[arg-type]
        self._buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        main_layout.addWidget(self._buttons)
        self._context_input.textChanged.connect(self._on_context_changed)  # type: ignore[arg-type]

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

    def _on_accept(self) -> None:
        """Validate inputs, build the prompt, and close the dialog."""

        prompt = self._build_prompt()
        if prompt is None:
            return
        self._result_prompt = prompt
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
            f"Source: {diff.source or 'builtin'}",
            "",
        ]
        entries.extend(_diff_entry_to_text(entry) for entry in diff.entries)
        return "\n".join(entries)

    def _on_accept(self) -> None:
        self._apply = True
        self.accept()


__all__ = [
    "CatalogPreviewDialog",
    "PromptDialog",
    "SaveResultDialog",
    "fallback_suggest_prompt_name",
    "fallback_generate_description",
]
