"""Dialog widgets used by the Prompt Manager GUI.

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
            return fallback_suggest_prompt_name(context)
        try:
            return self._name_generator(context)
        except NameGenerationError as exc:
            message = str(exc)
            if "not configured" in message:
                logger.info("LiteLLM disabled; using fallback name suggestion")
            else:
                logger.warning("Name generation failed; using fallback suggestion", exc_info=exc)
            return fallback_suggest_prompt_name(context)

    def _generate_description(self, context: str) -> str:
        """Generate description using LiteLLM when configured."""

        context = context.strip()
        if not context:
            return ""
        if self._description_generator is None:
            return fallback_generate_description(context)
        try:
            return self._description_generator(context)
        except DescriptionGenerationError as exc:
            message = str(exc)
            if "not configured" in message:
                logger.info("LiteLLM disabled; using fallback description summary")
            else:
                logger.warning("Description generation failed; using fallback summary", exc_info=exc)
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
            return Prompt(
                id=uuid.uuid4(),
                name=name,
                description=description,
                category=category,
                tags=tags,
                language=language,
                context=context,
                example_input=example_input,
                example_output=example_output,
            )

        base = self._source_prompt
        ext2_copy = deepcopy(base.ext2) if base.ext2 is not None else None
        ext5_copy = deepcopy(base.ext5) if base.ext5 is not None else None
        return Prompt(
            id=base.id,
            name=name,
            description=description,
            category=category,
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


__all__ = ["CatalogPreviewDialog", "PromptDialog", "fallback_suggest_prompt_name"]
