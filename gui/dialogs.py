"""Dialog widgets used by the Prompt Manager GUI.

Updates: v0.1.0 - 2025-11-04 - Implement create/edit prompt dialog backed by Prompt dataclass.
"""

from __future__ import annotations

import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Optional

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QVBoxLayout,
)

from models.prompt_model import Prompt


class PromptDialog(QDialog):
    """Modal dialog used for creating or editing prompt records."""

    def __init__(self, parent=None, prompt: Optional[Prompt] = None) -> None:
        super().__init__(parent)
        self._source_prompt = prompt
        self._result_prompt: Optional[Prompt] = None
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
        self._category_input = QLineEdit(self)
        self._language_input = QLineEdit(self)
        self._tags_input = QLineEdit(self)
        self._context_input = QPlainTextEdit(self)
        self._description_input = QPlainTextEdit(self)
        self._example_input = QPlainTextEdit(self)
        self._example_output = QPlainTextEdit(self)

        self._language_input.setPlaceholderText("en")
        self._tags_input.setPlaceholderText("tag-a, tag-b")

        form_layout.addRow("Name", self._name_input)
        form_layout.addRow("Category", self._category_input)
        form_layout.addRow("Language", self._language_input)
        form_layout.addRow("Tags", self._tags_input)
        form_layout.addRow("Context", self._context_input)
        form_layout.addRow("Description", self._description_input)
        form_layout.addRow("Example Input", self._example_input)
        form_layout.addRow("Example Output", self._example_output)

        main_layout.addLayout(form_layout)

        self._buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        self._buttons.accepted.connect(self._on_accept)  # type: ignore[arg-type]
        self._buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        main_layout.addWidget(self._buttons)

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

    def _build_prompt(self) -> Optional[Prompt]:
        """Construct a Prompt object from the dialog inputs."""

        name = self._name_input.text().strip()
        description = self._description_input.toPlainText().strip()
        category = self._category_input.text().strip()
        if not name or not description:
            QMessageBox.warning(self, "Missing fields", "Name and description are required.")
            return None

        language = self._language_input.text().strip() or "en"
        tags = [tag.strip() for tag in self._tags_input.text().split(",") if tag.strip()]
        context = self._context_input.toPlainText().strip() or None
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


__all__ = ["PromptDialog"]
