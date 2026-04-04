"""Compact dialog and helpers for promoting captured draft prompts.

Updates:
  v0.1.0 - 2026-04-04 - Add bounded draft promotion dialog that preserves prompt provenance.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from .quick_capture import parse_quick_capture_tags

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Sequence

    from models.prompt_model import Prompt


def is_prompt_draft(prompt: Prompt) -> bool:
    """Return ``True`` when the prompt still carries active draft capture metadata."""
    ext2 = prompt.ext2
    if not isinstance(ext2, Mapping):
        return False
    return str(ext2.get("capture_state") or "").strip() == "draft"


def build_promoted_prompt(
    prompt: Prompt,
    *,
    title: str,
    category: str,
    tags_text: str,
    source: str,
    description: str,
) -> Prompt:
    """Return an updated prompt with draft state cleared and provenance preserved."""
    cleaned_title = title.strip()
    if not cleaned_title:
        raise ValueError("Prompt title is required.")

    cleaned_ext2 = deepcopy(dict(prompt.ext2)) if isinstance(prompt.ext2, Mapping) else None
    if cleaned_ext2 is not None:
        cleaned_ext2.pop("capture_state", None)
        if not cleaned_ext2:
            cleaned_ext2 = None

    return replace(
        prompt,
        name=cleaned_title,
        description=description.strip(),
        category=category.strip() or prompt.category or "General",
        category_slug=None,
        tags=parse_quick_capture_tags(tags_text),
        last_modified=datetime.now(UTC),
        source=source.strip() or prompt.source,
        ext2=cleaned_ext2,
        ext4=list(prompt.ext4) if prompt.ext4 is not None else None,
        ext5=deepcopy(prompt.ext5) if prompt.ext5 is not None else None,
    )


class DraftPromoteDialog(QDialog):
    """Collect the minimal metadata needed to promote a draft into a reusable prompt."""

    def __init__(
        self,
        prompt: Prompt,
        *,
        categories: Sequence[str] = (),
        parent: QWidget | None = None,
    ) -> None:
        """Initialise the compact promotion form from the existing prompt state."""
        super().__init__(parent)
        self._source_prompt = prompt
        self._result_prompt: Prompt | None = None
        self.setWindowTitle("Promote Draft")
        self.setMinimumSize(560, 360)
        self.resize(640, 420)
        self._build_ui(categories)
        self._populate(prompt)

    @property
    def result_prompt(self) -> Prompt | None:
        """Return the promoted prompt after acceptance."""
        return self._result_prompt

    def _build_ui(self, categories: Sequence[str]) -> None:
        """Construct the compact bounded form."""
        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setContentsMargins(16, 16, 16, 8)
        form.setSpacing(10)

        self._title_input = QLineEdit(self)
        form.addRow("Title", self._title_input)

        self._category_input = QComboBox(self)
        self._category_input.setEditable(True)
        self._category_input.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        for category in categories:
            cleaned = category.strip()
            if cleaned:
                self._category_input.addItem(cleaned)
        form.addRow("Category", self._category_input)

        self._tags_input = QLineEdit(self)
        self._tags_input.setPlaceholderText("Optional tags, comma-separated")
        form.addRow("Tags", self._tags_input)

        self._source_input = QLineEdit(self)
        self._source_input.setPlaceholderText("Where this prompt came from")
        form.addRow("Source", self._source_input)

        self._description_input = QPlainTextEdit(self)
        self._description_input.setPlaceholderText("Short description or note")
        self._description_input.setFixedHeight(90)
        form.addRow("Note", self._description_input)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        promote_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        if promote_button is not None:
            promote_button.setText("Promote Draft")
        buttons.accepted.connect(self._on_accept)  # type: ignore[arg-type]
        buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        layout.addWidget(buttons)

    def _populate(self, prompt: Prompt) -> None:
        """Fill the form from the current prompt metadata."""
        self._title_input.setText(prompt.name)
        category = prompt.category.strip() or "General"
        index = self._category_input.findText(category)
        if index >= 0:
            self._category_input.setCurrentIndex(index)
        else:
            self._category_input.setEditText(category)
        self._tags_input.setText(", ".join(prompt.tags))
        self._source_input.setText(prompt.source)
        self._description_input.setPlainText(prompt.description)

    def _on_accept(self) -> None:
        """Validate the bounded fields and keep the promoted prompt for callers."""
        try:
            self._result_prompt = build_promoted_prompt(
                self._source_prompt,
                title=self._title_input.text(),
                category=self._category_input.currentText(),
                tags_text=self._tags_input.text(),
                source=self._source_input.text(),
                description=self._description_input.toPlainText(),
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Title required", str(exc))
            return
        self.accept()


__all__ = ["DraftPromoteDialog", "build_promoted_prompt", "is_prompt_draft"]
