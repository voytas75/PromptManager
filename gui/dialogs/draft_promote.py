"""Compact dialog and helpers for promoting captured draft prompts.

Updates:
  v0.1.5 - 2026-04-11 - Nudge very close promote-time matches
  with a stronger open-existing button label.
  v0.1.4 - 2026-04-11 - Add one bounded visible
  similarity-strength cue for very close promote-time matches.
  v0.1.3 - 2026-04-10 - Show one bounded distinguishing preview cue for similar prompt matches.
  v0.1.2 - 2026-04-06 - Improve untouched placeholder/raw draft titles with the shared heuristic.
  v0.1.1 - 2026-04-04 - Add advisory similar-prompt review actions to draft promotion.
  v0.1.0 - 2026-04-04 - Add bounded draft promotion dialog that preserves prompt provenance.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..prompt_preview import build_prompt_preview
from .base import resolve_draft_origin_title
from .quick_capture import parse_quick_capture_tags

VERY_CLOSE_SIMILARITY_THRESHOLD = 0.85
VERY_CLOSE_MATCH_CUE = "Very close match"
OPEN_SIMILAR_EXISTING_LABEL = "Open Existing Match"
OPEN_VERY_CLOSE_EXISTING_LABEL = "Open Very Close Match"

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Sequence
    from uuid import UUID

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
    allow_title_improvement: bool = False,
) -> Prompt:
    """Return an updated prompt with draft state cleared and provenance preserved."""
    cleaned_title = " ".join(title.strip().split())
    if allow_title_improvement:
        cleaned_title = resolve_draft_origin_title(
            cleaned_title,
            prompt.context or "",
        )
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
        similar_prompts: Sequence[Prompt] = (),
        parent: QWidget | None = None,
    ) -> None:
        """Initialise the compact promotion form from the existing prompt state."""
        super().__init__(parent)
        self._source_prompt = prompt
        self._similar_prompts = list(similar_prompts)
        self._result_prompt: Prompt | None = None
        self._selected_existing_prompt_id: UUID | None = None
        self.setWindowTitle("Promote Draft")
        self.setMinimumSize(560, 360)
        self.resize(640, 460)
        self._build_ui(categories)
        self._populate(prompt)
        self._populate_similar_prompts()

    @property
    def result_prompt(self) -> Prompt | None:
        """Return the promoted prompt after acceptance."""
        return self._result_prompt

    @property
    def selected_existing_prompt_id(self) -> UUID | None:
        """Return the similar prompt selected for opening, if any."""
        return self._selected_existing_prompt_id

    def _build_ui(self, categories: Sequence[str]) -> None:
        """Construct the compact bounded form."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        self._similarity_summary = QLabel(self)
        self._similarity_summary.setWordWrap(True)
        layout.addWidget(self._similarity_summary)

        self._similar_prompts_list = QListWidget(self)
        self._similar_prompts_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._similar_prompts_list.itemDoubleClicked.connect(self._open_selected_existing_prompt)  # type: ignore[arg-type]
        self._similar_prompts_list.itemSelectionChanged.connect(self._sync_open_existing_button)  # type: ignore[arg-type]
        layout.addWidget(self._similar_prompts_list)

        form = QFormLayout()
        form.setContentsMargins(0, 6, 0, 0)
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

        self._buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel, self)
        self._promote_button = self._buttons.addButton(
            "Promote as New",
            QDialogButtonBox.ButtonRole.AcceptRole,
        )
        self._open_existing_button = self._buttons.addButton(
            OPEN_SIMILAR_EXISTING_LABEL,
            QDialogButtonBox.ButtonRole.ActionRole,
        )
        self._buttons.accepted.connect(self._on_accept)  # type: ignore[arg-type]
        self._buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        self._open_existing_button.clicked.connect(self._open_selected_existing_prompt)  # type: ignore[arg-type]
        layout.addWidget(self._buttons)

    def _populate(self, prompt: Prompt) -> None:
        """Fill the form from the current prompt metadata."""
        self._title_input.setText(resolve_draft_origin_title(prompt.name, prompt.context or ""))
        self._title_input.setModified(False)
        category = prompt.category.strip() or "General"
        index = self._category_input.findText(category)
        if index >= 0:
            self._category_input.setCurrentIndex(index)
        else:
            self._category_input.setEditText(category)
        self._tags_input.setText(", ".join(prompt.tags))
        self._source_input.setText(prompt.source)
        self._description_input.setPlainText(prompt.description)

    def _populate_similar_prompts(self) -> None:
        """Render the advisory similar-prompt section."""
        self._similar_prompts_list.clear()
        if not self._similar_prompts:
            self._similarity_summary.setText(
                "No similar prompts found. You can promote this draft as a new prompt."
            )
            self._similar_prompts_list.hide()
            self._open_existing_button.hide()
            return

        if any(self._build_similarity_strength_cue(prompt) for prompt in self._similar_prompts):
            self._similarity_summary.setText(
                "A very close existing match may already exist. "
                "Review it before promoting this draft as a new prompt."
            )
        else:
            self._similarity_summary.setText(
                "Similar prompts already exist. "
                "Review one or continue promoting this draft as a new prompt."
            )
        self._similar_prompts_list.show()
        self._open_existing_button.show()

        for prompt in self._similar_prompts:
            item = QListWidgetItem(
                self._build_similar_prompt_label(prompt),
                self._similar_prompts_list,
            )
            item.setData(Qt.ItemDataRole.UserRole, str(prompt.id))
            item.setToolTip(self._build_similar_prompt_tooltip(prompt))

        if self._similar_prompts:
            self._similar_prompts_list.setCurrentRow(0)
        self._sync_open_existing_button()

    def _sync_open_existing_button(self) -> None:
        """Enable the open-existing action only when a similar prompt is selected."""
        row = self._similar_prompts_list.currentRow()
        self._open_existing_button.setEnabled(row >= 0)
        if row < 0 or row >= len(self._similar_prompts):
            self._open_existing_button.setText(OPEN_SIMILAR_EXISTING_LABEL)
            return
        strength_cue = self._build_similarity_strength_cue(self._similar_prompts[row])
        if strength_cue is not None:
            self._open_existing_button.setText(OPEN_VERY_CLOSE_EXISTING_LABEL)
            return
        self._open_existing_button.setText(OPEN_SIMILAR_EXISTING_LABEL)

    def _open_selected_existing_prompt(self) -> None:
        """Accept the dialog with the selected similar prompt target."""
        row = self._similar_prompts_list.currentRow()
        if row < 0 or row >= len(self._similar_prompts):
            return
        self._selected_existing_prompt_id = self._similar_prompts[row].id
        self._result_prompt = None
        self.accept()

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
                allow_title_improvement=not self._title_input.isModified(),
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Title required", str(exc))
            return
        self._selected_existing_prompt_id = None
        self.accept()

    @staticmethod
    def _build_similar_prompt_label(prompt: Prompt) -> str:
        """Return a compact identifying label for a similar prompt match."""
        category = prompt.category.strip() or "Uncategorised"
        label = f"{prompt.name} — {category}"
        cue_parts: list[str] = []
        strength_cue = DraftPromoteDialog._build_similarity_strength_cue(prompt)
        if strength_cue:
            cue_parts.append(strength_cue)
        preview = build_prompt_preview(prompt)
        if preview:
            cue_parts.append(preview)
        if cue_parts:
            return f"{label} · {' · '.join(cue_parts)}"
        return label

    @staticmethod
    def _build_similarity_strength_cue(prompt: Prompt) -> str | None:
        """Return one quiet visible cue only for very close similarity matches."""
        similarity = getattr(prompt, "similarity", None)
        if not isinstance(similarity, int | float):
            return None
        if float(similarity) < VERY_CLOSE_SIMILARITY_THRESHOLD:
            return None
        return VERY_CLOSE_MATCH_CUE

    @staticmethod
    def _build_similar_prompt_tooltip(prompt: Prompt) -> str:
        """Return a compact identifying tooltip for a similar prompt."""
        parts = [
            f"Category: {prompt.category.strip() or 'Uncategorised'}",
            f"Last modified: {DraftPromoteDialog._format_timestamp(prompt.last_modified)}",
        ]
        if prompt.tags:
            parts.append(f"Tags: {', '.join(prompt.tags)}")
        similarity = getattr(prompt, "similarity", None)
        if isinstance(similarity, int | float):
            parts.append(f"Similarity: {float(similarity):.2f}")
        return "\n".join(parts)

    @staticmethod
    def _format_timestamp(value: datetime) -> str:
        """Format timestamps in a compact UTC form for the advisory list."""
        return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")


__all__ = ["DraftPromoteDialog", "build_promoted_prompt", "is_prompt_draft"]
