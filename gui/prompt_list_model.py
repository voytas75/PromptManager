"""Qt list model that exposes prompt summaries for list views.

Updates:
  v0.1.2 - 2026-04-06 - Add bounded retrieval-preview roles derived from existing prompt data.
  v0.1.1 - 2025-12-08 - Align Qt override signatures and guard similarity conversion.
  v0.1.0 - 2025-11-30 - Extract PromptListModel into its own module.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from PySide6.QtCore import QAbstractListModel, QModelIndex, QPersistentModelIndex, Qt

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from collections.abc import Iterable, Sequence

    from models.prompt_model import Prompt

_PREVIEW_MAX_LENGTH = 96
_SOURCE_PREFIX = "Source: "
_LOW_SIGNAL_SOURCE_VALUES = {
    "",
    "-",
    "local",
    "n/a",
    "na",
    "none",
    "promptmanager",
    "prompt manager",
    "quick_capture",
    "unknown",
}


def _flatten_preview_text(value: str) -> str:
    """Collapse multi-line prompt metadata into a single readable preview line."""
    return re.sub(r"\s+", " ", value).strip()


def _truncate_preview_text(value: str, *, limit: int = _PREVIEW_MAX_LENGTH) -> str:
    """Return a deterministically truncated preview string."""
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip(" ,.;:-") + "..."


def _is_credible_preview_text(value: str, *, minimum_length: int = 10) -> bool:
    """Return whether *value* is strong enough to use as retrieval preview text."""
    if len(value) < minimum_length:
        return False
    if not any(character.isalpha() for character in value):
        return False
    return True


def _build_prompt_preview(prompt: Prompt) -> str | None:
    """Derive one compact preview from existing prompt data in priority order."""
    name_key = prompt.name.strip().casefold()

    description = _flatten_preview_text(prompt.description)
    if (
        description
        and description.casefold() != name_key
        and _is_credible_preview_text(description)
    ):
        return _truncate_preview_text(description)

    for scenario in prompt.scenarios:
        normalized = _flatten_preview_text(str(scenario))
        if normalized and _is_credible_preview_text(normalized):
            return _truncate_preview_text(normalized)

    source = _flatten_preview_text(prompt.source)
    if source and source.casefold() not in _LOW_SIGNAL_SOURCE_VALUES:
        preview = _SOURCE_PREFIX + source
        if _is_credible_preview_text(preview, minimum_length=len(_SOURCE_PREFIX) + 3):
            return _truncate_preview_text(preview)
    return None


class PromptListModel(QAbstractListModel):
    """List model providing prompt summaries for the QListView."""

    PromptRole = int(Qt.ItemDataRole.UserRole)
    PreviewRole = int(Qt.ItemDataRole.UserRole) + 1
    PreviewMaxLength = _PREVIEW_MAX_LENGTH

    def __init__(self, prompts: Sequence[Prompt] | None = None, parent=None) -> None:
        """Initialise the model with optional starting *prompts*."""
        super().__init__(parent)
        self._prompts: list[Prompt] = list(prompts or [])

    def rowCount(
        self,
        parent: QModelIndex | QPersistentModelIndex | None = None,
    ) -> int:  # noqa: N802 - Qt API
        """Return the number of prompts available for the view."""
        parent_index = parent or QModelIndex()
        if parent_index.isValid():
            return 0
        return len(self._prompts)

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> object | None:  # noqa: N802
        """Return the decorated prompt label for the requested index."""
        if not index.isValid() or index.row() >= len(self._prompts):
            return None
        prompt = self._prompts[index.row()]
        if role in {Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole}:
            category = f" ({prompt.category})" if prompt.category else ""
            similarity_suffix = ""
            similarity_value = getattr(prompt, "similarity", None)
            if isinstance(similarity_value, (int, float)):
                similarity = float(similarity_value)
                similarity_suffix = f" [{similarity:.4f}]"
            return f"{prompt.name}{category}{similarity_suffix}"
        if role == self.PromptRole:
            return prompt
        if role == self.PreviewRole:
            return _build_prompt_preview(prompt)
        return None

    def prompt_at(self, row: int) -> Prompt | None:
        """Return the prompt at the given list index."""
        if 0 <= row < len(self._prompts):
            return self._prompts[row]
        return None

    def set_prompts(self, prompts: Iterable[Prompt]) -> None:
        """Replace the backing prompt list and notify listeners."""
        self.beginResetModel()
        self._prompts = list(prompts)
        self.endResetModel()

    def prompts(self) -> Sequence[Prompt]:
        """Expose the underlying prompts for selection helpers."""
        return tuple(self._prompts)


__all__ = ["PromptListModel"]
