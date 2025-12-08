"""Qt list model that exposes prompt summaries for list views.

Updates:
  v0.1.1 - 2025-12-08 - Align Qt override signatures and guard similarity conversion.
  v0.1.0 - 2025-11-30 - Extract PromptListModel into its own module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QAbstractListModel, QModelIndex, QPersistentModelIndex, Qt

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from collections.abc import Iterable, Sequence

    from models.prompt_model import Prompt


class PromptListModel(QAbstractListModel):
    """List model providing prompt summaries for the QListView."""

    def __init__(self, prompts: Sequence[Prompt] | None = None, parent=None) -> None:
        """Initialise the model with optional starting *prompts*."""
        super().__init__(parent)
        self._prompts: list[Prompt] = list(prompts or [])

    def rowCount(
        self,
        parent: QModelIndex | QPersistentModelIndex = QModelIndex(),
    ) -> int:  # noqa: N802 - Qt API
        """Return the number of prompts available for the view."""
        if parent.isValid():
            return 0
        return len(self._prompts)

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> str | None:  # noqa: N802
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
