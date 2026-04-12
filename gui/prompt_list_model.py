"""Qt list model that exposes prompt summaries for list views.

Updates:
  v0.1.4 - 2026-04-12 - Expose bounded active-search match spans for prompt title and preview text.
  v0.1.3 - 2026-04-10 - Reuse a shared prompt-preview helper across UI surfaces.
  v0.1.2 - 2026-04-06 - Add bounded retrieval-preview roles derived from existing prompt data.
  v0.1.1 - 2025-12-08 - Align Qt override signatures and guard similarity conversion.
  v0.1.0 - 2025-11-30 - Extract PromptListModel into its own module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QAbstractListModel, QModelIndex, QPersistentModelIndex, Qt

from .prompt_preview import PREVIEW_MAX_LENGTH, build_prompt_preview

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from collections.abc import Iterable, Sequence

    from models.prompt_model import Prompt


class PromptListModel(QAbstractListModel):
    """List model providing prompt summaries for the QListView."""

    PromptRole = int(Qt.ItemDataRole.UserRole)
    PreviewRole = int(Qt.ItemDataRole.UserRole) + 1
    TitleMatchRole = int(Qt.ItemDataRole.UserRole) + 2
    PreviewMatchRole = int(Qt.ItemDataRole.UserRole) + 3
    PreviewMaxLength = PREVIEW_MAX_LENGTH

    def __init__(self, prompts: Sequence[Prompt] | None = None, parent=None) -> None:
        """Initialise the model with optional starting *prompts*."""
        super().__init__(parent)
        self._prompts: list[Prompt] = list(prompts or [])
        self._active_search_terms: tuple[str, ...] = ()

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
            return self._display_text(prompt)
        if role == self.PromptRole:
            return prompt
        if role == self.PreviewRole:
            return build_prompt_preview(prompt)
        if role == self.TitleMatchRole:
            return self._match_ranges(self._display_text(prompt))
        if role == self.PreviewMatchRole:
            return self._match_ranges(build_prompt_preview(prompt))
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

    def set_active_search_text(self, search_text: str) -> None:
        """Store the current plain-text search so rows can expose bounded match spans."""
        active_terms = self._normalise_search_terms(search_text)
        if active_terms == self._active_search_terms:
            return
        self._active_search_terms = active_terms
        if not self._prompts:
            return

        top_left = self.index(0, 0)
        bottom_right = self.index(len(self._prompts) - 1, 0)
        self.dataChanged.emit(
            top_left,
            bottom_right,
            [self.TitleMatchRole, self.PreviewMatchRole],
        )

    def prompts(self) -> Sequence[Prompt]:
        """Expose the underlying prompts for selection helpers."""
        return tuple(self._prompts)

    @staticmethod
    def _display_text(prompt: Prompt) -> str:
        """Return the existing single-line prompt label used by the list view."""
        category = f" ({prompt.category})" if prompt.category else ""
        similarity_suffix = ""
        similarity_value = getattr(prompt, "similarity", None)
        if isinstance(similarity_value, (int, float)):
            similarity = float(similarity_value)
            similarity_suffix = f" [{similarity:.4f}]"
        return f"{prompt.name}{category}{similarity_suffix}"

    @staticmethod
    def _normalise_search_terms(search_text: str) -> tuple[str, ...]:
        """Return unique case-folded plain-text search terms."""
        terms: list[str] = []
        for raw_term in search_text.split():
            term = raw_term.strip().casefold()
            if term and term not in terms:
                terms.append(term)
        return tuple(terms)

    def _match_ranges(self, text: str | None) -> tuple[tuple[int, int], ...]:
        """Return non-overlapping text spans matching the active plain-text search."""
        if not text or not self._active_search_terms:
            return ()

        lowered_text = text.casefold()
        occupied = [False] * len(text)
        spans: list[tuple[int, int]] = []
        for term in sorted(self._active_search_terms, key=lambda value: (-len(value), value)):
            start = 0
            while True:
                index = lowered_text.find(term, start)
                if index < 0:
                    break
                end = index + len(term)
                if not any(occupied[index:end]):
                    spans.append((index, len(term)))
                    for offset in range(index, end):
                        occupied[offset] = True
                start = end
        spans.sort()
        return tuple(spans)


__all__ = ["PromptListModel"]
