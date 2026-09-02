"""Qt list model that exposes prompt summaries for list views.

Updates:
  v0.1.6 - 2026-09-02 - Expose description-match reason and inspect-first handoff cues.
  v0.1.5 - 2026-04-12 - Let active-search source-matched previews reuse the existing preview role.
  v0.1.4 - 2026-04-12 - Expose bounded active-search match spans for prompt title and preview text.
  v0.1.3 - 2026-04-10 - Reuse a shared prompt-preview helper across UI surfaces.
  v0.1.2 - 2026-04-06 - Add bounded retrieval-preview roles derived from existing prompt data.
  v0.1.1 - 2025-12-08 - Align Qt override signatures and guard similarity conversion.
  v0.1.0 - 2025-11-30 - Extract PromptListModel into its own module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QAbstractListModel, QModelIndex, QObject, QPersistentModelIndex, Qt

from .prompt_preview import (
    PREVIEW_MAX_LENGTH,
    build_prompt_preview,
    build_prompt_source_cue,
    flatten_preview_text,
    is_credible_preview_text,
    truncate_preview_text,
)

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from collections.abc import Iterable, Sequence

    from models.prompt_model import Prompt


class PromptListModel(QAbstractListModel):
    """List model providing prompt summaries for the QListView."""

    PromptRole = int(Qt.ItemDataRole.UserRole)
    PreviewRole = int(Qt.ItemDataRole.UserRole) + 1
    TitleMatchRole = int(Qt.ItemDataRole.UserRole) + 2
    PreviewMatchRole = int(Qt.ItemDataRole.UserRole) + 3
    MatchReasonRole = int(Qt.ItemDataRole.UserRole) + 4
    HandoffCueRole = int(Qt.ItemDataRole.UserRole) + 5
    PreviewMaxLength = PREVIEW_MAX_LENGTH

    def __init__(
        self,
        prompts: Sequence[Prompt] | None = None,
        parent: QObject | None = None,
    ) -> None:
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
            return self._preview_text(prompt)
        if role == self.MatchReasonRole:
            return self._match_reason_text(prompt)
        if role == self.HandoffCueRole:
            return self._handoff_cue_text(prompt)
        if role == self.TitleMatchRole:
            return self._match_ranges(self._display_text(prompt))
        if role == self.PreviewMatchRole:
            return self._match_ranges(self._preview_text(prompt))
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
            [
                self.PreviewRole,
                self.MatchReasonRole,
                self.HandoffCueRole,
                self.TitleMatchRole,
                self.PreviewMatchRole,
            ],
        )

    def prompts(self) -> Sequence[Prompt]:
        """Expose the underlying prompts for selection helpers."""
        return tuple(self._prompts)

    def _preview_text(self, prompt: Prompt) -> str | None:
        """Return the current bounded preview text for *prompt*."""
        return build_prompt_preview(prompt, active_search_terms=self._active_search_terms)

    def _match_reason_text(self, prompt: Prompt) -> str | None:
        """Return one compact active-search reason cue for the current visible preview path."""
        if not self._active_search_terms:
            return None

        preview = self._preview_text(prompt)
        if not preview:
            return None
        if preview == build_prompt_source_cue(prompt.source):
            return "Matched in source"
        if preview in self._matching_scenario_previews(prompt):
            return "Matched in scenario"
        if self._match_ranges(self._display_text(prompt)):
            return "Matched in title"
        if self._description_matches_active_search(prompt, preview):
            return "Matched in description"
        return None

    def _description_matches_active_search(self, prompt: Prompt, preview: str) -> bool:
        """Return whether the visible preview is the description with an active search match."""
        description = flatten_preview_text(prompt.description)
        if not description or not is_credible_preview_text(description):
            return False
        if preview != truncate_preview_text(description):
            return False
        lowered_description = description.casefold()
        return any(term in lowered_description for term in self._active_search_terms)

    def _matching_scenario_previews(self, prompt: Prompt) -> tuple[str, ...]:
        """Return truncated scenario previews that match the active search terms."""
        if not self._active_search_terms:
            return ()

        previews: list[str] = []
        for scenario in prompt.scenarios:
            normalized = flatten_preview_text(str(scenario))
            if not normalized or not is_credible_preview_text(normalized):
                continue
            lowered = normalized.casefold()
            if not any(term in lowered for term in self._active_search_terms):
                continue
            previews.append(truncate_preview_text(normalized))
        return tuple(previews)

    def _handoff_cue_text(self, prompt: Prompt) -> str | None:
        """Return one compact list-local handoff cue."""
        match_reason = self._match_reason_text(prompt)
        if match_reason == "Matched in title":
            return "Ready to reuse"
        if match_reason in {"Matched in source", "Matched in scenario", "Matched in description"}:
            return "Inspect before reuse"
        return None

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
