"""Prompt note helpers for Prompt Manager.

Updates:
  v0.1.0 - 2025-12-02 - Extract prompt note APIs into mixin for modularisation.
"""
from __future__ import annotations

import uuid

from models.prompt_note import PromptNote

from ..exceptions import (
    PromptNoteNotFoundError,
    PromptNoteStorageError,
)
from ..repository import PromptRepository, RepositoryError, RepositoryNotFoundError

__all__ = ["PromptNoteSupport"]


class PromptNoteSupport:
    """Mixin exposing prompt note CRUD helpers backed by the repository."""

    _repository: PromptRepository

    def list_prompt_notes(self) -> list[PromptNote]:
        """Return stored prompt notes ordered by recency."""
        try:
            return self._repository.list_prompt_notes()
        except RepositoryError as exc:
            raise PromptNoteStorageError("Unable to list prompt notes") from exc

    def get_prompt_note(self, note_id: uuid.UUID) -> PromptNote:
        """Return a single prompt note by identifier."""
        try:
            return self._repository.get_prompt_note(note_id)
        except RepositoryNotFoundError as exc:
            raise PromptNoteNotFoundError(str(exc)) from exc
        except RepositoryError as exc:
            raise PromptNoteStorageError(f"Unable to load prompt note {note_id}") from exc

    def create_prompt_note(self, note: PromptNote) -> PromptNote:
        """Persist a new prompt note."""
        note.touch()
        try:
            return self._repository.add_prompt_note(note)
        except RepositoryError as exc:
            raise PromptNoteStorageError(f"Failed to persist prompt note {note.id}") from exc

    def update_prompt_note(self, note: PromptNote) -> PromptNote:
        """Update an existing prompt note."""
        note.touch()
        try:
            return self._repository.update_prompt_note(note)
        except RepositoryNotFoundError as exc:
            raise PromptNoteNotFoundError(str(exc)) from exc
        except RepositoryError as exc:
            raise PromptNoteStorageError(f"Failed to update prompt note {note.id}") from exc

    def delete_prompt_note(self, note_id: uuid.UUID) -> None:
        """Delete a stored prompt note."""
        try:
            self._repository.delete_prompt_note(note_id)
        except RepositoryNotFoundError as exc:
            raise PromptNoteNotFoundError(str(exc)) from exc
        except RepositoryError as exc:
            raise PromptNoteStorageError(f"Failed to delete prompt note {note_id}") from exc
