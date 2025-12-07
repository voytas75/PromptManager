"""Prompt note persistence helpers.

Updates:
  v0.11.0 - 2025-12-04 - Extract prompt note CRUD into mixin.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from models.prompt_note import PromptNote

from .base import (
    RepositoryError,
    RepositoryNotFoundError,
    connect as _connect,
    stringify_uuid as _stringify_uuid,
)

if TYPE_CHECKING:
    import uuid


class PromptNoteStoreMixin:
    """CRUD helpers for prompt notes."""

    _db_path: Path

    _NOTE_COLUMNS: ClassVar[tuple[str, ...]] = ("id", "note", "created_at", "last_modified")

    def add_prompt_note(self, note: PromptNote) -> PromptNote:
        """Insert a new prompt note."""
        payload = self._note_to_row(note)
        column_list = ", ".join(self._NOTE_COLUMNS)
        placeholders = ", ".join(f":{column}" for column in self._NOTE_COLUMNS)
        query = f"INSERT INTO prompt_notes ({column_list}) VALUES ({placeholders});"
        try:
            with _connect(self._db_path) as conn:
                conn.execute(query, payload)
        except sqlite3.IntegrityError as exc:
            raise RepositoryError(f"Prompt note {note.id} already exists") from exc
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to insert prompt note {note.id}") from exc
        return note

    def update_prompt_note(self, note: PromptNote) -> PromptNote:
        """Update an existing prompt note."""
        payload = self._note_to_row(note)
        assignments = ", ".join(
            f"{column} = :{column}" for column in self._NOTE_COLUMNS if column != "id"
        )
        query = f"UPDATE prompt_notes SET {assignments} WHERE id = :id;"
        try:
            with _connect(self._db_path) as conn:
                updated = conn.execute(query, payload).rowcount
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to update prompt note {note.id}") from exc
        if updated == 0:
            raise RepositoryNotFoundError(f"Prompt note {note.id} not found")
        return note

    def get_prompt_note(self, note_id: uuid.UUID) -> PromptNote:
        """Return a prompt note by identifier."""
        try:
            with _connect(self._db_path) as conn:
                row = conn.execute(
                    "SELECT * FROM prompt_notes WHERE id = ?;",
                    (_stringify_uuid(note_id),),
                ).fetchone()
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to load prompt note {note_id}") from exc
        if row is None:
            raise RepositoryNotFoundError(f"Prompt note {note_id} not found")
        return self._row_to_note(row)

    def list_prompt_notes(self) -> list[PromptNote]:
        """Return stored prompt notes ordered by modification time."""
        try:
            with _connect(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT * FROM prompt_notes ORDER BY last_modified DESC;"
                ).fetchall()
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to list prompt notes") from exc
        return [self._row_to_note(row) for row in rows]

    def delete_prompt_note(self, note_id: uuid.UUID) -> None:
        """Delete a prompt note."""
        try:
            with _connect(self._db_path) as conn:
                deleted = conn.execute(
                    "DELETE FROM prompt_notes WHERE id = ?;",
                    (_stringify_uuid(note_id),),
                ).rowcount
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to delete prompt note {note_id}") from exc
        if deleted == 0:
            raise RepositoryNotFoundError(f"Prompt note {note_id} not found")

    def _note_to_row(self, note: PromptNote) -> dict[str, str]:
        """Serialise PromptNote to SQLite mapping."""
        record = note.to_record()
        return {
            "id": record["id"],
            "note": record["note"],
            "created_at": record["created_at"],
            "last_modified": record["last_modified"],
        }

    def _row_to_note(self, row: sqlite3.Row) -> PromptNote:
        """Hydrate PromptNote from SQLite row."""
        payload = {column: row[column] for column in row.keys()}
        return PromptNote.from_record(payload)


__all__ = ["PromptNoteStoreMixin"]
