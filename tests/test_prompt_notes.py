"""Prompt note data model and repository tests.

Updates: v0.1.0 - 2025-12-06 - Cover PromptNote dataclass and CRUD workflows.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from core.repository import PromptRepository, RepositoryNotFoundError
from models.prompt_note import PromptNote


def _make_note(text: str = "Remember to test edge cases") -> PromptNote:
    timestamp = datetime.now(UTC)
    return PromptNote(id=uuid.uuid4(), note=text, created_at=timestamp, last_modified=timestamp)


def test_prompt_note_roundtrip() -> None:
    note = _make_note()
    record = note.to_record()
    loaded = PromptNote.from_record(record)
    assert loaded.id == note.id
    assert loaded.note == note.note


def test_repository_crud(tmp_path) -> None:
    repo = PromptRepository(str(tmp_path / "notes.db"))
    note = _make_note("Ship release notes")

    repo.add_prompt_note(note)
    stored = repo.get_prompt_note(note.id)
    assert stored.note == "Ship release notes"

    note.note = "Ship release notes + QA checklist"
    note.touch()
    repo.update_prompt_note(note)

    updated = repo.get_prompt_note(note.id)
    assert "QA" in updated.note

    notes = repo.list_prompt_notes()
    assert notes and notes[0].id == note.id

    repo.delete_prompt_note(note.id)
    with pytest.raises(RepositoryNotFoundError):
        repo.get_prompt_note(note.id)
