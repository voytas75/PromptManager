"""ResponseStyle data model and repository integration tests.

Updates: v0.1.0 - 2025-12-05 - Cover ResponseStyle dataclass and CRUD workflows.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from core.repository import PromptRepository, RepositoryNotFoundError
from models.response_style import ResponseStyle


def _make_response_style(name: str = "Friendly Reviewer") -> ResponseStyle:
    """Return a populated ResponseStyle instance for tests."""
    now = datetime.now(UTC)
    return ResponseStyle(
        id=uuid.uuid4(),
        name=name,
        description="Short, friendly summaries.",
        prompt_part="System Instruction",
        tone="friendly",
        voice="mentor",
        format_instructions="Use bullet lists.",
        guidelines="Keep explanations under 3 sentences.",
        tags=["friendly", "summary"],
        examples=["Example response"],
        metadata={"format": "markdown"},
        version="1.0",
        created_at=now,
        last_modified=now,
    )


def test_response_style_roundtrip() -> None:
    """Ensure ResponseStyle serialization stays lossless."""
    style = _make_response_style()
    record = style.to_record()
    loaded = ResponseStyle.from_record(record)

    assert loaded.id == style.id
    assert loaded.tags == style.tags
    assert loaded.metadata == style.metadata
    assert loaded.prompt_part == style.prompt_part


def test_repository_crud(tmp_path: Path) -> None:
    """Persist response styles and verify CRUD operations."""
    repo = PromptRepository(str(tmp_path / "repo.db"))
    style = _make_response_style()

    repo.add_response_style(style)
    stored = repo.get_response_style(style.id)
    assert stored.name == style.name
    assert stored.prompt_part == style.prompt_part

    style.description = "Updated description"
    style.tags.append("detailed")
    style.prompt_part = "Output Formatter"
    style.touch()
    repo.update_response_style(style)

    updated = repo.get_response_style(style.id)
    assert updated.description == "Updated description"
    assert "detailed" in updated.tags
    assert updated.prompt_part == "Output Formatter"

    repo.delete_response_style(style.id)
    with pytest.raises(RepositoryNotFoundError):
        repo.get_response_style(style.id)


def test_repository_filters_and_search(tmp_path: Path) -> None:
    """List response styles with inactive and search filters."""
    repo = PromptRepository(str(tmp_path / "repo.db"))
    active = _make_response_style("Active Style")
    active.prompt_part = "Output Formatter"
    inactive = _make_response_style("Inactive Style")
    inactive.is_active = False
    inactive.description = "Formal legal voice."
    inactive.prompt_part = "Legal Brief"

    repo.add_response_style(active)
    repo.add_response_style(inactive)

    visible = repo.list_response_styles()
    assert [style.name for style in visible] == ["Active Style"]

    all_styles = repo.list_response_styles(include_inactive=True)
    assert {style.name for style in all_styles} == {"Active Style", "Inactive Style"}

    searched = repo.list_response_styles(include_inactive=True, search="legal")
    assert len(searched) == 1
    assert searched[0].name == "Inactive Style"

    part_search = repo.list_response_styles(include_inactive=True, search="formatter")
    assert len(part_search) == 1
    assert part_search[0].name == "Active Style"


def test_response_style_snippet_is_independent_of_supporting_fields(tmp_path: Path) -> None:
    """The canonical fragment must survive a SQLite round trip unchanged."""
    repo = PromptRepository(str(tmp_path / "repo.db"))
    style = _make_response_style()
    style.snippet = "Original system instruction"
    repo.add_response_style(style)

    loaded = repo.get_response_style(style.id)
    assert loaded.snippet == "Original system instruction"
    assert loaded.description == "Short, friendly summaries."
    assert loaded.format_instructions == "Use bullet lists."
    assert loaded.examples == ["Example response"]

    loaded.snippet = "Revised system instruction"
    repo.update_response_style(loaded)
    assert repo.get_response_style(style.id).snippet == "Revised system instruction"


def test_legacy_response_style_migration_preserves_original_fields(tmp_path: Path) -> None:
    """Old records acquire the best available text exactly once on open."""
    db = tmp_path / "legacy.db"
    repo = PromptRepository(str(db))
    formatted = _make_response_style("Formatted")
    formatted.format_instructions = "  Keep spacing.\n  Follow this.  "
    described = _make_response_style("Described")
    described.format_instructions = None
    repo.add_response_style(formatted)
    repo.add_response_style(described)
    with sqlite3.connect(db) as conn:
        conn.execute("ALTER TABLE response_styles DROP COLUMN snippet")

    migrated = PromptRepository(str(db))
    assert migrated.get_response_style(formatted.id).snippet == formatted.format_instructions
    assert migrated.get_response_style(described.id).snippet == described.description
    assert migrated.get_response_style(described.id).format_instructions is None
    reopened = PromptRepository(str(db))
    assert reopened.get_response_style(formatted.id).snippet == formatted.format_instructions
