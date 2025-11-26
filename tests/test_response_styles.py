"""ResponseStyle data model and repository integration tests.

Updates: v0.1.0 - 2025-12-05 - Cover ResponseStyle dataclass and CRUD workflows.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from core.repository import PromptRepository, RepositoryNotFoundError
from models.response_style import ResponseStyle


def _make_response_style(name: str = "Friendly Reviewer") -> ResponseStyle:
    """Return a populated ResponseStyle instance for tests."""

    now = datetime.now(timezone.utc)
    return ResponseStyle(
        id=uuid.uuid4(),
        name=name,
        description="Short, friendly summaries.",
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


def test_repository_crud(tmp_path) -> None:
    """Persist response styles and verify CRUD operations."""

    repo = PromptRepository(str(tmp_path / "repo.db"))
    style = _make_response_style()

    repo.add_response_style(style)
    stored = repo.get_response_style(style.id)
    assert stored.name == style.name

    style.description = "Updated description"
    style.tags.append("detailed")
    style.touch()
    repo.update_response_style(style)

    updated = repo.get_response_style(style.id)
    assert updated.description == "Updated description"
    assert "detailed" in updated.tags

    repo.delete_response_style(style.id)
    with pytest.raises(RepositoryNotFoundError):
        repo.get_response_style(style.id)


def test_repository_filters_and_search(tmp_path) -> None:
    """List response styles with inactive and search filters."""

    repo = PromptRepository(str(tmp_path / "repo.db"))
    active = _make_response_style("Active Style")
    inactive = _make_response_style("Inactive Style")
    inactive.is_active = False
    inactive.description = "Formal legal voice."

    repo.add_response_style(active)
    repo.add_response_style(inactive)

    visible = repo.list_response_styles()
    assert [style.name for style in visible] == ["Active Style"]

    all_styles = repo.list_response_styles(include_inactive=True)
    assert {style.name for style in all_styles} == {"Active Style", "Inactive Style"}

    searched = repo.list_response_styles(include_inactive=True, search="legal")
    assert len(searched) == 1
    assert searched[0].name == "Inactive Style"
