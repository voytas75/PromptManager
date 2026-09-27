"""CAS-style prompt activity transition shared by repository and local CLI."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from .base import RepositoryError, RepositoryNotFoundError

if TYPE_CHECKING:
    import sqlite3
    import uuid


class PromptStatusConflictError(RepositoryError):
    """The caller's expected activity state is no longer current."""


def set_prompt_active(
    conn: sqlite3.Connection,
    prompt_id: uuid.UUID,
    *,
    active: bool,
    expect_active: bool,
) -> tuple[bool, str]:
    """Update only the status and timestamp under one SQLite write lock."""
    conn.execute("BEGIN IMMEDIATE")
    row = conn.execute(
        "SELECT is_active, last_modified FROM prompts WHERE id=?", (str(prompt_id),)
    ).fetchone()
    if row is None:
        raise RepositoryNotFoundError("Prompt not found")
    if row["is_active"] not in (0, 1) or not isinstance(row["last_modified"], str):
        raise RepositoryError("Prompt status metadata is invalid")
    try:
        previous = datetime.fromisoformat(row["last_modified"])
        if previous.tzinfo is None:
            raise ValueError("Timezone required")
        next_stamp = previous + timedelta(microseconds=1)
    except (ValueError, OverflowError) as exc:
        raise RepositoryError("Prompt status metadata is invalid") from exc
    current = bool(row["is_active"])
    if current != expect_active:
        raise PromptStatusConflictError("Prompt activity changed; reload before retrying")
    if current == active:
        return False, row["last_modified"]
    stamp = max(datetime.now(UTC), next_stamp).isoformat()
    updated = conn.execute(
        "UPDATE prompts SET is_active=?, last_modified=? WHERE id=? AND is_active=?",
        (int(active), stamp, str(prompt_id), int(expect_active)),
    )
    if updated.rowcount != 1:
        raise PromptStatusConflictError("Prompt activity changed; reload before retrying")
    return True, stamp
