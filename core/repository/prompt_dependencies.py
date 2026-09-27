"""Fail-closed relation checks before physically deleting a prompt asset."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, cast

from .base import RepositoryError

if TYPE_CHECKING:
    import sqlite3


class PromptDeleteBlockedError(RepositoryError):
    """A prompt has dependents; no prompt body or invalid raw metadata is exposed."""

    def __init__(self, kinds: tuple[str, ...]) -> None:
        """Capture bounded dependency kinds for the refused deletion."""
        self.kinds = kinds
        super().__init__("Prompt has dependencies: " + ", ".join(kinds) + ". Deactivate instead.")


def deletion_dependencies(conn: sqlite3.Connection, prompt_id: uuid.UUID | str) -> tuple[str, ...]:
    """Return dependency kinds from one SQLite view, refusing ambiguous metadata."""
    target = str(prompt_id)
    kinds: list[str] = []
    if conn.execute(
        "SELECT 1 FROM prompt_forks WHERE source_prompt_id = ? LIMIT 1", (target,)
    ).fetchone():
        kinds.append("fork")
    if conn.execute(
        "SELECT 1 FROM prompt_chain_steps WHERE prompt_id = ? LIMIT 1", (target,)
    ).fetchone():
        kinds.append("chain")
    related_found = False
    for row in conn.execute("SELECT related_prompts FROM prompts"):
        raw: object = row[0]
        if raw is None:
            continue
        if not isinstance(raw, str):
            raise RepositoryError("Cannot validate prompt relations; catalog metadata is invalid")
        try:
            values: object = json.loads(raw)
        except (ValueError, TypeError) as exc:
            raise RepositoryError(
                "Cannot validate prompt relations; catalog metadata is invalid"
            ) from exc
        if not isinstance(values, list):
            raise RepositoryError("Cannot validate prompt relations; catalog metadata is invalid")
        raw_items = cast("list[object]", values)
        if not all(isinstance(item, str) for item in raw_items):
            raise RepositoryError("Cannot validate prompt relations; catalog metadata is invalid")
        for item in cast("list[str]", raw_items):
            try:
                normalized = str(uuid.UUID(item))
            except (ValueError, TypeError):
                # An unrelated legacy non-UUID reference cannot point to a canonical UUID.
                continue
            if normalized == target:
                related_found = True
    if related_found:
        kinds.append("related")
    return tuple(kinds)
