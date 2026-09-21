"""Durable prompt-asset activity records."""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from .base import RepositoryError, connect as _connect, stringify_uuid as _stringify_uuid

if TYPE_CHECKING:
    from pathlib import Path

PromptActivityOperation = Literal["created", "updated", "forked", "restored", "deleted"]
PromptActivityOrigin = Literal["cli", "gui"]


@dataclass(frozen=True, slots=True)
class PromptActivity:
    """One successful prompt-asset mutation recorded locally."""

    id: int
    prompt_id: uuid.UUID
    operation: PromptActivityOperation
    origin: PromptActivityOrigin
    occurred_at: datetime


class PromptActivityStoreMixin:
    """Persist and retrieve compact prompt mutation evidence."""

    _db_path: Path

    def record_prompt_activity(
        self,
        prompt_id: uuid.UUID,
        *,
        operation: PromptActivityOperation,
        origin: PromptActivityOrigin,
    ) -> PromptActivity:
        """Record one successful prompt mutation without storing prompt content."""
        timestamp = datetime.now(UTC).isoformat()
        try:
            with _connect(self._db_path) as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO prompt_activity_events (prompt_id, operation, origin, occurred_at)
                    VALUES (?, ?, ?, ?);
                    """,
                    (_stringify_uuid(prompt_id), operation, origin, timestamp),
                )
                row = conn.execute(
                    """
                    SELECT id, prompt_id, operation, origin, occurred_at
                    FROM prompt_activity_events WHERE id = ?;
                    """,
                    (cursor.lastrowid,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to record prompt activity") from exc
        if row is None:  # pragma: no cover - defensive
            raise RepositoryError("Prompt activity insert succeeded but row missing")
        return self._row_to_prompt_activity(row)

    def list_prompt_activity(
        self,
        prompt_id: uuid.UUID | None = None,
        *,
        limit: int | None = None,
    ) -> list[PromptActivity]:
        """Return newest-first activity, optionally limited to one prompt."""
        query = "SELECT id, prompt_id, operation, origin, occurred_at FROM prompt_activity_events"
        params: list[object] = []
        if prompt_id is not None:
            query += " WHERE prompt_id = ?"
            params.append(_stringify_uuid(prompt_id))
        query += " ORDER BY id DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(0, int(limit)))
        try:
            with _connect(self._db_path) as conn:
                rows = conn.execute(query + ";", params).fetchall()
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to load prompt activity") from exc
        return [self._row_to_prompt_activity(row) for row in rows]

    @staticmethod
    def _row_to_prompt_activity(row: sqlite3.Row) -> PromptActivity:
        return PromptActivity(
            id=int(row["id"]),
            prompt_id=uuid.UUID(str(row["prompt_id"])),
            operation=row["operation"],
            origin=row["origin"],
            occurred_at=datetime.fromisoformat(str(row["occurred_at"])),
        )
