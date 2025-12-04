"""Response style persistence helpers.

Updates:
  v0.11.0 - 2025-12-04 - Extract response style CRUD into mixin.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any, ClassVar

from models.response_style import ResponseStyle

from .base import (
    RepositoryError,
    RepositoryNotFoundError,
    connect as _connect,
    json_dumps as _json_dumps,
    json_loads_list as _json_loads_list,
    json_loads_optional as _json_loads_optional,
    stringify_uuid as _stringify_uuid,
)

if TYPE_CHECKING:
    import uuid


class ResponseStyleStoreMixin:
    """CRUD helpers for response style definitions."""

    _RESPONSE_STYLE_COLUMNS: ClassVar[tuple[str, ...]] = (
        "id",
        "name",
        "description",
        "prompt_part",
        "tone",
        "voice",
        "format_instructions",
        "guidelines",
        "tags",
        "examples",
        "metadata",
        "is_active",
        "version",
        "created_at",
        "last_modified",
        "ext1",
        "ext2",
        "ext3",
    )

    def add_response_style(self, style: ResponseStyle) -> ResponseStyle:
        """Insert a new response style record."""
        payload = self._response_style_to_row(style)
        placeholders = ", ".join(f":{column}" for column in self._RESPONSE_STYLE_COLUMNS)
        query = (
            f"INSERT INTO response_styles ({', '.join(self._RESPONSE_STYLE_COLUMNS)}) "
            f"VALUES ({placeholders});"
        )
        try:
            with _connect(self._db_path) as conn:
                conn.execute(query, payload)
        except sqlite3.IntegrityError as exc:
            raise RepositoryError(f"Response style {style.id} already exists") from exc
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to insert response style {style.id}") from exc
        return style

    def update_response_style(self, style: ResponseStyle) -> ResponseStyle:
        """Update an existing response style record."""
        payload = self._response_style_to_row(style)
        assignments = ", ".join(
            f"{column} = :{column}" for column in self._RESPONSE_STYLE_COLUMNS if column != "id"
        )
        query = f"UPDATE response_styles SET {assignments} WHERE id = :id;"
        try:
            with _connect(self._db_path) as conn:
                updated = conn.execute(query, payload).rowcount
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to update response style {style.id}") from exc
        if updated == 0:
            raise RepositoryNotFoundError(f"Response style {style.id} not found")
        return style

    def get_response_style(self, style_id: uuid.UUID) -> ResponseStyle:
        """Return a response style by identifier."""
        try:
            with _connect(self._db_path) as conn:
                row = conn.execute(
                    "SELECT * FROM response_styles WHERE id = ?;",
                    (_stringify_uuid(style_id),),
                ).fetchone()
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to load response style {style_id}") from exc
        if row is None:
            raise RepositoryNotFoundError(f"Response style {style_id} not found")
        return self._row_to_response_style(row)

    def list_response_styles(
        self,
        *,
        include_inactive: bool = False,
        search: str | None = None,
    ) -> list[ResponseStyle]:
        """Return stored response styles ordered by case-insensitive name."""
        query = "SELECT * FROM response_styles"
        clauses: list[str] = []
        params: list[Any] = []
        if not include_inactive:
            clauses.append("is_active = 1")
        if search:
            clauses.append(
                "(LOWER(name) LIKE ? OR LOWER(description) LIKE ? OR LOWER(prompt_part) LIKE ?)"
            )
            pattern = f"%{search.lower()}%"
            params.extend([pattern, pattern, pattern])
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY name COLLATE NOCASE;"
        try:
            with _connect(self._db_path) as conn:
                rows = conn.execute(query, params).fetchall()
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to list response styles") from exc
        return [self._row_to_response_style(row) for row in rows]

    def delete_response_style(self, style_id: uuid.UUID) -> None:
        """Delete a stored response style."""
        try:
            with _connect(self._db_path) as conn:
                deleted = conn.execute(
                    "DELETE FROM response_styles WHERE id = ?;",
                    (_stringify_uuid(style_id),),
                ).rowcount
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to delete response style {style_id}") from exc
        if deleted == 0:
            raise RepositoryNotFoundError(f"Response style {style_id} not found")

    def _response_style_to_row(self, style: ResponseStyle) -> dict[str, Any]:
        """Serialise ResponseStyle into SQLite-compatible mapping."""
        record = style.to_record()
        return {
            "id": record["id"],
            "name": record["name"],
            "description": record["description"],
            "prompt_part": record["prompt_part"],
            "tone": record["tone"],
            "voice": record["voice"],
            "format_instructions": record["format_instructions"],
            "guidelines": record["guidelines"],
            "tags": _json_dumps(record["tags"]) or "[]",
            "examples": _json_dumps(record["examples"]) or "[]",
            "metadata": _json_dumps(record["metadata"]),
            "is_active": record["is_active"],
            "version": record["version"],
            "created_at": record["created_at"],
            "last_modified": record["last_modified"],
            "ext1": record["ext1"],
            "ext2": _json_dumps(record["ext2"]),
            "ext3": _json_dumps(record["ext3"]),
        }

    def _row_to_response_style(self, row: sqlite3.Row) -> ResponseStyle:
        """Hydrate ResponseStyle from SQLite row."""
        payload: dict[str, Any] = {column: row[column] for column in row.keys()}
        payload["tags"] = _json_loads_list(row["tags"])
        payload["examples"] = _json_loads_list(row["examples"])
        payload["metadata"] = _json_loads_optional(row["metadata"])
        payload["ext2"] = _json_loads_optional(row["ext2"])
        payload["ext3"] = _json_loads_optional(row["ext3"])
        payload["is_active"] = row["is_active"]
        payload.setdefault("prompt_part", "Response Style")
        return ResponseStyle.from_record(payload)


__all__ = ["ResponseStyleStoreMixin"]
