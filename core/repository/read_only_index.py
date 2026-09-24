"""Inspect Chroma SQLite metadata without starting its writable client."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path


class IndexReadError(Exception):
    """The existing index metadata cannot be safely inspected."""


def read_index_metadata_ids(path: Path) -> set[str]:
    """Read exact IDs from the known Chroma 1.5 metadata segment, not HNSW."""
    db = path / "chroma.sqlite3"
    sidecars = (Path(f"{db}-wal"), Path(f"{db}-journal"))
    try:
        if not db.is_file():
            raise IndexReadError("Index metadata is missing")
        if any(item.exists() and item.stat().st_size for item in sidecars):
            raise IndexReadError("Index metadata has pending journal data")
        uri = db.resolve().as_uri() + "?mode=ro&immutable=1"
        with closing(sqlite3.connect(uri, uri=True, timeout=1)) as connection:
            columns = {
                table: {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
                for table in ("collections", "segments", "embeddings")
            }
            expected = {
                "collections": {"id", "name"},
                "segments": {"id", "scope", "collection"},
                "embeddings": {"segment_id", "embedding_id"},
            }
            if any(not fields.issubset(columns[table]) for table, fields in expected.items()):
                raise IndexReadError("Unsupported index metadata schema")
            collections = connection.execute(
                "SELECT id FROM collections WHERE name=?", ("prompt_manager",)
            ).fetchall()
            if len(collections) != 1:
                raise IndexReadError("Expected collection is unavailable or ambiguous")
            segments = connection.execute(
                "SELECT id FROM segments WHERE collection=? AND scope=?",
                (collections[0][0], "METADATA"),
            ).fetchall()
            if len(segments) != 1:
                raise IndexReadError("Index metadata segment is unavailable or ambiguous")
            rows = connection.execute(
                "SELECT embedding_id FROM embeddings WHERE segment_id=?", (segments[0][0],)
            ).fetchall()
        if any(item.exists() and item.stat().st_size for item in sidecars):
            raise IndexReadError("Index metadata changed during inspection")
        ids = [row[0] for row in rows]
        if any(not isinstance(value, str) or not value for value in ids):
            raise IndexReadError("Index metadata contains invalid identifiers")
        if len(ids) != len(set(ids)):
            raise IndexReadError("Index metadata contains duplicate identifiers")
        return set(ids)
    except (sqlite3.Error, OSError, ValueError, TypeError) as exc:
        raise IndexReadError("Existing index metadata cannot be read") from exc
