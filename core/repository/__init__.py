"""SQLite-backed repository for persistent prompt storage.

Updates:
  v0.11.1 - 2025-12-04 - Modularize repository via prompt/execution/profile/note/maintenance mixins.
  v0.11.0 - 2025-12-04 - Begin modularization by extracting base helpers.
  v0.10.5 - 2025-11-29 - Gate Sequence import for typing-only usage and wrap fork queries.
  v0.10.4 - 2025-11-29 - Gate typing-only imports and add repository logger.
  v0.10.3 - 2025-11-28 - Add usage and benchmark analytics queries.
  v0.10.2 - 2025-11-28 - Add category aggregation helpers for maintenance.
  v0.10.1 - 2025-11-27 - Add prompt part column to response styles.
  v0.10.0 - 2025-11-22 - Add prompt versioning and fork lineage tables.
  v0.9.0 - 2025-12-08 - Remove task template persistence after retirement.
  v0.8.0 - 2025-12-06 - Add PromptNote persistence with CRUD helpers.
  pre-v0.8.0 - 2025-11-19 - Consolidated history covering releases v0.1.0–v0.7.0.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .base import (
    PromptCatalogueStats,
    RepositoryError,
    RepositoryNotFoundError,
    connect as _connect,
    ensure_directory as _ensure_directory,
    json_dumps as _json_dumps,
    json_loads_dict as _json_loads_dict,
    json_loads_list as _json_loads_list,
    json_loads_optional as _json_loads_optional,
    parse_optional_datetime as _parse_optional_datetime,
)
from .chains import ChainStoreMixin
from .executions import ExecutionStoreMixin
from .maintenance import RepositoryMaintenanceMixin
from .notes import PromptNoteStoreMixin
from .profiles import ProfileStoreMixin
from .prompts import PromptStoreMixin
from .response_styles import ResponseStyleStoreMixin


class PromptRepository(
    RepositoryMaintenanceMixin,
    PromptStoreMixin,
    ExecutionStoreMixin,
    ProfileStoreMixin,
    ResponseStyleStoreMixin,
    PromptNoteStoreMixin,
    ChainStoreMixin,
):
    """Compose repository mixins for SQLite-backed storage."""

    def __init__(self, db_path: str) -> None:
        """Initialise repository storage and ensure the schema exists."""
        self._db_path = Path(db_path)
        _ensure_directory(self._db_path)
        try:
            with _connect(self._db_path) as conn:
                self._ensure_schema(conn)
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            raise RepositoryError("Failed to initialise SQLite schema") from exc


__all__ = [
    "PromptRepository",
    "PromptCatalogueStats",
    "RepositoryError",
    "RepositoryNotFoundError",
    "_connect",
    "_json_dumps",
    "_json_loads_dict",
    "_json_loads_list",
    "_json_loads_optional",
    "_parse_optional_datetime",
]
