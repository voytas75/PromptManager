"""Shared repository helpers, dataclasses, and error hierarchy.

Updates:
  v0.11.0 - 2025-12-04 - Extract logger, helpers, and exceptions from monolith.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

logger = logging.getLogger("prompt_manager.repository")


@dataclass(slots=True, frozen=True)
class PromptCatalogueStats:
    """Aggregate prompt metadata metrics for maintenance views."""

    total_prompts: int
    active_prompts: int
    inactive_prompts: int
    distinct_categories: int
    distinct_tags: int
    prompts_without_category: int
    prompts_without_tags: int
    average_tags_per_prompt: float
    stale_prompts: int
    last_modified_at: datetime | None


class RepositoryError(Exception):
    """Base exception for repository failures."""


class RepositoryNotFoundError(RepositoryError):
    """Raised when a requested record cannot be located."""


def ensure_directory(path: Path) -> None:
    """Ensure the directory for the SQLite database exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def connect(db_path: Path) -> sqlite3.Connection:
    """Return a configured SQLite connection."""
    conn = sqlite3.connect(str(db_path), detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def stringify_uuid(value: uuid.UUID | str) -> str:
    """Return a canonical UUID string for storage."""
    if isinstance(value, uuid.UUID):
        return str(value)
    return str(uuid.UUID(str(value)))


def json_dumps(value: Any | None) -> str | None:
    """Serialize arbitrary values to JSON strings (or None)."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def json_loads_list(value: str | None) -> list[str]:
    """Deserialize JSON-encoded lists stored in SQLite into Python lists."""
    if value is None:
        return []
    if value in ("", "null"):
        return []
    try:
        parsed: object = json.loads(value)
    except json.JSONDecodeError:
        return [str(value)]  # degraded fallback
    if isinstance(parsed, list):
        entries = cast("Sequence[object]", parsed)
        return [str(item) for item in entries]
    return [str(parsed)]


def json_loads_optional(value: str | None) -> Any | None:
    """Deserialize JSON strings while tolerating plain-text fallbacks."""
    if value is None:
        return None
    if value in ("", "null"):
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def json_loads_dict(value: str | None) -> dict[str, Any]:
    """Deserialize JSON strings into dictionaries."""
    if value is None or value in ("", "null"):
        return {}
    try:
        parsed_obj: object = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed_obj, dict):
        parsed_map = cast("Mapping[str, Any]", parsed_obj)
        return {str(key): parsed_map[key] for key in parsed_map}
    return {}


def parse_optional_datetime(value: Any) -> datetime | None:
    """Return a timezone-aware datetime parsed from SQLite rows when possible."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed
    return None


__all__ = [
    "PromptCatalogueStats",
    "RepositoryError",
    "RepositoryNotFoundError",
    "connect",
    "ensure_directory",
    "json_dumps",
    "json_loads_dict",
    "json_loads_list",
    "json_loads_optional",
    "logger",
    "parse_optional_datetime",
    "stringify_uuid",
]
