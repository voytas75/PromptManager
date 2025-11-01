"""SQLite-backed repository for persistent prompt storage.

Updates: v0.4.0 - 2025-11-11 - Persist single-user profile preferences alongside prompts.
Updates: v0.3.0 - 2025-11-09 - Add rating aggregation columns and execution rating support.
Updates: v0.2.0 - 2025-11-08 - Add prompt execution history persistence APIs.
Updates: v0.1.0 - 2025-10-31 - Introduce PromptRepository syncing Prompt dataclass with SQLite.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from models.prompt_model import DEFAULT_PROFILE_ID, Prompt, PromptExecution, UserProfile


class RepositoryError(Exception):
    """Base exception for repository failures."""


class RepositoryNotFoundError(RepositoryError):
    """Raised when a requested record cannot be located."""


def _ensure_directory(path: Path) -> None:
    """Ensure the directory for the SQLite database exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _connect(db_path: Path) -> sqlite3.Connection:
    """Return a configured SQLite connection."""
    conn = sqlite3.connect(str(db_path), detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def _json_dumps(value: Optional[Any]) -> Optional[str]:
    """Serialize arbitrary values to JSON strings (or None)."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _json_loads_list(value: Optional[str]) -> List[str]:
    """Deserialize JSON-encoded lists stored in SQLite into Python lists."""
    if value is None:
        return []
    if value in ("", "null"):
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return [str(value)]  # degraded fallback
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def _json_loads_optional(value: Optional[str]) -> Optional[Any]:
    """Deserialize JSON strings while tolerating plain-text fallbacks."""
    if value is None:
        return None
    if value in ("", "null"):
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _json_loads_dict(value: Optional[str]) -> Dict[str, Any]:
    """Deserialize JSON strings into dictionaries."""

    if value is None or value in ("", "null"):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, dict):
        return {str(key): parsed[key] for key in parsed}
    return {}


class PromptRepository:
    """Persist prompts to SQLite and hydrate `Prompt` objects."""

    _COLUMNS: Sequence[str] = (
        "id",
        "name",
        "description",
        "category",
        "tags",
        "language",
        "context",
        "example_input",
        "example_output",
        "last_modified",
        "version",
        "author",
        "quality_score",
        "usage_count",
        "rating_count",
        "rating_sum",
        "related_prompts",
        "created_at",
        "modified_by",
        "is_active",
        "source",
        "checksum",
        "ext1",
        "ext2",
        "ext3",
        "ext4",
        "ext5",
    )

    _EXECUTION_COLUMNS: Sequence[str] = (
        "id",
        "prompt_id",
        "request_text",
        "response_text",
        "status",
        "error_message",
        "duration_ms",
        "executed_at",
        "input_hash",
        "rating",
        "metadata",
    )

    _PROFILE_COLUMNS: Sequence[str] = (
        "id",
        "username",
        "preferred_language",
        "category_weights",
        "tag_weights",
        "recent_prompts",
        "settings",
        "updated_at",
        "ext1",
        "ext2",
        "ext3",
    )

    def __init__(self, db_path: str) -> None:
        """Initialise the repository and ensure schema exists."""
        self._db_path = Path(db_path)
        _ensure_directory(self._db_path)
        try:
            with _connect(self._db_path) as conn:
                self._ensure_schema(conn)
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            raise RepositoryError("Failed to initialise SQLite schema") from exc

    def add(self, prompt: Prompt) -> Prompt:
        """Insert a new prompt record."""
        payload = self._prompt_to_row(prompt)
        placeholders = ", ".join(f":{column}" for column in self._COLUMNS)
        query = f"INSERT INTO prompts ({', '.join(self._COLUMNS)}) VALUES ({placeholders});"
        try:
            with _connect(self._db_path) as conn:
                conn.execute(query, payload)
        except sqlite3.IntegrityError as exc:
            raise RepositoryError(f"Prompt {prompt.id} already exists") from exc
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to insert prompt {prompt.id}") from exc
        return prompt

    def get(self, prompt_id: uuid.UUID) -> Prompt:
        """Fetch a prompt by UUID."""
        try:
            with _connect(self._db_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM prompts WHERE id = ?;",
                    (str(prompt_id),),
                )
                row = cursor.fetchone()
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to load prompt {prompt_id}") from exc
        if row is None:
            raise RepositoryNotFoundError(f"Prompt {prompt_id} not found")
        return self._row_to_prompt(row)

    def update(self, prompt: Prompt) -> Prompt:
        """Persist an existing prompt."""
        payload = self._prompt_to_row(prompt)
        assignments = ", ".join(
            f"{column} = :{column}"
            for column in self._COLUMNS
            if column != "id"
        )
        query = f"UPDATE prompts SET {assignments} WHERE id = :id;"
        try:
            with _connect(self._db_path) as conn:
                cursor = conn.execute(query, payload)
                if cursor.rowcount == 0:
                    raise RepositoryNotFoundError(f"Prompt {prompt.id} not found")
        except RepositoryNotFoundError:
            raise
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to update prompt {prompt.id}") from exc
        return prompt

    def delete(self, prompt_id: uuid.UUID) -> None:
        """Delete a prompt by UUID."""
        try:
            with _connect(self._db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM prompts WHERE id = ?;",
                    (str(prompt_id),),
                )
                if cursor.rowcount == 0:
                    raise RepositoryNotFoundError(f"Prompt {prompt_id} not found")
        except RepositoryNotFoundError:
            raise
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to delete prompt {prompt_id}") from exc

    def list(self, limit: Optional[int] = None) -> List[Prompt]:
        """Return prompts ordered by most recently modified."""
        clause = "ORDER BY datetime(last_modified) DESC"
        if limit is not None:
            clause += f" LIMIT {int(limit)}"
        query = f"SELECT * FROM prompts {clause};"
        try:
            with _connect(self._db_path) as conn:
                cursor = conn.execute(query)
                rows = cursor.fetchall()
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to fetch prompt list") from exc
        return [self._row_to_prompt(row) for row in rows]

    def add_execution(self, execution: PromptExecution) -> PromptExecution:
        """Persist a prompt execution history entry."""
        payload = self._execution_to_row(execution)
        placeholders = ", ".join(f":{column}" for column in self._EXECUTION_COLUMNS)
        query = (
            f"INSERT INTO prompt_executions ({', '.join(self._EXECUTION_COLUMNS)}) "
            f"VALUES ({placeholders});"
        )
        try:
            with _connect(self._db_path) as conn:
                conn.execute(query, payload)
        except sqlite3.IntegrityError as exc:
            raise RepositoryError(f"Execution {execution.id} already exists") from exc
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to insert execution {execution.id}") from exc
        return execution

    def get_execution(self, execution_id: uuid.UUID) -> PromptExecution:
        """Fetch a single execution by identifier."""
        try:
            with _connect(self._db_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM prompt_executions WHERE id = ?;",
                    (str(execution_id),),
                )
                row = cursor.fetchone()
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to load execution {execution_id}") from exc
        if row is None:
            raise RepositoryNotFoundError(f"Execution {execution_id} not found")
        return self._row_to_execution(row)

    def list_executions(
        self,
        *,
        limit: Optional[int] = None,
    ) -> List[PromptExecution]:
        """Return executions ordered by most recent first."""
        clause = "ORDER BY datetime(executed_at) DESC"
        if limit is not None:
            clause += f" LIMIT {int(limit)}"
        query = f"SELECT * FROM prompt_executions {clause};"
        try:
            with _connect(self._db_path) as conn:
                cursor = conn.execute(query)
                rows = cursor.fetchall()
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to fetch executions") from exc
        return [self._row_to_execution(row) for row in rows]

    def list_executions_for_prompt(
        self,
        prompt_id: uuid.UUID,
        *,
        limit: Optional[int] = None,
    ) -> List[PromptExecution]:
        """Return execution history for a given prompt."""
        clause = "WHERE prompt_id = ? ORDER BY datetime(executed_at) DESC"
        params: List[Any] = [str(prompt_id)]
        if limit is not None:
            clause += f" LIMIT {int(limit)}"
        query = f"SELECT * FROM prompt_executions {clause};"
        try:
            with _connect(self._db_path) as conn:
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to fetch execution history for {prompt_id}") from exc
        return [self._row_to_execution(row) for row in rows]

    def list_executions_filtered(
        self,
        *,
        status: Optional[str] = None,
        prompt_id: Optional[uuid.UUID] = None,
        search: Optional[str] = None,
        limit: Optional[int] = None,
        order_desc: bool = True,
    ) -> List[PromptExecution]:
        """Return executions filtered by status, prompt, and search term."""

        query = "SELECT * FROM prompt_executions"
        conditions: List[str] = []
        params: List[Any] = []

        if status:
            conditions.append("status = ?")
            params.append(status)
        if prompt_id:
            conditions.append("prompt_id = ?")
            params.append(str(prompt_id))
        if search:
            like = f"%{search.lower()}%"
            conditions.append(
                "("
                "LOWER(request_text) LIKE ? OR "
                "LOWER(response_text) LIKE ? OR "
                "LOWER(COALESCE(metadata, '')) LIKE ?"
                ")"
            )
            params.extend([like, like, like])

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        order = "DESC" if order_desc else "ASC"
        query += f" ORDER BY datetime(executed_at) {order}"
        if limit is not None:
            query += f" LIMIT {int(limit)}"

        try:
            with _connect(self._db_path) as conn:
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to fetch filtered executions") from exc
        return [self._row_to_execution(row) for row in rows]

    def update_execution(self, execution: PromptExecution) -> PromptExecution:
        """Persist changes to an existing execution entry."""
        payload = self._execution_to_row(execution)
        assignments = ", ".join(
            f"{column} = :{column}"
            for column in self._EXECUTION_COLUMNS
            if column != "id"
        )
        query = f"UPDATE prompt_executions SET {assignments} WHERE id = :id;"
        try:
            with _connect(self._db_path) as conn:
                cursor = conn.execute(query, payload)
                if cursor.rowcount == 0:
                    raise RepositoryNotFoundError(f"Execution {execution.id} not found")
        except RepositoryNotFoundError:
            raise
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to update execution {execution.id}") from exc
        return execution

    def _prompt_to_row(self, prompt: Prompt) -> Dict[str, Any]:
        """Convert Prompt into SQLite-friendly mapping."""
        return {
            "id": str(prompt.id),
            "name": prompt.name,
            "description": prompt.description,
            "category": prompt.category,
            "tags": _json_dumps(prompt.tags),
            "language": prompt.language,
            "context": prompt.context,
            "example_input": prompt.example_input,
            "example_output": prompt.example_output,
            "last_modified": prompt.last_modified.isoformat(),
            "version": prompt.version,
            "author": prompt.author,
            "quality_score": prompt.quality_score,
            "usage_count": prompt.usage_count,
            "rating_count": prompt.rating_count,
            "rating_sum": prompt.rating_sum,
            "related_prompts": _json_dumps(prompt.related_prompts),
            "created_at": prompt.created_at.isoformat(),
            "modified_by": prompt.modified_by,
            "is_active": int(prompt.is_active),
            "source": prompt.source,
            "checksum": prompt.checksum,
            "ext1": prompt.ext1,
            "ext2": _json_dumps(prompt.ext2),
            "ext3": prompt.ext3,
            "ext4": _json_dumps(list(prompt.ext4) if prompt.ext4 is not None else None),
            "ext5": _json_dumps(prompt.ext5),
        }

    def _row_to_prompt(self, row: sqlite3.Row) -> Prompt:
        """Transform a SQLite row into a Prompt dataclass."""
        record: Dict[str, Any] = {
            "id": row["id"],
            "name": row["name"],
            "description": row["description"],
            "category": row["category"],
            "tags": _json_loads_list(row["tags"]),
            "language": row["language"],
            "context": row["context"],
            "example_input": row["example_input"],
            "example_output": row["example_output"],
            "last_modified": row["last_modified"],
            "version": row["version"],
            "author": row["author"],
            "quality_score": row["quality_score"],
            "usage_count": row["usage_count"],
            "rating_count": row["rating_count"],
            "rating_sum": row["rating_sum"],
            "related_prompts": _json_loads_list(row["related_prompts"]),
            "created_at": row["created_at"],
            "modified_by": row["modified_by"],
            "is_active": row["is_active"],
            "source": row["source"],
            "checksum": row["checksum"],
            "ext1": row["ext1"],
            "ext2": row["ext2"],
            "ext3": row["ext3"],
            "ext4": row["ext4"],
            "ext5": row["ext5"],
        }
        return Prompt.from_record(record)

    def _execution_to_row(self, execution: PromptExecution) -> Dict[str, Any]:
        """Convert PromptExecution into SQLite-compatible mapping."""
        record = execution.to_record()
        record["metadata"] = _json_dumps(record["metadata"])
        return record

    def _row_to_execution(self, row: sqlite3.Row) -> PromptExecution:
        """Hydrate PromptExecution from a SQLite row."""
        payload: Dict[str, Any] = {
            "id": row["id"],
            "prompt_id": row["prompt_id"],
            "request_text": row["request_text"],
            "response_text": row["response_text"],
            "status": row["status"],
            "error_message": row["error_message"],
            "duration_ms": row["duration_ms"],
            "executed_at": row["executed_at"],
            "input_hash": row["input_hash"],
            "rating": row["rating"],
            "metadata": row["metadata"],
        }
        return PromptExecution.from_record(payload)

    def _user_profile_to_row(self, profile: UserProfile) -> Dict[str, Any]:
        """Serialize UserProfile to SQLite row mapping."""

        record = profile.to_record()
        return {
            "id": record["id"],
            "username": record["username"],
            "preferred_language": record["preferred_language"],
            "category_weights": _json_dumps(record["category_weights"]),
            "tag_weights": _json_dumps(record["tag_weights"]),
            "recent_prompts": _json_dumps(record["recent_prompts"]),
            "settings": _json_dumps(record["settings"]),
            "updated_at": record["updated_at"],
            "ext1": record["ext1"],
            "ext2": _json_dumps(record["ext2"]),
            "ext3": _json_dumps(record["ext3"]),
        }

    def _row_to_user_profile(self, row: sqlite3.Row) -> UserProfile:
        """Hydrate UserProfile from SQLite row."""

        payload: Dict[str, Any] = {
            "id": row["id"],
            "username": row["username"],
            "preferred_language": row["preferred_language"],
            "category_weights": _json_loads_dict(row["category_weights"]),
            "tag_weights": _json_loads_dict(row["tag_weights"]),
            "recent_prompts": _json_loads_list(row["recent_prompts"]),
            "settings": _json_loads_optional(row["settings"]),
            "updated_at": row["updated_at"],
            "ext1": row["ext1"],
            "ext2": _json_loads_optional(row["ext2"]),
            "ext3": _json_loads_optional(row["ext3"]),
        }
        return UserProfile.from_record(payload)

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        """Create required tables if they do not exist."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prompts (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                category TEXT,
                tags TEXT,
                language TEXT,
                context TEXT,
                example_input TEXT,
                example_output TEXT,
                last_modified TEXT NOT NULL,
                version TEXT NOT NULL,
                author TEXT,
                quality_score REAL,
                usage_count INTEGER NOT NULL DEFAULT 0,
                related_prompts TEXT,
                created_at TEXT NOT NULL,
                modified_by TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                source TEXT,
                checksum TEXT,
                ext1 TEXT,
                ext2 TEXT,
                ext3 TEXT,
                ext4 TEXT,
                ext5 TEXT
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_prompts_category ON prompts(category);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_prompts_name ON prompts(name);"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prompt_executions (
                id TEXT PRIMARY KEY,
                prompt_id TEXT NOT NULL,
                request_text TEXT NOT NULL,
                response_text TEXT,
                status TEXT NOT NULL,
                error_message TEXT,
                duration_ms INTEGER,
                executed_at TEXT NOT NULL,
                input_hash TEXT NOT NULL,
                rating REAL,
                metadata TEXT,
                FOREIGN KEY(prompt_id) REFERENCES prompts(id) ON DELETE CASCADE
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_prompt_executions_prompt_id "
            "ON prompt_executions(prompt_id);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_prompt_executions_executed_at "
            "ON prompt_executions(executed_at);"
        )
        prompt_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(prompts);")
        }
        if "rating_count" not in prompt_columns:
            conn.execute("ALTER TABLE prompts ADD COLUMN rating_count INTEGER NOT NULL DEFAULT 0;")
        if "rating_sum" not in prompt_columns:
            conn.execute("ALTER TABLE prompts ADD COLUMN rating_sum REAL NOT NULL DEFAULT 0.0;")
        execution_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(prompt_executions);")
        }
        if "rating" not in execution_columns:
            conn.execute("ALTER TABLE prompt_executions ADD COLUMN rating REAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profile (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                preferred_language TEXT,
                category_weights TEXT,
                tag_weights TEXT,
                recent_prompts TEXT,
                settings TEXT,
                updated_at TEXT NOT NULL,
                ext1 TEXT,
                ext2 TEXT,
                ext3 TEXT
            );
            """
        )
        profile_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(user_profile);")
        }
        for column, ddl in (
            ("category_weights", "ALTER TABLE user_profile ADD COLUMN category_weights TEXT;"),
            ("tag_weights", "ALTER TABLE user_profile ADD COLUMN tag_weights TEXT;"),
            ("recent_prompts", "ALTER TABLE user_profile ADD COLUMN recent_prompts TEXT;"),
            ("settings", "ALTER TABLE user_profile ADD COLUMN settings TEXT;"),
            ("ext1", "ALTER TABLE user_profile ADD COLUMN ext1 TEXT;"),
            ("ext2", "ALTER TABLE user_profile ADD COLUMN ext2 TEXT;"),
            ("ext3", "ALTER TABLE user_profile ADD COLUMN ext3 TEXT;"),
        ):
            if column not in profile_columns:
                conn.execute(ddl)

        existing_profile = conn.execute("SELECT 1 FROM user_profile LIMIT 1;").fetchone()
        if existing_profile is None:
            conn.execute(
                "INSERT INTO user_profile (id, username, updated_at) VALUES (?, ?, ?);",
                (str(DEFAULT_PROFILE_ID), "default", datetime.now(timezone.utc).isoformat()),
            )

    # User profile helpers ------------------------------------------------- #

    def get_user_profile(self) -> UserProfile:
        """Return the single stored user profile, creating a default if missing."""

        try:
            with _connect(self._db_path) as conn:
                cursor = conn.execute("SELECT * FROM user_profile LIMIT 1;")
                row = cursor.fetchone()
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to load user profile") from exc

        if row is None:
            profile = UserProfile.create_default()
            return self.save_user_profile(profile)

        return self._row_to_user_profile(row)

    def save_user_profile(self, profile: UserProfile) -> UserProfile:
        """Insert or update the singleton user profile."""

        payload = self._user_profile_to_row(profile)
        assignments = ", ".join(
            f"{column} = excluded.{column}" for column in self._PROFILE_COLUMNS if column != "id"
        )
        query = (
            f"INSERT INTO user_profile ({', '.join(self._PROFILE_COLUMNS)}) "
            f"VALUES ({', '.join(f':{column}' for column in self._PROFILE_COLUMNS)}) "
            f"ON CONFLICT(id) DO UPDATE SET {assignments};"
        )

        try:
            with _connect(self._db_path) as conn:
                conn.execute(query, payload)
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to persist user profile") from exc

        return profile

    def record_user_prompt_usage(self, prompt: Prompt, *, max_recent: int = 20) -> UserProfile:
        """Update stored profile preferences using a prompt usage event."""

        profile = self.get_user_profile()
        profile.record_prompt_usage(prompt, max_recent=max_recent)
        return self.save_user_profile(profile)


__all__ = ["PromptRepository", "RepositoryError", "RepositoryNotFoundError"]
