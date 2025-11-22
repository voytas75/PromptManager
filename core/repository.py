"""SQLite-backed repository for persistent prompt storage.

Updates: v0.10.0 - 2025-11-22 - Add prompt versioning and fork lineage persistence tables.
Updates: v0.9.0 - 2025-12-08 - Remove task template persistence after feature retirement.
Updates: v0.8.0 - 2025-12-06 - Add PromptNote persistence with CRUD helpers.
Updates: v0.7.0 - 2025-12-05 - Add ResponseStyle persistence with CRUD helpers.
Updates: v0.6.1 - 2025-11-30 - Add repository reset helper for maintenance workflows.
Updates: v0.6.0 - 2025-11-25 - Add prompt catalogue statistics accessor for maintenance UI.
Updates: v0.5.1 - 2025-11-19 - Persist prompt usage scenarios alongside prompt records.
Updates: v0.4.0 - 2025-11-11 - Persist single-user profile preferences alongside prompts.
Updates: v0.3.0 - 2025-11-09 - Add rating aggregation columns and execution rating support.
Updates: v0.2.0 - 2025-11-08 - Add prompt execution history persistence APIs.
Updates: v0.1.0 - 2025-10-31 - Introduce PromptRepository syncing Prompt dataclass with SQLite.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from models.prompt_model import (
    DEFAULT_PROFILE_ID,
    Prompt,
    PromptExecution,
    PromptVersion,
    PromptForkLink,
    UserProfile,
)
from models.category_model import PromptCategory, slugify_category
from models.response_style import ResponseStyle
from models.prompt_note import PromptNote


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
    last_modified_at: Optional[datetime]


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


def _prompt_snapshot_json(prompt: Prompt) -> str:
    """Return a canonical JSON payload describing the prompt state."""

    record = prompt.to_record()
    return json.dumps(record, ensure_ascii=False, sort_keys=True)


def _parse_optional_datetime(value: Any) -> Optional[datetime]:
    """Return a timezone-aware datetime parsed from SQLite rows when possible."""

    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    return None


class PromptRepository:
    """Persist prompts to SQLite and hydrate `Prompt` objects."""

    _COLUMNS: Sequence[str] = (
        "id",
        "name",
        "description",
        "category",
        "category_slug",
        "tags",
        "language",
        "context",
        "example_input",
        "example_output",
        "scenarios",
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

    _RESPONSE_STYLE_COLUMNS: Sequence[str] = (
        "id",
        "name",
        "description",
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

    _NOTE_COLUMNS: Sequence[str] = (
        "id",
        "note",
        "created_at",
        "last_modified",
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

    def get_prompt_catalogue_stats(self) -> PromptCatalogueStats:
        """Return aggregate prompt statistics for maintenance workflows."""

        total = 0
        active = 0
        categories: Set[str] = set()
        tags: Set[str] = set()
        missing_category = 0
        missing_tags = 0
        tag_total = 0
        stale_count = 0
        latest_modified: Optional[datetime] = None
        now = datetime.now(timezone.utc)
        stale_cutoff = now - timedelta(days=30)

        try:
            with _connect(self._db_path) as conn:
                cursor = conn.execute(
                    "SELECT category, tags, is_active, last_modified FROM prompts;"
                )
                for row in cursor:
                    total += 1
                    if bool(row["is_active"]):
                        active += 1

                    category = (row["category"] or "").strip()
                    if category:
                        categories.add(category)
                    else:
                        missing_category += 1

                    raw_tags = _json_loads_list(row["tags"])
                    cleaned_tags = [tag.strip() for tag in raw_tags if tag and tag.strip()]
                    if cleaned_tags:
                        tags.update(cleaned_tags)
                        tag_total += len(cleaned_tags)
                    else:
                        missing_tags += 1

                    timestamp = _parse_optional_datetime(row["last_modified"])
                    if timestamp is not None:
                        if latest_modified is None or timestamp > latest_modified:
                            latest_modified = timestamp
                        if timestamp < stale_cutoff:
                            stale_count += 1
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to compute prompt statistics") from exc

        inactive = max(total - active, 0)
        average_tags = round(tag_total / total, 2) if total else 0.0

        return PromptCatalogueStats(
            total_prompts=total,
            active_prompts=active,
            inactive_prompts=inactive,
            distinct_categories=len(categories),
            distinct_tags=len(tags),
            prompts_without_category=missing_category,
            prompts_without_tags=missing_tags,
            average_tags_per_prompt=average_tags,
            stale_prompts=stale_count,
            last_modified_at=latest_modified,
        )

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

    # Category management ------------------------------------------------ #

    def list_categories(self, include_archived: bool = False) -> List[PromptCategory]:
        """Return all stored categories."""

        query = "SELECT * FROM prompt_categories"
        params: List[Any] = []
        if not include_archived:
            query += " WHERE is_active = 1"
        query += " ORDER BY label COLLATE NOCASE;"
        try:
            with _connect(self._db_path) as conn:
                rows = conn.execute(query, params).fetchall()
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to list categories") from exc
        return [self._row_to_category(row) for row in rows]

    def get_category(self, slug: str) -> PromptCategory:
        """Return a category by slug."""

        try:
            with _connect(self._db_path) as conn:
                row = conn.execute(
                    "SELECT * FROM prompt_categories WHERE slug = ?;",
                    (slugify_category(slug),),
                ).fetchone()
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to load category {slug}") from exc
        if row is None:
            raise RepositoryNotFoundError(f"Category {slug} not found")
        return self._row_to_category(row)

    def create_category(self, category: PromptCategory) -> PromptCategory:
        """Persist a new category definition."""

        payload = self._category_to_row(category)
        try:
            with _connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO prompt_categories (
                        slug,
                        label,
                        description,
                        parent_slug,
                        color,
                        icon,
                        min_quality,
                        default_tags,
                        is_active,
                        created_at,
                        updated_at
                    ) VALUES (
                        :slug,
                        :label,
                        :description,
                        :parent_slug,
                        :color,
                        :icon,
                        :min_quality,
                        :default_tags,
                        :is_active,
                        :created_at,
                        :updated_at
                    );
                    """,
                    payload,
                )
        except sqlite3.IntegrityError as exc:
            raise RepositoryError(f"Category {category.slug} already exists") from exc
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to create category {category.slug}") from exc
        return category

    def update_category(self, category: PromptCategory) -> PromptCategory:
        """Update an existing category definition."""

        payload = self._category_to_row(category)
        slug = payload.pop("slug")
        try:
            with _connect(self._db_path) as conn:
                existing = conn.execute(
                    "SELECT label FROM prompt_categories WHERE slug = ?;",
                    (slug,),
                ).fetchone()
                if existing is None:
                    raise RepositoryNotFoundError(f"Category {slug} not found")
                conn.execute(
                    """
                    UPDATE prompt_categories
                    SET label = :label,
                        description = :description,
                        parent_slug = :parent_slug,
                        color = :color,
                        icon = :icon,
                        min_quality = :min_quality,
                        default_tags = :default_tags,
                        is_active = :is_active,
                        updated_at = :updated_at
                    WHERE slug = :slug;
                    """,
                    {**payload, "slug": slug},
                )
                if existing["label"] != category.label:
                    self._update_prompt_labels_for_category(conn, slug, category.label)
        except RepositoryNotFoundError:
            raise
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to update category {slug}") from exc
        return category

    def set_category_active(self, slug: str, is_active: bool) -> PromptCategory:
        """Enable or disable a category."""

        try:
            with _connect(self._db_path) as conn:
                cursor = conn.execute(
                    """
                    UPDATE prompt_categories
                    SET is_active = ?, updated_at = ?
                    WHERE slug = ?;
                    """,
                    (int(is_active), datetime.now(timezone.utc).isoformat(), slugify_category(slug)),
                )
                if cursor.rowcount == 0:
                    raise RepositoryNotFoundError(f"Category {slug} not found")
                row = conn.execute(
                    "SELECT * FROM prompt_categories WHERE slug = ?;",
                    (slugify_category(slug),),
                ).fetchone()
        except RepositoryNotFoundError:
            raise
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to update category {slug}") from exc
        if row is None:  # pragma: no cover - defensive
            raise RepositoryError(f"Failed to load category {slug} after update")
        return self._row_to_category(row)

    def sync_category_definitions(
        self,
        categories: Sequence[PromptCategory],
    ) -> List[PromptCategory]:
        """Ensure the provided categories exist; return any newly created ones."""

        existing = {category.slug for category in self.list_categories(include_archived=True)}
        created: List[PromptCategory] = []
        for category in categories:
            if category.slug in existing:
                continue
            created.append(self.create_category(category))
            existing.add(category.slug)
        return created

    def update_prompt_category_labels(self, slug: str, label: str) -> None:
        """Rename prompt records linked to the specified category slug."""

        try:
            with _connect(self._db_path) as conn:
                self._update_prompt_labels_for_category(conn, slugify_category(slug), label)
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to propagate prompt category label changes") from exc

    # Prompt versioning ------------------------------------------------- #

    def record_prompt_version(
        self,
        prompt: Prompt,
        *,
        commit_message: Optional[str] = None,
        parent_version_id: Optional[int] = None,
    ) -> PromptVersion:
        """Persist a snapshot of the prompt for version history tracking."""

        snapshot_json = _prompt_snapshot_json(prompt)
        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            with _connect(self._db_path) as conn:
                parent_id = parent_version_id
                if parent_id is None:
                    parent_id = self._get_latest_version_id(conn, prompt.id)
                version_number = self._next_version_number(conn, prompt.id)
                cursor = conn.execute(
                    """
                    INSERT INTO prompt_versions (
                        prompt_id,
                        parent_version,
                        version_number,
                        created_at,
                        commit_message,
                        snapshot_json
                    ) VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    (
                        str(prompt.id),
                        parent_id,
                        version_number,
                        timestamp,
                        commit_message,
                        snapshot_json,
                    ),
                )
                version_id = cursor.lastrowid
                row = conn.execute(
                    "SELECT version_id, prompt_id, parent_version, version_number, created_at, commit_message, snapshot_json "
                    "FROM prompt_versions WHERE version_id = ?;",
                    (version_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to record prompt version") from exc

        if row is None:  # pragma: no cover - defensive
            raise RepositoryError("Prompt version insert succeeded but row missing")

        return PromptVersion.from_row(row)

    def list_prompt_versions(
        self,
        prompt_id: uuid.UUID,
        *,
        limit: Optional[int] = None,
    ) -> List[PromptVersion]:
        """Return stored versions for a prompt ordered by newest first."""

        query = (
            "SELECT version_id, prompt_id, parent_version, version_number, created_at, commit_message, snapshot_json "
            "FROM prompt_versions WHERE prompt_id = ? ORDER BY version_number DESC"
        )
        params: List[Any] = [str(prompt_id)]
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))

        try:
            with _connect(self._db_path) as conn:
                cursor = conn.execute(query + ";", params)
                rows = cursor.fetchall()
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to load prompt versions") from exc

        return [PromptVersion.from_row(row) for row in rows]

    def get_prompt_version(self, version_id: int) -> PromptVersion:
        """Return a specific prompt version by identifier."""

        try:
            with _connect(self._db_path) as conn:
                row = conn.execute(
                    "SELECT version_id, prompt_id, parent_version, version_number, created_at, commit_message, snapshot_json "
                    "FROM prompt_versions WHERE version_id = ?;",
                    (int(version_id),),
                ).fetchone()
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to load prompt version {version_id}") from exc

        if row is None:
            raise RepositoryNotFoundError(f"Prompt version {version_id} not found")

        return PromptVersion.from_row(row)

    def get_prompt_latest_version(self, prompt_id: uuid.UUID) -> Optional[PromptVersion]:
        """Return the most recent version entry for the given prompt, if any."""

        try:
            with _connect(self._db_path) as conn:
                row = conn.execute(
                    "SELECT version_id, prompt_id, parent_version, version_number, created_at, commit_message, snapshot_json "
                    "FROM prompt_versions WHERE prompt_id = ? ORDER BY version_number DESC LIMIT 1;",
                    (str(prompt_id),),
                ).fetchone()
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to load latest prompt version") from exc

        if row is None:
            return None
        return PromptVersion.from_row(row)

    def record_prompt_fork(
        self,
        source_prompt_id: uuid.UUID,
        child_prompt_id: uuid.UUID,
    ) -> PromptForkLink:
        """Persist lineage information between a prompt and its fork."""

        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            with _connect(self._db_path) as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO prompt_forks (source_prompt_id, child_prompt_id, created_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(child_prompt_id) DO UPDATE SET
                        source_prompt_id = excluded.source_prompt_id,
                        created_at = excluded.created_at;
                    """,
                    (str(source_prompt_id), str(child_prompt_id), timestamp),
                )
                fork_id = cursor.lastrowid
                row = conn.execute(
                    "SELECT fork_id, source_prompt_id, child_prompt_id, created_at FROM prompt_forks WHERE child_prompt_id = ?;",
                    (str(child_prompt_id),),
                ).fetchone()
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to record prompt fork") from exc

        if row is None:  # pragma: no cover - defensive
            raise RepositoryError("Prompt fork insert succeeded but row missing")

        return PromptForkLink.from_row(row)

    def get_prompt_parent_fork(self, prompt_id: uuid.UUID) -> Optional[PromptForkLink]:
        """Return the lineage entry that links the prompt to its source."""

        try:
            with _connect(self._db_path) as conn:
                row = conn.execute(
                    "SELECT fork_id, source_prompt_id, child_prompt_id, created_at FROM prompt_forks "
                    "WHERE child_prompt_id = ? ORDER BY datetime(created_at) DESC LIMIT 1;",
                    (str(prompt_id),),
                ).fetchone()
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to load prompt parent fork") from exc

        if row is None:
            return None
        return PromptForkLink.from_row(row)

    def list_prompt_children(self, prompt_id: uuid.UUID) -> List[PromptForkLink]:
        """Return lineage entries for prompts forked from the provided prompt."""

        try:
            with _connect(self._db_path) as conn:
                cursor = conn.execute(
                    "SELECT fork_id, source_prompt_id, child_prompt_id, created_at FROM prompt_forks "
                    "WHERE source_prompt_id = ? ORDER BY datetime(created_at) DESC;",
                    (str(prompt_id),),
                )
                rows = cursor.fetchall()
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to list prompt forks") from exc

        return [PromptForkLink.from_row(row) for row in rows]

    def list(self, limit: Optional[int] = None) -> List[Prompt]:
        """Return prompts ordered by most recently modified."""
        # For deterministic unit tests ensure stable ordering when a small
        # *limit* is requested – return oldest first so callers get predictable
        # results even when multiple prompts share the same timestamp.
        order = "ASC" if limit is not None else "DESC"
        clause = f"ORDER BY datetime(last_modified) {order}"
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
            "category_slug": prompt.category_slug,
            "tags": _json_dumps(prompt.tags),
            "language": prompt.language,
            "context": prompt.context,
            "example_input": prompt.example_input,
            "example_output": prompt.example_output,
            "scenarios": _json_dumps(prompt.scenarios),
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
            "category_slug": row["category_slug"],
            "tags": _json_loads_list(row["tags"]),
            "language": row["language"],
            "context": row["context"],
            "example_input": row["example_input"],
            "example_output": row["example_output"],
            "scenarios": _json_loads_list(row["scenarios"]),
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

    def _category_to_row(self, category: PromptCategory) -> Dict[str, Any]:
        """Serialize a PromptCategory into SQLite-friendly mapping."""

        return {
            "slug": category.slug,
            "label": category.label,
            "description": category.description,
            "parent_slug": category.parent_slug,
            "color": category.color,
            "icon": category.icon,
            "min_quality": category.min_quality,
            "default_tags": _json_dumps(category.default_tags),
            "is_active": int(category.is_active),
            "created_at": category.created_at.isoformat(),
            "updated_at": category.updated_at.isoformat(),
        }

    def _row_to_category(self, row: sqlite3.Row) -> PromptCategory:
        """Hydrate PromptCategory from SQLite row."""

        payload: Dict[str, Any] = {
            "slug": row["slug"],
            "label": row["label"],
            "description": row["description"],
            "parent_slug": row["parent_slug"],
            "color": row["color"],
            "icon": row["icon"],
            "min_quality": row["min_quality"],
            "default_tags": _json_loads_list(row["default_tags"]),
            "is_active": row["is_active"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        return PromptCategory.from_record(payload)

    def _update_prompt_labels_for_category(
        self,
        conn: sqlite3.Connection,
        slug: str,
        label: str,
    ) -> None:
        """Propagate category label updates to prompt records."""

        conn.execute(
            "UPDATE prompts SET category = ? WHERE category_slug = ?;",
            (label, slug),
        )

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

    def _next_version_number(self, conn: sqlite3.Connection, prompt_id: uuid.UUID) -> int:
        """Return the next version number for the given prompt."""

        row = conn.execute(
            "SELECT COALESCE(MAX(version_number), 0) AS current FROM prompt_versions WHERE prompt_id = ?;",
            (str(prompt_id),),
        ).fetchone()
        current = row["current"] if row is not None else 0
        current_value = int(current or 0)
        return current_value + 1

    def _get_latest_version_id(
        self,
        conn: sqlite3.Connection,
        prompt_id: uuid.UUID,
    ) -> Optional[int]:
        """Return the version_id for the latest version of a prompt."""

        row = conn.execute(
            "SELECT version_id FROM prompt_versions WHERE prompt_id = ? ORDER BY version_number DESC LIMIT 1;",
            (str(prompt_id),),
        ).fetchone()
        if row is None:
            return None
        return int(row["version_id"])

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        """Create required tables if they do not exist."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prompts (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                category TEXT,
                category_slug TEXT,
                tags TEXT,
                language TEXT,
                context TEXT,
                example_input TEXT,
                example_output TEXT,
                scenarios TEXT,
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
            CREATE TABLE IF NOT EXISTS prompt_categories (
                slug TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                description TEXT NOT NULL,
                parent_slug TEXT,
                color TEXT,
                icon TEXT,
                min_quality REAL,
                default_tags TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(parent_slug) REFERENCES prompt_categories(slug) ON DELETE SET NULL
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_prompt_categories_active ON prompt_categories(is_active);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_prompt_categories_parent ON prompt_categories(parent_slug);"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prompt_versions (
                version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id TEXT NOT NULL,
                parent_version INTEGER,
                version_number INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                commit_message TEXT,
                snapshot_json TEXT NOT NULL,
                FOREIGN KEY(prompt_id) REFERENCES prompts(id) ON DELETE CASCADE,
                FOREIGN KEY(parent_version) REFERENCES prompt_versions(version_id) ON DELETE SET NULL
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_prompt_versions_prompt_id ON prompt_versions(prompt_id);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_prompt_versions_created_at ON prompt_versions(created_at);"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prompt_forks (
                fork_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_prompt_id TEXT NOT NULL,
                child_prompt_id TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                FOREIGN KEY(source_prompt_id) REFERENCES prompts(id) ON DELETE CASCADE,
                FOREIGN KEY(child_prompt_id) REFERENCES prompts(id) ON DELETE CASCADE
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_prompt_forks_source ON prompt_forks(source_prompt_id);"
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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS response_styles (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                tone TEXT,
                voice TEXT,
                format_instructions TEXT,
                guidelines TEXT,
                tags TEXT,
                examples TEXT,
                metadata TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_modified TEXT NOT NULL,
                ext1 TEXT,
                ext2 TEXT,
                ext3 TEXT
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_response_styles_name "
            "ON response_styles(name);"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prompt_notes (
                id TEXT PRIMARY KEY,
                note TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_modified TEXT NOT NULL
            );
            """
        )
        prompt_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(prompts);")
        }
        if "category_slug" not in prompt_columns:
            conn.execute("ALTER TABLE prompts ADD COLUMN category_slug TEXT;")
        if "scenarios" not in prompt_columns:
            conn.execute("ALTER TABLE prompts ADD COLUMN scenarios TEXT;")
        if "rating_count" not in prompt_columns:
            conn.execute("ALTER TABLE prompts ADD COLUMN rating_count INTEGER NOT NULL DEFAULT 0;")
        if "rating_sum" not in prompt_columns:
            conn.execute("ALTER TABLE prompts ADD COLUMN rating_sum REAL NOT NULL DEFAULT 0.0;")
        self._backfill_category_slugs(conn)
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

    def _backfill_category_slugs(self, conn: sqlite3.Connection) -> None:
        """Populate missing prompt category slugs for legacy records."""

        try:
            cursor = conn.execute(
                "SELECT id, category FROM prompts WHERE (category_slug IS NULL OR category_slug = '') "
                "AND category IS NOT NULL AND TRIM(category) != '';"
            )
            rows = cursor.fetchall()
            for row in rows:
                slug = slugify_category(row["category"])
                if not slug:
                    continue
                conn.execute(
                    "UPDATE prompts SET category_slug = ? WHERE id = ?;",
                    (slug, row["id"]),
                )
        except sqlite3.Error:  # pragma: no cover - defensive
            logger.warning("Unable to backfill prompt category slugs", exc_info=True)

    def reset_all_data(self) -> None:
        """Clear all persisted prompts, executions, and user profile data."""

        try:
            with _connect(self._db_path) as conn:
                conn.execute("DELETE FROM prompt_versions;")
                conn.execute("DELETE FROM prompt_forks;")
                conn.execute("DELETE FROM prompt_executions;")
                conn.execute("DELETE FROM response_styles;")
                conn.execute("DELETE FROM prompt_notes;")
                conn.execute("DELETE FROM prompts;")
                conn.execute("DELETE FROM user_profile;")
                conn.execute(
                    "INSERT INTO user_profile (id, username, updated_at) VALUES (?, ?, ?);",
                    (str(DEFAULT_PROFILE_ID), "default", datetime.now(timezone.utc).isoformat()),
                )
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to reset repository data") from exc

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

    def _response_style_to_row(self, style: ResponseStyle) -> Dict[str, Any]:
        """Serialise ResponseStyle into SQLite-compatible mapping."""

        record = style.to_record()
        return {
            "id": record["id"],
            "name": record["name"],
            "description": record["description"],
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

        payload: Dict[str, Any] = {
            column: row[column]
            for column in row.keys()
        }
        payload["tags"] = _json_loads_list(row["tags"])
        payload["examples"] = _json_loads_list(row["examples"])
        payload["metadata"] = _json_loads_optional(row["metadata"])
        payload["ext2"] = _json_loads_optional(row["ext2"])
        payload["ext3"] = _json_loads_optional(row["ext3"])
        payload["is_active"] = row["is_active"]
        return ResponseStyle.from_record(payload)

    def _note_to_row(self, note: PromptNote) -> Dict[str, Any]:
        """Serialise PromptNote to SQLite mapping."""

        record = note.to_record()
        return {
            "id": record["id"],
            "note": record["note"],
            "created_at": record["created_at"],
            "last_modified": record["last_modified"],
        }

    def _row_to_note(self, row: sqlite3.Row) -> PromptNote:
        """Hydrate PromptNote from SQLite row."""

        payload: Dict[str, Any] = {column: row[column] for column in row.keys()}
        return PromptNote.from_record(payload)

    # Response style helpers -------------------------------------------- #

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
            f"{column} = :{column}"
            for column in self._RESPONSE_STYLE_COLUMNS
            if column != "id"
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
                cursor = conn.execute(
                    "SELECT * FROM response_styles WHERE id = ?;",
                    (str(style_id),),
                )
                row = cursor.fetchone()
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to load response style {style_id}") from exc
        if row is None:
            raise RepositoryNotFoundError(f"Response style {style_id} not found")
        return self._row_to_response_style(row)

    def list_response_styles(
        self,
        *,
        include_inactive: bool = False,
        search: Optional[str] = None,
    ) -> List[ResponseStyle]:
        """Return stored response styles ordered by case-insensitive name."""

        query = "SELECT * FROM response_styles"
        clauses: List[str] = []
        params: List[Any] = []
        if not include_inactive:
            clauses.append("is_active = 1")
        if search:
            clauses.append("(LOWER(name) LIKE ? OR LOWER(description) LIKE ?)")
            pattern = f"%{search.lower()}%"
            params.extend([pattern, pattern])
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
                    (str(style_id),),
                ).rowcount
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to delete response style {style_id}") from exc
        if deleted == 0:
            raise RepositoryNotFoundError(f"Response style {style_id} not found")

    # Prompt note helpers ------------------------------------------------- #

    def add_prompt_note(self, note: PromptNote) -> PromptNote:
        """Insert a new prompt note."""

        payload = self._note_to_row(note)
        placeholders = ", ".join(f":{column}" for column in self._NOTE_COLUMNS)
        query = f"INSERT INTO prompt_notes ({', '.join(self._NOTE_COLUMNS)}) VALUES ({placeholders});"
        try:
            with _connect(self._db_path) as conn:
                conn.execute(query, payload)
        except sqlite3.IntegrityError as exc:
            raise RepositoryError(f"Prompt note {note.id} already exists") from exc
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to insert prompt note {note.id}") from exc
        return note

    def update_prompt_note(self, note: PromptNote) -> PromptNote:
        """Update an existing prompt note."""

        payload = self._note_to_row(note)
        assignments = ", ".join(
            f"{column} = :{column}" for column in self._NOTE_COLUMNS if column != "id"
        )
        query = f"UPDATE prompt_notes SET {assignments} WHERE id = :id;"
        try:
            with _connect(self._db_path) as conn:
                updated = conn.execute(query, payload).rowcount
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to update prompt note {note.id}") from exc
        if updated == 0:
            raise RepositoryNotFoundError(f"Prompt note {note.id} not found")
        return note

    def get_prompt_note(self, note_id: uuid.UUID) -> PromptNote:
        """Return a prompt note by identifier."""

        try:
            with _connect(self._db_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM prompt_notes WHERE id = ?;",
                    (str(note_id),),
                )
                row = cursor.fetchone()
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to load prompt note {note_id}") from exc
        if row is None:
            raise RepositoryNotFoundError(f"Prompt note {note_id} not found")
        return self._row_to_note(row)

    def list_prompt_notes(self) -> List[PromptNote]:
        """Return stored prompt notes ordered by modification time."""

        try:
            with _connect(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT * FROM prompt_notes ORDER BY last_modified DESC;"
                ).fetchall()
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to list prompt notes") from exc
        return [self._row_to_note(row) for row in rows]

    def delete_prompt_note(self, note_id: uuid.UUID) -> None:
        """Delete a prompt note."""

        try:
            with _connect(self._db_path) as conn:
                deleted = conn.execute(
                    "DELETE FROM prompt_notes WHERE id = ?;",
                    (str(note_id),),
                ).rowcount
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to delete prompt note {note_id}") from exc
        if deleted == 0:
            raise RepositoryNotFoundError(f"Prompt note {note_id} not found")

    def get_prompts_for_ids(self, prompt_ids: Sequence[uuid.UUID]) -> List[Prompt]:
        """Return prompts that match provided identifiers in input order."""

        if not prompt_ids:
            return []
        ids = [str(pid) for pid in prompt_ids]
        placeholders = ", ".join("?" for _ in ids)
        query = f"SELECT * FROM prompts WHERE id IN ({placeholders});"
        try:
            with _connect(self._db_path) as conn:
                rows = conn.execute(query, tuple(ids)).fetchall()
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to load prompts for identifiers") from exc
        prompt_map = {row["id"]: self._row_to_prompt(row) for row in rows}
        ordered: List[Prompt] = []
        for pid in ids:
            prompt = prompt_map.get(pid)
            if prompt:
                ordered.append(prompt)
        return ordered


__all__ = ["PromptRepository", "RepositoryError", "RepositoryNotFoundError"]
