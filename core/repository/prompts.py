"""Prompt persistence, categories, versions, and list helpers.

Updates:
  v0.11.0 - 2025-12-04 - Extract prompt CRUD/category/version helpers into mixin.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, ClassVar

from models.category_model import PromptCategory, slugify_category
from models.prompt_model import Prompt, PromptForkLink, PromptVersion

from .base import (
    PromptCatalogueStats,
    RepositoryError,
    RepositoryNotFoundError,
    connect as _connect,
    json_dumps as _json_dumps,
    json_loads_list as _json_loads_list,
    parse_optional_datetime as _parse_optional_datetime,
    stringify_uuid as _stringify_uuid,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import Sequence


class PromptStoreMixin:
    """Shared prompt/category/version persistence helpers."""

    _COLUMNS: ClassVar[Sequence[str]] = (
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

    def get_prompt_catalogue_stats(self) -> PromptCatalogueStats:
        """Return aggregate prompt statistics for maintenance workflows."""
        total = 0
        active = 0
        categories: set[str] = set()
        tags: set[str] = set()
        missing_category = 0
        missing_tags = 0
        tag_total = 0
        stale_count = 0
        latest_modified: datetime | None = None
        now = datetime.now(UTC)
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

    # Prompt CRUD -------------------------------------------------------- #

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
                    (_stringify_uuid(prompt_id),),
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
            f"{column} = :{column}" for column in self._COLUMNS if column != "id"
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
                    (_stringify_uuid(prompt_id),),
                )
                if cursor.rowcount == 0:
                    raise RepositoryNotFoundError(f"Prompt {prompt_id} not found")
        except RepositoryNotFoundError:
            raise
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to delete prompt {prompt_id}") from exc

    def list(self, limit: int | None = None) -> list[Prompt]:
        """Return prompts ordered by most recently modified."""
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

    def get_prompts_for_ids(self, prompt_ids: Sequence[uuid.UUID]) -> list[Prompt]:
        """Return prompts that match provided identifiers in input order."""
        if not prompt_ids:
            return []
        ids = [_stringify_uuid(pid) for pid in prompt_ids]
        placeholders = ", ".join("?" for _ in ids)
        query = f"SELECT * FROM prompts WHERE id IN ({placeholders});"
        try:
            with _connect(self._db_path) as conn:
                rows = conn.execute(query, tuple(ids)).fetchall()
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to load prompts for identifiers") from exc
        prompt_map = {row["id"]: self._row_to_prompt(row) for row in rows}
        ordered: list[Prompt] = []
        for pid in ids:
            prompt = prompt_map.get(pid)
            if prompt:
                ordered.append(prompt)
        return ordered

    # Category management ------------------------------------------------ #

    def list_categories(self, include_archived: bool = False) -> list[PromptCategory]:
        """Return all stored categories."""
        query = "SELECT * FROM prompt_categories"
        params: list[Any] = []
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
                    (int(is_active), datetime.now(UTC).isoformat(), slugify_category(slug)),
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
    ) -> list[PromptCategory]:
        """Ensure the provided categories exist; return any newly created ones."""
        existing = {category.slug for category in self.list_categories(include_archived=True)}
        created: list[PromptCategory] = []
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

    def get_category_prompt_counts(self) -> dict[str, dict[str, Any]]:
        """Return prompt totals per category (active/inactive)."""
        query = (
            "SELECT "
            "category_slug AS slug, "
            "COUNT(*) AS total_prompts, "
            "SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END) AS active_prompts "
            "FROM prompts "
            "GROUP BY category_slug;"
        )
        try:
            with _connect(self._db_path) as conn:
                rows = conn.execute(query).fetchall()
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to compute category prompt counts") from exc
        counts: dict[str, dict[str, Any]] = {}
        for row in rows:
            slug = row["slug"] if row["slug"] is not None else ""
            counts[slug] = {
                "total_prompts": int(row["total_prompts"]),
                "active_prompts": int(row["active_prompts"]),
            }
        return counts

    # Prompt versioning -------------------------------------------------- #

    def record_prompt_version(
        self,
        prompt: Prompt,
        *,
        commit_message: str | None = None,
        parent_version_id: int | None = None,
    ) -> PromptVersion:
        """Persist a snapshot of the prompt for version history tracking."""
        snapshot_json = _prompt_snapshot_json(prompt)
        timestamp = datetime.now(UTC).isoformat()

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
                        _stringify_uuid(prompt.id),
                        parent_id,
                        version_number,
                        timestamp,
                        commit_message,
                        snapshot_json,
                    ),
                )
                version_id = cursor.lastrowid
                row = conn.execute(
                    (
                        "SELECT version_id, prompt_id, parent_version, version_number, created_at, "
                        "commit_message, snapshot_json FROM prompt_versions WHERE version_id = ?;"
                    ),
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
        limit: int | None = None,
    ) -> list[PromptVersion]:
        """Return stored versions for a prompt ordered by newest first."""
        query = (
            "SELECT version_id, prompt_id, parent_version, version_number, created_at, "
            "commit_message, snapshot_json FROM prompt_versions WHERE prompt_id = ? "
            "ORDER BY version_number DESC"
        )
        params: list[Any] = [_stringify_uuid(prompt_id)]
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
                    (
                        "SELECT version_id, prompt_id, parent_version, version_number, created_at, "
                        "commit_message, snapshot_json FROM prompt_versions WHERE version_id = ?;"
                    ),
                    (int(version_id),),
                ).fetchone()
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to load prompt version {version_id}") from exc

        if row is None:
            raise RepositoryNotFoundError(f"Prompt version {version_id} not found")

        return PromptVersion.from_row(row)

    def get_prompt_latest_version(self, prompt_id: uuid.UUID) -> PromptVersion | None:
        """Return the most recent version entry for the given prompt, if any."""
        try:
            with _connect(self._db_path) as conn:
                row = conn.execute(
                    (
                        "SELECT version_id, prompt_id, parent_version, version_number, created_at, "
                        "commit_message, snapshot_json FROM prompt_versions WHERE prompt_id = ? "
                        "ORDER BY version_number DESC LIMIT 1;"
                    ),
                    (_stringify_uuid(prompt_id),),
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
        timestamp = datetime.now(UTC).isoformat()
        try:
            with _connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO prompt_forks (source_prompt_id, child_prompt_id, created_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(child_prompt_id) DO UPDATE SET
                        source_prompt_id = excluded.source_prompt_id,
                        created_at = excluded.created_at;
                    """,
                    (
                        _stringify_uuid(source_prompt_id),
                        _stringify_uuid(child_prompt_id),
                        timestamp,
                    ),
                )
                row = conn.execute(
                    (
                        "SELECT fork_id, source_prompt_id, child_prompt_id, created_at "
                        "FROM prompt_forks WHERE child_prompt_id = ?;"
                    ),
                    (_stringify_uuid(child_prompt_id),),
                ).fetchone()
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to record prompt fork") from exc

        if row is None:  # pragma: no cover - defensive
            raise RepositoryError("Prompt fork insert succeeded but row missing")

        return PromptForkLink.from_row(row)

    def get_prompt_parent_fork(self, prompt_id: uuid.UUID) -> PromptForkLink | None:
        """Return the lineage entry that links the prompt to its source."""
        try:
            with _connect(self._db_path) as conn:
                row = conn.execute(
                    (
                        "SELECT fork_id, source_prompt_id, child_prompt_id, created_at "
                        "FROM prompt_forks WHERE child_prompt_id = ? "
                        "ORDER BY datetime(created_at) DESC LIMIT 1;"
                    ),
                    (_stringify_uuid(prompt_id),),
                ).fetchone()
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to load prompt parent fork") from exc

        if row is None:
            return None
        return PromptForkLink.from_row(row)

    def list_prompt_children(self, prompt_id: uuid.UUID) -> list[PromptForkLink]:
        """Return lineage entries for prompts forked from the provided prompt."""
        try:
            with _connect(self._db_path) as conn:
                cursor = conn.execute(
                    (
                        "SELECT fork_id, source_prompt_id, child_prompt_id, created_at "
                        "FROM prompt_forks WHERE source_prompt_id = ? "
                        "ORDER BY datetime(created_at) DESC;"
                    ),
                    (_stringify_uuid(prompt_id),),
                )
                rows = cursor.fetchall()
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to list prompt forks") from exc

        return [PromptForkLink.from_row(row) for row in rows]

    # Serialization helpers --------------------------------------------- #

    def _prompt_to_row(self, prompt: Prompt) -> dict[str, Any]:
        """Serialise Prompt into SQLite mapping."""
        return {
            "id": _stringify_uuid(prompt.id),
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
        """Hydrate Prompt from SQLite row."""
        payload: dict[str, Any] = {
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
        return Prompt.from_record(payload)

    def _category_to_row(self, category: PromptCategory) -> dict[str, Any]:
        """Serialise PromptCategory into SQLite mapping."""
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
        payload: dict[str, Any] = {
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
        """Update prompts to reflect renamed category labels."""
        conn.execute(
            "UPDATE prompts SET category = ? WHERE category_slug = ?;",
            (label, slug),
        )

    def _next_version_number(self, conn: sqlite3.Connection, prompt_id: uuid.UUID) -> int:
        """Return the next version number for the given prompt."""
        row = conn.execute(
            (
                "SELECT COALESCE(MAX(version_number), 0) AS current FROM prompt_versions "
                "WHERE prompt_id = ?;"
            ),
            (_stringify_uuid(prompt_id),),
        ).fetchone()
        current = row["current"] if row is not None else 0
        current_value = int(current or 0)
        return current_value + 1

    def _get_latest_version_id(
        self,
        conn: sqlite3.Connection,
        prompt_id: uuid.UUID,
    ) -> int | None:
        """Return the version_id for the latest version of a prompt."""
        row = conn.execute(
            (
                "SELECT version_id FROM prompt_versions WHERE prompt_id = ? "
                "ORDER BY version_number DESC LIMIT 1;"
            ),
            (_stringify_uuid(prompt_id),),
        ).fetchone()
        if row is None:
            return None
        return int(row["version_id"])


def _prompt_snapshot_json(prompt: Prompt) -> str:
    """Return a canonical JSON payload describing the prompt state."""
    record = prompt.to_record()
    return json.dumps(record, ensure_ascii=False, sort_keys=True)


__all__ = ["PromptStoreMixin"]
