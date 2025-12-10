"""Schema bootstrap and maintenance helpers for the repository.

Updates:
  v0.11.2 - 2025-12-10 - Add execution session identifier column and index.
  v0.11.1 - 2025-12-07 - Restrict Path import to type checking contexts.
  v0.11.0 - 2025-12-04 - Extract schema management and reset helpers.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from models.category_model import slugify_category
from models.prompt_model import DEFAULT_PROFILE_ID

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from pathlib import Path

from .base import RepositoryError, connect as _connect, logger, stringify_uuid as _stringify_uuid


class RepositoryMaintenanceMixin:
    """Tasks that create, migrate, and reset repository storage."""

    _db_path: Path

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
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prompts_category ON prompts(category);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prompts_name ON prompts(name);")
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
            "CREATE INDEX IF NOT EXISTS idx_prompt_categories_active "
            "ON prompt_categories(is_active);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_prompt_categories_parent "
            "ON prompt_categories(parent_slug);"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prompt_chains (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                variables_schema TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prompt_chain_steps (
                id TEXT PRIMARY KEY,
                chain_id TEXT NOT NULL,
                prompt_id TEXT NOT NULL,
                order_index INTEGER NOT NULL,
                input_template TEXT NOT NULL,
                output_variable TEXT NOT NULL,
                condition TEXT,
                stop_on_failure INTEGER NOT NULL DEFAULT 1,
                metadata TEXT,
                FOREIGN KEY(chain_id) REFERENCES prompt_chains(id) ON DELETE CASCADE,
                FOREIGN KEY(prompt_id) REFERENCES prompts(id) ON DELETE CASCADE
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_prompt_chain_steps_chain "
            "ON prompt_chain_steps(chain_id);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_prompt_chain_steps_order "
            "ON prompt_chain_steps(chain_id, order_index);"
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
                FOREIGN KEY(parent_version) REFERENCES prompt_versions(version_id)
                    ON DELETE SET NULL
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_prompt_versions_prompt_id "
            "ON prompt_versions(prompt_id);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_prompt_versions_created_at "
            "ON prompt_versions(created_at);"
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
                session_id TEXT,
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
                prompt_part TEXT NOT NULL DEFAULT 'Response Style',
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
            "CREATE INDEX IF NOT EXISTS idx_response_styles_name ON response_styles(name);"
        )
        response_style_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(response_styles);")
        }
        if "prompt_part" not in response_style_columns:
            conn.execute(
                "ALTER TABLE response_styles ADD COLUMN prompt_part TEXT DEFAULT 'Response Style';"
            )
            conn.execute(
                "UPDATE response_styles SET prompt_part = 'Response Style' "
                "WHERE prompt_part IS NULL OR TRIM(prompt_part) = '';"
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
        prompt_columns = {row["name"] for row in conn.execute("PRAGMA table_info(prompts);")}
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
        if "session_id" not in execution_columns:
            conn.execute("ALTER TABLE prompt_executions ADD COLUMN session_id TEXT;")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_prompt_executions_session_id "
            "ON prompt_executions(session_id);"
        )
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
        profile_columns = {row["name"] for row in conn.execute("PRAGMA table_info(user_profile);")}
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
                (_stringify_uuid(DEFAULT_PROFILE_ID), "default", datetime.now(UTC).isoformat()),
            )

    def _backfill_category_slugs(self, conn: sqlite3.Connection) -> None:
        """Populate missing prompt category slugs for legacy records."""
        try:
            cursor = conn.execute(
                "SELECT id, category FROM prompts WHERE (category_slug IS NULL OR "
                "category_slug = '') AND category IS NOT NULL AND TRIM(category) != '';"
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
                    (_stringify_uuid(DEFAULT_PROFILE_ID), "default", datetime.now(UTC).isoformat()),
                )
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to reset repository data") from exc


__all__ = ["RepositoryMaintenanceMixin"]
