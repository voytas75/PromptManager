"""User profile persistence helpers for Prompt Manager repository.

Updates:
  v0.11.0 - 2025-12-04 - Extract profile CRUD into dedicated mixin.
"""

from __future__ import annotations

import sqlite3
from typing import Any, ClassVar

from models.prompt_model import Prompt, UserProfile

from .base import (
    RepositoryError,
    connect as _connect,
    json_dumps as _json_dumps,
    json_loads_dict as _json_loads_dict,
    json_loads_list as _json_loads_list,
    json_loads_optional as _json_loads_optional,
)


class ProfileStoreMixin:
    """Profile CRUD helpers shared across repository implementations."""

    _PROFILE_COLUMNS: ClassVar[tuple[str, ...]] = (
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

    def get_user_profile(self) -> UserProfile:
        """Return the single stored user profile, creating a default if missing."""
        try:
            with _connect(self._db_path) as conn:
                row = conn.execute("SELECT * FROM user_profile LIMIT 1;").fetchone()
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

    def _user_profile_to_row(self, profile: UserProfile) -> dict[str, Any]:
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
        payload: dict[str, Any] = {
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


__all__ = ["ProfileStoreMixin"]
