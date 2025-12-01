"""Prompt note data model definitions.

Updates: v0.1.0 - 2025-12-06 - Add PromptNote dataclass for simple note storage.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _utc_now() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(UTC)


@dataclass(slots=True)
class PromptNote:
    """Single-field note persisted for quick reference."""
    id: uuid.UUID
    note: str
    created_at: datetime = field(default_factory=_utc_now)
    last_modified: datetime = field(default_factory=_utc_now)

    def touch(self) -> None:
        """Update last_modified timestamp."""
        self.last_modified = _utc_now()

    def to_record(self) -> dict[str, Any]:
        """Return a mapping suitable for SQLite persistence."""
        return {
            "id": str(self.id),
            "note": self.note,
            "created_at": self.created_at.isoformat(),
            "last_modified": self.last_modified.isoformat(),
        }

    @classmethod
    def from_record(cls, data: dict[str, Any]) -> PromptNote:
        """Hydrate a PromptNote from a stored mapping."""
        created_raw = data.get("created_at")
        last_raw = data.get("last_modified")
        created_at = datetime.fromisoformat(str(created_raw)) if created_raw else _utc_now()
        last_modified = datetime.fromisoformat(str(last_raw)) if last_raw else created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        if last_modified.tzinfo is None:
            last_modified = last_modified.replace(tzinfo=UTC)
        return cls(
            id=uuid.UUID(str(data.get("id") or uuid.uuid4())),
            note=str(data.get("note") or "").strip(),
            created_at=created_at,
            last_modified=last_modified,
        )


__all__ = ["PromptNote"]
