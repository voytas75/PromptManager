"""Response style data model definitions.

Updates: v0.1.1 - 2025-11-27 - Add prompt part classification so entries cover any prompt segment.
Updates: v0.1.0 - 2025-12-05 - Introduce ResponseStyle dataclass for formatting presets.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(UTC)


def ensure_uuid(value: Any) -> uuid.UUID:
    """Parse arbitrary UUID representations into a uuid.UUID instance."""
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


def ensure_datetime(value: Any) -> datetime:
    """Parse incoming datetime values (isoformat strings or datetime)."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if value is None:
        return utc_now()
    parsed = datetime.fromisoformat(str(value))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def serialize_list(items: Iterable[Any] | None) -> list[Any]:
    """Normalize iterable inputs into JSON-serialisable lists."""
    if items is None:
        return []
    if isinstance(items, (list, tuple, set)):
        return list(items)
    if isinstance(items, str):
        return [items]
    return list(items)


def deserialize_metadata(value: object) -> Any | None:
    """Deserialize metadata stored as JSON strings."""
    if value is None:
        return None
    if isinstance(value, list):
        return cast("list[Any]", value)
    if isinstance(value, dict):
        return cast("dict[str, Any]", value)
    if isinstance(value, str):
        if value in ("", "null"):
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


@dataclass(slots=True)
class ResponseStyle:
    """Describe reusable response formatting and tone preferences."""

    id: uuid.UUID
    name: str
    description: str
    prompt_part: str = "Response Style"
    tone: str | None = None
    voice: str | None = None
    format_instructions: str | None = None
    guidelines: str | None = None
    tags: list[str] = field(default_factory=lambda: cast("list[str]", []))
    examples: list[str] = field(default_factory=lambda: cast("list[str]", []))
    metadata: MutableMapping[str, Any] | None = None
    is_active: bool = True
    version: str = "1.0"
    created_at: datetime = field(default_factory=utc_now)
    last_modified: datetime = field(default_factory=utc_now)
    ext1: str | None = None
    ext2: MutableMapping[str, Any] | None = None
    ext3: MutableMapping[str, Any] | None = None

    def touch(self) -> None:
        """Refresh the modification timestamp."""
        self.last_modified = utc_now()

    def to_record(self) -> dict[str, Any]:
        """Return a serialisable mapping suitable for SQLite persistence."""
        metadata_payload: dict[str, Any] | None
        if isinstance(self.metadata, Mapping):
            metadata_payload = dict(self.metadata)
        else:
            metadata_payload = None

        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "prompt_part": self.prompt_part,
            "tone": self.tone,
            "voice": self.voice,
            "format_instructions": self.format_instructions,
            "guidelines": self.guidelines,
            "tags": list(self.tags),
            "examples": list(self.examples),
            "metadata": metadata_payload,
            "is_active": int(self.is_active),
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "ext1": self.ext1,
            "ext2": dict(self.ext2) if self.ext2 else None,
            "ext3": dict(self.ext3) if self.ext3 else None,
        }

    @classmethod
    def from_record(cls, data: Mapping[str, Any]) -> ResponseStyle:
        """Hydrate a ResponseStyle from a mapping."""
        metadata_value = data.get("metadata")
        metadata_dict: dict[str, Any] | None
        if isinstance(metadata_value, Mapping):
            metadata_items = cast("Iterable[tuple[object, Any]]", metadata_value.items())
            metadata_dict = {str(key): value for key, value in metadata_items}
        else:
            deserialized_metadata = deserialize_metadata(metadata_value)
            if isinstance(deserialized_metadata, Mapping):
                deserialized_items = cast(
                    "Iterable[tuple[object, Any]]",
                    deserialized_metadata.items(),
                )
                metadata_dict = {str(key): value for key, value in deserialized_items}
            else:
                metadata_dict = None

        return cls(
            id=ensure_uuid(data.get("id") or uuid.uuid4()),
            name=str(data.get("name") or ""),
            description=str(data.get("description") or ""),
            prompt_part=str(data.get("prompt_part") or "Response Style"),
            tone=str(data.get("tone") or "") or None,
            voice=str(data.get("voice") or "") or None,
            format_instructions=str(data.get("format_instructions") or "") or None,
            guidelines=str(data.get("guidelines") or "") or None,
            tags=[str(tag) for tag in serialize_list(data.get("tags"))],
            examples=[str(example) for example in serialize_list(data.get("examples"))],
            metadata=metadata_dict,
            is_active=bool(int(data.get("is_active", 1))),
            version=str(data.get("version") or "1.0"),
            created_at=ensure_datetime(data.get("created_at")),
            last_modified=ensure_datetime(data.get("last_modified")),
            ext1=data.get("ext1"),
            ext2=cast(
                "MutableMapping[str, Any] | None",
                deserialize_metadata(cast("str | None", data.get("ext2"))),
            ),
            ext3=cast(
                "MutableMapping[str, Any] | None",
                deserialize_metadata(cast("str | None", data.get("ext3"))),
            ),
        )
