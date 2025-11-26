"""Response style data model definitions.

Updates: v0.1.0 - 2025-12-05 - Introduce ResponseStyle dataclass for formatting presets.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

from .prompt_model import (
    _deserialize_metadata,
    _ensure_datetime,
    _ensure_uuid,
    _serialize_list,
    _utc_now,
)


@dataclass(slots=True)
class ResponseStyle:
    """Describe reusable response formatting and tone preferences."""

    id: uuid.UUID
    name: str
    description: str
    tone: Optional[str] = None
    voice: Optional[str] = None
    format_instructions: Optional[str] = None
    guidelines: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    metadata: Optional[MutableMapping[str, Any]] = None
    is_active: bool = True
    version: str = "1.0"
    created_at: datetime = field(default_factory=_utc_now)
    last_modified: datetime = field(default_factory=_utc_now)
    ext1: Optional[str] = None
    ext2: Optional[MutableMapping[str, Any]] = None
    ext3: Optional[MutableMapping[str, Any]] = None

    def touch(self) -> None:
        """Refresh the modification timestamp."""

        self.last_modified = _utc_now()

    def to_record(self) -> Dict[str, Any]:
        """Return a serialisable mapping suitable for SQLite persistence."""

        metadata_payload: Optional[Dict[str, Any]]
        if isinstance(self.metadata, Mapping):
            metadata_payload = dict(self.metadata)
        else:
            metadata_payload = None

        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
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
    def from_record(cls, data: Mapping[str, Any]) -> "ResponseStyle":
        """Hydrate a ResponseStyle from a mapping."""

        metadata_value = data.get("metadata")
        if isinstance(metadata_value, Mapping):
            metadata_dict = {str(key): metadata_value[key] for key in metadata_value}
        else:
            metadata_dict = _deserialize_metadata(metadata_value)
            metadata_dict = dict(metadata_dict) if isinstance(metadata_dict, Mapping) else None

        return cls(
            id=_ensure_uuid(data.get("id") or uuid.uuid4()),
            name=str(data.get("name") or ""),
            description=str(data.get("description") or ""),
            tone=str(data.get("tone") or "") or None,
            voice=str(data.get("voice") or "") or None,
            format_instructions=str(data.get("format_instructions") or "") or None,
            guidelines=str(data.get("guidelines") or "") or None,
            tags=[str(tag) for tag in _serialize_list(data.get("tags"))],
            examples=[str(example) for example in _serialize_list(data.get("examples"))],
            metadata=metadata_dict,
            is_active=bool(int(data.get("is_active", 1))),
            version=str(data.get("version") or "1.0"),
            created_at=_ensure_datetime(data.get("created_at")),
            last_modified=_ensure_datetime(data.get("last_modified")),
            ext1=data.get("ext1"),
            ext2=_deserialize_metadata(data.get("ext2")),
            ext3=_deserialize_metadata(data.get("ext3")),
        )

