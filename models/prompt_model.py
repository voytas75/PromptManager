"""Prompt data model definitions.

Updates: v0.2.0 - 2025-11-08 - Add prompt execution records for history logging.
Updates: v0.1.1 - 2025-11-01 - Filtered null metadata fields for Chroma compatibility.
Updates: v0.1.0 - 2025-10-30 - Initial Prompt schema with serialization helpers.
"""

from __future__ import annotations

import json
import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence


def _utc_now() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _ensure_uuid(value: Any) -> uuid.UUID:
    """Parse arbitrary UUID representations into a uuid.UUID instance."""
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


def _ensure_datetime(value: Any) -> datetime:
    """Parse incoming datetime values (isoformat strings or datetime)."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if value is None:
        return _utc_now()
    return datetime.fromisoformat(str(value))


def _serialize_list(items: Optional[Iterable[Any]]) -> List[Any]:
    """Normalize iterable inputs into JSON-serialisable lists."""
    if items is None:
        return []
    if isinstance(items, (list, tuple, set)):
        return list(items)
    if isinstance(items, str):
        return [items]
    return list(items)


def _serialize_metadata(value: Optional[Any]) -> Optional[str]:
    """Serialize complex metadata (dict/list) to JSON strings for storage."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def _deserialize_metadata(value: Optional[str]) -> Optional[Any]:
    """Deserialize metadata stored as JSON strings."""
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    if value in ("", "null"):
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _deserialize_list(value: Any) -> List[str]:
    """Coerce metadata list fields into lists of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            return [value]
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]

def _hash_text(value: str) -> str:
    """Return a stable SHA-256 digest for the provided text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class ExecutionStatus(str, Enum):
    """Enumerate prompt execution outcomes for history tracking."""

    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass(slots=True)
class Prompt:
    """Dataclass representation of a prompt entry."""

    id: uuid.UUID
    name: str
    description: str
    category: str
    tags: List[str] = field(default_factory=list)
    language: str = "en"
    context: Optional[str] = None
    example_input: Optional[str] = None
    example_output: Optional[str] = None
    last_modified: datetime = field(default_factory=_utc_now)
    version: str = "1.1"
    author: Optional[str] = None
    quality_score: Optional[float] = None
    usage_count: int = 0
    related_prompts: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utc_now)
    modified_by: Optional[str] = None
    is_active: bool = True
    source: str = "local"
    checksum: Optional[str] = None
    ext1: Optional[str] = None
    ext2: Optional[MutableMapping[str, Any]] = None
    ext3: Optional[str] = None
    ext4: Optional[Sequence[float]] = None
    ext5: Optional[MutableMapping[str, Any]] = None

    @property
    def document(self) -> str:
        """Return a text representation suitable for vector embedding."""
        sections = [
            f"Name: {self.name}",
            f"Description: {self.description}",
            f"Category: {self.category}",
            f"Tags: {', '.join(self.tags)}" if self.tags else "",
            f"Context: {self.context}" if self.context else "",
            f"Example Input: {self.example_input}" if self.example_input else "",
            f"Example Output: {self.example_output}" if self.example_output else "",
        ]
        return "\n".join(filter(None, sections))

    def to_metadata(self) -> Dict[str, Any]:
        """Return metadata dictionary compatible with ChromaDB."""
        metadata = {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "tags": json.dumps(self.tags, ensure_ascii=False),
            "language": self.language,
            "context": self.context,
            "example_input": self.example_input,
            "example_output": self.example_output,
            "version": self.version,
            "author": self.author,
            "quality_score": self.quality_score,
            "usage_count": self.usage_count,
            "related_prompts": json.dumps(self.related_prompts, ensure_ascii=False),
            "created_at": self.created_at.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "modified_by": self.modified_by,
            "is_active": self.is_active,
            "source": self.source,
            "checksum": self.checksum,
            "ext1": self.ext1,
            "ext2": _serialize_metadata(self.ext2),
            "ext3": self.ext3,
            "ext4": _serialize_metadata(self.ext4),
            "ext5": _serialize_metadata(self.ext5),
        }
        return {key: value for key, value in metadata.items() if value is not None}

    def to_record(self) -> Dict[str, Any]:
        """Return a plain dictionary representation for caching."""
        record = {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "tags": self.tags,
            "language": self.language,
            "context": self.context,
            "example_input": self.example_input,
            "example_output": self.example_output,
            "last_modified": self.last_modified.isoformat(),
            "version": self.version,
            "author": self.author,
            "quality_score": self.quality_score,
            "usage_count": self.usage_count,
            "related_prompts": self.related_prompts,
            "created_at": self.created_at.isoformat(),
            "modified_by": self.modified_by,
            "is_active": self.is_active,
            "source": self.source,
            "checksum": self.checksum,
            "ext1": self.ext1,
            "ext2": self.ext2,
            "ext3": self.ext3,
            "ext4": list(self.ext4) if self.ext4 is not None else None,
            "ext5": self.ext5,
        }
        return record

    @classmethod
    def from_record(cls, data: Mapping[str, Any]) -> "Prompt":
        """Create a Prompt from a dictionary record."""
        return cls(
            id=_ensure_uuid(data.get("id") or uuid.uuid4()),
            name=str(data["name"]),
            description=str(data["description"]),
            category=str(data.get("category") or ""),
            tags=_serialize_list(data.get("tags")),
            language=str(data.get("language") or "en"),
            context=data.get("context"),
            example_input=data.get("example_input"),
            example_output=data.get("example_output"),
            last_modified=_ensure_datetime(data.get("last_modified")),
            version=str(data.get("version") or "1.1"),
            author=data.get("author"),
            quality_score=(
                float(data["quality_score"]) if data.get("quality_score") is not None else None
            ),
            usage_count=int(data.get("usage_count") or 0),
            related_prompts=_serialize_list(data.get("related_prompts")),
            created_at=_ensure_datetime(data.get("created_at")),
            modified_by=data.get("modified_by"),
            is_active=bool(data.get("is_active", True)),
            source=str(data.get("source") or "local"),
            checksum=data.get("checksum"),
            ext1=data.get("ext1"),
            ext2=_deserialize_metadata(data.get("ext2")),
            ext3=data.get("ext3"),
            ext4=_deserialize_metadata(data.get("ext4")),
            ext5=_deserialize_metadata(data.get("ext5")),
        )

    @classmethod
    def from_chroma(cls, record: Mapping[str, Any]) -> "Prompt":
        """Instantiate from a ChromaDB metadata record."""
        metadata = record.get("metadata") or {}
        base = {
            "id": record.get("id"),
            "name": metadata.get("name"),
            "description": metadata.get("description"),
            "category": metadata.get("category"),
            "tags": _deserialize_list(metadata.get("tags")),
            "language": metadata.get("language"),
            "context": metadata.get("context"),
            "example_input": metadata.get("example_input"),
            "example_output": metadata.get("example_output"),
            "last_modified": metadata.get("last_modified"),
            "version": metadata.get("version"),
            "author": metadata.get("author"),
            "quality_score": metadata.get("quality_score"),
            "usage_count": metadata.get("usage_count"),
            "related_prompts": _deserialize_list(metadata.get("related_prompts")),
            "created_at": metadata.get("created_at"),
            "modified_by": metadata.get("modified_by"),
            "is_active": metadata.get("is_active", True),
            "source": metadata.get("source"),
            "checksum": metadata.get("checksum"),
            "ext1": metadata.get("ext1"),
            "ext2": metadata.get("ext2"),
            "ext3": metadata.get("ext3"),
            "ext4": metadata.get("ext4"),
            "ext5": metadata.get("ext5"),
        }
        return cls.from_record(base)


@dataclass(slots=True)
class PromptExecution:
    """Dataclass representing a single prompt execution event."""

    id: uuid.UUID
    prompt_id: uuid.UUID
    request_text: str
    response_text: Optional[str] = None
    status: ExecutionStatus = ExecutionStatus.SUCCESS
    error_message: Optional[str] = None
    duration_ms: Optional[int] = None
    executed_at: datetime = field(default_factory=_utc_now)
    input_hash: str = ""
    metadata: Optional[MutableMapping[str, Any]] = None

    def __post_init__(self) -> None:
        if not self.input_hash:
            self.input_hash = _hash_text(self.request_text)

    def to_record(self) -> Dict[str, Any]:
        """Return a dictionary representation suitable for persistence."""
        return {
            "id": str(self.id),
            "prompt_id": str(self.prompt_id),
            "request_text": self.request_text,
            "response_text": self.response_text,
            "status": self.status.value,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
            "executed_at": self.executed_at.isoformat(),
            "input_hash": self.input_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_record(cls, data: Mapping[str, Any]) -> "PromptExecution":
        """Hydrate a PromptExecution from a mapping."""
        status_value = str(data.get("status") or ExecutionStatus.SUCCESS.value)
        try:
            parsed_status = ExecutionStatus(status_value)
        except ValueError:
            parsed_status = ExecutionStatus.SUCCESS
        return cls(
            id=_ensure_uuid(data.get("id") or uuid.uuid4()),
            prompt_id=_ensure_uuid(data["prompt_id"]),
            request_text=str(data.get("request_text") or ""),
            response_text=data.get("response_text"),
            status=parsed_status,
            error_message=data.get("error_message"),
            duration_ms=(
                int(data["duration_ms"])
                if data.get("duration_ms") not in (None, "")
                else None
            ),
            executed_at=_ensure_datetime(data.get("executed_at")),
            input_hash=str(data.get("input_hash") or ""),
            metadata=_deserialize_metadata(data.get("metadata")),
        )

__all__ = ["Prompt", "PromptExecution", "ExecutionStatus"]
