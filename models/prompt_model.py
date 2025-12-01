"""Prompt data model definitions.

Updates: v0.8.0 - 2025-11-22 - Persist prompt category slugs for taxonomy management.
Updates: v0.7.1 - 2025-11-22 - Normalize prompt version labels to integer strings.
Updates: v0.7.0 - 2025-11-22 - Add prompt versioning and fork lineage dataclasses.
Updates: v0.6.0 - 2025-12-08 - Remove deprecated task template dataclass.
Updates: v0.5.1 - 2025-11-19 - Add persisted usage scenarios metadata for prompts.
Updates: v0.4.0 - 2025-11-11 - Add single-user profile dataclass with preference helpers.
Updates: v0.3.0 - 2025-11-09 - Add rating aggregates and execution rating support.
Updates: v0.2.0 - 2025-11-08 - Add prompt execution records for history logging.
Updates: v0.1.1 - 2025-11-01 - Filtered null metadata fields for Chroma compatibility.
Updates: v0.1.0 - 2025-10-30 - Initial Prompt schema with serialization helpers.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from .category_model import slugify_category


def _utc_now() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(UTC)


def _ensure_uuid(value: Any) -> uuid.UUID:
    """Parse arbitrary UUID representations into a uuid.UUID instance."""
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


def _ensure_datetime(value: Any) -> datetime:
    """Parse incoming datetime values (isoformat strings or datetime)."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if value is None:
        return _utc_now()
    return datetime.fromisoformat(str(value))


def _serialize_list(items: Iterable[Any] | None) -> list[Any]:
    """Normalize iterable inputs into JSON-serialisable lists."""
    if items is None:
        return []
    if isinstance(items, (list, tuple, set)):
        return list(items)
    if isinstance(items, str):
        return [items]
    return list(items)


def _sanitize_scenarios(items: Iterable[Any] | None) -> list[str]:
    """Return a deduplicated, trimmed list of scenario strings."""
    scenarios: list[str] = []
    seen: set[str] = set()
    for raw in _serialize_list(items):
        text = str(raw).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        scenarios.append(text)
    return scenarios


def _serialize_metadata(value: Any | None) -> str | None:
    """Serialize complex metadata (dict/list) to JSON strings for storage."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def _deserialize_metadata(value: str | None) -> Any | None:
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


def _deserialize_list(value: Any) -> list[str]:
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


def _normalize_version_label(value: Any) -> str:
    """Return an integer-only version label derived from mixed inputs."""
    if value is None:
        return "1"
    text = str(value).strip()
    if not text:
        return "1"
    try:
        number = int(float(text))
        return str(number)
    except ValueError:
        match = re.search(r"\d+", text)
        if match:
            return str(int(match.group(0)))
    return "1"


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
    category_slug: str | None = None
    tags: list[str] = field(default_factory=list)
    language: str = "en"
    context: str | None = None
    example_input: str | None = None
    example_output: str | None = None
    scenarios: list[str] = field(default_factory=list)
    last_modified: datetime = field(default_factory=_utc_now)
    version: str = "1"
    author: str | None = None
    quality_score: float | None = None
    usage_count: int = 0
    rating_count: int = 0
    rating_sum: float = 0.0
    related_prompts: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utc_now)
    modified_by: str | None = None
    is_active: bool = True
    source: str = "local"
    checksum: str | None = None
    ext1: str | None = None
    ext2: MutableMapping[str, Any] | None = None
    ext3: str | None = None
    ext4: Sequence[float] | None = None
    ext5: MutableMapping[str, Any] | None = None

    # Transient search-only attribute populated at runtime; excluded from
    # persistence and comparisons.
    similarity: float | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Normalise stored scenarios and mirror them into ext5 metadata."""
        ext5_mapping: MutableMapping[str, Any] | None
        if isinstance(self.ext5, MutableMapping):
            ext5_mapping = self.ext5
        elif isinstance(self.ext5, Mapping):
            ext5_mapping = dict(self.ext5)
        else:
            ext5_mapping = None

        combined_sources: list[Any] = []
        if self.scenarios:
            combined_sources.extend(self.scenarios)
        if ext5_mapping is not None and "scenarios" in ext5_mapping:
            combined_sources.extend(_serialize_list(ext5_mapping.get("scenarios")))

        normalised = _sanitize_scenarios(combined_sources)
        self.scenarios = normalised

        if normalised:
            if ext5_mapping is None:
                ext5_mapping = {}
            ext5_mapping["scenarios"] = list(normalised)
        elif ext5_mapping is not None and "scenarios" in ext5_mapping:
            ext5_mapping.pop("scenarios", None)
            if not ext5_mapping:
                ext5_mapping = None

        self.ext5 = ext5_mapping
        slug_value = slugify_category(self.category_slug or self.category)
        self.category_slug = slug_value or None
        self.version = _normalize_version_label(self.version)

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
            (
                "Scenarios:\n" + "\n".join(f"- {scenario}" for scenario in self.scenarios)
            )
            if self.scenarios
            else "",
        ]
        return "\n".join(filter(None, sections))

    def to_metadata(self) -> dict[str, Any]:
        """Return metadata dictionary compatible with ChromaDB."""
        metadata = {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "category_slug": self.category_slug,
            "tags": json.dumps(self.tags, ensure_ascii=False),
            "language": self.language,
            "context": self.context,
            "example_input": self.example_input,
            "example_output": self.example_output,
            "scenarios": json.dumps(self.scenarios, ensure_ascii=False),
            "version": self.version,
            "author": self.author,
            "quality_score": self.quality_score,
            "usage_count": self.usage_count,
            "rating_count": self.rating_count,
            "rating_sum": self.rating_sum,
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

    def to_record(self) -> dict[str, Any]:
        """Return a plain dictionary representation for caching."""
        record = {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "category_slug": self.category_slug,
            "tags": self.tags,
            "language": self.language,
            "context": self.context,
            "example_input": self.example_input,
            "example_output": self.example_output,
            "scenarios": self.scenarios,
            "last_modified": self.last_modified.isoformat(),
            "version": self.version,
            "author": self.author,
            "quality_score": self.quality_score,
            "usage_count": self.usage_count,
            "rating_count": self.rating_count,
            "rating_sum": self.rating_sum,
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
    def from_record(cls, data: Mapping[str, Any]) -> Prompt:
        """Create a Prompt from a dictionary record."""
        return cls(
            id=_ensure_uuid(data.get("id") or uuid.uuid4()),
            name=str(data["name"]),
            description=str(data["description"]),
            category=str(data.get("category") or ""),
            category_slug=str(data.get("category_slug") or "").strip() or None,
            tags=_serialize_list(data.get("tags")),
            language=str(data.get("language") or "en"),
            context=data.get("context"),
            example_input=data.get("example_input"),
            example_output=data.get("example_output"),
            scenarios=_deserialize_list(data.get("scenarios")),
            last_modified=_ensure_datetime(data.get("last_modified")),
            version=str(data.get("version") or "1"),
            author=data.get("author"),
            quality_score=(
                float(data["quality_score"]) if data.get("quality_score") is not None else None
            ),
            usage_count=int(data.get("usage_count") or 0),
            rating_count=int(data.get("rating_count") or 0),
            rating_sum=float(data.get("rating_sum") or 0.0),
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
    def from_chroma(cls, record: Mapping[str, Any]) -> Prompt:
        """Instantiate from a ChromaDB metadata record."""
        metadata = record.get("metadata") or {}
        base = {
            "id": record.get("id"),
            "name": metadata.get("name"),
            "description": metadata.get("description"),
            "category": metadata.get("category"),
            "category_slug": metadata.get("category_slug"),
            "tags": _deserialize_list(metadata.get("tags")),
            "language": metadata.get("language"),
            "context": metadata.get("context"),
            "example_input": metadata.get("example_input"),
            "example_output": metadata.get("example_output"),
            "scenarios": _deserialize_list(metadata.get("scenarios")),
            "last_modified": metadata.get("last_modified"),
            "version": metadata.get("version"),
            "author": metadata.get("author"),
            "quality_score": metadata.get("quality_score"),
            "usage_count": metadata.get("usage_count"),
            "rating_count": metadata.get("rating_count"),
            "rating_sum": metadata.get("rating_sum"),
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
    response_text: str | None = None
    status: ExecutionStatus = ExecutionStatus.SUCCESS
    error_message: str | None = None
    duration_ms: int | None = None
    executed_at: datetime = field(default_factory=_utc_now)
    input_hash: str = ""
    rating: float | None = None
    metadata: MutableMapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.input_hash:
            self.input_hash = _hash_text(self.request_text)

    def to_record(self) -> dict[str, Any]:
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
            "rating": self.rating,
            "metadata": self.metadata,
        }

    @classmethod
    def from_record(cls, data: Mapping[str, Any]) -> PromptExecution:
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
            rating=(
                float(data["rating"])
                if data.get("rating") not in (None, "")
                else None
            ),
            metadata=_deserialize_metadata(data.get("metadata")),
        )


@dataclass(slots=True)
class PromptVersion:
    """Snapshot of a prompt captured for version history tracking."""
    id: int
    prompt_id: uuid.UUID
    version_number: int
    created_at: datetime
    parent_version_id: int | None
    commit_message: str | None
    snapshot: dict[str, Any]

    def to_prompt(self) -> Prompt:
        """Hydrate the stored snapshot into a :class:`Prompt`."""
        return Prompt.from_record(self.snapshot)

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> PromptVersion:
        """Instantiate a version from a SQLite row payload."""
        raw_snapshot = row["snapshot_json"]
        if raw_snapshot is None:
            raise ValueError("Prompt version snapshot is missing")
        if isinstance(raw_snapshot, (bytes, bytearray)):
            snapshot_text = raw_snapshot.decode("utf-8")
        else:
            snapshot_text = str(raw_snapshot)
        try:
            snapshot = json.loads(snapshot_text)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise ValueError("Invalid prompt snapshot payload") from exc

        parent_value = row["parent_version"] if "parent_version" in row.keys() else None
        parent_version_id = int(parent_value) if parent_value not in (None, "") else None

        return cls(
            id=int(row["version_id"]),
            prompt_id=_ensure_uuid(row["prompt_id"]),
            version_number=int(row["version_number"]),
            created_at=_ensure_datetime(row["created_at"]),
            parent_version_id=parent_version_id,
            commit_message=row["commit_message"] if "commit_message" in row.keys() else None,
            snapshot=snapshot,
        )


@dataclass(slots=True)
class PromptForkLink:
    """Describe lineage between a source prompt and its fork."""
    id: int
    source_prompt_id: uuid.UUID
    child_prompt_id: uuid.UUID
    created_at: datetime

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> PromptForkLink:
        """Hydrate a fork link from a SQLite row payload."""
        return cls(
            id=int(row["fork_id"]),
            source_prompt_id=_ensure_uuid(row["source_prompt_id"]),
            child_prompt_id=_ensure_uuid(row["child_prompt_id"]),
            created_at=_ensure_datetime(row["created_at"]),
        )

DEFAULT_PROFILE_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


@dataclass(slots=True)
class UserProfile:
    """Lightweight preference profile for the single Prompt Manager user."""
    id: uuid.UUID
    username: str = "default"
    preferred_language: str | None = None
    category_weights: MutableMapping[str, int] = field(default_factory=dict)
    tag_weights: MutableMapping[str, int] = field(default_factory=dict)
    recent_prompts: list[str] = field(default_factory=list)
    settings: MutableMapping[str, Any] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=_utc_now)
    ext1: str | None = None
    ext2: MutableMapping[str, Any] | None = None
    ext3: MutableMapping[str, Any] | None = None

    def touch(self) -> None:
        """Refresh the profile timestamp."""
        self.updated_at = _utc_now()

    def record_prompt_usage(self, prompt: Prompt, *, max_recent: int = 20) -> None:
        """Update preference counters and recency list from a prompt usage."""
        category = (prompt.category or "").strip()
        if category:
            self.category_weights[category] = int(self.category_weights.get(category, 0)) + 1

        for raw_tag in prompt.tags or []:
            tag = raw_tag.strip()
            if not tag:
                continue
            self.tag_weights[tag] = int(self.tag_weights.get(tag, 0)) + 1

        prompt_id = str(prompt.id)
        if prompt_id in self.recent_prompts:
            self.recent_prompts.remove(prompt_id)
        self.recent_prompts.insert(0, prompt_id)
        if len(self.recent_prompts) > max_recent:
            del self.recent_prompts[max_recent:]

        self.touch()

    def favorite_categories(self, *, limit: int = 3) -> list[str]:
        """Return the most frequently used categories."""
        if not self.category_weights:
            return []
        ordered = sorted(
            self.category_weights.items(),
            key=lambda item: (int(item[1]), item[0].lower()),
            reverse=True,
        )
        return [name for name, _ in ordered[:limit]]

    def favorite_tags(self, *, limit: int = 5) -> list[str]:
        """Return the most frequently used tags."""
        if not self.tag_weights:
            return []
        ordered = sorted(
            self.tag_weights.items(),
            key=lambda item: (int(item[1]), item[0].lower()),
            reverse=True,
        )
        return [name for name, _ in ordered[:limit]]

    def to_record(self) -> dict[str, Any]:
        """Serialise the profile into a plain mapping."""
        return {
            "id": str(self.id),
            "username": self.username,
            "preferred_language": self.preferred_language,
            "category_weights": dict(self.category_weights),
            "tag_weights": dict(self.tag_weights),
            "recent_prompts": list(self.recent_prompts),
            "settings": dict(self.settings),
            "updated_at": self.updated_at.isoformat(),
            "ext1": self.ext1,
            "ext2": self.ext2,
            "ext3": self.ext3,
        }

    @classmethod
    def from_record(cls, data: Mapping[str, Any]) -> UserProfile:
        """Hydrate a profile from a mapping."""
        settings_value = data.get("settings")
        if isinstance(settings_value, Mapping):
            settings_dict = {str(key): settings_value[key] for key in settings_value}
        else:
            settings_dict = {}

        return cls(
            id=_ensure_uuid(data.get("id") or DEFAULT_PROFILE_ID),
            username=str(data.get("username") or "default"),
            preferred_language=(
                str(data["preferred_language"])
                if data.get("preferred_language") not in (None, "")
                else None
            ),
            category_weights={
                str(key): int(value)
                for key, value in dict(data.get("category_weights") or {}).items()
            },
            tag_weights={
                str(key): int(value)
                for key, value in dict(data.get("tag_weights") or {}).items()
            },
            recent_prompts=_serialize_list(data.get("recent_prompts")),
            settings=settings_dict,
            updated_at=_ensure_datetime(data.get("updated_at")),
            ext1=data.get("ext1"),
            ext2=_deserialize_metadata(data.get("ext2")),
            ext3=_deserialize_metadata(data.get("ext3")),
        )

    @classmethod
    def create_default(cls, username: str = "default") -> UserProfile:
        """Return a default profile for single-user deployments."""
        return cls(id=DEFAULT_PROFILE_ID, username=username)


__all__ = [
    "Prompt",
    "PromptExecution",
    "PromptVersion",
    "PromptForkLink",
    "ExecutionStatus",
    "UserProfile",
    "DEFAULT_PROFILE_ID",
]
