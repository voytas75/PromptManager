"""Category metadata models and helpers.

Updates: v0.1.0 - 2025-11-22 - Introduce PromptCategory dataclass and helpers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional

_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def _utc_now() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(timezone.utc)


def slugify_category(value: Optional[str]) -> str:
    """Return a URL-safe slug derived from the provided value."""

    text = (value or "").strip().lower()
    if not text:
        return ""
    slug = _SLUG_PATTERN.sub("-", text).strip("-")
    return slug


def _parse_datetime(value: Any) -> datetime:
    """Return a timezone-aware datetime from unstructured inputs."""

    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if value is None:
        return _utc_now()
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _normalise_tags(value: Optional[Iterable[Any]]) -> List[str]:
    """Coerce iterable tag inputs into a deduplicated list of strings."""

    tags: List[str] = []
    seen: set[str] = set()
    if value is None:
        return tags
    for raw in value:
        text = str(raw).strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        tags.append(text)
    return tags


def _clean_optional_text(value: Optional[str]) -> Optional[str]:
    """Strip whitespace from optional string inputs."""
    if value is None:
        return None
    text = value.strip()
    return text or None


@dataclass(slots=True)
class PromptCategory:
    """Structured representation of a prompt category."""

    slug: str
    label: str
    description: str
    parent_slug: Optional[str] = None
    color: Optional[str] = None
    icon: Optional[str] = None
    min_quality: Optional[float] = None
    default_tags: List[str] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        """Normalise text fields and ensure slug validity."""

        if not self.slug:
            self.slug = slugify_category(self.label)
        else:
            self.slug = slugify_category(self.slug)
        if not self.slug:
            raise ValueError("category slug cannot be empty")
        label = self.label.strip()
        if not label:
            label = self.slug.replace("-", " ").title()
        self.label = label
        description = self.description.strip()
        if not description:
            description = f"{self.label} prompts"
        self.description = description
        parent_slug = slugify_category(self.parent_slug)
        self.parent_slug = parent_slug or None
        self.color = _clean_optional_text(self.color)
        self.icon = _clean_optional_text(self.icon)
        if isinstance(self.min_quality, str):
            try:
                self.min_quality = float(self.min_quality)
            except ValueError:
                self.min_quality = None
        self.default_tags = _normalise_tags(self.default_tags)

    def to_record(self) -> Dict[str, Any]:
        """Serialize the category into a plain dictionary."""

        return {
            "slug": self.slug,
            "label": self.label,
            "description": self.description,
            "parent_slug": self.parent_slug,
            "color": self.color,
            "icon": self.icon,
            "min_quality": self.min_quality,
            "default_tags": list(self.default_tags),
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_record(cls, data: Mapping[str, Any]) -> "PromptCategory":
        """Hydrate a PromptCategory from a mapping."""

        return cls(
            slug=str(data["slug"]),
            label=str(data.get("label") or data["slug"]),
            description=str(data.get("description") or ""),
            parent_slug=_clean_optional_text(data.get("parent_slug")),
            color=_clean_optional_text(data.get("color")),
            icon=_clean_optional_text(data.get("icon")),
            min_quality=data.get("min_quality"),
            default_tags=_normalise_tags(data.get("default_tags")),
            is_active=bool(data.get("is_active", True)),
            created_at=_parse_datetime(data.get("created_at")),
            updated_at=_parse_datetime(data.get("updated_at")),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PromptCategory":
        """Create a category from loosely structured configuration."""

        if "label" not in payload and "slug" not in payload:
            raise ValueError("categories require at least a label or slug")
        return cls(
            slug=str(payload.get("slug") or payload.get("label") or ""),
            label=str(payload.get("label") or payload.get("slug") or ""),
            description=str(payload.get("description") or payload.get("label") or ""),
            parent_slug=payload.get("parent_slug"),
            color=payload.get("color"),
            icon=payload.get("icon"),
            min_quality=payload.get("min_quality"),
            default_tags=payload.get("default_tags") or [],
            is_active=bool(payload.get("is_active", True)),
        )


__all__ = ["PromptCategory", "slugify_category"]
