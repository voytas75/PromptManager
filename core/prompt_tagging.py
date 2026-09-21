"""Deterministic local tag catalogue and single-prompt tag mutation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from models.prompt_model import Prompt


@dataclass(frozen=True, slots=True)
class TagRecord:
    """One logical tag with aggregate prompt counts."""

    tag: str
    prompt_count: int
    active_prompt_count: int

    def to_record(self) -> dict[str, str | int]:
        """Return a stable JSON-ready record."""
        return asdict(self)


def normalize_tag(value: object) -> str:
    """Return a trimmed tag or raise for an unusable operator value."""
    tag = str(value).strip()
    if not tag:
        raise ValueError("Tag must not be blank.")
    return tag


def build_tag_catalog(prompts: Iterable[Prompt]) -> tuple[TagRecord, ...]:
    """Aggregate logical tags once per prompt and sort predictably."""
    counts: dict[str, TagRecord] = {}
    for prompt in prompts:
        seen: set[str] = set()
        for raw_tag in prompt.tags or []:
            tag = str(raw_tag).strip()
            if not tag:
                continue
            key = tag.casefold()
            if key in seen:
                continue
            seen.add(key)
            current = counts.get(key)
            if current is None:
                current = TagRecord(tag=tag, prompt_count=0, active_prompt_count=0)
            counts[key] = TagRecord(
                tag=min(current.tag, tag, key=lambda value: (value.casefold(), value)),
                prompt_count=current.prompt_count + 1,
                active_prompt_count=current.active_prompt_count + int(bool(prompt.is_active)),
            )
    return tuple(
        sorted(
            counts.values(),
            key=lambda record: (-record.prompt_count, record.tag.casefold(), record.tag),
        )
    )


def find_tagged_prompts(prompts: Iterable[Prompt], tag: object) -> tuple[Prompt, ...]:
    """Return prompts having the requested logical tag in stable display order."""
    target = normalize_tag(tag).casefold()
    matches = [
        prompt
        for prompt in prompts
        if any(str(raw_tag).strip().casefold() == target for raw_tag in prompt.tags or [])
    ]
    return tuple(sorted(matches, key=lambda prompt: (prompt.name.casefold(), str(prompt.id))))


def apply_prompt_tag(prompt: Prompt, action: str, tag: object) -> bool:
    """Apply an idempotent add/remove action and return whether tags changed."""
    normalized_tag = normalize_tag(tag)
    target = normalized_tag.casefold()
    current = [str(raw_tag).strip() for raw_tag in prompt.tags or [] if str(raw_tag).strip()]

    if action == "add":
        if any(existing.casefold() == target for existing in current):
            return False
        prompt.tags = [*current, normalized_tag]
        return True
    if action == "remove":
        updated = [existing for existing in current if existing.casefold() != target]
        if updated == current:
            return False
        prompt.tags = updated
        return True
    raise ValueError(f"Unsupported tag action: {action}")


__all__ = [
    "TagRecord",
    "apply_prompt_tag",
    "build_tag_catalog",
    "find_tagged_prompts",
    "normalize_tag",
]
