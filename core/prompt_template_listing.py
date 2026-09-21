"""Read-only reporting for effective built-in workflow prompt templates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from prompt_templates import (
    DEFAULT_PROMPT_TEMPLATES,
    PROMPT_TEMPLATE_DESCRIPTIONS,
    PROMPT_TEMPLATE_KEYS,
    PROMPT_TEMPLATE_LABELS,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class PromptTemplateRecord:
    """One canonical template with its effective runtime text and provenance."""

    key: str
    label: str
    description: str
    source: str
    text: str
    characters: int
    lines: int

    def to_record(self) -> dict[str, str | int]:
        """Return a stable JSON-ready template record."""
        return asdict(self)


def list_effective_prompt_templates(
    overrides: Mapping[str, object] | None = None,
) -> tuple[PromptTemplateRecord, ...]:
    """Return canonical templates in fixed order with valid non-default overrides applied."""
    supplied = overrides or {}
    records: list[PromptTemplateRecord] = []
    for key in PROMPT_TEMPLATE_KEYS:
        default_text = DEFAULT_PROMPT_TEMPLATES[key]
        candidate = supplied.get(key)
        text = _effective_text(candidate, default_text)
        source = "override" if text != default_text else "default"
        records.append(
            PromptTemplateRecord(
                key=key,
                label=PROMPT_TEMPLATE_LABELS.get(key, key),
                description=PROMPT_TEMPLATE_DESCRIPTIONS.get(key, ""),
                source=source,
                text=text,
                characters=len(text),
                lines=_line_count(text),
            )
        )
    return tuple(records)


def _effective_text(candidate: object, default_text: str) -> str:
    if not isinstance(candidate, str):
        return default_text
    return candidate if candidate.strip() and candidate != default_text else default_text


def _line_count(text: str) -> int:
    return len(text.splitlines()) or 1


__all__ = ["PromptTemplateRecord", "list_effective_prompt_templates"]
