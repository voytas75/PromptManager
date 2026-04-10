"""Shared bounded preview helpers for prompt retrieval and ingest advisory surfaces.

Updates:
  v0.1.0 - 2026-04-10 - Extract shared preview selection and truncation logic.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.prompt_model import Prompt

PREVIEW_MAX_LENGTH = 96
_SOURCE_PREFIX = "Source: "
_LOW_SIGNAL_SOURCE_VALUES = {
    "",
    "-",
    "local",
    "n/a",
    "na",
    "none",
    "promptmanager",
    "prompt manager",
    "quick_capture",
    "unknown",
}


def flatten_preview_text(value: str) -> str:
    """Collapse multi-line prompt metadata into a single readable preview line."""
    return re.sub(r"\s+", " ", value).strip()


def truncate_preview_text(value: str, *, limit: int = PREVIEW_MAX_LENGTH) -> str:
    """Return a deterministically truncated preview string."""
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip(" ,.;:-") + "..."


def is_credible_preview_text(value: str, *, minimum_length: int = 10) -> bool:
    """Return whether *value* is strong enough to use as bounded preview text."""
    if len(value) < minimum_length:
        return False
    if not any(character.isalpha() for character in value):
        return False
    return True


def build_prompt_preview(prompt: Prompt) -> str | None:
    """Derive one compact preview from existing prompt data in priority order."""
    name_key = prompt.name.strip().casefold()

    description = flatten_preview_text(prompt.description)
    if (
        description
        and description.casefold() != name_key
        and is_credible_preview_text(description)
    ):
        return truncate_preview_text(description)

    for scenario in prompt.scenarios:
        normalized = flatten_preview_text(str(scenario))
        if normalized and is_credible_preview_text(normalized):
            return truncate_preview_text(normalized)

    source = flatten_preview_text(prompt.source)
    if source and source.casefold() not in _LOW_SIGNAL_SOURCE_VALUES:
        preview = _SOURCE_PREFIX + source
        if is_credible_preview_text(preview, minimum_length=len(_SOURCE_PREFIX) + 3):
            return truncate_preview_text(preview)
    return None


__all__ = [
    "PREVIEW_MAX_LENGTH",
    "build_prompt_preview",
    "flatten_preview_text",
    "is_credible_preview_text",
    "truncate_preview_text",
]
