"""Shared local-only presentation for compact prompt fit evidence.

This module deliberately summarizes only evidence already persisted on a prompt
asset. It does not calculate a confidence score or alter retrieval ranking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from models.prompt_model import Prompt


class _PromptFitEvidence(Protocol):
    usage_count: int
    rating_count: int
    quality_score: float | None


def build_prompt_fit_summary(prompt: Prompt | _PromptFitEvidence) -> str:
    """Return one truthful, compact fit line from persisted prompt aggregates."""
    parts: list[str] = []
    usage_count = max(0, int(prompt.usage_count or 0))
    if usage_count:
        parts.append(f"Used {usage_count}×")

    if prompt.rating_count > 0 and prompt.quality_score is not None:
        parts.append(f"Rated {prompt.quality_score:.1f}/10")
    elif prompt.rating_count > 0:
        parts.append("Rating recorded · score unavailable")

    return " · ".join(parts) if parts else "No run evidence yet"


__all__ = ["build_prompt_fit_summary"]
