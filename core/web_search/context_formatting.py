"""Helpers for formatting injected web search context blocks.

Updates:
  v0.1.0 - 2025-12-07 - Introduce shared numbering and wrapper helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

SEARCH_RESULTS_START_MARKER = "--- Search Results Start ---"
SEARCH_RESULTS_END_MARKER = "--- Search Results End ---"
SEARCH_RESULTS_NOTE = (
    "Note: Use these results to inform your response. Reference by result number and "
    "prioritize recent, reliable sources. Do not add unverified information."
)

__all__ = [
    "SEARCH_RESULTS_END_MARKER",
    "SEARCH_RESULTS_NOTE",
    "SEARCH_RESULTS_START_MARKER",
    "build_numbered_search_results",
    "format_search_results_block",
    "wrap_search_results_block",
]


def build_numbered_search_results(lines: Sequence[str]) -> str:
    """Return newline-wrapped lines prefixed with incremental result numbers."""
    entries: list[str] = []
    counter = 0
    for line in lines:
        text = (line or "").strip()
        if not text:
            continue
        counter += 1
        entries.append(f"{counter}. {text}")
    return "\n".join(entries).strip()


def wrap_search_results_block(content: str) -> str:
    """Wrap *content* with standardized markers and guidance note."""
    body = (content or "").strip()
    if not body:
        return ""
    return "\n".join(
        (
            SEARCH_RESULTS_START_MARKER,
            SEARCH_RESULTS_NOTE,
            body,
            SEARCH_RESULTS_END_MARKER,
        )
    )


def format_search_results_block(lines: Sequence[str]) -> str:
    """Produce a fully wrapped search block from *lines*."""
    numbered = build_numbered_search_results(lines)
    if not numbered:
        return ""
    return wrap_search_results_block(numbered)
