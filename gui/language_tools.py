"""Heuristic language detection utilities for the Prompt Manager workspace.

Updates:
  v0.1.1 - 2025-11-29 - Wrap detection result creation for Ruff line-length compliance.
  v0.1.0 - 2025-11-10 - Introduce lightweight language detection for query input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(slots=True)
class DetectedLanguage:
    """Structured result describing the detected language."""

    code: str
    name: str
    confidence: float


@dataclass(frozen=True, slots=True)
class _LanguagePattern:
    code: str
    name: str
    keyword_patterns: tuple[re.Pattern[str], ...]
    weight: float = 1.0

    def score(self, text: str) -> float:
        return sum(len(pattern.findall(text)) for pattern in self.keyword_patterns) * self.weight


def _compile(patterns: Iterable[str]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in patterns)


_LANGUAGE_PATTERNS: list[_LanguagePattern] = [
    _LanguagePattern(
        code="python",
        name="Python",
        keyword_patterns=_compile(
            (
                r"\bdef\s+\w+",
                r"\bclass\s+\w+",
                r"\bimport\s+\w+",
                r"\basync\s+def\b",
                r"\bself\.",
                r"^\s*#",
            )
        ),
        weight=1.2,
    ),
    _LanguagePattern(
        code="powershell",
        name="PowerShell",
        keyword_patterns=_compile(
            (
                r"\b(Get|Set|New|Invoke|Test|Start|Stop)-[A-Za-z]+",
                r"\$[A-Za-z_][\w:]*",
                r"\[Parameter",
                r"#",
            )
        ),
        weight=1.2,
    ),
    _LanguagePattern(
        code="bash",
        name="Bash",
        keyword_patterns=_compile(
            (
                r"^#!\s*/bin/(ba)?sh",
                r"\b(if|for|while)\s+.+;\s*do",
                r"\bfi\b",
                r"\bdone\b",
            )
        ),
    ),
    _LanguagePattern(
        code="markdown",
        name="Markdown",
        keyword_patterns=_compile(
            (
                r"(^|\n)#{1,6}\s",
                r"```[a-zA-Z0-9_-]*",
                r"\*\*[^\*]+\*\*",
                r"\[[^\]]+\]\([^\)]+\)",
            )
        ),
        weight=0.8,
    ),
    _LanguagePattern(
        code="json",
        name="JSON",
        keyword_patterns=_compile(
            (
                r"\{\s*\"[^\"]+\"\s*:",
                r"\[\s*\{",
                r"\}\s*\]",
            )
        ),
        weight=0.9,
    ),
    _LanguagePattern(
        code="yaml",
        name="YAML",
        keyword_patterns=_compile(
            (
                r"^\s*-\s+\w+:",
                r"^\s*\w+:\s*\|",
                r"^\s*\w+:\s*\>",
                r"^---\s*$",
            )
        ),
        weight=0.9,
    ),
]

_LANGUAGE_ALIASES = {
    "python": "Python",
    "powershell": "PowerShell",
    "bash": "Bash",
    "markdown": "Markdown",
    "json": "JSON",
    "yaml": "YAML",
    "plain": "Plain Text",
}


def detect_language(text: str) -> DetectedLanguage:
    """Return a best-effort guess at the language contained in `text`."""
    stripped = text.strip()
    if not stripped:
        return DetectedLanguage(code="plain", name=_LANGUAGE_ALIASES["plain"], confidence=0.0)

    lowered = stripped.lower()
    best_code = "plain"
    best_score = 0.0

    for pattern in _LANGUAGE_PATTERNS:
        score = pattern.score(lowered)
        if score > best_score:
            best_score = score
            best_code = pattern.code

    if best_score == 0.0:
        return DetectedLanguage(code="plain", name=_LANGUAGE_ALIASES["plain"], confidence=0.0)

    confidence = min(1.0, 0.3 + (best_score / max(len(lowered) / 40.0, 1.0)))
    return DetectedLanguage(
        code=best_code,
        name=_LANGUAGE_ALIASES[best_code],
        confidence=confidence,
    )


__all__ = ["DetectedLanguage", "detect_language"]
