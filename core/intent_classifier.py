"""Intent classification helpers for intent-aware prompt suggestions.

Updates:
  v0.6.2 - 2025-11-30 - Remove extraneous docstring spacing to satisfy Ruff D202.
  v0.6.1 - 2025-11-29 - Adopt PEP 695 generics and gate typing-only imports for Ruff.
  v0.6.0 - 2025-11-06 - Provide rule-based classifier aligned with hybrid retrieval plan.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

logger = logging.getLogger(__name__)


class IntentClassifierError(Exception):
    """Raised when intent classification cannot be completed."""


class IntentLabel(str, Enum):
    """Canonical labels representing prompt intent groupings."""

    GENERAL = "general"
    ANALYSIS = "analysis"
    DEBUG = "debug"
    REFACTOR = "refactor"
    ENHANCEMENT = "enhancement"
    DOCUMENTATION = "documentation"
    REPORTING = "reporting"


_CATEGORY_HINTS = {
    IntentLabel.ANALYSIS: ["Code Analysis"],
    IntentLabel.DEBUG: ["Reasoning / Debugging"],
    IntentLabel.REFACTOR: ["Refactoring"],
    IntentLabel.ENHANCEMENT: ["Enhancement"],
    IntentLabel.DOCUMENTATION: ["Documentation"],
    IntentLabel.REPORTING: ["Reporting"],
    IntentLabel.GENERAL: [],
}

_TAG_HINTS = {
    IntentLabel.ANALYSIS: ["analysis", "tests", "review"],
    IntentLabel.DEBUG: ["debugging", "ci", "failure"],
    IntentLabel.REFACTOR: ["refactor", "cleanup", "modular"],
    IntentLabel.ENHANCEMENT: ["enhancement", "feature", "improve"],
    IntentLabel.DOCUMENTATION: ["documentation", "docs", "changelog"],
    IntentLabel.REPORTING: ["reporting", "summary", "status"],
    IntentLabel.GENERAL: [],
}


def _collect_keywords(keywords: Sequence[str], text: str) -> int:
    pattern = r"|".join(re.escape(keyword) for keyword in keywords)
    if not pattern:
        return 0
    matches = re.findall(pattern, text)
    return len(matches)


_HEURISTIC_KEYWORDS = {
    IntentLabel.DEBUG: (
        "bug",
        "debug",
        "error",
        "traceback",
        "exception",
        "stack",
        "fail",
        "broken",
        "diagnos",
        "issue",
    ),
    IntentLabel.REFACTOR: (
        "refactor",
        "cleanup",
        "modular",
        "extract",
        "rename",
        "split",
        "restructure",
        "simplify",
    ),
    IntentLabel.ENHANCEMENT: (
        "enhance",
        "feature",
        "improve",
        "optimiz",
        "extend",
        "speed",
        "support",
        "add",
        "implement",
    ),
    IntentLabel.DOCUMENTATION: (
        "doc",
        "documentation",
        "readme",
        "comment",
        "guide",
        "explain",
        "describe",
        "write-up",
        "docstring",
    ),
    IntentLabel.REPORTING: (
        "summary",
        "report",
        "changelog",
        "status",
        "bullet",
        "update",
        "overview",
    ),
    IntentLabel.ANALYSIS: (
        "analyse",
        "analyze",
        "analysis",
        "investig",
        "review",
        "audit",
        "assess",
        "inspect",
        "profil",
        "test",
    ),
}


_LANGUAGE_HINTS = {
    "python": ("python", "py ", "py3", "pytest", "fastapi", "pydantic"),
    "powershell": ("powershell", "ps1", "pwsh", "get-", "set-", "invoke-"),
    "bash": ("bash", "shell", "sh", "#!/bin/bash"),
    "javascript": ("javascript", "js", "node", "react", "next.js"),
    "terraform": ("terraform", "tf", "hcl"),
}


@dataclass(slots=True)
class IntentPrediction:
    """Result of the classifier containing hints for retrieval."""

    label: IntentLabel
    confidence: float
    rationale: str | None = None
    category_hints: list[str] = field(default_factory=list)
    tag_hints: list[str] = field(default_factory=list)
    language_hints: list[str] = field(default_factory=list)

    @classmethod
    def general(cls) -> IntentPrediction:
        """Return a baseline prediction representing a general intent."""
        return cls(IntentLabel.GENERAL, confidence=0.0)


class IntentClassifier:
    """Classify free-form queries into prompt intent categories."""

    def classify(self, query: str) -> IntentPrediction:
        """Return intent prediction for the supplied query text."""
        stripped = query.strip()
        if not stripped:
            return IntentPrediction.general()

        lower = stripped.lower()
        best_label = IntentLabel.GENERAL
        best_score = 0

        for label, keywords in _HEURISTIC_KEYWORDS.items():
            score = _collect_keywords(keywords, lower)
            if score > best_score:
                best_label = label
                best_score = score

        confidence = min(0.9, 0.35 + (best_score * 0.15)) if best_score else 0.25

        language_hints: list[str] = []
        for language, candidates in _LANGUAGE_HINTS.items():
            if _collect_keywords(candidates, lower):
                language_hints.append(language)

        prediction = IntentPrediction(
            label=best_label,
            confidence=confidence,
            rationale=None,
            category_hints=list(_CATEGORY_HINTS.get(best_label, [])),
            tag_hints=list(_TAG_HINTS.get(best_label, [])),
            language_hints=language_hints,
        )

        if best_label is IntentLabel.GENERAL and language_hints:
            prediction.confidence = max(prediction.confidence, 0.3)

        logger.debug(
            "Intent classification",
            extra={
                "label": prediction.label.value,
                "confidence": prediction.confidence,
                "languages": prediction.language_hints,
            },
        )

        return prediction


class PromptLikeProtocol(Protocol):
    """Structural type describing objects accepted by `rank_by_hints`."""

    @property
    def id(self) -> object:  # pragma: no cover - attribute hints only
        """Return the identifier for the prompt-like instance."""

    @property
    def category(self) -> str:  # pragma: no cover - attribute hints only
        """Return the category slug associated with the prompt-like instance."""

    @property
    def tags(self) -> Sequence[str]:  # pragma: no cover - attribute hints only
        """Return the tag sequence for the prompt-like instance."""


def rank_by_hints[PromptLikeT: PromptLikeProtocol](
    prompts: Iterable[PromptLikeT],
    *,
    category_hints: Sequence[str],
    tag_hints: Sequence[str],
) -> list[PromptLikeT]:
    """Return prompts ordered by category/tag hints while preserving stability."""
    matched: list[PromptLikeT] = []
    secondary: list[PromptLikeT] = []
    remainder: list[PromptLikeT] = []

    normalized_categories = {hint.lower() for hint in category_hints}
    normalized_tags = {hint.lower() for hint in tag_hints}

    for prompt in prompts:
        category = getattr(prompt, "category", "") or ""
        tags = getattr(prompt, "tags", []) or []

        category_match = category.lower() in normalized_categories if category else False
        tag_match = any(tag.lower() in normalized_tags for tag in tags)

        if category_match:
            matched.append(prompt)
        elif tag_match:
            secondary.append(prompt)
        else:
            remainder.append(prompt)

    return [*matched, *secondary, *remainder]


__all__ = [
    "IntentClassifier",
    "IntentClassifierError",
    "IntentLabel",
    "IntentPrediction",
    "rank_by_hints",
    "PromptLikeProtocol",
]
