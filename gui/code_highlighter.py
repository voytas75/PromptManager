"""Syntax highlighting helpers for the query workspace.

Updates:
  v0.1.1 - 2025-11-29 - Wrap highlight rule definitions to satisfy Ruff line-length rules.
  v0.1.0 - 2025-11-10 - Provide keyword-based highlighting for common languages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PySide6.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(slots=True)
class _HighlightRule:
    pattern: re.Pattern[str]
    fmt: QTextCharFormat


def _format(color: str, *, bold: bool = False) -> QTextCharFormat:
    text_format = QTextCharFormat()
    text_format.setForeground(QColor(color))
    if bold:
        text_format.setFontWeight(QFont.Weight.Bold)
    return text_format


def _compile_keywords(keywords: Iterable[str]) -> re.Pattern[str]:
    escaped = "|".join(re.escape(keyword) for keyword in keywords)
    return re.compile(rf"\b({escaped})\b") if escaped else re.compile(r"^$")


def _compile_pattern(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


def _python_rules() -> list[_HighlightRule]:
    return [
        _HighlightRule(
            _compile_keywords(
                (
                    "def",
                    "class",
                    "import",
                    "from",
                    "async",
                    "await",
                    "return",
                    "yield",
                    "try",
                    "except",
                    "with",
                    "for",
                    "while",
                    "if",
                    "else",
                    "elif",
                    "raise",
                )
            ),
            _format("#007acc", bold=True),
        ),
        _HighlightRule(_compile_pattern(r"#[^\n]*"), _format("#6a9955")),
        _HighlightRule(_compile_pattern(r"\b(True|False|None)\b"), _format("#569cd6")),
        _HighlightRule(_compile_pattern(r"\b(self)\b"), _format("#dcdcaa")),
    ]


def _powershell_rules() -> list[_HighlightRule]:
    return [
        _HighlightRule(
            _compile_pattern(r"\b(Get|Set|New|Invoke|Start|Stop|Test)-[A-Za-z]+"),
            _format("#c586c0", bold=True),
        ),
        _HighlightRule(_compile_pattern(r"\$[A-Za-z_][\w:]*"), _format("#9cdcfe")),
        _HighlightRule(_compile_pattern(r"#[^\n]*"), _format("#6a9955")),
        _HighlightRule(
            _compile_pattern(r"\[(string|int|bool|array|hashtable)\]"),
            _format("#4fc1ff"),
        ),
    ]


def _bash_rules() -> list[_HighlightRule]:
    return [
        _HighlightRule(_compile_pattern(r"^#!.*$"), _format("#6a9955")),
        _HighlightRule(
            _compile_keywords(("if", "then", "fi", "for", "do", "done", "elif", "else", "while")),
            _format("#d19a66", bold=True),
        ),
        _HighlightRule(_compile_pattern(r"#[^\n]*"), _format("#6a9955")),
        _HighlightRule(_compile_pattern(r"\$[A-Za-z_][\w]*"), _format("#9cdcfe")),
    ]


def _markdown_rules() -> list[_HighlightRule]:
    return [
        _HighlightRule(_compile_pattern(r"(^|\n)#{1,6}[^\n]*"), _format("#dcdcaa", bold=True)),
        _HighlightRule(_compile_pattern(r"\*\*[^\*]+\*\*"), _format("#c586c0")),
        _HighlightRule(_compile_pattern(r"`[^`]+`"), _format("#4fc1ff")),
        _HighlightRule(_compile_pattern(r"\[[^\]]+\]\([^\)]+\)"), _format("#ce9178")),
    ]


def _json_rules() -> list[_HighlightRule]:
    return [
        _HighlightRule(_compile_pattern(r"\"[^\"]+\"(?=\s*:)"), _format("#9cdcfe")),
        _HighlightRule(_compile_pattern(r"\b(true|false|null)\b"), _format("#569cd6")),
        _HighlightRule(_compile_pattern(r"\d+"), _format("#b5cea8")),
    ]


def _yaml_rules() -> list[_HighlightRule]:
    return [
        _HighlightRule(_compile_pattern(r"^\s*\w+:"), _format("#9cdcfe", bold=True)),
        _HighlightRule(_compile_pattern(r"^\s*-\s"), _format("#ce9178", bold=True)),
        _HighlightRule(_compile_pattern(r"#.*"), _format("#6a9955")),
    ]


_RULE_BUILDERS = {
    "python": _python_rules,
    "powershell": _powershell_rules,
    "bash": _bash_rules,
    "markdown": _markdown_rules,
    "json": _json_rules,
    "yaml": _yaml_rules,
}


class CodeHighlighter(QSyntaxHighlighter):
    """Keyword-based syntax highlighter for the free-form query input."""

    def __init__(self, document) -> None:  # type: ignore[override]
        """Prepare the highlighter with a target QTextDocument."""
        super().__init__(document)
        self._language = "plain"
        self._rules: list[_HighlightRule] = []

    @property
    def language(self) -> str:
        """Return the currently active language identifier."""
        return self._language

    def set_language(self, language: str) -> None:
        """Update the language and rebuild syntax rules as needed."""
        normalized = (language or "plain").lower()
        if normalized == self._language:
            return
        self._language = normalized
        builder = _RULE_BUILDERS.get(normalized)
        self._rules = builder() if builder else []
        self.rehighlight()

    def highlightBlock(self, text: str) -> None:  # noqa: N802 - Qt API
        """Apply syntax formats for the current block of text."""
        if not self._rules or not text:
            return
        for rule in self._rules:
            for match in rule.pattern.finditer(text):
                start, end = match.span()
                if start < end:
                    self.setFormat(start, end - start, rule.fmt)


__all__ = ["CodeHighlighter"]
