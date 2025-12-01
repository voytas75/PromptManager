"""Tests for heuristic language detection utilities."""
from __future__ import annotations

from gui.language_tools import detect_language


def test_detect_language_returns_plain_for_empty_input() -> None:
    detection = detect_language("")
    assert detection.code == "plain"
    assert detection.confidence == 0.0


def test_detect_language_identifies_python_snippet() -> None:
    snippet = """
def add(a, b):
    return a + b
"""
    detection = detect_language(snippet)
    assert detection.code == "python"
    assert detection.name == "Python"
    assert detection.confidence > 0.3


def test_detect_language_identifies_markdown() -> None:
    snippet = "# Title\n\nSome **bold** text and a [link](https://example.com)."
    detection = detect_language(snippet)
    assert detection.code == "markdown"
    assert detection.confidence > 0.2
