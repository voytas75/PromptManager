"""Tests for helper utilities in gui.main_window."""

from __future__ import annotations

from gui.diff_utils import build_diff_preview


def test_build_diff_preview_handles_empty_output() -> None:
    result = build_diff_preview("print('hello')\n", "")
    assert "output is empty" in result


def test_build_diff_preview_detects_identical_text() -> None:
    original = "line 1\nline 2"
    result = build_diff_preview(original, original)
    assert "No differences" in result


def test_build_diff_preview_returns_unified_diff() -> None:
    original = "line 1\nline 2\n"
    generated = "line 1\nline changed\n"
    diff = build_diff_preview(original, generated)
    assert diff.startswith("--- input")
    assert "@@" in diff
    assert "+line changed" in diff
