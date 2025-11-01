"""Utility helpers for rendering text diffs in the Prompt Manager GUI.

Updates: v0.1.0 - 2025-11-10 - Introduce diff preview builder for GUI result comparisons.
"""

from __future__ import annotations

import difflib


def build_diff_preview(original: str, generated: str) -> str:
    """Return a unified diff (or explanatory message) comparing input/output."""

    original_text = original.strip("\n")
    generated_text = generated.strip("\n")
    if not generated_text:
        return "No diff available because the prompt output is empty."
    if original_text == generated_text:
        return "No differences detected between the input and the generated output."

    diff_lines = list(
        difflib.unified_diff(
            original_text.splitlines(),
            generated_text.splitlines(),
            fromfile="input",
            tofile="result",
            lineterm="",
        )
    )
    if diff_lines:
        return "\n".join(diff_lines)
    return "No differences detected between the input and the generated output."


__all__ = ["build_diff_preview"]
