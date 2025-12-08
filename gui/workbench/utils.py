"""Utility helpers shared across Enhanced Prompt Workbench widgets.

Updates:
  v0.1.1 - 2025-12-08 - Cast application palette retrieval for Pyright.
  v0.1.0 - 2025-12-04 - Extract palette helpers, snippet presets, and token parsing.
"""

from __future__ import annotations

import re
import textwrap
from typing import TYPE_CHECKING, cast

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QWidget

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Mapping

    from PySide6.QtGui import QTextCursor
else:  # pragma: no cover - runtime placeholders for type-only imports
    from typing import Any as _Any

    Mapping = _Any
    QTextCursor = _Any

__all__ = [
    "BLOCK_SNIPPETS",
    "StreamRelay",
    "inherit_palette",
    "normalise_variable_token",
    "variable_at_cursor",
]


class StreamRelay(QObject):
    """Signal relay used to forward streaming CodexExecutor output."""

    chunk = Signal(str)


def inherit_palette(widget: QWidget) -> None:
    """Copy the parent or application palette into ``widget`` when available."""
    parent = widget.parent()
    palette = parent.palette() if isinstance(parent, QWidget) else None
    if palette is None:
        app = cast("QGuiApplication | None", QGuiApplication.instance())
        palette = app.palette() if app is not None else None
    if palette is None:
        return
    widget.setPalette(palette)
    widget.setAutoFillBackground(True)


BLOCK_SNIPPETS: Mapping[str, str] = {
    "System Role": textwrap.dedent(
        """### System Role\nYou are a meticulous assistant that follows instructions exactly."""
    ),
    "Context": textwrap.dedent(
        """### Context\nDescribe the task, background knowledge, and any important caveats here."""
    ),
    "Constraints": textwrap.dedent(
        """### Constraints\n- Limit answers to 200 words.\n- Use professional tone."""
    ),
    "Examples": textwrap.dedent(
        """### Examples\n**Input**\n<example input>\n\n**Expected Output**\n<example output>"""
    ),
    "JSON Response": textwrap.dedent(
        """### Output Format
Return a JSON object with these keys:
- `summary`: One sentence overview.
- `steps`: Array of ordered actions."""
    ),
}


_JINJA_PATTERN = re.compile(r"{{\s*(?P<name>[A-Za-z0-9_\.]+)\s*(?:\|[^}]*)?}}")


def variable_at_cursor(cursor: QTextCursor) -> str | None:
    """Return the variable token intersecting the cursor selection, if any."""
    block_text = cursor.block().text()
    column = cursor.positionInBlock()
    for match in _JINJA_PATTERN.finditer(block_text):
        start, end = match.span()
        if start <= column <= end:
            return match.group("name")
    return None


def normalise_variable_token(text: str) -> str | None:
    """Return the canonical token name when ``text`` looks like a variable."""
    stripped = text.strip()
    if not stripped:
        return None
    match = _JINJA_PATTERN.search(stripped)
    if match:
        return match.group("name")
    if stripped.isidentifier():
        return stripped
    return None
