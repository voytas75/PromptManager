"""Minimal PySide6.QtGui stubs so Pyright tolerates Qt-heavy modules.

Updates:
  v0.1.0 - 2025-12-02 - Shim exported QtGui symbols referenced by the GUI package.
"""
from __future__ import annotations

from typing import Any

QClipboard: Any
QCloseEvent: Any
QColor: Any
QFont: Any
QGuiApplication: Any
QIcon: Any
QKeySequence: Any
QPaintEvent: Any
QPainter: Any
QPalette: Any
QResizeEvent: Any
QShortcut: Any
QShowEvent: Any
QSyntaxHighlighter: Any
QTextCharFormat: Any
QTextCursor: Any

__all__ = [
    "QClipboard",
    "QCloseEvent",
    "QColor",
    "QFont",
    "QGuiApplication",
    "QIcon",
    "QKeySequence",
    "QPaintEvent",
    "QPainter",
    "QPalette",
    "QResizeEvent",
    "QShortcut",
    "QShowEvent",
    "QSyntaxHighlighter",
    "QTextCharFormat",
    "QTextCursor",
]
