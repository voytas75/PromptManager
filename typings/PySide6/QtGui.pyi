"""Minimal PySide6.QtGui stubs to keep Pyright strict mode happy.

Updates:
  v0.1.1 - 2025-12-02 - Expand PySide6.QtGui stub classes and expose referenced attributes.
  v0.1.0 - 2025-12-02 - Initial shim exposing Qt types used by the GUI.
"""
from __future__ import annotations

from typing import Any


class _QtBase:
    """Catch-all base providing permissive attribute access on Qt objects."""

    def __init__(self, *args: object, **kwargs: object) -> None: ...

    def __call__(self, *args: object, **kwargs: object) -> Any: ...

    def __getattr__(self, name: str, /) -> Any: ...


class QClipboard(_QtBase):
    ...


class QCloseEvent(_QtBase):
    ...


class QColor(_QtBase):
    ...


class QFont(_QtBase):
    Weight: Any


class QGuiApplication(_QtBase):
    clipboard: Any
    instance: Any
    primaryScreen: Any
    processEvents: Any


class QIcon(_QtBase):
    ...


class QKeySequence(_QtBase):
    ...


class QPaintEvent(_QtBase):
    ...


class QPainter(_QtBase):
    Antialiasing: Any


class QPalette(_QtBase):
    AlternateBase: Any
    Base: Any
    BrightText: Any
    Button: Any
    ButtonText: Any
    Disabled: Any
    Highlight: Any
    HighlightedText: Any
    Link: Any
    Mid: Any
    Midlight: Any
    PlaceholderText: Any
    Text: Any
    ToolTipBase: Any
    ToolTipText: Any
    Window: Any
    WindowText: Any


class QResizeEvent(_QtBase):
    ...


class QShortcut(_QtBase):
    ...


class QShowEvent(_QtBase):
    ...


class QSyntaxHighlighter(_QtBase):
    ...


class QTextCharFormat(_QtBase):
    ...


class QTextCursor(_QtBase):
    End: Any
    EndOfBlock: Any
    KeepAnchor: Any


__all__ = ["QClipboard", "QCloseEvent", "QColor", "QFont", "QGuiApplication", "QIcon", "QKeySequence", "QPaintEvent", "QPainter", "QPalette", "QResizeEvent", "QShortcut", "QShowEvent", "QSyntaxHighlighter", "QTextCharFormat", "QTextCursor"]