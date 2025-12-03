from __future__ import annotations

from typing import Any

class QColor:
    def __init__(self, value: Any = ...) -> None: ...
    def name(self) -> str: ...
    def isValid(self) -> bool: ...

class QGuiApplication:
    @staticmethod
    def primaryScreen() -> Any: ...
    @staticmethod
    def clipboard() -> Any: ...

class QPalette:
    Button: int
    ButtonText: int
    Mid: int
    Highlight: int
    HighlightedText: int
    def color(self, role: Any) -> QColor: ...

class QTextCursor:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
