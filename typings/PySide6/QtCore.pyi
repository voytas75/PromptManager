from __future__ import annotations

from typing import Any

class QObject:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __getattr__(self, name: str) -> Any: ...

class QEvent:
    PaletteChange: int
    StyleChange: int
    FocusOut: int
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def type(self) -> int: ...

class QSettings(QObject):
    def value(self, key: str, type: type | None = None) -> Any: ...
    def setValue(self, key: str, value: Any) -> None: ...

class QUrl:
    def __init__(self, url: str | None = None) -> None: ...

class Signal:
    def __init__(self, *types: Any) -> None: ...
    def connect(self, slot, /) -> None: ...
    def emit(self, *args: Any, **kwargs: Any) -> None: ...

class Qt:
    AlignLeft: int
    AlignRight: int
    AlignTop: int
    AlignVCenter: int
    AlignHCenter: int
    AlignBottom: int
    Horizontal: int
    Vertical: int
    RightArrow: int
    DownArrow: int
    ToolButtonTextBesideIcon: int
    OtherFocusReason: int
    ScrollBarAlwaysOff: int
    RichText: int
    TextSelectableByMouse: int
    TextBrowserInteraction: int
    PointingHandCursor: int
    MatchFixedString: int
    WaitCursor: int
