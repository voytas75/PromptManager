"""Minimal PySide6.QtCore stubs to keep Pyright strict mode happy.

Updates:
  v0.1.1 - 2025-12-02 - Expand PySide6.QtCore stub classes and expose referenced attributes.
  v0.1.0 - 2025-12-02 - Initial shim exposing Qt types used by the GUI.
"""

from __future__ import annotations

from typing import Any

class _QtBase:
    """Catch-all base providing permissive attribute access on Qt objects."""

    def __init__(self, *args: object, **kwargs: object) -> None: ...
    def __call__(self, *args: object, **kwargs: object) -> Any: ...
    def __getattr__(self, name: str, /) -> Any: ...

class QAbstractListModel(_QtBase): ...
class QByteArray(_QtBase): ...

class QEvent(_QtBase):
    ApplicationPaletteChange: Any
    FocusOut: Any
    PaletteChange: Any
    Resize: Any
    Show: Any
    StyleChange: Any

class QEventLoop(_QtBase):
    AllEvents: Any

class QModelIndex(_QtBase): ...
class QObject(_QtBase): ...
class QPoint(_QtBase): ...
class QRect(_QtBase): ...
class QSettings(_QtBase): ...
class QSize(_QtBase): ...

class QTimer(_QtBase):
    singleShot: Any

class Qt:
    AA_EnableHighDpiScaling: Any
    AA_UseHighDpiPixmaps: Any
    AlignCenter: Any
    AlignHCenter: Any
    AlignLeft: Any
    AlignRight: Any
    AlignTop: Any
    AlignVCenter: Any
    AlignmentFlag: Any
    ApplicationModal: Any
    CustomContextMenu: Any
    CustomizeWindowHint: Any
    Dialog: Any
    DisplayRole: Any
    DownArrow: Any
    EditRole: Any
    Horizontal: Any
    ItemIsEditable: Any
    MatchFixedString: Any
    Orientation: Any
    Orientations: Any
    OtherFocusReason: Any
    PointingHandCursor: Any
    RichText: Any
    RightArrow: Any
    ScrollBarAlwaysOff: Any
    ShortcutFocusReason: Any
    TabFocusReason: Any
    TextBrowserInteraction: Any
    TextSelectableByMouse: Any
    ToolButtonTextBesideIcon: Any
    UserRole: Any
    Vertical: Any
    WA_StyledBackground: Any
    WA_TransparentForMouseEvents: Any
    WaitCursor: Any
    WindowTitleHint: Any
    white: Any
    yellow: Any

class Signal(_QtBase):
    def __init__(self, *args: object, **kwargs: object) -> None: ...
    def emit(self, *args: object, **kwargs: object) -> None: ...

__all__ = [
    "QAbstractListModel",
    "QByteArray",
    "QEvent",
    "QEventLoop",
    "QModelIndex",
    "QObject",
    "QPoint",
    "QRect",
    "QSettings",
    "QSize",
    "QTimer",
    "Qt",
    "Signal",
]
