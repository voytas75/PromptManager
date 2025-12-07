"""Appearance helpers for Prompt Manager windows.

Updates:
  v0.1.0 - 2025-12-01 - Introduced AppearanceController managing theme + palette utilities.
"""

from __future__ import annotations

from collections import abc
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication, QPalette

from config import (
    DEFAULT_CHAT_ASSISTANT_BUBBLE_COLOR,
    DEFAULT_CHAT_USER_BUBBLE_COLOR,
    DEFAULT_THEME_MODE,
)

if TYPE_CHECKING:
    from PySide6.QtWidgets import QMainWindow, QWidget

_CHAT_PALETTE_KEYS = {"user", "assistant"}


def _default_chat_palette() -> dict[str, str]:
    return {
        "user": QColor(DEFAULT_CHAT_USER_BUBBLE_COLOR).name().lower(),
        "assistant": QColor(DEFAULT_CHAT_ASSISTANT_BUBBLE_COLOR).name().lower(),
    }


def normalise_chat_palette(palette: abc.Mapping[str, object] | None) -> dict[str, str]:
    """Return a validated chat palette containing user/assistant colours.

    Invalid entries, unsupported roles, and malformed colour values are ignored.
    Hex codes are normalised to lowercase ``#rrggbb`` strings for consistency.
    """
    cleaned: dict[str, str] = {}
    if not palette or not isinstance(palette, abc.Mapping):
        return cleaned
    for role, value in palette.items():
        if role not in _CHAT_PALETTE_KEYS:
            continue
        text = str(value).strip()
        if not text:
            continue
        candidate = QColor(text)
        if candidate.isValid():
            cleaned[role] = candidate.name().lower()
    return cleaned


def palette_differs_from_defaults(palette: abc.Mapping[str, str] | None) -> bool:
    """Return True when the provided palette differs from default chat colours."""
    if not palette or not isinstance(palette, abc.Mapping):
        return False
    defaults = _default_chat_palette()
    for role, default_hex in defaults.items():
        value = palette.get(role)
        if value is not None and value != default_hex:
            return True
    return False


class AppearanceController:
    """Manage theming and palette styling for the main window."""

    def __init__(self, window: QMainWindow, runtime_settings: dict[str, object | None]) -> None:
        """Store the target window and runtime settings reference."""
        self._window = window
        self._runtime_settings = runtime_settings
        self._container: QWidget | None = None

    def set_container(self, container: QWidget | None) -> None:
        """Track the main container so palette refresh can update borders."""
        self._container = container
        self.refresh_theme_styles()

    def apply_theme(self, mode: str | None = None) -> str:
        """Apply the provided or configured theme and return the active mode."""
        app = QGuiApplication.instance()
        if app is None:
            return DEFAULT_THEME_MODE

        raw_theme = mode or self._runtime_settings.get("theme_mode") or DEFAULT_THEME_MODE
        theme = str(raw_theme).strip().lower()
        if theme not in {"light", "dark"}:
            theme = DEFAULT_THEME_MODE

        if theme == "dark":
            palette = QPalette()
            palette.setColor(QPalette.Window, QColor(31, 41, 51))
            palette.setColor(QPalette.WindowText, Qt.white)
            palette.setColor(QPalette.Base, QColor(24, 31, 41))
            palette.setColor(QPalette.AlternateBase, QColor(37, 46, 59))
            palette.setColor(QPalette.ToolTipBase, QColor(45, 55, 68))
            palette.setColor(QPalette.ToolTipText, Qt.white)
            palette.setColor(QPalette.Text, QColor(232, 235, 244))
            palette.setColor(QPalette.Button, QColor(45, 55, 68))
            palette.setColor(QPalette.ButtonText, Qt.white)
            palette.setColor(QPalette.BrightText, QColor(255, 107, 107))
            palette.setColor(QPalette.Link, QColor(140, 180, 255))
            palette.setColor(QPalette.Highlight, QColor(99, 102, 241))
            palette.setColor(QPalette.HighlightedText, Qt.white)
            palette.setColor(QPalette.PlaceholderText, QColor(156, 163, 175))
            palette.setColor(QPalette.Disabled, QPalette.Text, QColor(110, 115, 125))
            palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(110, 115, 125))
            app.setPalette(palette)
            app.setStyleSheet(
                "QToolTip { color: #f9fafc; background-color: #1f2933; border: 1px solid #3b4252; }"
            )
        else:
            palette = app.style().standardPalette()
            app.setPalette(palette)
            app.setStyleSheet("")
            theme = "light"

        self._window.setPalette(app.palette())
        self._runtime_settings["theme_mode"] = theme
        self.refresh_theme_styles()
        return theme

    def refresh_theme_styles(self) -> None:
        """Update widgets styled using palette-derived colours."""
        container = self._container
        if container is None:
            return

        app = QGuiApplication.instance()
        palette = app.palette() if app is not None else container.palette()
        window_color = palette.color(QPalette.Window)
        border_color = QColor(
            255 - window_color.red(),
            255 - window_color.green(),
            255 - window_color.blue(),
        )
        border_color.setAlpha(255)
        container.setStyleSheet(
            "#mainContainer { "
            f"border: 1px solid {border_color.name()}; "
            "border-radius: 6px; background-color: palette(base); }"
        )


__all__ = [
    "AppearanceController",
    "normalise_chat_palette",
    "palette_differs_from_defaults",
]
