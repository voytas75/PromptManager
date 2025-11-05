"""Qt application helpers for Prompt Manager GUI.

Updates: v0.1.2 - 2025-11-05 - Apply packaged application icon for desktop builds.
Updates: v0.1.1 - 2025-11-05 - Detect display server before forcing offscreen backend.
Updates: v0.1.0 - 2025-11-04 - Provide QApplication factory and launch routine.
"""

from __future__ import annotations

import os
import sys
from typing import MutableMapping, Optional, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from core import PromptManager
from config import PromptManagerSettings

from .main_window import MainWindow
from .resources import load_application_icon


_DISPLAY_ENV_VARS = ("DISPLAY", "WAYLAND_DISPLAY", "MIR_SOCKET")


def _should_force_offscreen(env: MutableMapping[str, str]) -> bool:
    """Return True when we should default Qt to the offscreen platform plugin."""
    if env.get("QT_QPA_PLATFORM"):
        return False

    if sys.platform.startswith(("win", "cygwin")) or sys.platform == "darwin":
        return False

    return not any(env.get(var) for var in _DISPLAY_ENV_VARS)


def create_qapplication(argv: Optional[Sequence[str]] = None) -> QApplication:
    """Return an existing QApplication or create a new one with sensible defaults."""

    app = QApplication.instance()
    if app is not None:
        return app  # Reuse existing instance when running inside tests/tools

    if _should_force_offscreen(os.environ):
        # Allow running in headless environments by defaulting to the offscreen plugin.
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    return QApplication(list(argv or []))


def launch_prompt_manager(
    prompt_manager: PromptManager, settings: Optional[PromptManagerSettings] = None
) -> int:
    """Create the Qt event loop, show the main window, and enter the GUI."""

    app = create_qapplication()
    icon = load_application_icon()
    if icon is not None:
        app.setWindowIcon(icon)

    window = MainWindow(prompt_manager, settings=settings)
    if icon is not None and not window.windowIcon().cacheKey():
        window.setWindowIcon(icon)
    window.show()
    return app.exec()


__all__ = ["create_qapplication", "launch_prompt_manager"]
