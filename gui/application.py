"""Qt application helpers for Prompt Manager GUI.

Updates:
  v0.1.5 - 2025-12-08 - Guard ApplicationAttribute and style lookups for stubbed Qt modules.
  v0.1.4 - 2025-11-30 - Make Fusion style optional for stubbed Qt environments.
  v0.1.3 - 2025-11-29 - Apply Fusion style globally and log active GUI style for debugging.
  v0.1.2 - 2025-11-05 - Apply packaged application icon for desktop builds.
  v0.1.1 - 2025-11-05 - Detect display server before forcing offscreen backend.
  v0.1.0 - 2025-11-04 - Provide QApplication factory and launch routine.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING, Any, cast

from PySide6.QtCore import Qt

try:  # pragma: no cover - guard for stubbed PySide6 modules during tests
    from PySide6.QtWidgets import QApplication, QStyleFactory
except ImportError:  # pragma: no cover - stub environments may omit QStyleFactory
    from PySide6.QtWidgets import QApplication

    QStyleFactory = None  # type: ignore[assignment]

from .main_window import MainWindow
from .resources import load_application_icon

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence

    from config import PromptManagerSettings
    from core import PromptManager

_DISPLAY_ENV_VARS = ("DISPLAY", "WAYLAND_DISPLAY", "MIR_SOCKET")
logger = logging.getLogger("prompt_manager.gui.application")


def _should_force_offscreen(env: MutableMapping[str, str]) -> bool:
    """Return True when we should default Qt to the offscreen platform plugin."""
    if env.get("QT_QPA_PLATFORM"):
        return False

    if sys.platform.startswith(("win", "cygwin")) or sys.platform == "darwin":
        return False

    return not any(env.get(var) for var in _DISPLAY_ENV_VARS)


def _set_application_attribute(attribute: str, enable: bool) -> None:
    """Best-effort setter for Qt application attributes in stubbed environments."""
    enum = getattr(Qt, "ApplicationAttribute", None)
    if enum is None:
        return
    flag = getattr(enum, attribute, None)
    if flag is None:
        return
    QApplication.setAttribute(flag, enable)


def create_qapplication(argv: Sequence[str] | None = None) -> QApplication:
    """Return an existing QApplication or create a new one with sensible defaults."""
    existing = QApplication.instance()
    if existing is not None:
        return cast(QApplication, existing)  # Reuse existing instance when running inside tests/tools

    if _should_force_offscreen(os.environ):
        # Allow running in headless environments by defaulting to the offscreen plugin.
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    _set_application_attribute("AA_EnableHighDpiScaling", True)
    _set_application_attribute("AA_UseHighDpiPixmaps", True)
    app = QApplication(list(argv or []))
    fusion: Any | None = None
    if QStyleFactory is not None:  # pragma: no branch - branch depends on import success
        fusion = QStyleFactory.create("Fusion")
    if fusion is not None:
        app.setStyle(fusion)
    style_name = "<unknown>"
    style_obj: Any | None = None
    if hasattr(app, "style"):
        try:
            style_obj = app.style()
        except Exception:  # pragma: no cover - stubbed Qt may lack style()
            style_obj = None
    if style_obj is not None:
        try:
            style_name = style_obj.metaObject().className()
        except AttributeError:  # pragma: no cover - stubbed Qt lacks metaObject
            style_name = "<unavailable>"
    else:
        style_name = "<unavailable>"
    logger.debug("GUI_STYLE active_style=%s", style_name)
    return app


def launch_prompt_manager(
    prompt_manager: PromptManager, settings: PromptManagerSettings | None = None
) -> int:
    """Create the Qt event loop, show the main window, and enter the GUI."""
    app = create_qapplication()
    icon = load_application_icon()
    if icon is not None:
        app.setWindowIcon(icon)

    window = MainWindow(prompt_manager, settings=settings)
    if icon is not None:
        window.setWindowIcon(icon)
    window.show()
    return app.exec()


__all__ = ["create_qapplication", "launch_prompt_manager"]
