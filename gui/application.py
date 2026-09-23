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
import re
import subprocess
import sys
from pathlib import Path
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
_XCB_MISSING_LIBRARY_PATTERN = re.compile(r"^\s*(\S+)\s+=>\s+not found\s*$", re.MULTILINE)
_XCB_RUNTIME_PACKAGES = "libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 libxkbcommon-x11-0"
logger = logging.getLogger("prompt_manager.gui.application")


class GuiRuntimeError(RuntimeError):
    """Raised when the local desktop environment cannot start the Qt GUI."""


def _xcb_platform_plugin_path() -> Path | None:
    """Return PySide6's bundled Linux xcb platform plugin when available."""
    try:
        import PySide6
    except ModuleNotFoundError:  # pragma: no cover - handled by gui package fallback
        return None
    package_file = getattr(PySide6, "__file__", None)
    if not isinstance(package_file, str):
        return None
    plugin_path = (
        Path(package_file).resolve().parent / "Qt" / "plugins" / "platforms" / "libqxcb.so"
    )
    return plugin_path if plugin_path.is_file() else None


def linux_xcb_runtime_issues(env: MutableMapping[str, str] | None = None) -> list[str]:
    """Return unresolved xcb-plugin libraries before Qt can abort the process.

    Qt terminates the process when an xcb plugin dependency is absent, which makes
    the normal Python launcher error handling unreachable. The probe is Linux-only
    and deliberately skips explicit non-xcb platforms and Wayland sessions.
    """
    environment = os.environ if env is None else env
    if not sys.platform.startswith("linux"):
        return []
    requested_platform = environment.get("QT_QPA_PLATFORM", "").strip().lower()
    if requested_platform and requested_platform != "xcb":
        return []
    no_x11_display = not environment.get("DISPLAY")
    if not requested_platform and (environment.get("WAYLAND_DISPLAY") or no_x11_display):
        return []

    plugin_path = _xcb_platform_plugin_path()
    if plugin_path is None:
        return []
    try:
        result = subprocess.run(
            ("ldd", str(plugin_path)),
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []

    missing: list[str] = []
    for library in _XCB_MISSING_LIBRARY_PATTERN.findall(result.stdout):
        if library not in missing:
            missing.append(library)
    return missing


def ensure_gui_runtime(env: MutableMapping[str, str] | None = None) -> None:
    """Fail closed with actionable guidance before Qt loads an incomplete xcb plugin."""
    missing_libraries = linux_xcb_runtime_issues(env)
    if not missing_libraries:
        return
    missing = ", ".join(missing_libraries)
    raise GuiRuntimeError(
        "PromptManager GUI cannot start because the Qt xcb plugin is missing: "
        f"{missing}. On Ubuntu/Debian install the Qt X11 runtime libraries with: "
        f"sudo apt install {_XCB_RUNTIME_PACKAGES}"
    )


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
        return cast(
            "QApplication", existing
        )  # Reuse existing instance when running inside tests/tools

    if _should_force_offscreen(os.environ):
        # Allow running in headless environments by defaulting to the offscreen plugin.
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    ensure_gui_runtime()

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
