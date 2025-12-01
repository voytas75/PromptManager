"""Helpers for accessing bundled GUI resources.

Updates:
  v0.1.2 - 2025-11-29 - Remove unused Optional import flagged by Ruff.
  v0.1.1 - 2025-11-05 - Ship stylised Prompt Manager icon and expose loader.
  v0.1.0 - 2025-11-05 - Provide accessor for packaged application icon.
"""
from __future__ import annotations

import importlib.resources as importlib_resources

from PySide6.QtGui import QIcon


def load_application_icon() -> QIcon | None:
    """Return the application icon packaged with the GUI, if available."""
    try:
        icon_resource = importlib_resources.files(__name__).joinpath("prompt_manager.ico")
    except (FileNotFoundError, ModuleNotFoundError):
        return None

    try:
        with importlib_resources.as_file(icon_resource) as icon_path:
            icon = QIcon(str(icon_path))
    except FileNotFoundError:
        return None

    if icon.isNull():
        return None
    return icon


__all__ = ["load_application_icon"]
