"""GUI module namespace for Prompt Manager.

Updates: v0.3.1 - 2025-11-14 - Align dependency guidance with unified requirements install.
Updates: v0.3.0 - 2025-11-05 - Handle missing PySide6 dependency with friendly error.
Updates: v0.2.0 - 2025-11-04 - Expose PySide6 launcher utilities for prompt CRUD UI.
Updates: v0.1.0 - 2025-10-30 - Package scaffold.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from collections.abc import Sequence

    from config import PromptManagerSettings
    from core import PromptManager


class GuiDependencyError(RuntimeError):
    """Raised when the GUI cannot start because optional dependencies are absent."""


_MISSING_PYSIDE6_MESSAGE = (
    "PySide6 is not installed. Install dependencies with `pip install -r requirements.txt` "
    "before launching the GUI, or rerun with --no-gui."
)

try:
    from .application import create_qapplication, launch_prompt_manager
except ModuleNotFoundError as exc:  # pragma: no cover - exercised via main unit tests
    if exc.name != "PySide6":
        raise

    def _raise_create_qapplication(_: Sequence[str] | None = None) -> NoReturn:
        raise GuiDependencyError(_MISSING_PYSIDE6_MESSAGE)

    def _raise_launch_prompt_manager(
        _: PromptManager,
        __: PromptManagerSettings | None = None,
    ) -> NoReturn:
        raise GuiDependencyError(_MISSING_PYSIDE6_MESSAGE)

    create_qapplication = _raise_create_qapplication
    launch_prompt_manager = _raise_launch_prompt_manager


__all__ = ["create_qapplication", "launch_prompt_manager", "GuiDependencyError"]
