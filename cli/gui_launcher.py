"""Default CLI behaviour for launching the Prompt Manager GUI."""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, cast

from .utils import print_and_log

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Callable


def run_default_mode(
    manager,
    settings,
    args,
    logger: logging.Logger,
) -> int:
    """Print readiness messages and optionally launch the GUI."""
    if manager is None:
        raise ValueError("Prompt Manager must be initialised before launching GUI mode.")

    print_and_log(logger, logging.INFO, f"Prompt Manager ready. Database at {settings.db_path}")
    print_and_log(logger, logging.INFO, f"ChromaDB at {settings.chroma_path}")
    launch_requested = args.gui if args.gui is not None else True
    if not launch_requested:
        return 0

    try:
        gui_module = importlib.import_module("gui")
    except ModuleNotFoundError as exc:  # pragma: no cover - import failure path
        logger.error(
            "GUI launch requested but dependency %s is missing. Install requirements with "
            "`pip install -r requirements.txt` or rerun without --gui.",
            exc.name,
        )
        return 4
    try:
        launch_gui_callable = gui_module.launch_prompt_manager
    except AttributeError:  # pragma: no cover - misconfigured GUI package
        logger.error(
            "GUI module is missing the launch_prompt_manager entry point. "
            "Reinstall dependencies with `pip install -r requirements.txt` "
            "or launch without --gui."
        )
        return 4
    if not callable(launch_gui_callable):  # pragma: no cover - misconfigured entrypoint
        logger.error(
            "GUI module launch_prompt_manager entry point is not callable. "
            "Reinstall dependencies with `pip install -r requirements.txt` "
            "or launch without --gui."
        )
        return 4
    launch_callable = cast("Callable[[object, object | None], int]", launch_gui_callable)

    dependency_error_type = getattr(gui_module, "GuiDependencyError", RuntimeError)
    if not isinstance(dependency_error_type, type) or not issubclass(
        dependency_error_type,
        BaseException,
    ):
        dependency_error_type = RuntimeError

    try:
        return launch_callable(manager, settings)
    except Exception as exc:  # pragma: no cover - GUI runtime error path
        if isinstance(exc, dependency_error_type):
            logger.error("Unable to start GUI: %s", exc)
            return 4
        logger.error("Unexpected error while starting GUI: %s", exc)
        return 4
