"""Application entry point for Prompt Manager.

Updates: v0.5.0 - 2025-11-05 - Seed prompt repository from packaged catalogue before launch.
Updates: v0.4.1 - 2025-11-05 - Launch GUI by default and add --no-gui flag.
Updates: v0.4.0 - 2025-11-05 - Ensure manager shutdown occurs on exit and update GUI guidance.
Updates: v0.3.0 - 2025-11-05 - Gracefully handle missing GUI dependencies.
Updates: v0.2.0 - 2025-11-04 - Add optional PySide6 GUI launcher toggle.
Updates: v0.1.0 - 2025-10-30 - Initial CLI bootstrap loading settings and building services.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import logging.config
from pathlib import Path
from typing import Callable, Optional, cast

from config import load_settings
from core import build_prompt_manager
from core.catalog_importer import import_prompt_catalog


def _setup_logging(logging_conf_path: Optional[Path]) -> None:
    """Configure logging from a dictConfig file when present."""
    # Default to config/logging.conf if not provided
    path = logging_conf_path or Path("config/logging.conf")
    if path.exists():
        try:
            logging.config.fileConfig(path, disable_existing_loggers=False)
            return
        except Exception:  # pragma: no cover - logging config errors are non-critical
            pass
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prompt Manager launcher")
    parser.add_argument(
        "--logging-config",
        type=Path,
        default=None,
        help="Path to logging configuration file (INI format)",
    )
    parser.add_argument(
        "--print-settings",
        action="store_true",
        help="Print resolved settings and exit",
    )
    parser.add_argument(
        "--gui",
        dest="gui",
        action="store_true",
        default=None,
        help="Launch the PySide6 interface after services are initialised (default behaviour).",
    )
    parser.add_argument(
        "--no-gui",
        dest="gui",
        action="store_false",
        help="Skip launching the GUI and exit once services are initialised.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _setup_logging(args.logging_config)

    logger = logging.getLogger("prompt_manager.main")
    try:
        settings = load_settings()
    except Exception as exc:
        logger.error("Failed to load settings: %s", exc)
        return 2

    if args.print_settings:
        logger.info(
            "Resolved settings: db=%s chroma=%s redis=%s ttl=%s",
            settings.db_path,
            settings.chroma_path,
            settings.redis_dsn,
            settings.cache_ttl_seconds,
        )
        return 0

    # Build core services; GUI wiring will attach here in later milestones
    try:
        manager = build_prompt_manager(settings)
    except Exception as exc:
        logger.error("Failed to initialise services: %s", exc)
        return 3

    catalog_result = None
    try:
        catalog_result = import_prompt_catalog(manager, settings.catalog_path)
        if catalog_result.added or catalog_result.updated:
            logger.info(
                "Prompt catalogue synced: added=%d updated=%d skipped=%d",
                catalog_result.added,
                catalog_result.updated,
                catalog_result.skipped,
            )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to import prompt catalogue: %s", exc)

    try:
        # Minimal interactive stub to verify bootstrap until GUI arrives
        logger.info("Prompt Manager ready. Database at %s", settings.db_path)
        logger.info("ChromaDB at %s", settings.chroma_path)
        launch_gui = args.gui if args.gui is not None else True
        if launch_gui:
            try:
                gui_module = importlib.import_module("gui")
            except ModuleNotFoundError as exc:
                logger.error(
                    "GUI launch requested but optional dependency %s is missing. "
                    "Install GUI extras with `pip install -r requirements-gui.txt` or rerun without --gui.",
                    exc.name,
                )
                return 4
            try:
                launch_gui = getattr(gui_module, "launch_prompt_manager")
            except AttributeError:
                logger.error(
                    "GUI module is missing the launch_prompt_manager entry point. "
                    "Reinstall GUI extras or launch without --gui."
                )
                return 4
            if not callable(launch_gui):
                logger.error(
                    "GUI module launch_prompt_manager entry point is not callable. "
                    "Reinstall GUI extras or launch without --gui."
                )
                return 4
            launch_callable = cast(Callable[[object], int], launch_gui)

            dependency_error_type = getattr(gui_module, "GuiDependencyError", RuntimeError)
            if not isinstance(dependency_error_type, type) or not issubclass(
                dependency_error_type, BaseException
            ):
                dependency_error_type = RuntimeError

            try:
                return launch_callable(manager)
            except Exception as exc:
                if isinstance(exc, dependency_error_type):
                    logger.error("Unable to start GUI: %s", exc)
                    return 4
                logger.error("Unexpected error while starting GUI: %s", exc)
                return 4

        _ = manager  # placeholder to avoid unused variable until GUI implemented
        return 0
    finally:
        manager.close()


if __name__ == "__main__":
    raise SystemExit(main())
