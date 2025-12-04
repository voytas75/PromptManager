"""Application entry point for Prompt Manager.

Updates:
  v0.9.0 - 2025-12-04 - Modularise CLI parsing, commands, and GUI launcher helpers.
  v0.8.3 - 2025-11-30 - Restore stdout messaging for CLI commands and relax history imports.
  v0.8.2 - 2025-11-29 - Reformat CLI summary output and modernise type hints.
  v0.8.1 - 2025-11-28 - Add analytics diagnostics target with dashboard export.
  v0.8.0 - 2025-02-14 - Add embedding diagnostics CLI command.
  v0.7.9 - 2025-11-28 - Add benchmark and scenario refresh CLI commands.
  v0.7.8 - 2025-12-07 - Add CLI command to rebuild embeddings from scratch.
  v0.7.7 - 2025-11-05 - Surface LiteLLM workflow routing details in CLI summaries.
  v0.7.6 - 2025-11-05 - Expand CLI settings summary to list fast and inference models.
  v0.7.5 - 2025-11-30 - Remove catalogue import command and startup messaging.
  v0.7.4 - 2025-11-26 - Surface LiteLLM streaming configuration in CLI summaries.
  v0.7.3 - 2025-11-17 - Require explicit catalogue paths; skip built-in seeding.
  v0.7.2 - 2025-11-15 - Extend --print-settings with health checks and masked output.
  v0.7.1 - 2025-11-14 - Simplify GUI dependency guidance for unified installs.
  v0.7.0 - 2025-11-07 - Add semantic suggestion CLI to verify embedding backends.
  v0.4.1 - 2025-11-05 - Launch GUI by default and add --no-gui flag.
  v0.4.0 - 2025-11-05 - Ensure manager shutdown occurs on exit and update GUI guidance.
  v0.3.0 - 2025-11-05 - Gracefully handle missing GUI dependencies.
  v0.2.0 - 2025-11-04 - Add optional PySide6 GUI launcher toggle.
  v0.1.0 - 2025-10-30 - Initial CLI bootstrap loading settings and building services.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cli.commands import COMMAND_SPECS
from cli.gui_launcher import run_default_mode
from cli.parser import parse_args
from cli.runtime import setup_logging as _runtime_setup_logging
from cli.settings_summary import print_settings_summary
from config import PromptManagerSettings, load_settings
from core import (
    build_analytics_snapshot as _core_build_analytics_snapshot,
    build_prompt_manager,
    export_prompt_catalog as _core_export_prompt_catalog,
    snapshot_dataset_rows as _core_snapshot_dataset_rows,
)

# Backwards-compatible re-exports for tests/legacy entry points.
build_analytics_snapshot = _core_build_analytics_snapshot
export_prompt_catalog = _core_export_prompt_catalog
snapshot_dataset_rows = _core_snapshot_dataset_rows


def _setup_logging(config_path) -> None:
    """Compatibility wrapper for older test harnesses expecting main._setup_logging."""
    _runtime_setup_logging(config_path)

from cli.commands import COMMAND_SPECS  # noqa: E402  (import moved for compatibility)

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from core.prompt_manager import PromptManager


def _initialise_manager(
    settings: PromptManagerSettings,
    logger: logging.Logger,
) -> PromptManager | None:
    try:
        return build_prompt_manager(settings)
    except Exception as exc:  # pragma: no cover - surfaced to CLI
        logger.error("Failed to initialise services: %s", exc)
        return None


def main() -> int:
    """Entrypoint that wires settings, services, and CLI commands."""
    args = parse_args()
    _runtime_setup_logging(args.logging_config)

    logger = logging.getLogger("prompt_manager.main")
    try:
        settings = load_settings()
    except Exception as exc:  # pragma: no cover - surfaced to CLI
        logger.error("Failed to load settings: %s", exc)
        return 2

    if args.print_settings:
        print_settings_summary(settings)
        return 0

    command = getattr(args, "command", None)
    spec = COMMAND_SPECS.get(command)

    manager = None
    manager_required = spec is None or spec.requires_manager
    if manager_required:
        manager = _initialise_manager(settings, logger)
        if manager is None:
            return 3

    try:
        if spec is not None:
            return spec.handler(manager, args, logger)

        if manager is None:
            manager = _initialise_manager(settings, logger)
            if manager is None:
                return 3
        return run_default_mode(manager, settings, args, logger)
    finally:
        if manager is not None:
            manager.close()


if __name__ == "__main__":
    raise SystemExit(main())
