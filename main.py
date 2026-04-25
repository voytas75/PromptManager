"""Application entry point for Prompt Manager.

Updates:
  v0.9.3 - 2025-12-10 - Apply LiteLLM logging toggle from settings.
  v0.9.2 - 2025-12-09 - Offer to create config/config.json from template when missing.
  v0.9.1 - 2025-12-05 - Remove duplicate COMMAND_SPECS import flagged by Ruff.
  v0.9.0 - 2025-12-04 - Modularise CLI parsing, commands, and GUI launcher helpers.
  v0.8.3 - 2025-11-30 - Restore stdout messaging for CLI commands and relax history imports.
  v0.8.2 - 2025-11-29 - Reformat CLI summary output and modernise type hints.
  v0.8.1 - 2025-11-28 - Add analytics diagnostics CLI command.
  v0.8.0 - 2025-02-14 - Add embedding diagnostics CLI command.
  v0.7.9 - 2025-11-28 - Add benchmark and scenario refresh CLI commands.
  v0.7.8 - 2025-12-07 - Add CLI command to rebuild embeddings from scratch.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    from config import load_settings
except (ImportError, AttributeError):  # pragma: no cover - fallback for test stubs
    import config as _config

    load_settings = _config.load_settings

if TYPE_CHECKING:
    from config import PromptManagerSettings
else:
    PromptManagerSettings = Any
from cli.gui_launcher import run_default_mode
from cli.parser import parse_args
from cli.runtime import configure_litellm_logging, setup_logging as _runtime_setup_logging
from cli.settings_summary import print_settings_summary
from gui.runtime_settings_service import build_config_diagnostics_items
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

DEFAULT_CONFIG_PATH = Path("config/config.json")
CONFIG_TEMPLATE_PATH = Path("config/config.template.json")


_setup_logging = _runtime_setup_logging


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


def _should_offer_config_creation(exc: Exception) -> bool:
    """Return True when settings loading failed because the default config is missing."""
    config_path_override = os.getenv("PROMPT_MANAGER_CONFIG_JSON")
    if config_path_override and config_path_override != str(DEFAULT_CONFIG_PATH):
        return False
    config_path = DEFAULT_CONFIG_PATH.expanduser()
    template_path = CONFIG_TEMPLATE_PATH.expanduser()
    try:
        config_path = config_path.resolve(strict=False)
        template_path = template_path.resolve(strict=False)
    except OSError:
        pass
    return (not config_path.exists()) and template_path.exists()


def _prompt_create_default_config(logger: logging.Logger) -> bool:
    """Ask the user to create config/config.json from the template when absent."""
    if not sys.stdin.isatty():
        logger.error(
            "Configuration file %s is missing. Create it from %s or set "
            "PROMPT_MANAGER_CONFIG_JSON to continue.",
            DEFAULT_CONFIG_PATH,
            CONFIG_TEMPLATE_PATH,
        )
        return False
    prompt = (
        f"Configuration file not found at {DEFAULT_CONFIG_PATH}. "
        f"Create it from {CONFIG_TEMPLATE_PATH}? [Y/n]: "
    )
    response = input(prompt).strip().lower()
    if response not in {"", "y", "yes"}:
        logger.error(
            "Configuration file is required. Create %s or point "
            "PROMPT_MANAGER_CONFIG_JSON to an existing JSON file.",
            DEFAULT_CONFIG_PATH,
        )
        return False
    contents = "{}\n"
    if CONFIG_TEMPLATE_PATH.exists():
        contents = CONFIG_TEMPLATE_PATH.read_text(encoding="utf-8")
    try:
        DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_CONFIG_PATH.write_text(contents, encoding="utf-8")
    except OSError as write_error:
        logger.error("Unable to create configuration file: %s", write_error)
        return False
    logger.info(
        "Created configuration at %s. Update it with your LiteLLM credentials and preferences.",
        DEFAULT_CONFIG_PATH,
    )
    print(f"Created configuration at {DEFAULT_CONFIG_PATH}")
    return True


def _blocking_runtime_issues(settings: PromptManagerSettings) -> list[str]:
    """Return critical runtime issues that should block normal startup."""
    diagnostics = build_config_diagnostics_items(
        litellm_model=getattr(settings, "litellm_model", None),
        litellm_inference_model=getattr(settings, "litellm_inference_model", None),
        litellm_api_key=getattr(settings, "litellm_api_key", None),
        embedding_backend=getattr(settings, "embedding_backend", None),
        embedding_model=getattr(settings, "embedding_model", None),
        litellm_tts_model=getattr(settings, "litellm_tts_model", None),
        redis_status=None,
    )
    issues: list[str] = []
    for raw_item in diagnostics.get("items", []):
        if not isinstance(raw_item, dict):
            continue
        label = str(raw_item.get("label") or "").strip()
        status = str(raw_item.get("status") or "").upper()
        detail = str(raw_item.get("detail") or "").strip()
        if status != "FAIL":
            continue
        if label == "Fast model" and detail == "missing":
            issues.append("LiteLLM fast model is missing.")
        elif label == "API key" and detail == "missing":
            issues.append("LiteLLM API key is missing.")
    return issues


def main() -> int:
    """Entrypoint that wires settings, services, and CLI commands."""
    args = parse_args()
    _runtime_setup_logging(args.logging_config)

    logger = logging.getLogger("prompt_manager.main")
    try:
        settings = load_settings()
    except Exception as exc:  # pragma: no cover - surfaced to CLI
        if _should_offer_config_creation(exc):
            created = _prompt_create_default_config(logger)
            if created:
                try:
                    settings = load_settings()
                except Exception as retry_error:  # pragma: no cover - surfaced to CLI
                    logger.error("Failed to load settings: %s", retry_error)
                    return 2
            else:
                return 2
        else:
            logger.error("Failed to load settings: %s", exc)
            return 2

    configure_litellm_logging(bool(getattr(settings, "litellm_logging_enabled", False)))
    if args.print_settings:
        print_settings_summary(settings)
        return 0

    blocking_issues = _blocking_runtime_issues(settings)
    command = getattr(args, "command", None)
    if blocking_issues and command is None:
        print("Blocking configuration issues:")
        for issue in blocking_issues:
            print(f"- {issue}")
        print("Run with --print-settings to inspect the effective configuration.")
        return 2

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
