"""Application entry point for Prompt Manager.

Updates: v0.1.0 - 2025-10-30 - Initial CLI bootstrap loading settings and building services.
"""

from __future__ import annotations

import argparse
import logging
import logging.config
from pathlib import Path
from typing import Optional

from config import load_settings
from core import build_prompt_manager


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

    # Minimal interactive stub to verify bootstrap until GUI arrives
    logger.info("Prompt Manager ready. Database at %s", settings.db_path)
    logger.info("ChromaDB at %s", settings.chroma_path)
    _ = manager  # placeholder to avoid unused variable until GUI implemented
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
