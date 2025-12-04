"""Runtime boot helpers for Prompt Manager CLI.

Updates:
  v0.1.0 - 2025-12-04 - Extract logging configuration helpers.
"""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path


def setup_logging(logging_conf_path: Path | None) -> None:
    """Configure logging using *logging_conf_path* when available."""
    path = logging_conf_path or Path("config/logging.conf")
    if path.exists():
        try:
            logging.config.fileConfig(path, disable_existing_loggers=False)
            return
        except Exception:  # pragma: no cover - configuration fallback
            pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
