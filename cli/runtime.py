"""Runtime boot helpers for Prompt Manager CLI.

Updates:
  v0.1.1 - 2025-12-10 - Add LiteLLM logging toggle helper.
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


def configure_litellm_logging(enabled: bool) -> None:
    """Enable or disable upstream LiteLLM library logs."""
    litellm_loggers = (
        logging.getLogger("litellm"),
        logging.getLogger("litellm.proxy"),
        logging.getLogger("litellm.proxy.proxy_server"),
    )
    for litellm_logger in litellm_loggers:
        litellm_logger.propagate = True
        if enabled:
            litellm_logger.disabled = False
            litellm_logger.setLevel(logging.NOTSET)
        else:
            litellm_logger.disabled = True
            litellm_logger.setLevel(logging.CRITICAL)
