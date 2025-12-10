"""Pytest configuration for shared test fixtures and environment hooks.

Updates:
  v0.1.0 - 2025-12-10 - Force Qt offscreen platform for headless test runs.
"""

import os
from typing import Any


def pytest_configure(config: Any) -> None:
    """Ensure Qt uses the offscreen platform during tests to avoid GUI aborts."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
