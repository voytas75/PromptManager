"""Storage helpers for Prompt Manager.

This sub‑module encapsulates interactions with persistence back‑ends (SQLite
DB, ChromaDB, Redis, etc.).  For the first incremental step we merely provide a
`PromptStorage` façade that wraps the existing :class:`core.repository.PromptRepository`.

In future iterations the detailed CRUD operations living in
`core.prompt_manager.__init__` will migrate here, but introducing the façade
now allows gradual refactors without impacting public APIs or tests.

Usage (internal):
    >>> from core.prompt_manager.storage import PromptStorage
    >>> storage = PromptStorage(db_path="data/prompt_manager.db")
    >>> prompt = storage.get("prompt_123")

Updates:
  v0.14.1 - 2025-11-29 - Move typing-only imports behind TYPE_CHECKING and wrap init.
  v0.14.0 - 2025-11-18 - Initial scaffold with proxy implementation.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..repository import PromptRepository  # relative to core package

if TYPE_CHECKING:
    from collections.abc import Iterator

    from models.prompt_model import Prompt

# Public re‑export to keep mypy/pyright consumers happy
__all__ = ["PromptStorage"]


class PromptStorage:
    """Thread‑safe façade over :class:`~core.repository.PromptRepository`.

    This wrapper exists so that higher‑level components (e.g.
    :class:`core.prompt_manager.PromptManager`) can depend on an abstraction
    that may later gain smarter coordination logic (caching, batching, async
    I/O) without changing their public signature.
    """

    _lock: threading.RLock

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        repository: PromptRepository | None = None,
    ) -> None:
        self._lock = threading.RLock()

        if repository is not None:
            self._repo = repository
        else:
            resolved = Path(db_path or "data/prompt_manager.db").expanduser().resolve()
            self._repo = PromptRepository(str(resolved))

    # ------------------------------------------------------------------
    # Transparent proxy methods – delegate to underlying repository.
    # These will be replaced with richer logic in subsequent refactors.
    # ------------------------------------------------------------------

    def __getattr__(self, item: str) -> Any:  # noqa: D401,E501  (simple delegation)
        return getattr(self._repo, item)

    # Example explicit wrapper – others can be added incrementally
    def list_prompts(self, limit: int | None = None) -> list[Prompt]:  # noqa: D401,E501
        """Return list of prompts (proxy)."""

        with self._lock:
            return self._repo.list(limit=limit)

    # Implementing *iter* provides transparent iteration support
    def __iter__(self) -> Iterator[Prompt]:  # noqa: D401
        return iter(self.list_prompts())

    # Length for convenience
    def __len__(self) -> int:  # noqa: D401
        return len(self.list_prompts())
