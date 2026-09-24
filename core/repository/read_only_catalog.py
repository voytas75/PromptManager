"""Read the existing catalog for diagnostics without bootstrapping repository storage."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

from .chains import ChainStoreMixin
from .prompts import PromptStoreMixin

if TYPE_CHECKING:
    from models.prompt_chain_model import PromptChain, PromptChainStep
    from models.prompt_model import Prompt


class CatalogReadError(Exception):
    """An existing catalog cannot be inspected safely or completely."""


class _Hydrator(PromptStoreMixin, ChainStoreMixin):
    """Reuse production row hydration without constructing a writing repository."""


def read_existing_catalog(path: Path) -> tuple[list[Prompt], list[PromptChain]]:
    """Read all prompts and chains through one immutable, sidecar-free SQLite snapshot."""
    try:
        if not path.is_file():
            raise CatalogReadError("Catalog file does not exist")
        wal = Path(f"{path}-wal")
        journal = Path(f"{path}-journal")
        if any(sidecar.exists() and sidecar.stat().st_size for sidecar in (wal, journal)):
            raise CatalogReadError("Catalog has pending journal data")
        reader = _Hydrator()
        uri = path.resolve().as_uri() + "?mode=ro&immutable=1"
        with closing(sqlite3.connect(uri, uri=True, timeout=1)) as connection:
            connection.row_factory = sqlite3.Row
            prompts = [
                reader._row_to_prompt(row)  # pyright: ignore[reportPrivateUsage]
                for row in connection.execute(
                    "SELECT * FROM prompts ORDER BY datetime(last_modified) DESC"
                )
            ]
            chain_rows = connection.execute(
                "SELECT * FROM prompt_chains ORDER BY datetime(created_at) DESC"
            ).fetchall()
            steps: dict[str, list[PromptChainStep]] = {}
            for start in range(0, len(chain_rows), 500):
                batch = [row["id"] for row in chain_rows[start : start + 500]]
                steps.update(
                    reader._load_steps_for_chains(connection, batch)  # pyright: ignore[reportPrivateUsage]
                )
            chains = [
                reader._row_to_chain(row, steps=steps.get(row["id"], []))  # pyright: ignore[reportPrivateUsage]
                for row in chain_rows
            ]
        if any(sidecar.exists() and sidecar.stat().st_size for sidecar in (wal, journal)):
            raise CatalogReadError("Catalog changed during inspection")
    except (sqlite3.Error, ValueError, TypeError, KeyError, IndexError, OSError) as exc:
        raise CatalogReadError("Existing catalog cannot be fully read") from exc
    return prompts, chains
