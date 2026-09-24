"""Small provider-free local execution summary without opening a writing repository."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from .read_only_catalog import CatalogReadError


def read_execution_counts(path: Path) -> dict[str, int]:
    """Count local execution statuses in an immutable snapshot; never read request text."""
    try:
        if not path.is_file():
            raise CatalogReadError("Catalog file does not exist")
        sidecars = (Path(f"{path}-wal"), Path(f"{path}-journal"))
        if any(item.exists() and item.stat().st_size for item in sidecars):
            raise CatalogReadError("Catalog has pending journal data")
        uri = path.resolve().as_uri() + "?mode=ro&immutable=1"
        with closing(sqlite3.connect(uri, uri=True, timeout=1)) as connection:
            row = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(CASE WHEN status='success' THEN 1 ELSE 0 END),0) "
                "FROM prompt_executions"
            ).fetchone()
        if any(item.exists() and item.stat().st_size for item in sidecars):
            raise CatalogReadError("Catalog changed during inspection")
        if row is None:
            raise CatalogReadError("Execution summary unavailable")
        return {"total_runs": int(row[0]), "success_runs": int(row[1])}
    except (sqlite3.Error, OSError, ValueError, TypeError) as exc:
        raise CatalogReadError("Existing execution history cannot be read") from exc
