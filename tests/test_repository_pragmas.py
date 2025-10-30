"""Integration tests around SQLite connection pragmas."""

from __future__ import annotations

from pathlib import Path

from core.repository import _connect


def test_connect_configures_sqlite_pragmas(tmp_path: Path) -> None:
    db_path = tmp_path / "pragmas.db"
    with _connect(db_path) as conn:
        foreign_keys = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        synchronous = conn.execute("PRAGMA synchronous;").fetchone()[0]

    assert foreign_keys == 1
    assert journal_mode.lower() == "wal"
    assert synchronous in (1, 2)  # NORMAL maps to 1 on SQLite 3.44+
