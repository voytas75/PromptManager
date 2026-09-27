"""Provider-free status command against an existing SQLite prompt catalog."""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import uuid
from contextlib import closing
from dataclasses import dataclass
from typing import TYPE_CHECKING

from config import SettingsError, load_settings
from core.repository import RepositoryError, RepositoryNotFoundError
from core.repository.prompt_status import PromptStatusConflictError, set_prompt_active
from core.repository.prompts import PromptStoreMixin

if TYPE_CHECKING:
    from argparse import Namespace
    from pathlib import Path


@dataclass(frozen=True)
class StatusError(Exception):
    """Bounded status error without prompt bodies or local paths."""

    code: str
    message: str

    def __str__(self) -> str:
        """Render only the bounded operator-safe message."""
        return self.message


def _id(raw: str) -> uuid.UUID:
    try:
        prompt_id = uuid.UUID(raw)
    except (TypeError, ValueError) as exc:
        raise StatusError("INVALID_ID", "A full canonical prompt UUID is required.") from exc
    if str(prompt_id) != raw:
        raise StatusError("INVALID_ID", "A full canonical prompt UUID is required.")
    return prompt_id


def _catalog() -> Path:
    try:
        logger = logging.getLogger("prompt_manager.settings")
        previous = logger.level
        logger.setLevel(logging.ERROR)
        try:
            path = load_settings().db_path.expanduser()
        finally:
            logger.setLevel(previous)
        if not path.is_file() or path.is_symlink():
            raise StatusError("CATALOG_UNAVAILABLE", "Selected catalog is unavailable.")
        return path
    except StatusError:
        raise
    except (SettingsError, OSError, ValueError, TypeError) as exc:
        raise StatusError("CONFIG_UNAVAILABLE", "Catalog configuration is unavailable.") from exc


def _execute(args: Namespace) -> dict[str, object]:
    prompt_id = _id(args.prompt_id)
    path = _catalog()
    try:
        with closing(
            sqlite3.connect(path.resolve().as_uri() + "?mode=rw", uri=True, timeout=3)
        ) as conn:
            conn.row_factory = sqlite3.Row
            # Do not mutate arbitrary SQLite files with a coincidental prompts table.
            required = set(PromptStoreMixin._COLUMNS)  # pyright: ignore[reportPrivateUsage]
            columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(prompts)")}
            tables = {
                str(row[0])
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if not required.issubset(columns) or not {
                "prompt_versions",
                "prompt_forks",
                "prompt_chain_steps",
            }.issubset(tables):
                raise StatusError(
                    "CATALOG_INVALID", "Selected catalog has incompatible prompt schema."
                )
            try:
                changed, stamp = set_prompt_active(
                    conn,
                    prompt_id,
                    active=args.status_action == "activate",
                    expect_active=args.expect_active == "true",
                )
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
    except PromptStatusConflictError as exc:
        raise StatusError(
            "STATUS_CONFLICT", "Prompt activity changed; reload before retrying."
        ) from exc
    except RepositoryNotFoundError as exc:
        raise StatusError("PROMPT_NOT_FOUND", "Prompt UUID does not exist.") from exc
    except (RepositoryError, sqlite3.Error) as exc:
        raise StatusError(
            "CATALOG_INVALID", "Selected catalog cannot update prompt status."
        ) from exc
    return {
        "ok": True,
        "id": str(prompt_id),
        "active": args.status_action == "activate",
        "changed": changed,
        "last_modified": stamp,
    }


def run_prompt_status(args: Namespace) -> int:
    """Emit one sanitized result on stdout, or one bounded error on stderr."""
    try:
        outcome = _execute(args)
    except StatusError as exc:
        if args.json:
            print(
                json.dumps({"ok": False, "error": {"code": exc.code, "message": exc.message}}),
                file=sys.stderr,
            )
        else:
            print(f"Prompt status ({exc.code}): {exc.message}", file=sys.stderr)
        return 2 if exc.code == "INVALID_ID" else 4
    if args.json:
        print(json.dumps(outcome))
    else:
        print(
            f"Prompt {outcome['id']} is {'active' if outcome['active'] else 'inactive'}"
            f" ({'changed' if outcome['changed'] else 'unchanged'})."
        )
    return 0
