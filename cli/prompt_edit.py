"""Provider-free, guarded editing of allowlisted prompt metadata."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import uuid
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, cast

from config import load_settings

if TYPE_CHECKING:
    from argparse import Namespace


class EditError(Exception):
    """An edit was rejected without exposing catalog content."""

    def __init__(self, code: str, message: str) -> None:
        """Store a stable error code and bounded, data-free message."""
        super().__init__(message)
        self.code = code


def _parse_list(raw: str, *, allow_invalid_entries: bool) -> list[str]:
    try:
        value: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EditError("INVALID_VALUE", "Value must be valid JSON.") from exc
    if not isinstance(value, list):
        raise EditError("INVALID_VALUE", "related_prompts must be a JSON array of strings.")
    items = cast("list[object]", value)
    if not all(isinstance(item, str) for item in items):
        raise EditError("INVALID_VALUE", "related_prompts must be a JSON array of strings.")
    result = cast("list[str]", items)
    if not allow_invalid_entries:
        canonical: set[str] = set()
        for item in result:
            try:
                parsed = uuid.UUID(item)
            except (ValueError, TypeError) as exc:
                raise EditError("INVALID_REFERENCE", "New references must be UUIDs.") from exc
            if str(parsed) != item or item in canonical:
                raise EditError(
                    "INVALID_REFERENCE", "New references must be unique canonical UUIDs."
                )
            canonical.add(item)
    return result


def _read_catalog_path() -> Path:
    try:
        path = load_settings().db_path.expanduser()
    except Exception as exc:
        raise EditError("CONFIG_UNAVAILABLE", "Catalog configuration is unavailable.") from exc
    if not path.is_file():
        raise EditError("CATALOG_UNAVAILABLE", "Selected catalog does not exist.")
    if path.is_symlink() or path.stat().st_nlink != 1:
        raise EditError("CATALOG_UNAVAILABLE", "Selected catalog must not be a link.")
    return path


def _open_catalog(path: Path, *, write: bool) -> sqlite3.Connection:
    # mode=rw refuses to create a missing catalog; read-only preview uses no writing connection.
    mode = "rw" if write else "ro"
    uri = path.resolve().as_uri() + f"?mode={mode}"
    conn = sqlite3.connect(uri, uri=True, timeout=2)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 2000")
    return conn


def _read_field(conn: sqlite3.Connection, prompt_id: str) -> tuple[str, list[str]]:
    row = conn.execute("SELECT related_prompts FROM prompts WHERE id = ?", (prompt_id,)).fetchone()
    if row is None:
        raise EditError("PROMPT_NOT_FOUND", "Prompt UUID does not exist in the selected catalog.")
    raw: object = row[0]
    if not isinstance(raw, str):
        raise EditError("CATALOG_INVALID", "Stored related_prompts is not JSON text.")
    return raw, _parse_list(raw, allow_invalid_entries=True)


def _validate_targets(conn: sqlite3.Connection, references: list[str]) -> None:
    if not references:
        return
    params = ",".join("?" for _ in references)
    found = conn.execute(f"SELECT id FROM prompts WHERE id IN ({params})", references).fetchall()
    if len(found) != len(references):
        raise EditError(
            "REFERENCE_NOT_FOUND", "New reference does not exist in the selected catalog."
        )


def _backup(path: Path, destination: Path) -> None:
    if destination.resolve() == path.resolve() or destination.resolve() in {
        Path(f"{path.resolve()}{suffix}") for suffix in ("-wal", "-shm", "-journal")
    }:
        raise EditError("INVALID_BACKUP", "Backup must differ from the selected catalog.")
    # Reserve the name without following a symlink or replacing an existing file.
    try:
        fd = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except OSError as exc:
        raise EditError(
            "BACKUP_UNAVAILABLE", "Backup destination cannot be created exclusively."
        ) from exc
    os.close(fd)
    try:
        # Backing up on the same connection as BEGIN IMMEDIATE can block forever;
        # use a second reader, while the writer holds the reserved write lock.
        with closing(_open_catalog(path, write=False)) as reader:
            with closing(sqlite3.connect(destination)) as backup:
                reader.backup(backup, pages=64, sleep=0.05)
                if backup.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise EditError("BACKUP_INVALID", "Backup integrity check failed.")
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def _ensure_uncontended(path: Path) -> None:
    """Reject a snapshot with outstanding journal data (close other writers first)."""
    for suffix in ("-wal", "-journal"):
        sidecar = Path(f"{path}{suffix}")
        if sidecar.exists() and sidecar.stat().st_size:
            raise EditError(
                "CATALOG_BUSY", "Catalog has pending journal data; close other writers."
            )


def _edit(args: Namespace) -> dict[str, object]:
    if args.attr != "related_prompts":
        raise EditError("UNSUPPORTED_ATTRIBUTE", "Only related_prompts is supported in v1.")
    try:
        prompt_id = str(uuid.UUID(args.prompt_id))
    except (ValueError, TypeError) as exc:
        raise EditError("INVALID_PROMPT_ID", "Prompt ID must be a canonical UUID.") from exc
    if args.prompt_id != prompt_id:
        raise EditError("INVALID_PROMPT_ID", "Prompt ID must be a canonical UUID.")
    desired = _parse_list(args.value, allow_invalid_entries=False)
    expected = (
        _parse_list(args.expect_value, allow_invalid_entries=True)
        if args.expect_value is not None
        else None
    )
    backup_to: Path | None = args.backup_to
    if args.apply and (expected is None or backup_to is None):
        raise EditError("MISSING_PRECONDITION", "--apply requires --expect-value and --backup-to.")
    if not args.apply and args.backup_to is not None:
        raise EditError("INVALID_OPTION", "--backup-to is only valid with --apply.")
    db_path = _read_catalog_path()
    try:
        if args.apply:
            _ensure_uncontended(db_path)
        with closing(_open_catalog(db_path, write=bool(args.apply))) as conn:
            if args.apply:
                conn.execute("BEGIN IMMEDIATE")
            raw, before = _read_field(conn, prompt_id)
            _validate_targets(conn, desired)
            if expected is not None and before != expected:
                raise EditError(
                    "STALE_VALUE", "Stored value differs from --expect-value; no change applied."
                )
            changed = before != desired
            applied = False
            if args.apply and changed:
                assert backup_to is not None
                _backup(db_path, backup_to)
                updated = conn.execute(
                    "UPDATE prompts SET related_prompts = ? WHERE id = ? AND related_prompts = ?",
                    (json.dumps(desired, ensure_ascii=False), prompt_id, raw),
                )
                if updated.rowcount != 1:
                    raise EditError("STALE_VALUE", "Stored value changed; no change applied.")
                conn.commit()
                applied = True
            elif args.apply:
                conn.rollback()
            if applied:
                _, verified = _read_field(conn, prompt_id)
                if verified != desired:
                    raise EditError(
                        "READBACK_FAILED", "Committed value did not match the requested value."
                    )
            elif not args.apply:
                conn.rollback()
    except (sqlite3.Error, OSError) as exc:
        raise EditError("CATALOG_ERROR", "Catalog could not be read or changed safely.") from exc
    return {
        "command": "prompt-edit",
        "prompt_id": prompt_id,
        "attr": args.attr,
        "before": before,
        "after": desired,
        "changed": changed,
        "applied": applied,
    }


def run_prompt_edit(args: Namespace) -> int:
    """Print a bounded outcome; never print prompt body or unchecked input."""
    try:
        result = _edit(args)
    except EditError as exc:
        if args.json:
            print(
                json.dumps(
                    {"ok": False, "command": "prompt-edit", "code": exc.code, "message": str(exc)}
                ),
                file=sys.stderr,
            )
        else:
            print(f"Prompt edit: FAIL ({exc.code}) — {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        label = (
            "Applied" if result["applied"] else "No change" if not result["changed"] else "Preview"
        )
        print(f"{label}: {result['attr']} for {result['prompt_id']}")
        print(f"Before: {json.dumps(result['before'], ensure_ascii=False)}")
        print(f"After: {json.dumps(result['after'], ensure_ascii=False)}")
    return 0
