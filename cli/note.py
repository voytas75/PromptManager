"""Lightweight standalone-note CLI over the GUI's existing prompt_notes table.

Updates: v0.1.0 - 2026-09-25 - Add provider-free note management without manager bootstrap.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import uuid
from contextlib import closing
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from config import SettingsError, load_settings
from models.prompt_note import PromptNote

if TYPE_CHECKING:
    from argparse import Namespace
    from pathlib import Path

_MAX_BODY_BYTES = 1024 * 1024
_COLUMNS = ("id", "note", "created_at", "last_modified")


@dataclass(frozen=True)
class NoteError(Exception):
    """Bounded CLI error which does not disclose note text or storage paths."""

    code: str
    message: str

    def __str__(self) -> str:
        """Render bounded operator-safe message."""
        return self.message


def _catalog_path() -> Path:
    try:
        settings_logger = logging.getLogger("prompt_manager.settings")
        previous_level = settings_logger.level
        settings_logger.setLevel(logging.ERROR)
        try:
            path = load_settings().db_path.expanduser()
        finally:
            settings_logger.setLevel(previous_level)
        if not path.is_file() or path.is_symlink():
            raise NoteError(
                "CATALOG_UNAVAILABLE", "Selected catalog does not exist or is not a regular file."
            )
        return path
    except NoteError:
        raise
    except (SettingsError, OSError, ValueError, TypeError) as exc:
        raise NoteError("CONFIG_UNAVAILABLE", "Catalog configuration is unavailable.") from exc


def _connect(path: Path, *, write: bool) -> sqlite3.Connection:
    mode = "rw" if write else "ro"
    try:
        conn = sqlite3.connect(path.resolve().as_uri() + f"?mode={mode}", uri=True, timeout=3)
        conn.row_factory = sqlite3.Row
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(prompt_notes)")}
        if not set(_COLUMNS).issubset(columns):
            conn.close()
            raise NoteError(
                "CATALOG_INVALID", "Selected catalog has no compatible prompt_notes table."
            )
        return conn
    except sqlite3.Error as exc:
        raise NoteError("CATALOG_UNAVAILABLE", "Unable to open the selected catalog.") from exc


def _id(raw: str) -> str:
    try:
        value = str(uuid.UUID(raw))
    except (ValueError, TypeError) as exc:
        raise NoteError("INVALID_ID", "A full canonical note UUID is required.") from exc
    if value != raw:
        raise NoteError("INVALID_ID", "A full canonical note UUID is required.")
    return value


def _limit(raw: int) -> int:
    if not 1 <= raw <= 100:
        raise NoteError("INVALID_LIMIT", "Limit must be between 1 and 100.")
    return raw


def _body(args: Namespace) -> str:
    sources = [
        getattr(args, "text", None) is not None,
        args.body is not None,
        args.file is not None,
        args.from_stdin,
    ]
    if sum(sources) != 1:
        raise NoteError("INVALID_INPUT", "Provide exactly one text source.")
    if args.file is not None:
        try:
            if not args.file.is_file() or args.file.stat().st_size > _MAX_BODY_BYTES:
                raise NoteError("INVALID_INPUT", "Note file is missing or too large.")
            data = args.file.read_bytes()
        except OSError as exc:
            raise NoteError("INVALID_INPUT", "Unable to read note file.") from exc
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise NoteError("INVALID_INPUT", "Note file must be UTF-8 text.") from exc
    elif args.from_stdin:
        data = sys.stdin.buffer.read(_MAX_BODY_BYTES + 1)
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise NoteError("INVALID_INPUT", "Stdin must be UTF-8 text.") from exc
    else:
        content = args.body if args.body is not None else args.text
        try:
            data = content.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise NoteError("INVALID_INPUT", "Note text must be UTF-8.") from exc
    if len(data) > _MAX_BODY_BYTES or not content.strip():
        raise NoteError("INVALID_INPUT", "Note text must be nonempty and at most 1 MiB.")
    return content.strip()


def _hydrate(row: sqlite3.Row) -> PromptNote:
    """Reject malformed existing rows without leaking their contents."""
    try:
        return PromptNote.from_record(dict(row))
    except (ValueError, TypeError) as exc:
        raise NoteError("CATALOG_INVALID", "Selected catalog has an invalid note record.") from exc


def _one(conn: sqlite3.Connection, note_id: str) -> PromptNote:
    row = conn.execute(
        "SELECT id, note, created_at, last_modified FROM prompt_notes WHERE id=?", (note_id,)
    ).fetchone()
    if row is None:
        raise NoteError("NOTE_NOT_FOUND", "Note UUID was not found.")
    return _hydrate(row)


def _rows(conn: sqlite3.Connection, limit: int, *, query: str | None = None) -> list[PromptNote]:
    if query is None:
        rows = conn.execute(
            "SELECT id, note, created_at, last_modified FROM prompt_notes "
            "ORDER BY last_modified DESC, id ASC LIMIT ?",
            (limit,),
        ).fetchall()
    else:
        literal = query.replace("!", "!!").replace("%", "!%").replace("_", "!_")
        rows = conn.execute(
            "SELECT id, note, created_at, last_modified FROM prompt_notes "
            "WHERE note LIKE ? ESCAPE '!' ORDER BY last_modified DESC, id ASC LIMIT ?",
            (f"%{literal}%", limit),
        ).fetchall()
    return [_hydrate(row) for row in rows]


def _preview(note: PromptNote) -> dict[str, str]:
    first = note.note.splitlines()[0] if note.note else ""
    return {
        "id": str(note.id),
        "preview": first[:100],
        "last_modified": note.last_modified.isoformat(),
    }


def _emit(payload: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps({"ok": True, **payload}, ensure_ascii=False))
        return
    if "notes" in payload:
        notes = cast("list[dict[str, str]]", payload["notes"])
        print(
            "\n".join(f"{item['id']} | {item['preview']}" for item in notes)
            if notes
            else "No notes."
        )
    elif "note" in payload:
        note = payload["note"]
        assert isinstance(note, dict)
        print(
            f"ID: {note['id']}\n"
            f"Created: {note['created_at']}\n"
            f"Modified: {note['last_modified']}\n\n"
            f"{note['note']}"
        )
    else:
        print(f"Deleted note: {payload['id']}")


def _execute(args: Namespace) -> dict[str, object]:
    action = args.note_action
    note_id: str = ""
    limit = 20
    query: str | None = None
    text = ""
    if action in (None, "find"):
        limit = _limit(args.limit)
        query = None
        if action == "find":
            query = args.query.strip()
            if not query:
                raise NoteError("INVALID_INPUT", "Search text must not be blank.")
    if action in ("show", "edit", "delete"):
        note_id = _id(args.id)
    if action in ("add", "edit"):
        text = _body(args)
    if action == "delete" and not args.yes and (args.json or not sys.stdin.isatty()):
        raise NoteError(
            "CONFIRM_REQUIRED", "Deletion requires --yes in non-interactive or JSON mode."
        )
    with closing(_connect(_catalog_path(), write=action in {"add", "edit", "delete"})) as conn:
        try:
            if action in (None, "find"):
                notes = _rows(conn, limit, query=query)
                return {"notes": [_preview(note) for note in notes]}
            if action == "show":
                return {"note": _one(conn, note_id).to_record()}
            if action == "add":
                note = PromptNote(id=uuid.uuid4(), note=text)
                note.touch()
                record = note.to_record()
                with conn:
                    conn.execute(
                        "INSERT INTO prompt_notes (id, note, created_at, last_modified) "
                        "VALUES (:id, :note, :created_at, :last_modified)",
                        record,
                    )
                return {"note": note.to_record()}
            if action == "edit":
                with conn:
                    previous = _one(conn, note_id)
                    updated = PromptNote(id=previous.id, note=text, created_at=previous.created_at)
                    updated.touch()
                    record = updated.to_record()
                    conn.execute(
                        "UPDATE prompt_notes SET note=:note, last_modified=:last_modified "
                        "WHERE id=:id",
                        record,
                    )
                return {"note": updated.to_record()}
            if action == "delete":
                if not args.yes:
                    _one(conn, note_id)
                    try:
                        reply = input(f"Permanently delete note {note_id}? [y/N] ")
                    except EOFError:
                        reply = ""
                    if reply.strip().lower() not in {"y", "yes"}:
                        raise NoteError("CANCELLED", "Deletion cancelled.")
                with conn:
                    deleted = conn.execute(
                        "DELETE FROM prompt_notes WHERE id=?", (note_id,)
                    ).rowcount
                    if not deleted:
                        raise NoteError("NOTE_NOT_FOUND", "Note UUID was not found.")
                return {"id": note_id}
            raise NoteError("INVALID_USAGE", "Unknown note action.")
        except sqlite3.Error as exc:
            raise NoteError(
                "CATALOG_UNAVAILABLE", "Unable to access the selected catalog."
            ) from exc


def run_note(args: Namespace) -> int:
    """Dispatch notes without importing the full provider/index-backed manager."""
    try:
        result = _execute(args)
        _emit(result, as_json=args.json)
        return 0
    except NoteError as exc:
        if args.json:
            print(
                json.dumps({"ok": False, "error": {"code": exc.code, "message": exc.message}}),
                file=sys.stderr,
            )
        else:
            print(f"Note error ({exc.code}): {exc.message}", file=sys.stderr)
        return (
            2
            if exc.code.startswith("INVALID") or exc.code in {"CONFIRM_REQUIRED", "CANCELLED"}
            else 4
        )
