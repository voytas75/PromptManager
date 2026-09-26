"""Provider-free local CLI for the GUI's response_styles catalog."""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast

from config import SettingsError, load_settings
from models.response_style import ResponseStyle

if TYPE_CHECKING:
    from argparse import Namespace
    from pathlib import Path

_MAX_BODY_BYTES = 1024 * 1024

_COLUMNS = (
    "id",
    "name",
    "description",
    "prompt_part",
    "snippet",
    "tone",
    "voice",
    "format_instructions",
    "guidelines",
    "tags",
    "examples",
    "metadata",
    "is_active",
    "version",
    "created_at",
    "last_modified",
)
_STORAGE_COLUMNS = (*_COLUMNS, "ext1", "ext2", "ext3")


@dataclass(frozen=True)
class PartError(Exception):
    """Bounded error that never exposes catalog paths or fragment text."""

    code: str
    message: str

    def __str__(self) -> str:
        """Render only the bounded operator-safe message."""
        return self.message


def _catalog_path() -> Path:
    try:
        logger = logging.getLogger("prompt_manager.settings")
        previous = logger.level
        logger.setLevel(logging.ERROR)
        try:
            path = load_settings().db_path.expanduser()
        finally:
            logger.setLevel(previous)
        if path.is_symlink() or not path.is_file():
            raise PartError(
                "CATALOG_UNAVAILABLE", "Selected catalog does not exist or is not a regular file."
            )
        return path
    except PartError:
        raise
    except (SettingsError, OSError, ValueError, TypeError) as exc:
        raise PartError("CONFIG_UNAVAILABLE", "Catalog configuration is unavailable.") from exc


def _connect(path: Path, *, write: bool = False) -> sqlite3.Connection:
    try:
        mode = "rw" if write else "ro"
        conn = sqlite3.connect(path.resolve().as_uri() + f"?mode={mode}", uri=True, timeout=3)
        conn.row_factory = sqlite3.Row
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(response_styles)")}
        if "snippet" not in columns and "response_styles" in {
            str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }:
            conn.close()
            raise PartError(
                "CATALOG_MIGRATION_REQUIRED",
                "Prompt parts schema needs migration. Back up the catalog and open the GUI.",
            )
        if not set(_STORAGE_COLUMNS).issubset(columns):
            conn.close()
            raise PartError(
                "CATALOG_INVALID", "Selected catalog has no compatible prompt parts table."
            )
        return conn
    except sqlite3.Error as exc:
        raise PartError("CATALOG_UNAVAILABLE", "Unable to read the selected catalog.") from exc


def _id(raw: str) -> str:
    try:
        value = str(uuid.UUID(raw))
    except (ValueError, TypeError) as exc:
        raise PartError("INVALID_ID", "A full canonical prompt part UUID is required.") from exc
    if value != raw:
        raise PartError("INVALID_ID", "A full canonical prompt part UUID is required.")
    return value


def _limit(raw: int) -> int:
    if not 1 <= raw <= 100:
        raise PartError("INVALID_LIMIT", "Limit must be between 1 and 100.")
    return raw


def _hydrate(row: sqlite3.Row) -> ResponseStyle:
    """Check mandatory fields before applying model defaults to optional ones."""
    try:
        record = dict(row)
        for key in ("id", "name", "snippet", "created_at", "last_modified"):
            if not isinstance(record[key], str) or not record[key].strip():
                raise ValueError("Missing mandatory field")
        for key in ("created_at", "last_modified"):
            if datetime.fromisoformat(record[key]).tzinfo is None:
                raise ValueError("Stored timestamps need timezone offsets")
        if str(uuid.UUID(record["id"])) != record["id"]:
            raise ValueError("Invalid stored identifier")
        if record["is_active"] not in (0, 1):
            raise ValueError("Invalid active flag")
        for key in ("tags", "examples"):
            parsed = json.loads(record[key])
            if not isinstance(parsed, list) or not all(
                isinstance(item, str) for item in cast("list[object]", parsed)
            ):
                raise ValueError("Invalid list")
            record[key] = parsed
        if record["metadata"] is not None:
            parsed = json.loads(record["metadata"])
            if parsed is not None and not isinstance(parsed, dict):
                raise ValueError("Invalid metadata")
            record["metadata"] = parsed
        style = ResponseStyle.from_record(record)
        if style.created_at.utcoffset() is None or style.last_modified.utcoffset() is None:
            raise ValueError("Invalid timestamp")
        return style
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise PartError(
            "CATALOG_INVALID", "Selected catalog has an invalid prompt part record."
        ) from exc


def _preview(style: ResponseStyle) -> dict[str, object]:
    return {
        "id": str(style.id),
        "name": style.name,
        "prompt_part": style.prompt_part,
        "is_active": style.is_active,
        "preview": style.snippet.splitlines()[0][:100],
    }


def _show(style: ResponseStyle) -> dict[str, object]:
    record = style.to_record()
    result: dict[str, object] = {key: record[key] for key in _COLUMNS}
    result["is_active"] = style.is_active
    return result


def _label(raw: str, *, field: str) -> str:
    text = raw.strip()
    if not text or len(text.encode("utf-8")) > _MAX_BODY_BYTES:
        raise PartError("INVALID_INPUT", f"{field} must be nonblank and at most 1 MiB.")
    return text


def _body(args: Namespace) -> str:
    if sum((args.body is not None, args.file is not None, args.from_stdin)) != 1:
        raise PartError("INVALID_INPUT", "Provide exactly one snippet source.")
    try:
        if args.file is not None:
            if not args.file.is_file() or args.file.stat().st_size > _MAX_BODY_BYTES:
                raise PartError("INVALID_INPUT", "Snippet file is missing or too large.")
            data = args.file.read_bytes()
        elif args.from_stdin:
            data = sys.stdin.buffer.read(_MAX_BODY_BYTES + 1)
        else:
            data = args.body.encode("utf-8")
        text = data.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise PartError("INVALID_INPUT", "Snippet must be readable UTF-8 text.") from exc
    if not text.strip() or len(data) > _MAX_BODY_BYTES:
        raise PartError("INVALID_INPUT", "Snippet must be nonblank and at most 1 MiB.")
    return text


def _expected(raw: str) -> str:
    try:
        value = datetime.fromisoformat(raw)
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Timezone required")
        if value.isoformat() != raw:
            raise ValueError("Use the exact last_modified token from show")
    except ValueError as exc:
        raise PartError(
            "INVALID_INPUT", "Expected modified value must be an ISO8601 timestamp."
        ) from exc
    return raw


def _one(conn: sqlite3.Connection, identifier: str) -> ResponseStyle:
    row = conn.execute(
        f"SELECT {', '.join(_COLUMNS)} FROM response_styles WHERE id = ?", (identifier,)
    ).fetchone()
    if row is None:
        raise PartError("PART_NOT_FOUND", "Prompt part UUID was not found.")
    return _hydrate(row)


def _edit_values(args: Namespace) -> dict[str, object]:
    values: dict[str, object] = {}
    if args.name is not None:
        values["name"] = _label(args.name, field="Name")
    if args.part is not None:
        values["prompt_part"] = _label(args.part, field="Part label")
    if args.description is not None:
        if len(args.description.encode("utf-8")) > _MAX_BODY_BYTES:
            raise PartError("INVALID_INPUT", "Description must be at most 1 MiB.")
        values["description"] = args.description
    if args.active or args.inactive:
        values["is_active"] = int(args.active)
    if args.body is not None or args.file is not None or args.from_stdin:
        values["snippet"] = _body(args)
    if not values:
        raise PartError("INVALID_INPUT", "Provide at least one change.")
    return values


def _mutate(
    conn: sqlite3.Connection,
    *,
    action: str,
    identifier: str,
    expected: str,
    values: dict[str, object] | None = None,
) -> dict[str, object]:
    # Lock before reading: no GUI writer can slip between precondition and write.
    conn.execute("BEGIN IMMEDIATE")
    try:
        previous = _one(conn, identifier)
        stored_stamp = conn.execute(
            "SELECT last_modified FROM response_styles WHERE id=?", (identifier,)
        ).fetchone()["last_modified"]
        if stored_stamp != expected:
            raise PartError("PART_CONFLICT", "Prompt part changed; reload it before retrying.")
        result: dict[str, object]
        if action == "delete":
            deleted = conn.execute(
                "DELETE FROM response_styles WHERE id=? AND last_modified=?", (identifier, expected)
            ).rowcount
            if deleted != 1:
                raise PartError("PART_CONFLICT", "Prompt part changed; reload it before retrying.")
            result = {"id": identifier}
        else:
            assert values is not None
            current = previous.to_record()
            if all(current[key] == value for key, value in values.items()):
                result = {"part": _show(previous)}
            else:
                stamp = datetime.now(UTC)
                if stamp <= previous.last_modified:
                    stamp = previous.last_modified + timedelta(microseconds=1)
                values["last_modified"] = stamp.isoformat()
                assignments = ", ".join(f"{key}=:{key}" for key in values)
                updated = conn.execute(
                    f"UPDATE response_styles SET {assignments} "
                    "WHERE id=:id AND last_modified=:expected",
                    {**values, "id": identifier, "expected": expected},
                ).rowcount
                if updated != 1:
                    raise PartError(
                        "PART_CONFLICT", "Prompt part changed; reload it before retrying."
                    )
                result = {"part": _show(_one(conn, identifier))}
        conn.commit()
        return result
    except BaseException:
        conn.rollback()
        raise


def _execute(args: Namespace) -> dict[str, object]:
    action = args.part_action
    if action == "add":
        name = _label(args.name, field="Name")
        label = _label(args.part, field="Part label")
        body = _body(args)
        description = args.description
        if len(description.encode("utf-8")) > _MAX_BODY_BYTES:
            raise PartError("INVALID_INPUT", "Description must be at most 1 MiB.")
        style = ResponseStyle(
            id=uuid.uuid4(),
            name=name,
            description=description,
            prompt_part=label,
            snippet=body,
        )
        record = style.to_record()
        record["tags"] = "[]"
        record["examples"] = "[]"
        columns = tuple(record)
        placeholders = ", ".join(f":{column}" for column in columns)
        with closing(_connect(_catalog_path(), write=True)) as conn, conn:
            conn.execute(
                f"INSERT INTO response_styles ({', '.join(columns)}) VALUES ({placeholders})",
                record,
            )
        return {"part": _show(style)}
    if action in ("edit", "delete"):
        identifier = _id(args.id)
        expected = _expected(args.expect_modified)
        if action == "delete":
            if not args.yes and (args.json or not sys.stdin.isatty()):
                raise PartError(
                    "CONFIRM_REQUIRED", "Deletion requires --yes in JSON or non-TTY mode."
                )
            if not args.yes:
                try:
                    reply = input(f"Permanently delete prompt part {identifier}? [y/N] ")
                except EOFError:
                    reply = ""
                if reply.strip().lower() not in {"y", "yes"}:
                    raise PartError("CANCELLED", "Deletion cancelled.")
            values = None
        else:
            values = _edit_values(args)
        with closing(_connect(_catalog_path(), write=True)) as conn:
            return _mutate(
                conn, action=action, identifier=identifier, expected=expected, values=values
            )
    identifier = _id(args.id) if action == "show" else None
    limit = _limit(args.limit) if action != "show" else None
    query: str | None = None
    if action == "find":
        query = args.query.strip()
        if not query:
            raise PartError("INVALID_INPUT", "Search text must not be blank.")
    with closing(_connect(_catalog_path())) as conn:
        if identifier is not None:
            return {"part": _show(_one(conn, identifier))}
        clauses = [] if args.all else ["is_active = 1"]
        params: list[object] = []
        if query is not None:
            literal = query.replace("!", "!!").replace("%", "!%").replace("_", "!_")
            pattern = f"%{literal}%"
            clauses.append(
                "("
                + " OR ".join(
                    f"{key} LIKE ? ESCAPE '!'"
                    for key in ("name", "prompt_part", "description", "snippet")
                )
                + ")"
            )
            params.extend([pattern] * 4)
        statement = f"SELECT {', '.join(_COLUMNS)} FROM response_styles"
        if clauses:
            statement += " WHERE " + " AND ".join(clauses)
        statement += " ORDER BY name COLLATE NOCASE, id LIMIT ?"
        params.append(limit)
        rows = conn.execute(statement, params).fetchall()
        return {"parts": [_preview(_hydrate(row)) for row in rows]}


def _emit(payload: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps({"ok": True, **payload}, ensure_ascii=False))
    elif "parts" in payload:
        rows = cast("list[dict[str, object]]", payload["parts"])
        print(
            "\n".join(
                f"{row['id']} | {row['name']} | {row['prompt_part']} | {row['preview']}"
                for row in rows
            )
            if rows
            else "No prompt parts."
        )
    elif "part" in payload:
        row = cast("dict[str, object]", payload["part"])
        print(
            "\n".join(
                (
                    f"ID: {row['id']}",
                    f"Name: {row['name']}",
                    f"Part: {row['prompt_part']}",
                    f"Active: {row['is_active']}",
                    f"Version: {row['version']}",
                    f"Created: {row['created_at']}",
                    f"Modified: {row['last_modified']}",
                    f"Description: {row['description']}",
                    f"Tone: {row['tone'] or ''}",
                    f"Voice: {row['voice'] or ''}",
                    f"Format instructions: {row['format_instructions'] or ''}",
                    f"Guidelines: {row['guidelines'] or ''}",
                    f"Tags: {', '.join(cast('list[str]', row['tags']))}",
                    f"Examples: {', '.join(cast('list[str]', row['examples']))}",
                    f"Snippet:\n{row['snippet']}",
                )
            )
        )
    else:
        print(f"Deleted prompt part: {payload['id']}")


def run_prompt_part(args: Namespace) -> int:
    """Dispatch local catalog actions without application service startup."""
    try:
        payload = _execute(args)
        _emit(payload, as_json=bool(args.json))
        return 0
    except (PartError, sqlite3.Error) as exc:
        error = (
            exc
            if isinstance(exc, PartError)
            else PartError("CATALOG_UNAVAILABLE", "Unable to read the selected catalog.")
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": {
                            "code": error.code,
                            "message": error.message,
                        },
                    }
                ),
                file=sys.stderr,
            )
        else:
            print(f"Prompt part error ({error.code}): {error.message}", file=sys.stderr)
        return 2
