"""Local, provider-free CLI for draft prompts stored in the existing catalog.

Updates: v0.1.0 - 2026-09-26 - Capture/read/delete drafts without manager bootstrap.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import sys
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

from config import SettingsError, load_settings
from models.prompt_model import Prompt

if TYPE_CHECKING:
    from argparse import Namespace

_MAX_BODY_BYTES = 1024 * 1024
_COLUMNS = (
    "id",
    "name",
    "description",
    "category",
    "category_slug",
    "tags",
    "language",
    "context",
    "example_input",
    "example_output",
    "scenarios",
    "last_modified",
    "version",
    "author",
    "quality_score",
    "usage_count",
    "rating_count",
    "rating_sum",
    "related_prompts",
    "created_at",
    "modified_by",
    "is_active",
    "source",
    "checksum",
    "ext1",
    "ext2",
    "ext3",
    "ext4",
    "ext5",
)


@dataclass(frozen=True)
class DraftError(Exception):
    """Stable error without prompt bodies, catalog paths or provider diagnostics."""

    code: str
    message: str

    def __str__(self) -> str:
        """Render a bounded operator-safe message."""
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
        if not path.is_file() or path.is_symlink():
            raise DraftError("CATALOG_UNAVAILABLE", "Selected catalog is unavailable.")
        return path
    except DraftError:
        raise
    except (SettingsError, OSError, ValueError, TypeError) as exc:
        raise DraftError("CONFIG_UNAVAILABLE", "Catalog configuration is unavailable.") from exc


def _connect(path: Path, *, write: bool) -> sqlite3.Connection:
    try:
        conn = sqlite3.connect(
            path.resolve().as_uri() + ("?mode=rw" if write else "?mode=ro"), uri=True, timeout=3
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(prompts)")}
        if not set(_COLUMNS).issubset(columns):
            conn.close()
            raise DraftError("CATALOG_INVALID", "Selected catalog has no compatible prompts table.")
        if write:
            for table in ("prompt_versions", "prompt_activity_events"):
                if (
                    conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
                    ).fetchone()
                    is None
                ):
                    conn.close()
                    raise DraftError(
                        "CATALOG_INVALID", "Selected catalog lacks prompt lifecycle tables."
                    )
        return conn
    except sqlite3.Error as exc:
        raise DraftError("CATALOG_UNAVAILABLE", "Unable to open selected catalog.") from exc


def _id(raw: str) -> str:
    try:
        parsed = str(uuid.UUID(raw))
    except (ValueError, TypeError) as exc:
        raise DraftError("INVALID_ID", "A full canonical draft UUID is required.") from exc
    if parsed != raw:
        raise DraftError("INVALID_ID", "A full canonical draft UUID is required.")
    return parsed


def _limit(raw: int) -> int:
    if not 1 <= raw <= 100:
        raise DraftError("INVALID_LIMIT", "Limit must be between 1 and 100.")
    return raw


def _body(args: Namespace) -> str:
    sources = [args.text is not None, args.body is not None, args.file is not None, args.from_stdin]
    if sum(sources) != 1:
        raise DraftError("INVALID_INPUT", "Provide exactly one prompt body source.")
    if args.file is not None:
        try:
            if not args.file.is_file() or args.file.stat().st_size > _MAX_BODY_BYTES:
                raise DraftError("INVALID_INPUT", "Prompt file is missing or too large.")
            data = args.file.read_bytes()
        except OSError as exc:
            raise DraftError("INVALID_INPUT", "Unable to read prompt file.") from exc
    elif args.from_stdin:
        data = sys.stdin.buffer.read(_MAX_BODY_BYTES + 1)
    else:
        try:
            data = (args.body if args.body is not None else args.text).encode("utf-8")
        except UnicodeEncodeError as exc:
            raise DraftError("INVALID_INPUT", "Prompt body must be UTF-8.") from exc
    if len(data) > _MAX_BODY_BYTES:
        raise DraftError("INVALID_INPUT", "Prompt body exceeds 1 MiB.")
    try:
        text = data.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise DraftError("INVALID_INPUT", "Prompt body must be UTF-8 text.") from exc
    if not text:
        raise DraftError("INVALID_INPUT", "Prompt body must not be blank.")
    return text


def _title(body: str) -> str:
    """Derive from the first meaningful line, following GUI Quick Capture title rules."""
    for line in body.splitlines():
        candidate = " ".join(line.strip().split())
        if not candidate:
            continue
        previous = None
        while candidate != previous:
            previous = candidate
            candidate = candidate.strip("`'\" ")
            candidate = re.sub(r"^(?:title|prompt|subject)\s*:\s*", "", candidate, flags=re.I)
            candidate = re.sub(r"^(?:[#>*-]+|\d+[.)])\s+", "", candidate).strip("`'\" ")
        if not candidate or candidate.casefold() in {
            "captured draft",
            "draft",
            "new prompt",
            "prompt",
            "prompt draft",
            "quick capture draft",
            "subject",
            "tbd",
            "title",
            "todo",
            "untitled",
            "untitled prompt",
        }:
            continue
        if len(candidate) > 80:
            trimmed = candidate[:79].rstrip()
            if " " in trimmed:
                trimmed = trimmed.rsplit(" ", 1)[0]
            candidate = trimmed.rstrip() + "…"
        return candidate
    return "Quick Capture Draft"


def _state(row: sqlite3.Row) -> bool:
    raw: object = row["ext2"]
    try:
        metadata: object = json.loads(raw) if isinstance(raw, str) else None
    except (ValueError, TypeError) as exc:
        raise DraftError("CATALOG_INVALID", "Stored draft metadata is invalid.") from exc
    if raw is not None and not isinstance(raw, str):
        raise DraftError("CATALOG_INVALID", "Stored draft metadata is invalid.")
    if metadata is not None and not isinstance(metadata, dict):
        raise DraftError("CATALOG_INVALID", "Stored draft metadata is invalid.")
    if not isinstance(metadata, dict):
        return False
    typed_metadata = cast("dict[str, object]", metadata)
    return typed_metadata.get("capture_state") == "draft"


def _one(conn: sqlite3.Connection, draft_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM prompts WHERE id=?", (draft_id,)).fetchone()
    if row is None:
        raise DraftError("DRAFT_NOT_FOUND", "Prompt UUID was not found.")
    if not _state(row):
        raise DraftError("NOT_DRAFT", "Prompt UUID is not a draft.")
    return row


def _preview(row: sqlite3.Row) -> dict[str, str]:
    body = str(row["context"] or "")
    return {
        "id": str(row["id"]),
        "title": str(row["name"]),
        "preview": (body.splitlines()[0] if body else "")[:100],
        "last_modified": str(row["last_modified"]),
    }


def _detail(row: sqlite3.Row) -> dict[str, str]:
    return {
        "id": str(row["id"]),
        "title": str(row["name"]),
        "description": str(row["description"]),
        "body": str(row["context"] or ""),
        "source": str(row["source"] or ""),
        "created_at": str(row["created_at"]),
        "last_modified": str(row["last_modified"]),
    }


def _rows(conn: sqlite3.Connection, limit: int, query: str | None) -> list[sqlite3.Row]:
    sql = "SELECT * FROM prompts ORDER BY last_modified DESC, id ASC"
    params: tuple[object, ...] = ()
    if query is not None:
        literal = query.replace("!", "!!").replace("%", "!%").replace("_", "!_")
        sql = (
            "SELECT * FROM prompts WHERE (name LIKE ? ESCAPE '!' OR "
            "description LIKE ? ESCAPE '!' OR context LIKE ? ESCAPE '!') "
            "ORDER BY last_modified DESC, id ASC"
        )
        params = (f"%{literal}%",) * 3
    # Fetch in chunks: limit applies to drafts, not to all prompts.
    offset = 0
    result: list[sqlite3.Row] = []
    while len(result) < limit:
        page = conn.execute(sql + " LIMIT 128 OFFSET ?", (*params, offset)).fetchall()
        if not page:
            break
        for row in page:
            if _state(row):
                result.append(row)
                if len(result) == limit:
                    break
        offset += len(page)
    return result


def _save(
    conn: sqlite3.Connection, body: str, title: str | None, source: str | None
) -> sqlite3.Row:
    name = (title or "").strip() or _title(body)
    if not name or len(name.encode("utf-8")) > 1024:
        raise DraftError("INVALID_INPUT", "Title must be at most 1 KiB.")
    provenance = (source or "").strip() or "cli_capture"
    if len(provenance.encode("utf-8")) > 1024:
        raise DraftError("INVALID_INPUT", "Source must be at most 1 KiB.")
    now = datetime.now(UTC)
    prompt = Prompt(
        id=uuid.uuid4(),
        name=name,
        description="Quick capture draft.",
        category="General",
        context=body,
        created_at=now,
        last_modified=now,
        source=provenance,
        ext2={"capture_state": "draft", "capture_method": "cli"},
    )
    record = prompt.to_record()
    record["category_slug"] = prompt.category_slug
    record["tags"] = json.dumps(prompt.tags, ensure_ascii=False)
    record["scenarios"] = json.dumps(prompt.scenarios, ensure_ascii=False)
    record["related_prompts"] = json.dumps(prompt.related_prompts, ensure_ascii=False)
    for key in ("ext2", "ext4", "ext5"):
        record[key] = (
            json.dumps(record[key], ensure_ascii=False) if record[key] is not None else None
        )
    record["is_active"] = int(prompt.is_active)
    columns = ", ".join(_COLUMNS)
    values = ", ".join(f":{column}" for column in _COLUMNS)
    snapshot = json.dumps(prompt.to_record(), ensure_ascii=False, sort_keys=True)
    with conn:
        conn.execute(f"INSERT INTO prompts ({columns}) VALUES ({values})", record)
        conn.execute(
            "INSERT INTO prompt_versions (prompt_id, parent_version, version_number, created_at, "
            "commit_message, snapshot_json) VALUES (?, NULL, 1, ?, ?, ?)",
            (str(prompt.id), now.isoformat(), "CLI draft capture", snapshot),
        )
        conn.execute(
            "INSERT INTO prompt_activity_events (prompt_id, operation, origin, occurred_at) "
            "VALUES (?, 'created', 'cli', ?)",
            (str(prompt.id), now.isoformat()),
        )
    return _one(conn, str(prompt.id))


def _delete_index(index: Path, draft_id: str) -> bool:
    """Remove an existing vector without creating an index or calling an embedding provider."""
    if not index.exists():
        return False
    if not index.is_dir() or index.is_symlink():
        raise DraftError("INDEX_UNAVAILABLE", "Local vector index cannot be safely opened.")
    db = index / "chroma.sqlite3"
    if not db.is_file() or db.is_symlink():
        raise DraftError("INDEX_UNAVAILABLE", "Local vector index cannot be safely opened.")
    try:
        # Never start Chroma for a missing collection/ID: PersistentClient is a
        # writer and can bootstrap even an existing empty SQLite file.
        sidecars = [Path(f"{db}{suffix}") for suffix in ("-wal", "-journal")]
        if any(path.exists() and path.stat().st_size for path in sidecars):
            raise DraftError("INDEX_UNAVAILABLE", "Local index has pending journal data.")
        uri = db.resolve().as_uri() + "?mode=ro&immutable=1"
        with closing(sqlite3.connect(uri, uri=True)) as read:
            expected = {
                "collections": {"id", "name"},
                "segments": {"id", "scope", "collection"},
                "embeddings": {"segment_id", "embedding_id"},
            }
            for table, fields in expected.items():
                columns = {str(row[1]) for row in read.execute(f"PRAGMA table_info({table})")}
                if not fields.issubset(columns):
                    raise DraftError("INDEX_UNAVAILABLE", "Local index schema is unsupported.")
            collections = read.execute(
                "SELECT id FROM collections WHERE name=?", ("prompt_manager",)
            ).fetchall()
            if not collections:
                return False
            if len(collections) != 1:
                raise DraftError("INDEX_UNAVAILABLE", "Local index collection is ambiguous.")
            segments = read.execute(
                "SELECT id FROM segments WHERE collection=? AND scope='METADATA'",
                (collections[0][0],),
            ).fetchall()
            if len(segments) != 1:
                raise DraftError("INDEX_UNAVAILABLE", "Local index segment is unavailable.")
            found = (
                read.execute(
                    "SELECT 1 FROM embeddings WHERE segment_id=? AND embedding_id=? LIMIT 1",
                    (segments[0][0], draft_id),
                ).fetchone()
                is not None
            )
        if any(path.exists() and path.stat().st_size for path in sidecars):
            raise DraftError("INDEX_UNAVAILABLE", "Local index changed during inspection.")
        if not found:
            return False
    except DraftError:
        raise
    except (sqlite3.Error, OSError, ValueError, TypeError) as exc:
        raise DraftError("INDEX_UNAVAILABLE", "Local vector index cannot be inspected.") from exc
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(index))
        collection = client.get_collection(name="prompt_manager")
        collection.delete(ids=[draft_id])
        return True
    except Exception as exc:  # noqa: BLE001 - mutation may precede raised error
        raise DraftError(
            "PARTIAL_DELETE",
            "Local vector cleanup may have changed the index; inspect draft state.",
        ) from exc


def _remove(conn: sqlite3.Connection, index: Path, draft_id: str) -> None:
    conn.execute("BEGIN IMMEDIATE")
    index_removed = False
    try:
        _one(conn, draft_id)  # Recheck while holding the SQLite write lock.
        index_removed = _delete_index(index, draft_id)
        conn.execute("DELETE FROM prompts WHERE id=?", (draft_id,))
        conn.execute(
            "INSERT INTO prompt_activity_events (prompt_id, operation, origin, occurred_at) "
            "VALUES (?, 'deleted', 'cli', ?)",
            (draft_id, datetime.now(UTC).isoformat()),
        )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        if isinstance(exc, DraftError) and exc.code == "PARTIAL_DELETE":
            raise
        if index_removed:
            raise DraftError(
                "PARTIAL_DELETE",
                "Index cleanup succeeded but catalog deletion failed; inspect draft state.",
            ) from exc
        raise


def _execute(args: Namespace) -> dict[str, object]:
    action = args.draft_action
    limit = _limit(args.limit) if action in {None, "find"} else 20
    draft_id = _id(args.id) if action in {"show", "delete"} else ""
    query = None
    if action == "find":
        query = args.query.strip()
        if not query:
            raise DraftError("INVALID_INPUT", "Search text must not be blank.")
    body = _body(args) if action == "add" else ""
    if action == "delete" and not args.yes and (args.json or not sys.stdin.isatty()):
        raise DraftError(
            "CONFIRM_REQUIRED", "Deletion requires --yes in non-interactive or JSON mode."
        )
    path = _catalog_path()
    with closing(_connect(path, write=action in {"add", "delete"})) as conn:
        try:
            if action in {None, "find"}:
                return {"drafts": [_preview(row) for row in _rows(conn, limit, query)]}
            if action == "show":
                return {"draft": _detail(_one(conn, draft_id))}
            if action == "add":
                return {"draft": _detail(_save(conn, body, args.title, args.source))}
            if action == "delete":
                _one(conn, draft_id)
                if not args.yes:
                    try:
                        answer = input(f"Permanently delete draft {draft_id}? [y/N] ")
                    except EOFError:
                        answer = ""
                    if answer.strip().lower() not in {"y", "yes"}:
                        raise DraftError("CANCELLED", "Deletion cancelled.")
                try:
                    settings_logger = logging.getLogger("prompt_manager.settings")
                    previous = settings_logger.level
                    settings_logger.setLevel(logging.ERROR)
                    try:
                        settings = load_settings()
                    finally:
                        settings_logger.setLevel(previous)
                    if settings.redis_dsn:
                        raise DraftError(
                            "CACHE_CONFIGURED",
                            "Draft deletion is unavailable while Redis caching is configured.",
                        )
                    index = settings.chroma_path.expanduser()
                except DraftError:
                    raise
                except (SettingsError, OSError, ValueError, TypeError) as exc:
                    raise DraftError(
                        "CONFIG_UNAVAILABLE", "Index configuration is unavailable."
                    ) from exc
                _remove(conn, index, draft_id)
                return {"id": draft_id}
            raise DraftError("INVALID_USAGE", "Unknown draft action.")
        except sqlite3.Error as exc:
            raise DraftError("CATALOG_UNAVAILABLE", "Unable to access selected catalog.") from exc


def _emit(payload: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps({"ok": True, **payload}, ensure_ascii=False))
        return
    if "drafts" in payload:
        drafts = cast("list[dict[str, str]]", payload["drafts"])
        print(
            "\n".join(f"{item['id']} | {item['title']} | {item['preview']}" for item in drafts)
            if drafts
            else "No drafts."
        )
    elif "draft" in payload:
        draft = cast("dict[str, str]", payload["draft"])
        print(
            f"ID: {draft['id']}\nTitle: {draft['title']}\n"
            f"Source: {draft['source']}\n\n{draft['body']}"
        )
    else:
        print(f"Deleted draft: {payload['id']}")


def run_draft(args: Namespace) -> int:
    """Dispatch draft commands without GUI, full manager or provider startup."""
    try:
        result = _execute(args)
        _emit(result, as_json=args.json)
        return 0
    except DraftError as exc:
        if args.json:
            print(
                json.dumps({"ok": False, "error": {"code": exc.code, "message": exc.message}}),
                file=sys.stderr,
            )
        else:
            print(f"Draft error ({exc.code}): {exc.message}", file=sys.stderr)
        return (
            2
            if exc.code.startswith("INVALID") or exc.code in {"CONFIRM_REQUIRED", "CANCELLED"}
            else 4
        )
