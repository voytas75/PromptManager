"""Real-process contracts for local Prompt Parts over the GUI SQLite table."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from core.repository import PromptRepository
from models.response_style import ResponseStyle

ROOT = Path(__file__).resolve().parents[1]


def _setup(tmp_path: Path, *, create: bool = True) -> tuple[Path, Path]:
    db = tmp_path / "catalog.sqlite3"
    config = tmp_path / "settings.json"
    config.write_text(
        json.dumps(
            {
                "database_path": str(db),
                "chroma_path": str(tmp_path / "chroma"),
                "embedding_backend": "deterministic",
                "litellm_model": None,
                "litellm_inference_model": None,
                "redis_dsn": None,
            }
        ),
        encoding="utf-8",
    )
    if create:
        PromptRepository(str(db))
    return config, db


def _run(
    tmp_path: Path,
    config: Path,
    *args: str,
    module: bool = False,
    stdin: str = "",
) -> subprocess.CompletedProcess[str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not (
            key.startswith(("PROMPT_MANAGER_", "AZURE_OPENAI_"))
            or key in {"OPENAI_API_KEY", "LITELLM_API_KEY"}
        )
    }
    env.update(
        HOME=str(tmp_path),
        PYTHONPATH=str(ROOT),
        PROMPT_MANAGER_CONFIG_JSON=str(config),
        PROMPT_MANAGER_ENV_FILE="",
        PYTHONDONTWRITEBYTECODE="1",
    )
    command = [sys.executable, "-m", "main"] if module else [str(ROOT / ".venv/bin/prompt-manager")]
    return subprocess.run(
        command + list(args),
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        input=stdin,
        timeout=25,
        check=False,
    )


def _part(db: Path, *, name: str, snippet: str, active: bool = True) -> str:
    part = ResponseStyle(
        id=uuid.uuid4(),
        name=name,
        description="Usage notes",
        snippet=snippet,
        prompt_part="System Instruction",
        format_instructions="Use bullets",
        tags=["policy"],
        examples=["Example"],
        is_active=active,
    )
    PromptRepository(str(db)).add_response_style(part)
    return str(part.id)


@pytest.mark.parametrize("module", [False, True])
def test_prompt_part_read_only_entrypoints_and_help(tmp_path: Path, module: bool) -> None:
    config, db = _setup(tmp_path)
    first_id = _part(db, name="Policy", snippet="Żółw says 100%_ literal")
    second_id = _part(db, name="Policy", snippet="Second policy", active=False)
    for args in (
        ("--help",),
        ("prompt-part", "--help"),
        ("prompt-part", "show", "--help"),
        ("prompt-part", "find", "--help"),
        ("prompt-part", "add", "--help"),
        ("prompt-part", "edit", "--help"),
        ("prompt-part", "delete", "--help"),
    ):
        help_result = _run(tmp_path, config, *args, module=module)
        assert help_result.returncode == 0, (args, help_result.stderr)
        assert help_result.stderr == ""
        assert "prompt-part" in help_result.stdout
    listing = _run(tmp_path, config, "prompt-part", "--json", module=module)
    assert listing.returncode == 0 and listing.stderr == ""
    assert [row["id"] for row in json.loads(listing.stdout)["parts"]] == [first_id]
    assert "Second policy" not in listing.stdout
    all_result = _run(tmp_path, config, "prompt-part", "--all", "--json", module=module)
    assert [row["id"] for row in json.loads(all_result.stdout)["parts"]] == sorted(
        [first_id, second_id]
    )
    shown = _run(tmp_path, config, "prompt-part", "show", second_id, "--json", module=module)
    assert shown.stderr == "" and shown.returncode == 0
    record = json.loads(shown.stdout)["part"]
    assert record["snippet"] == "Second policy" and record["is_active"] is False
    assert record["format_instructions"] == "Use bullets" and record["tags"] == ["policy"]
    assert "ext1" not in record and record["metadata"] is None
    found = _run(tmp_path, config, "prompt-part", "find", "100%_", "--json", module=module)
    assert found.returncode == 0 and found.stderr == ""
    assert [row["id"] for row in json.loads(found.stdout)["parts"]] == [first_id]
    escaped = _run(tmp_path, config, "prompt-part", "find", "%_", "--json", module=module)
    assert [row["id"] for row in json.loads(escaped.stdout)["parts"]] == [first_id]
    hidden = _run(tmp_path, config, "prompt-part", "find", "Second", "--json", module=module)
    assert json.loads(hidden.stdout)["parts"] == []
    visible = _run(
        tmp_path,
        config,
        "prompt-part",
        "find",
        "Second",
        "--all",
        "--json",
        module=module,
    )
    assert [row["id"] for row in json.loads(visible.stdout)["parts"]] == [second_id]
    assert not (tmp_path / "chroma").exists()


@pytest.mark.parametrize("module", [False, True])
def test_prompt_part_read_missing_catalog_does_not_create_it(tmp_path: Path, module: bool) -> None:
    config, db = _setup(tmp_path, create=False)
    result = _run(tmp_path, config, "prompt-part", "--json", module=module)
    assert result.returncode != 0 and result.stdout == ""
    assert json.loads(result.stderr)["error"]["code"] == "CATALOG_UNAVAILABLE"
    assert not db.exists() and not (tmp_path / "chroma").exists()


def test_prompt_part_legacy_schema_fails_without_migration(tmp_path: Path) -> None:
    config, db = _setup(tmp_path)
    with sqlite3.connect(db) as conn:
        conn.execute("ALTER TABLE response_styles DROP COLUMN snippet")
    result = _run(tmp_path, config, "prompt-part", "--json")
    assert result.returncode != 0 and result.stdout == ""
    assert json.loads(result.stderr)["error"]["code"] == "CATALOG_MIGRATION_REQUIRED"
    with sqlite3.connect(db) as conn:
        assert "snippet" not in {
            row[1] for row in conn.execute("PRAGMA table_info(response_styles)")
        }


def test_prompt_part_bounded_and_sanitized_errors(tmp_path: Path) -> None:
    config, db = _setup(tmp_path)
    part_id = _part(db, name="Prompt", snippet="sensitive body")
    for args, code in (
        (("prompt-part", "--limit", "0", "--json"), "INVALID_LIMIT"),
        (("prompt-part", "show", "bad-token", "--json"), "INVALID_ID"),
        (("prompt-part", "show", str(uuid.uuid4()), "--json"), "PART_NOT_FOUND"),
        (("prompt-part", "find", " ", "--json"), "INVALID_INPUT"),
        (("prompt-part", "show", part_id, "--json", "sensitive body"), "INVALID_USAGE"),
    ):
        result = _run(tmp_path, config, *args)
        assert result.returncode != 0 and result.stdout == "", result
        assert json.loads(result.stderr)["error"]["code"] == code
        assert "sensitive body" not in result.stderr and "bad-token" not in result.stderr


@pytest.mark.parametrize("module", [False, True])
def test_prompt_part_add_round_trip_and_sources(tmp_path: Path, module: bool) -> None:
    config, db = _setup(tmp_path)
    body = "Żółw\n<instructions>100%_</instructions>\n"
    source = tmp_path / "part.txt"
    source.write_text(body, encoding="utf-8")
    for source_args, stdin in (
        (("--body", body), ""),
        (("--file", str(source)), ""),
        (("--from-stdin",), body),
    ):
        result = _run(
            tmp_path,
            config,
            "prompt-part",
            "add",
            "--name",
            " Fragment ",
            "--description",
            "Operator note",
            *source_args,
            "--json",
            module=module,
            stdin=stdin,
        )
        assert result.returncode == 0 and result.stderr == "", result
        record = json.loads(result.stdout)["part"]
        identifier = uuid.UUID(record["id"])
        assert record["name"] == "Fragment" and record["snippet"] == body
        assert record["description"] == "Operator note"
        assert record["format_instructions"] is None
        gui_record = PromptRepository(str(db)).get_response_style(identifier)
        assert gui_record.snippet == body and gui_record.description == "Operator note"
        assert gui_record.prompt_part == "Response Style"
    assert not (tmp_path / "chroma").exists()


def test_prompt_part_add_rejects_invalid_sources_without_writes(tmp_path: Path) -> None:
    config, db = _setup(tmp_path)
    bad_utf8 = tmp_path / "bad.txt"
    bad_utf8.write_bytes(b"\xff")
    too_large = tmp_path / "large.txt"
    too_large.write_bytes(b"x" * (1024 * 1024 + 1))
    cases = (
        ("--name", "valid", "--body", "  "),
        ("--name", " ", "--body", "ok"),
        ("--name", "valid", "--body", "ok", "--from-stdin"),
        ("--name", "valid", "--file", str(bad_utf8)),
        ("--name", "valid", "--file", str(too_large)),
        ("--name", "valid", "--file", str(tmp_path / "missing.txt")),
        ("--name", "valid"),
    )
    for options in cases:
        result = _run(tmp_path, config, "prompt-part", "add", *options, "--json")
        assert result.returncode != 0 and result.stdout == "", result
        assert json.loads(result.stderr)["error"]["code"] in {"INVALID_INPUT", "INVALID_USAGE"}
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM response_styles").fetchone()[0] == 0


def test_prompt_part_malformed_rows_fail_closed_without_leaking_text(tmp_path: Path) -> None:
    config, db = _setup(tmp_path)
    identifier = _part(db, name="Private", snippet="VERY_PRIVATE_BODY")
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE response_styles SET tags=?, created_at=? WHERE id=?",
            ("not json PRIVATE_TAG", "not timestamp", identifier),
        )
    for args in (("prompt-part", "--json"), ("prompt-part", "show", identifier, "--json")):
        result = _run(tmp_path, config, *args)
        assert result.returncode != 0 and result.stdout == ""
        assert json.loads(result.stderr)["error"]["code"] == "CATALOG_INVALID"
        assert "PRIVATE_TAG" not in result.stderr and "VERY_PRIVATE_BODY" not in result.stderr


def test_prompt_part_malformed_stored_identifier_is_catalog_error(tmp_path: Path) -> None:
    config, db = _setup(tmp_path)
    identifier = _part(db, name="Private", snippet="VERY_PRIVATE_BODY")
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE response_styles SET id=? WHERE id=?", ("not-an-id", identifier))
    result = _run(tmp_path, config, "prompt-part", "--json")
    assert result.returncode != 0 and result.stdout == ""
    assert json.loads(result.stderr)["error"]["code"] == "CATALOG_INVALID"
    assert "VERY_PRIVATE_BODY" not in result.stderr


def test_prompt_part_noop_and_gui_concurrent_change(tmp_path: Path) -> None:
    config, db = _setup(tmp_path)
    identifier = _part(db, name="Private", snippet="Original")
    timestamp = json.loads(
        _run(tmp_path, config, "prompt-part", "show", identifier, "--json").stdout
    )["part"]["last_modified"]
    noop = _run(
        tmp_path,
        config,
        "prompt-part",
        "edit",
        identifier,
        "--expect-modified",
        timestamp,
        "--name",
        "Private",
        "--json",
    )
    assert noop.returncode == 0 and json.loads(noop.stdout)["part"]["last_modified"] == timestamp
    style = PromptRepository(str(db)).get_response_style(uuid.UUID(identifier))
    style.name = "GUI changed"
    style.touch()
    PromptRepository(str(db)).update_response_style(style)
    conflict = _run(
        tmp_path,
        config,
        "prompt-part",
        "edit",
        identifier,
        "--expect-modified",
        timestamp,
        "--name",
        "CLI overwrite",
        "--json",
    )
    assert conflict.returncode != 0 and conflict.stdout == ""
    assert json.loads(conflict.stderr)["error"]["code"] == "PART_CONFLICT"
    assert PromptRepository(str(db)).get_response_style(uuid.UUID(identifier)).name == "GUI changed"


def test_prompt_part_missing_catalog_rejects_add_without_creation(tmp_path: Path) -> None:
    config, db = _setup(tmp_path, create=False)
    result = _run(
        tmp_path, config, "prompt-part", "add", "--name", "Example", "--body", "A", "--json"
    )
    assert result.returncode != 0 and result.stdout == ""
    assert json.loads(result.stderr)["error"]["code"] == "CATALOG_UNAVAILABLE"
    assert not db.exists() and not (tmp_path / "chroma").exists()


@pytest.mark.parametrize("module", [False, True])
def test_prompt_part_full_cli_lifecycle_on_one_temp_catalog(tmp_path: Path, module: bool) -> None:
    config, db = _setup(tmp_path)
    body = "Part lifecycle: %_!\n"
    added = _run(
        tmp_path,
        config,
        "prompt-part",
        "add",
        "--name",
        "Example",
        "--body",
        body,
        "--json",
        module=module,
    )
    assert added.returncode == 0 and added.stderr == ""
    part = json.loads(added.stdout)["part"]
    identifier = part["id"]
    listing = _run(tmp_path, config, "prompt-part", "--json", module=module)
    found = _run(tmp_path, config, "prompt-part", "find", "%_!", "--json", module=module)
    shown = _run(tmp_path, config, "prompt-part", "show", identifier, "--json", module=module)
    assert [row["id"] for row in json.loads(listing.stdout)["parts"]] == [identifier]
    assert [row["id"] for row in json.loads(found.stdout)["parts"]] == [identifier]
    assert json.loads(shown.stdout)["part"]["snippet"] == body
    edited = _run(
        tmp_path,
        config,
        "prompt-part",
        "edit",
        identifier,
        "--expect-modified",
        part["last_modified"],
        "--name",
        "Revised",
        "--json",
        module=module,
    )
    assert edited.returncode == 0 and edited.stderr == ""
    revised = json.loads(edited.stdout)["part"]
    assert revised["name"] == "Revised" and revised["snippet"] == body
    deleted = _run(
        tmp_path,
        config,
        "prompt-part",
        "delete",
        identifier,
        "--expect-modified",
        revised["last_modified"],
        "--yes",
        "--json",
        module=module,
    )
    assert deleted.returncode == 0 and deleted.stderr == ""
    assert json.loads(deleted.stdout)["id"] == identifier
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM response_styles").fetchone()[0] == 0
    assert not (tmp_path / "chroma").exists()


def test_prompt_part_text_show_exposes_named_fields_not_storage_extensions(
    tmp_path: Path,
) -> None:
    config, db = _setup(tmp_path)
    identifier = _part(db, name="GUI part", snippet="Canonical text")
    shown = _run(tmp_path, config, "prompt-part", "show", identifier)
    assert shown.returncode == 0 and shown.stderr == ""
    for field in (
        "Tone:",
        "Voice:",
        "Format instructions:",
        "Guidelines:",
        "Tags:",
        "Examples:",
        "Version:",
        "Created:",
        "Snippet:",
    ):
        assert field in shown.stdout
    assert "Canonical text" in shown.stdout and "Use bullets" in shown.stdout
    assert "ext1" not in shown.stdout


def test_prompt_part_locked_writer_fails_without_mutation(tmp_path: Path) -> None:
    config, db = _setup(tmp_path)
    identifier = _part(db, name="Before lock", snippet="Text")
    timestamp = json.loads(
        _run(tmp_path, config, "prompt-part", "show", identifier, "--json").stdout
    )["part"]["last_modified"]
    with sqlite3.connect(db) as conn:
        conn.execute("BEGIN IMMEDIATE")
        blocked = _run(
            tmp_path,
            config,
            "prompt-part",
            "edit",
            identifier,
            "--expect-modified",
            timestamp,
            "--name",
            "Should not land",
            "--json",
        )
        assert blocked.returncode != 0 and blocked.stdout == ""
        assert json.loads(blocked.stderr)["error"]["code"] == "CATALOG_UNAVAILABLE"
    assert PromptRepository(str(db)).get_response_style(uuid.UUID(identifier)).name == "Before lock"


def test_prompt_part_rejects_naive_stored_timestamps_and_missing_extension_schema(
    tmp_path: Path,
) -> None:
    config, db = _setup(tmp_path)
    identifier = _part(db, name="Timestamp", snippet="Body")
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE response_styles SET last_modified=? WHERE id=?",
            ("2026-09-26T12:00:00", identifier),
        )
    shown = _run(tmp_path, config, "prompt-part", "show", identifier, "--json")
    assert shown.returncode != 0 and shown.stdout == ""
    assert json.loads(shown.stderr)["error"]["code"] == "CATALOG_INVALID"
    with sqlite3.connect(db) as conn:
        conn.execute("ALTER TABLE response_styles DROP COLUMN ext3")
    added = _run(
        tmp_path, config, "prompt-part", "add", "--name", "New", "--body", "Text", "--json"
    )
    assert added.returncode != 0 and added.stdout == ""
    assert json.loads(added.stderr)["error"]["code"] == "CATALOG_INVALID"
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM response_styles").fetchone()[0] == 1


@pytest.mark.parametrize("module", [False, True])
def test_prompt_part_edit_preserves_gui_fields_and_guards_stale_writes(
    tmp_path: Path, module: bool
) -> None:
    config, db = _setup(tmp_path)
    identifier = _part(db, name="Original", snippet="Original snippet")
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE response_styles SET ext1=?, ext2=?, version=?, metadata=?, tone=? WHERE id=?",
            ("opaque", '{"x":1}', "v-custom", '{"owner":"gui"}', "Calm", identifier),
        )
    shown = _run(tmp_path, config, "prompt-part", "show", identifier, "--json", module=module)
    timestamp = json.loads(shown.stdout)["part"]["last_modified"]
    edited = _run(
        tmp_path,
        config,
        "prompt-part",
        "edit",
        identifier,
        "--expect-modified",
        timestamp,
        "--body",
        "Updated snippet\n",
        "--description",
        "",
        "--inactive",
        "--json",
        module=module,
    )
    assert edited.returncode == 0 and edited.stderr == "", edited
    record = json.loads(edited.stdout)["part"]
    assert record["snippet"] == "Updated snippet\n" and record["description"] == ""
    assert record["is_active"] is False and record["version"] == "v-custom"
    assert record["tone"] == "Calm" and record["metadata"] == {"owner": "gui"}
    assert record["last_modified"] != timestamp
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        after = dict(
            conn.execute("SELECT * FROM response_styles WHERE id=?", (identifier,)).fetchone()
        )
    assert after["ext1"] == "opaque" and after["ext2"] == '{"x":1}'
    assert after["format_instructions"] == "Use bullets"
    assert after["created_at"] == record["created_at"]
    stale = _run(
        tmp_path,
        config,
        "prompt-part",
        "edit",
        identifier,
        "--expect-modified",
        timestamp,
        "--name",
        "Stale",
        "--json",
        module=module,
    )
    assert stale.returncode != 0 and stale.stdout == ""
    assert json.loads(stale.stderr)["error"]["code"] == "PART_CONFLICT"
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        assert (
            dict(conn.execute("SELECT * FROM response_styles WHERE id=?", (identifier,)).fetchone())
            == after
        )


def test_prompt_part_edit_and_delete_validate_without_mutations(tmp_path: Path) -> None:
    config, db = _setup(tmp_path)
    identifier = _part(db, name="Original", snippet="Private original")
    timestamp = json.loads(
        _run(tmp_path, config, "prompt-part", "show", identifier, "--json").stdout
    )["part"]["last_modified"]
    cases = (
        ("edit", identifier, "--expect-modified", timestamp),
        ("edit", identifier, "--expect-modified", timestamp, "--body", "  "),
        ("edit", identifier, "--expect-modified", "not-a-timestamp", "--name", "X"),
        ("edit", "bad-id", "--expect-modified", timestamp, "--name", "X"),
        ("delete", identifier, "--expect-modified", timestamp),
        ("delete", identifier, "--expect-modified", timestamp, "--json"),
    )
    with sqlite3.connect(db) as conn:
        original = conn.execute(
            "SELECT * FROM response_styles WHERE id=?", (identifier,)
        ).fetchone()
    for options in cases:
        result = _run(tmp_path, config, "prompt-part", *options, "--json")
        assert result.returncode != 0 and result.stdout == "", result
        assert "Private original" not in result.stderr
    with sqlite3.connect(db) as conn:
        assert (
            conn.execute("SELECT * FROM response_styles WHERE id=?", (identifier,)).fetchone()
            == original
        )


@pytest.mark.parametrize("module", [False, True])
def test_prompt_part_confirmed_delete_and_stale_delete(tmp_path: Path, module: bool) -> None:
    config, db = _setup(tmp_path)
    identifier = _part(db, name="Delete me", snippet="Content")
    timestamp = json.loads(
        _run(tmp_path, config, "prompt-part", "show", identifier, "--json", module=module).stdout
    )["part"]["last_modified"]
    stale = _run(
        tmp_path,
        config,
        "prompt-part",
        "delete",
        identifier,
        "--expect-modified",
        "2020-01-01T00:00:00+00:00",
        "--yes",
        "--json",
        module=module,
    )
    assert json.loads(stale.stderr)["error"]["code"] == "PART_CONFLICT"
    assert PromptRepository(str(db)).get_response_style(uuid.UUID(identifier)).snippet == "Content"
    deleted = _run(
        tmp_path,
        config,
        "prompt-part",
        "delete",
        identifier,
        "--expect-modified",
        timestamp,
        "--yes",
        "--json",
        module=module,
    )
    assert deleted.returncode == 0 and deleted.stderr == "", deleted
    assert json.loads(deleted.stdout)["id"] == identifier
    absent = _run(tmp_path, config, "prompt-part", "show", identifier, "--json", module=module)
    assert json.loads(absent.stderr)["error"]["code"] == "PART_NOT_FOUND"
    with sqlite3.connect(db) as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM response_styles WHERE id=?", (identifier,)
            ).fetchone()[0]
            == 0
        )
