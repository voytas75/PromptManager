"""Real-process contracts for standalone notes sharing the GUI SQLite table."""

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

ROOT = Path(__file__).resolve().parents[1]


def _setup(tmp_path: Path, *, create: bool = True) -> tuple[Path, Path]:
    database = tmp_path / "catalog.sqlite3"
    config = tmp_path / "settings.json"
    database.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        json.dumps(
            {
                "database_path": str(database),
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
        with sqlite3.connect(database) as conn:
            conn.execute("CREATE TABLE prompts (id TEXT PRIMARY KEY, name TEXT)")
            conn.execute(
                "CREATE TABLE prompt_notes (id TEXT PRIMARY KEY, note TEXT NOT NULL, "
                "created_at TEXT NOT NULL, last_modified TEXT NOT NULL)"
            )
    return config, database


def _run(
    tmp_path: Path, config: Path, *args: str, module: bool = False, input_text: str = ""
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
        input=input_text,
        capture_output=True,
        text=True,
        timeout=25,
        check=False,
    )


def _insert(db: Path, note: str, *, modified: str = "2026-01-01T00:00:00+00:00") -> str:
    note_id = str(uuid.uuid4())
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO prompt_notes VALUES (?, ?, ?, ?)", (note_id, note, modified, modified)
        )
    return note_id


@pytest.mark.parametrize("module", [False, True])
def test_note_reads_existing_catalog_without_bootstrap(tmp_path: Path, module: bool) -> None:
    config, db = _setup(tmp_path)
    old_id = _insert(db, "First line\nmore")
    recent_id = _insert(db, "Żółw 100%_ literal", modified="2026-02-01T00:00:00+00:00")
    for args in [
        ("note",),
        ("note", "--json"),
        ("note", "show", old_id, "--json"),
        ("note", "find", "100%_", "--json"),
    ]:
        result = _run(tmp_path, config, *args, module=module)
        assert result.returncode == 0, result.stderr
        assert result.stderr == ""
        if "--json" in args:
            assert json.loads(result.stdout)["ok"] is True
    listing = json.loads(_run(tmp_path, config, "note", "--json", module=module).stdout)
    assert [item["id"] for item in listing["notes"]] == [recent_id, old_id]
    assert "more" not in listing["notes"][1]["preview"]
    shown = json.loads(
        _run(tmp_path, config, "note", "show", old_id, "--json", module=module).stdout
    )
    assert shown["note"]["note"] == "First line\nmore"
    found = json.loads(
        _run(tmp_path, config, "note", "find", "100%_", "--json", module=module).stdout
    )
    assert [item["id"] for item in found["notes"]] == [recent_id]
    assert not (tmp_path / "chroma").exists()
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM prompts").fetchone()[0] == 0


@pytest.mark.parametrize("module", [False, True])
def test_note_missing_db_and_help_are_read_only(tmp_path: Path, module: bool) -> None:
    config, db = _setup(tmp_path, create=False)
    for args in [
        ("--help",),
        ("note", "--help"),
        ("note", "add", "--help"),
        ("note", "show", "--help"),
        ("note", "find", "--help"),
        ("note", "edit", "--help"),
        ("note", "delete", "--help"),
    ]:
        result = _run(tmp_path, config, *args, module=module)
        assert result.returncode == 0, (args, result.stderr)
        assert result.stderr == "" and "note" in result.stdout.lower()
    result = _run(tmp_path, config, "note", "--json", module=module)
    assert result.returncode != 0 and result.stdout == ""
    assert json.loads(result.stderr)["ok"] is False
    assert not db.exists() and not (tmp_path / "chroma").exists()


def test_note_mutation_lifecycle_and_guardrails(tmp_path: Path) -> None:
    config, db = _setup(tmp_path)
    added = _run(tmp_path, config, "note", "add", "Hello\nnotes", "--json")
    assert added.returncode == 0, added.stderr
    note_id = json.loads(added.stdout)["note"]["id"]
    assert str(uuid.UUID(note_id)) == note_id
    assert _run(tmp_path, config, "note", "show", note_id).returncode == 0
    assert note_id in _run(tmp_path, config, "note").stdout
    assert note_id in _run(tmp_path, config, "note", "find", "Hello").stdout
    file = tmp_path / "note.md"
    file.write_text("Updated Żółw", encoding="utf-8")
    edited = _run(tmp_path, config, "note", "edit", note_id, "--file", str(file), "--json")
    assert edited.returncode == 0, edited.stderr
    assert json.loads(edited.stdout)["note"]["note"] == "Updated Żółw"
    refused = _run(tmp_path, config, "note", "delete", note_id, "--json")
    assert refused.returncode != 0 and refused.stdout == ""
    assert json.loads(refused.stderr)["ok"] is False
    assert _run(tmp_path, config, "note", "show", note_id).returncode == 0
    deleted = _run(tmp_path, config, "note", "delete", note_id, "--yes", "--json")
    assert deleted.returncode == 0, deleted.stderr
    assert json.loads(deleted.stdout)["id"] == note_id
    assert _run(tmp_path, config, "note", "show", note_id).returncode != 0
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM prompt_notes").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM prompts").fetchone()[0] == 0
    assert not (tmp_path / "chroma").exists()


def test_note_rejects_ambiguous_and_invalid_inputs(tmp_path: Path) -> None:
    config, db = _setup(tmp_path)
    note_id = _insert(db, "safe")
    for args in [
        ("note", "add", "a", "--body", "b"),
        ("note", "add", "--body", "   "),
        ("note", "edit", note_id, "--body", "x", "--from-stdin"),
        ("note", "edit", note_id, "--from-stdin"),
        ("note", "show", note_id[:8]),
        ("note", "find", " "),
        ("note", "--limit", "0"),
    ]:
        result = _run(tmp_path, config, *args)
        assert result.returncode != 0, args
    assert (
        _run(tmp_path, config, "note", "add", "--from-stdin", input_text="via stdin").returncode
        == 0
    )
    with sqlite3.connect(db) as conn:
        assert (
            conn.execute("SELECT note FROM prompt_notes WHERE id=?", (note_id,)).fetchone()[0]
            == "safe"
        )


def test_note_find_like_metacharacters_are_literal(tmp_path: Path) -> None:
    config, db = _setup(tmp_path)
    _insert(db, "ordinary text")
    expected = _insert(db, r"literal 100%_\\ note")
    for query in ("%_", r"\\"):
        result = _run(tmp_path, config, "note", "find", query, "--json")
        assert result.returncode == 0, result.stderr
        assert [item["id"] for item in json.loads(result.stdout)["notes"]] == [expected]


def test_note_uses_gui_repository_schema_and_refuses_missing_table(tmp_path: Path) -> None:
    config, db = _setup(tmp_path, create=False)
    PromptRepository(str(db))
    created = _run(tmp_path, config, "note", "add", "--body", "Shared GUI note", "--json")
    assert created.returncode == 0, created.stderr
    note_id = uuid.UUID(json.loads(created.stdout)["note"]["id"])
    assert PromptRepository(str(db)).get_prompt_note(note_id).note == "Shared GUI note"
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM prompts").fetchone()[0] == 0
    other_config, other_db = _setup(tmp_path / "other", create=False)
    with sqlite3.connect(other_db) as conn:
        conn.execute("CREATE TABLE unrelated (id TEXT)")
    failed = _run(tmp_path, other_config, "note", "add", "x", "--json")
    assert failed.returncode != 0 and failed.stdout == ""
    assert json.loads(failed.stderr)["error"]["code"] == "CATALOG_INVALID"
    with sqlite3.connect(other_db) as conn:
        assert (
            conn.execute("SELECT name FROM sqlite_master WHERE name='prompt_notes'").fetchone()
            is None
        )


def test_note_inputs_and_json_error_contract(tmp_path: Path) -> None:
    config, db = _setup(tmp_path)
    note_id = _insert(db, "original")
    missing = str(uuid.uuid4())
    bad_file = tmp_path / "invalid.txt"
    bad_file.write_bytes(b"\xff\xfe")
    for args, code in [
        (("note", "add", "--file", str(bad_file), "--json"), "INVALID_INPUT"),
        (("note", "add", "--file", str(tmp_path / "absent"), "--json"), "INVALID_INPUT"),
        (("note", "edit", missing, "--body", "new", "--json"), "NOTE_NOT_FOUND"),
        (("note", "show", missing, "--json"), "NOTE_NOT_FOUND"),
        (("note", "delete", missing, "--yes", "--json"), "NOTE_NOT_FOUND"),
        (("note", "find", " ", "--json"), "INVALID_INPUT"),
        (("note", "--limit", "101", "--json"), "INVALID_LIMIT"),
    ]:
        result = _run(tmp_path, config, *args)
        assert result.returncode != 0 and result.stdout == "", (args, result.stdout)
        assert json.loads(result.stderr)["error"]["code"] == code
        assert "original" not in result.stderr
    with sqlite3.connect(db) as conn:
        value = conn.execute("SELECT note FROM prompt_notes WHERE id=?", (note_id,)).fetchone()
        assert value[0] == "original"


def test_note_family_options_and_body_sources(tmp_path: Path) -> None:
    config, _ = _setup(tmp_path)
    first = _run(tmp_path, config, "note", "--json", "add", "--body", "first")
    assert first.returncode == 0 and first.stderr == ""
    note_id = json.loads(first.stdout)["note"]["id"]
    found = _run(tmp_path, config, "note", "--limit", "1", "--json", "find", "first")
    assert found.returncode == 0 and len(json.loads(found.stdout)["notes"]) == 1
    changed = _run(
        tmp_path, config, "note", "--json", "edit", note_id, "--from-stdin", input_text="second"
    )
    assert changed.returncode == 0, changed.stderr
    assert json.loads(changed.stdout)["note"]["note"] == "second"
    failed = _run(tmp_path, config, "note", "--json", "delete", note_id)
    assert failed.returncode != 0
    assert json.loads(failed.stderr)["error"]["code"] == "CONFIRM_REQUIRED"


def test_note_json_usage_errors_do_not_echo_private_input(tmp_path: Path) -> None:
    config, db = _setup(tmp_path)
    secret = "private-fixture-body"
    for args in [
        ("note", "add", secret, "--json", "--unexpected"),
        ("note", "--json", "add", secret, "--unexpected"),
        ("note", "show", "--json", secret, "--unexpected"),
    ]:
        result = _run(tmp_path, config, *args)
        assert result.returncode == 2 and result.stdout == ""
        assert json.loads(result.stderr)["error"]["code"] == "INVALID_USAGE"
        assert secret not in result.stderr
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM prompt_notes").fetchone()[0] == 0


def test_note_bounded_preview_and_no_mutation_on_invalid_body(tmp_path: Path) -> None:
    config, db = _setup(tmp_path)
    note_id = _insert(db, "Original\nPRIVATE-LINE", modified="2026-01-01T00:00:00+00:00")
    big_file = tmp_path / "big.md"
    big_file.write_bytes(b"x" * (1024 * 1024 + 1))
    for args in [
        ("note", "add", "--file", str(big_file), "--json"),
        ("note", "edit", note_id, "--body", "  ", "--json"),
    ]:
        failed = _run(tmp_path, config, *args)
        assert failed.returncode != 0 and failed.stdout == ""
        assert json.loads(failed.stderr)["error"]["code"] == "INVALID_INPUT"
    listing = _run(tmp_path, config, "note", "--json")
    assert listing.returncode == 0 and listing.stderr == ""
    assert json.loads(listing.stdout)["notes"][0]["preview"] == "Original"
    assert "PRIVATE-LINE" not in listing.stdout
    readback = json.loads(_run(tmp_path, config, "note", "show", note_id, "--json").stdout)
    assert readback["note"]["note"] == "Original\nPRIVATE-LINE"
    assert readback["note"]["created_at"] == "2026-01-01T00:00:00+00:00"


def test_note_json_unavailable_config_is_sanitized(tmp_path: Path) -> None:
    config, db = _setup(tmp_path, create=False)
    config.unlink()
    result = _run(tmp_path, config, "note", "--json")
    assert result.returncode != 0 and result.stdout == ""
    assert json.loads(result.stderr)["error"]["code"] == "CONFIG_UNAVAILABLE"
    assert str(config) not in result.stderr
    assert not db.exists() and not (tmp_path / "chroma").exists()


def test_note_success_json_stderr_stays_empty_with_ignored_config_credential(
    tmp_path: Path,
) -> None:
    config, _ = _setup(tmp_path)
    data = json.loads(config.read_text(encoding="utf-8"))
    data["litellm_api_key"] = "synthetic-not-a-real-key"
    config.write_text(json.dumps(data), encoding="utf-8")
    result = _run(tmp_path, config, "note", "--json")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["notes"] == []
    assert result.stderr == ""


def test_note_corrupt_row_returns_bounded_error_without_traceback(tmp_path: Path) -> None:
    config, db = _setup(tmp_path)
    note_id = _insert(db, "private-corrupt-row", modified="invalid date private marker")
    for args in [
        ("note", "--json"),
        ("note", "find", "private-corrupt-row", "--json"),
        ("note", "show", note_id, "--json"),
        ("note", "edit", note_id, "--body", "replacement", "--json"),
    ]:
        result = _run(tmp_path, config, *args)
        assert result.returncode != 0 and result.stdout == "", args
        assert json.loads(result.stderr)["error"]["code"] == "CATALOG_INVALID"
        assert "private marker" not in result.stderr and "Traceback" not in result.stderr
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE prompt_notes SET last_modified=? WHERE id=?", ("2026-01-01", note_id))
        conn.execute(
            "INSERT INTO prompt_notes VALUES (?, ?, ?, ?)",
            ("private-malformed-id", "private-corrupt-row", "2026-01-01", "2026-01-01"),
        )
    result = _run(tmp_path, config, "note", "--json")
    assert result.returncode != 0 and result.stdout == ""
    assert json.loads(result.stderr)["error"]["code"] == "CATALOG_INVALID"
    assert "private-malformed-id" not in result.stderr
    with sqlite3.connect(db) as conn:
        assert (
            conn.execute("SELECT note FROM prompt_notes WHERE id=?", (note_id,)).fetchone()[0]
            == "private-corrupt-row"
        )


def test_note_text_usage_error_never_echoes_extra_private_body(tmp_path: Path) -> None:
    config, db = _setup(tmp_path)
    secret = "private-fixture-second"
    result = _run(tmp_path, config, "note", "add", "harmless-first", secret)
    assert result.returncode == 2 and result.stdout == ""
    assert secret not in result.stderr and "usage:" not in result.stderr
    assert "INVALID_USAGE" in result.stderr
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM prompt_notes").fetchone()[0] == 0


def test_note_error_sanitizer_does_not_capture_other_commands(tmp_path: Path) -> None:
    config, _ = _setup(tmp_path)
    result = _run(tmp_path, config, "prompt-show", "note", "--bad-option")
    assert result.returncode == 2
    assert "Note error (INVALID_USAGE)" not in result.stderr
    assert "unrecognized arguments" in result.stderr


def test_note_edit_supports_single_positional_body_without_extra_sources(tmp_path: Path) -> None:
    config, db = _setup(tmp_path)
    note_id = _insert(db, "original")
    edited = _run(tmp_path, config, "note", "edit", note_id, "positional replacement", "--json")
    assert edited.returncode == 0, edited.stderr
    assert json.loads(edited.stdout)["note"]["note"] == "positional replacement"
    rejected = _run(
        tmp_path, config, "note", "edit", note_id, "another value", "--body", "conflict", "--json"
    )
    assert rejected.returncode == 2 and rejected.stdout == ""
    assert json.loads(rejected.stderr)["error"]["code"] == "INVALID_INPUT"
    with sqlite3.connect(db) as conn:
        assert (
            conn.execute("SELECT note FROM prompt_notes WHERE id=?", (note_id,)).fetchone()[0]
            == "positional replacement"
        )
