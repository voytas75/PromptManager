"""Real-process contracts for the local draft prompt CLI."""

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
from models.prompt_model import Prompt

ROOT = Path(__file__).resolve().parents[1]


def _setup(tmp_path: Path, *, create: bool = True) -> tuple[Path, Path, Path]:
    db = tmp_path / "catalog.sqlite3"
    index = tmp_path / "chroma"
    config = tmp_path / "settings.json"
    config.write_text(
        json.dumps(
            {
                "database_path": str(db),
                "chroma_path": str(index),
                "embedding_backend": "litellm",
                "litellm_model": None,
                "litellm_inference_model": None,
                "redis_dsn": None,
            }
        ),
        encoding="utf-8",
    )
    if create:
        PromptRepository(str(db))
    return config, db, index


def _run(
    tmp_path: Path, config: Path, *args: str, module: bool = False, input_text: str = ""
) -> subprocess.CompletedProcess[str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not (
            key.startswith(("PROMPT_MANAGER_", "AZURE_", "OPENAI_", "LITELLM_"))
            or key in {"OPENAI_API_KEY", "LITELLM_API_KEY"}
        )
    }
    env.update(
        HOME=str(tmp_path),
        PYTHONPATH=str(ROOT),
        PROMPT_MANAGER_CONFIG_JSON=str(config),
        PROMPT_MANAGER_ENV_FILE="",
        PYTHONDONTWRITEBYTECODE="1",
        CHROMA_ANONYMIZED_TELEMETRY="0",
    )
    front = [sys.executable, "-m", "main"] if module else [str(ROOT / ".venv/bin/prompt-manager")]
    return subprocess.run(
        [*front, *args],
        cwd=tmp_path,
        env=env,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=35,
        check=False,
    )


def _existing(db: Path, *, draft: bool = True, body: str = "GUI body") -> str:
    entry = Prompt(
        id=uuid.uuid4(),
        name="GUI draft",
        description="quick draft",
        category="General",
        context=body,
        ext2={"capture_state": "draft", "capture_method": "quick_capture"} if draft else None,
    )
    PromptRepository(str(db)).add(entry)
    return str(entry.id)


@pytest.mark.parametrize("module", [False, True])
def test_draft_help_and_missing_db_are_provider_free(tmp_path: Path, module: bool) -> None:
    config, db, index = _setup(tmp_path, create=False)
    for args in [
        ("--help",),
        ("draft", "--help"),
        ("draft", "add", "--help"),
        ("draft", "show", "--help"),
        ("draft", "find", "--help"),
        ("draft", "delete", "--help"),
    ]:
        result = _run(tmp_path, config, *args, module=module)
        assert result.returncode == 0, (args, result.stderr)
        assert result.stderr == "" and "draft" in result.stdout.lower()
    missing = _run(tmp_path, config, "draft", "--json", module=module)
    assert missing.returncode != 0 and missing.stdout == ""
    assert json.loads(missing.stderr)["error"]["code"] == "CATALOG_UNAVAILABLE"
    assert not db.exists() and not index.exists()


@pytest.mark.parametrize("module", [False, True])
def test_draft_capture_and_reads_share_gui_catalog(tmp_path: Path, module: bool) -> None:
    config, db, index = _setup(tmp_path)
    gui_id = _existing(db)
    ordinary = _existing(db, draft=False, body="ordinary %_ body")
    result = _run(
        tmp_path,
        config,
        "draft",
        "add",
        "--body",
        "Prompt: Build 100%_ guide\nSecond line",
        "--source",
        "cli-chat",
        "--json",
        module=module,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    added = json.loads(result.stdout)["draft"]
    draft_id = added["id"]
    assert str(uuid.UUID(draft_id)) == draft_id
    assert added["title"] == "Build 100%_ guide"
    assert added["source"] == "cli-chat"
    assert PromptRepository(str(db)).get(uuid.UUID(draft_id)).context == (
        "Prompt: Build 100%_ guide\nSecond line"
    )
    shown = _run(tmp_path, config, "draft", "show", draft_id, "--json", module=module)
    assert shown.returncode == 0 and shown.stderr == ""
    assert json.loads(shown.stdout)["draft"]["body"] == "Prompt: Build 100%_ guide\nSecond line"
    listing = _run(tmp_path, config, "draft", "--json", module=module)
    listed = [entry["id"] for entry in json.loads(listing.stdout)["drafts"]]
    assert draft_id in listed and gui_id in listed and ordinary not in listed
    found = _run(tmp_path, config, "draft", "find", "%_", "--json", module=module)
    assert [entry["id"] for entry in json.loads(found.stdout)["drafts"]] == [draft_id]
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT ext2, ext4 FROM prompts WHERE id=?", (draft_id,)).fetchone()
        assert json.loads(row[0]) == {"capture_state": "draft", "capture_method": "cli"}
        assert row[1] is None
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM prompt_versions WHERE prompt_id=?", (draft_id,)
            ).fetchone()[0]
            == 1
        )
        assert conn.execute(
            "SELECT operation,origin FROM prompt_activity_events WHERE prompt_id=?", (draft_id,)
        ).fetchone() == ("created", "cli")
        assert conn.execute("SELECT COUNT(*) FROM prompt_notes").fetchone()[0] == 0
    assert not index.exists()


def test_draft_add_sources_and_guards(tmp_path: Path) -> None:
    config, db, index = _setup(tmp_path)
    file = tmp_path / "draft.txt"
    file.write_text("Żółw\nbody", encoding="utf-8")
    for args, input_text in [
        (("draft", "add", "Inline", "--json"), ""),
        (("draft", "add", "--file", str(file), "--json"), ""),
        (("draft", "add", "--from-stdin", "--json"), "Via stdin"),
    ]:
        result = _run(tmp_path, config, *args, input_text=input_text)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["draft"]["id"]
    for args in [
        ("draft", "add", "one", "--body", "two", "--json"),
        ("draft", "add", "--body", " ", "--json"),
        ("draft", "add", "--from-stdin", "--json"),
        ("draft", "--limit", "0", "--json"),
        ("draft", "find", " ", "--json"),
        ("draft", "show", "short", "--json"),
    ]:
        result = _run(tmp_path, config, *args)
        assert result.returncode != 0 and result.stdout == "", (args, result.stdout)
        assert json.loads(result.stderr)["ok"] is False
    bad = _run(
        tmp_path, config, "draft", "add", "--body", "SECRET_BODY", "extra SECRET_BODY", "--json"
    )
    assert bad.returncode != 0 and bad.stdout == "" and "SECRET_BODY" not in bad.stderr
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM prompts").fetchone()[0] == 3
    assert not index.exists()


def test_draft_delete_is_guarded_and_records_activity(tmp_path: Path) -> None:
    config, db, index = _setup(tmp_path)
    gui_id = _existing(db)
    ordinary = _existing(db, draft=False)
    refused = _run(tmp_path, config, "draft", "delete", gui_id, "--json")
    assert json.loads(refused.stderr)["error"]["code"] == "CONFIRM_REQUIRED"
    protected = _run(tmp_path, config, "draft", "delete", ordinary, "--yes", "--json")
    assert protected.returncode != 0 and protected.stdout == ""
    assert json.loads(protected.stderr)["error"]["code"] == "NOT_DRAFT"
    gone = _run(tmp_path, config, "draft", "delete", gui_id, "--yes", "--json")
    assert gone.returncode == 0 and json.loads(gone.stdout)["id"] == gui_id
    assert _run(tmp_path, config, "draft", "show", gui_id, "--json").returncode != 0
    with sqlite3.connect(db) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM prompts WHERE id=?", (ordinary,)).fetchone()[0] == 1
        )
        assert conn.execute(
            "SELECT operation,origin FROM prompt_activity_events WHERE prompt_id=?", (gui_id,)
        ).fetchone() == ("deleted", "cli")
    assert not index.exists()


def test_draft_rejects_corrupt_state_without_deleting(tmp_path: Path) -> None:
    config, db, _ = _setup(tmp_path)
    draft_id = _existing(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE prompts SET ext2=? WHERE id=?", ('{"capture_state": "draft"', draft_id)
        )
    shown = _run(tmp_path, config, "draft", "show", draft_id, "--json")
    assert shown.returncode != 0 and shown.stdout == ""
    assert json.loads(shown.stderr)["error"]["code"] == "CATALOG_INVALID"
    deleted = _run(tmp_path, config, "draft", "delete", draft_id, "--yes", "--json")
    assert deleted.returncode != 0 and deleted.stdout == ""
    with sqlite3.connect(db) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM prompts WHERE id=?", (draft_id,)).fetchone()[0] == 1
        )


def test_draft_find_literal_and_scan_beyond_non_draft_page(tmp_path: Path) -> None:
    config, db, _ = _setup(tmp_path)
    expected = _existing(db, body="literal 100%_! body")
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE prompts SET last_modified='2020-01-01T00:00:00+00:00' WHERE id=?", (expected,)
        )
    for _ in range(130):
        _existing(db, draft=False)
    listing = _run(tmp_path, config, "draft", "--limit", "1", "--json")
    assert [item["id"] for item in json.loads(listing.stdout)["drafts"]] == [expected]
    for fragment in ("%_!", "100%_!", "GUI draft"):
        found = _run(tmp_path, config, "draft", "find", fragment, "--json")
        assert [item["id"] for item in json.loads(found.stdout)["drafts"]] == [expected]


def test_draft_delete_retains_catalog_if_existing_index_unavailable(tmp_path: Path) -> None:
    config, db, index = _setup(tmp_path)
    draft_id = _existing(db)
    index.mkdir()
    (index / "unexpected").write_text("sentinel", encoding="utf-8")
    result = _run(tmp_path, config, "draft", "delete", draft_id, "--yes", "--json")
    assert result.returncode != 0 and result.stdout == ""
    assert json.loads(result.stderr)["error"]["code"] == "INDEX_UNAVAILABLE"
    assert (index / "unexpected").read_text(encoding="utf-8") == "sentinel"
    assert PromptRepository(str(db)).get(uuid.UUID(draft_id)).context == "GUI body"


def test_draft_delete_removes_existing_local_index_entry(tmp_path: Path) -> None:
    import chromadb

    config, db, index = _setup(tmp_path)
    draft_id = _existing(db)
    unrelated = _existing(db, draft=False)
    client = chromadb.PersistentClient(path=str(index))
    collection = client.get_or_create_collection("prompt_manager")
    collection.add(ids=[draft_id, unrelated], embeddings=[[0.1, 0.2], [0.3, 0.4]])
    deleted = _run(tmp_path, config, "draft", "delete", draft_id, "--yes", "--json")
    assert deleted.returncode == 0, deleted.stderr
    assert json.loads(deleted.stdout)["id"] == draft_id
    assert collection.get(ids=[draft_id])["ids"] == []
    assert collection.get(ids=[unrelated])["ids"] == [unrelated]
    with sqlite3.connect(db) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM prompts WHERE id=?", (draft_id,)).fetchone()[0] == 0
        )


def test_draft_delete_with_existing_index_without_collection(tmp_path: Path) -> None:
    import chromadb

    config, db, index = _setup(tmp_path)
    draft_id = _existing(db)
    chromadb.PersistentClient(path=str(index))
    before = (index / "chroma.sqlite3").read_bytes()
    deleted = _run(tmp_path, config, "draft", "delete", draft_id, "--yes", "--json")
    assert deleted.returncode == 0, deleted.stderr
    assert json.loads(deleted.stdout)["id"] == draft_id
    assert (index / "chroma.sqlite3").read_bytes() == before


def test_draft_delete_does_not_write_index_without_matching_entry(tmp_path: Path) -> None:
    import chromadb

    config, db, index = _setup(tmp_path)
    draft_id = _existing(db)
    client = chromadb.PersistentClient(path=str(index))
    collection = client.get_or_create_collection("prompt_manager")
    unrelated = str(uuid.uuid4())
    collection.add(ids=[unrelated], embeddings=[[0.1, 0.2]])
    before = (index / "chroma.sqlite3").read_bytes()
    deleted = _run(tmp_path, config, "draft", "delete", draft_id, "--yes", "--json")
    assert deleted.returncode == 0, deleted.stderr
    assert (index / "chroma.sqlite3").read_bytes() == before
    assert collection.get(ids=[unrelated])["ids"] == [unrelated]


def test_draft_reports_partial_on_catalog_delete_failure(tmp_path: Path) -> None:
    import chromadb

    config, db, index = _setup(tmp_path)
    draft_id = _existing(db)
    client = chromadb.PersistentClient(path=str(index))
    collection = client.get_or_create_collection("prompt_manager")
    collection.add(ids=[draft_id], embeddings=[[0.1, 0.2]])
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TRIGGER prevent_delete BEFORE DELETE ON prompts "
            "BEGIN SELECT RAISE(ABORT, 'SECRET_BODY'); END"
        )
    failed = _run(tmp_path, config, "draft", "delete", draft_id, "--yes", "--json")
    assert failed.returncode != 0 and failed.stdout == ""
    assert json.loads(failed.stderr)["error"]["code"] == "PARTIAL_DELETE"
    assert "SECRET_BODY" not in failed.stderr
    assert collection.get(ids=[draft_id])["ids"] == []
    assert PromptRepository(str(db)).get(uuid.UUID(draft_id)).context == "GUI body"


def test_draft_catalog_delete_failure_without_index_is_not_partial(tmp_path: Path) -> None:
    config, db, index = _setup(tmp_path)
    draft_id = _existing(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TRIGGER prevent_delete BEFORE DELETE ON prompts "
            "BEGIN SELECT RAISE(ABORT, 'SECRET_BODY'); END"
        )
    failed = _run(tmp_path, config, "draft", "delete", draft_id, "--yes", "--json")
    assert failed.returncode != 0 and failed.stdout == ""
    assert json.loads(failed.stderr)["error"]["code"] == "CATALOG_UNAVAILABLE"
    assert "SECRET_BODY" not in failed.stderr
    assert PromptRepository(str(db)).get(uuid.UUID(draft_id)).context == "GUI body"
    assert not index.exists()


def test_draft_delete_fails_closed_if_redis_cache_configured(tmp_path: Path) -> None:
    config, db, index = _setup(tmp_path)
    draft_id = _existing(db)
    data = json.loads(config.read_text(encoding="utf-8"))
    data["redis_dsn"] = "redis://127.0.0.1:1/0"
    config.write_text(json.dumps(data), encoding="utf-8")
    deleted = _run(tmp_path, config, "draft", "delete", draft_id, "--yes", "--json")
    assert deleted.returncode != 0 and deleted.stdout == ""
    assert json.loads(deleted.stderr)["error"]["code"] == "CACHE_CONFIGURED"
    assert "redis://" not in deleted.stderr
    assert PromptRepository(str(db)).get(uuid.UUID(draft_id)).context == "GUI body"
    assert not index.exists()


def test_draft_delete_reports_uncertain_index_mutation_as_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import chromadb

    import cli.draft as draft_cli

    _, db, index = _setup(tmp_path)
    draft_id = _existing(db)
    client = chromadb.PersistentClient(path=str(index))
    collection = client.get_or_create_collection("prompt_manager")
    collection.add(ids=[draft_id], embeddings=[[0.1, 0.2]])

    def remove_then_raise(*_args: object, **_kwargs: object) -> None:
        collection.delete(ids=[draft_id])
        raise RuntimeError("SECRET_BODY")

    class FakeClient:
        def get_collection(self, *, name: str) -> object:
            assert name == "prompt_manager"
            return type("DeletingCollection", (), {"delete": remove_then_raise})()

    def fake_client(**_kwargs: object) -> FakeClient:
        return FakeClient()

    monkeypatch.setattr(chromadb, "PersistentClient", fake_client)

    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        with pytest.raises(draft_cli.DraftError) as raised:
            draft_cli._remove(conn, index, draft_id)  # pyright: ignore[reportPrivateUsage]
    assert raised.value.code == "PARTIAL_DELETE"
    assert "SECRET_BODY" not in raised.value.message
    assert collection.get(ids=[draft_id])["ids"] == []
    assert PromptRepository(str(db)).get(uuid.UUID(draft_id)).context == "GUI body"
