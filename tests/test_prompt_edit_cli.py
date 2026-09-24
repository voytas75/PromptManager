"""Real-process, isolated checks for guarded prompt metadata editing."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from uuid import UUID, uuid4

from core.repository import PromptRepository
from models.prompt_model import Prompt

ROOT = Path(__file__).resolve().parents[1]


def _fixture(tmp_path: Path, *, related: list[str] | None = None) -> tuple[Path, Path, str]:
    db = tmp_path / "catalog.db"
    config = tmp_path / "settings.json"
    config.write_text(
        json.dumps(
            {
                "database_path": str(db),
                "chroma_path": str(tmp_path / "chroma"),
                "embedding_backend": "deterministic",
                "redis_dsn": None,
                "litellm_model": None,
                "litellm_inference_model": None,
            }
        ),
        encoding="utf-8",
    )
    prompt_id = str(uuid4())
    PromptRepository(str(db)).add(
        Prompt(
            id=UUID(prompt_id),
            name="private-fixture",
            description="keep me",
            category="Test",
            context="safe body",
            related_prompts=related if related is not None else ["[]"],
        )
    )
    return config, db, prompt_id


def _run(
    tmp_path: Path, config: Path, *args: str, module: bool = False
) -> subprocess.CompletedProcess[str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not (
            key.startswith(("PROMPT_MANAGER_", "AZURE_OPENAI_"))
            or key in {"LITELLM_API_KEY", "OPENAI_API_KEY"}
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
        input="",
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )


def _stored(db: Path, prompt_id: str) -> tuple[str, str, str]:
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT related_prompts, description, context FROM prompts WHERE id=?", (prompt_id,)
        ).fetchone()
    assert row is not None
    return row


def test_prompt_edit_help_and_preview_are_read_only(tmp_path: Path) -> None:
    config, db, prompt_id = _fixture(tmp_path)
    before = db.read_bytes()
    help_result = _run(tmp_path, config, "prompt-edit", "--help")
    leaf_help = _run(tmp_path, config, "prompt-edit", prompt_id, "set", "--help", module=True)
    assert help_result.returncode == leaf_help.returncode == 0
    assert help_result.stderr == leaf_help.stderr == ""
    for flag in ("--attr", "--value", "--expect-value", "--apply", "--backup-to", "--json"):
        assert flag in leaf_help.stdout
    preview = _run(
        tmp_path,
        config,
        "prompt-edit",
        prompt_id,
        "set",
        "--attr",
        "related_prompts",
        "--value",
        "[]",
        "--json",
    )
    assert preview.returncode == 0, preview.stderr
    assert preview.stderr == ""
    report = json.loads(preview.stdout)
    assert report["before"] == ["[]"] and report["after"] == []
    assert report["changed"] is True and report["applied"] is False
    human = _run(
        tmp_path,
        config,
        "prompt-edit",
        prompt_id,
        "set",
        "--attr",
        "related_prompts",
        "--value",
        "[]",
    )
    assert human.returncode == 0 and "Preview: related_prompts" in human.stdout
    assert "Before:" in human.stdout and "After:" in human.stdout
    assert human.stderr == ""
    assert db.read_bytes() == before
    assert not (tmp_path / "chroma").exists()


def test_prompt_edit_apply_backup_readback_and_doctor(tmp_path: Path) -> None:
    config, db, prompt_id = _fixture(tmp_path)
    backup = tmp_path / "backup.db"
    with sqlite3.connect(db) as conn:
        before_version_count = conn.execute("SELECT COUNT(*) FROM prompt_versions").fetchone()[0]
        before_activity_count = conn.execute(
            "SELECT COUNT(*) FROM prompt_activity_events"
        ).fetchone()[0]
    args = (
        "prompt-edit",
        prompt_id,
        "set",
        "--attr",
        "related_prompts",
        "--value",
        "[]",
        "--expect-value",
        '["[]"]',
        "--apply",
        "--backup-to",
        str(backup),
        "--json",
    )
    applied = _run(tmp_path, config, *args)
    assert applied.returncode == 0, applied.stderr
    assert applied.stderr == ""
    report = json.loads(applied.stdout)
    assert report["changed"] is report["applied"] is True
    assert report["before"] == ["[]"] and report["after"] == []
    assert _stored(db, prompt_id) == ("[]", "keep me", "safe body")
    with sqlite3.connect(db) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM prompt_versions").fetchone()[0]
            == before_version_count
        )
        assert (
            conn.execute("SELECT COUNT(*) FROM prompt_activity_events").fetchone()[0]
            == before_activity_count
        )
    assert _stored(backup, prompt_id)[0] == '["[]"]'
    with sqlite3.connect(backup) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    # sqlite3.Connection context commits but does not close: close/checkpoint readers
    # before doctor opens its immutable, sidecar-free snapshot.
    with closing(sqlite3.connect(db)) as checkpoint:
        checkpoint.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    doctor = _run(tmp_path, config, "doctor", "catalog", "--json")
    assert doctor.returncode == 0, doctor.stdout
    assert all(issue["code"] != "CAT004" for issue in json.loads(doctor.stdout)["report"]["issues"])
    assert not (tmp_path / "chroma").exists()
    shown = _run(tmp_path, config, "prompt-show", prompt_id, "--json")
    assert shown.returncode == 0 and json.loads(shown.stdout)["related_prompts"] == []


def test_prompt_edit_rejects_stale_and_invalid_without_mutation(tmp_path: Path) -> None:
    config, db, prompt_id = _fixture(tmp_path)
    before = db.read_bytes()
    base = ("prompt-edit", prompt_id, "set", "--attr", "related_prompts", "--value", "[]")
    for extra in (
        ("--apply",),
        ("--apply", "--expect-value", "[]", "--backup-to", str(tmp_path / "stale.db")),
        ("--value", '["[]"]'),
        ("--value", '"[]"'),
        ("--attr", "context"),
        ("--value", json.dumps([str(uuid4())])),
    ):
        result = _run(tmp_path, config, *base, *extra, "--json")
        assert result.returncode != 0
        assert result.stdout == ""
        assert json.loads(result.stderr)["ok"] is False
        assert db.read_bytes() == before
    assert not (tmp_path / "chroma").exists()


def test_prompt_edit_valid_reference_noop_and_existing_backup(tmp_path: Path) -> None:
    config, db, source_id = _fixture(tmp_path)
    target_id = str(uuid4())
    PromptRepository(str(db)).add(
        Prompt(id=UUID(target_id), name="target", description="target", category="Test")
    )
    backup = tmp_path / "original.db"
    edit = (
        "prompt-edit",
        source_id,
        "set",
        "--attr",
        "related_prompts",
        "--value",
        json.dumps([target_id]),
        "--expect-value",
        '["[]"]',
        "--apply",
        "--backup-to",
        str(backup),
        "--json",
    )
    applied = _run(tmp_path, config, *edit)
    assert applied.returncode == 0, applied.stderr
    assert _stored(db, source_id)[0] == json.dumps([target_id])
    assert _stored(backup, source_id)[0] == '["[]"]'
    original_backup = backup.read_bytes()
    again = _run(tmp_path, config, *edit)
    assert again.returncode != 0
    assert json.loads(again.stderr)["code"] == "STALE_VALUE"
    assert backup.read_bytes() == original_backup
    noop = _run(
        tmp_path,
        config,
        "prompt-edit",
        source_id,
        "set",
        "--attr",
        "related_prompts",
        "--value",
        json.dumps([target_id]),
        "--expect-value",
        json.dumps([target_id]),
        "--apply",
        "--backup-to",
        str(tmp_path / "unused.db"),
        "--json",
    )
    assert noop.returncode == 0, noop.stderr
    assert json.loads(noop.stdout)["applied"] is False
    assert not (tmp_path / "unused.db").exists()


def test_prompt_edit_missing_catalog_id_and_backup_collision(tmp_path: Path) -> None:
    config, db, prompt_id = _fixture(tmp_path)
    before = db.read_bytes()
    base = (
        "prompt-edit",
        prompt_id,
        "set",
        "--attr",
        "related_prompts",
        "--value",
        "[]",
        "--expect-value",
        '["[]"]',
        "--apply",
        "--json",
    )
    collision = tmp_path / "reserved.db"
    collision.write_text("leave me alone", encoding="utf-8")
    for args, code in (
        ((*base, "--backup-to", str(collision)), "BACKUP_UNAVAILABLE"),
        ((*base, "--backup-to", str(db)), "INVALID_BACKUP"),
        (
            (*base[:1], str(uuid4()), *base[2:], "--backup-to", str(tmp_path / "none.db")),
            "PROMPT_NOT_FOUND",
        ),
        (
            (
                "prompt-edit",
                "not-a-uuid",
                "set",
                "--attr",
                "related_prompts",
                "--value",
                "[]",
                "--json",
            ),
            "INVALID_PROMPT_ID",
        ),
    ):
        result = _run(tmp_path, config, *args)
        assert result.returncode == 2
        assert result.stdout == ""
        assert json.loads(result.stderr)["code"] == code
        assert db.read_bytes() == before
    assert collision.read_text(encoding="utf-8") == "leave me alone"
    assert not (tmp_path / "none.db").exists()
    db.unlink()
    missing = _run(
        tmp_path,
        config,
        "prompt-edit",
        prompt_id,
        "set",
        "--attr",
        "related_prompts",
        "--value",
        "[]",
        "--json",
    )
    assert missing.returncode == 2
    assert json.loads(missing.stderr)["code"] == "CATALOG_UNAVAILABLE"
    assert not db.exists()


def test_prompt_edit_rejects_pending_wal_before_apply(tmp_path: Path) -> None:
    config, db, prompt_id = _fixture(tmp_path)
    before = _stored(db, prompt_id)
    backup = tmp_path / "backup.db"
    with sqlite3.connect(db) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute(
            "UPDATE prompts SET description = ? WHERE id = ?", ("uncheckpointed", prompt_id)
        )
        writer.commit()
        sidecar = Path(f"{db}-wal")
        assert sidecar.exists() and sidecar.stat().st_size > 0
        result = _run(
            tmp_path,
            config,
            "prompt-edit",
            prompt_id,
            "set",
            "--attr",
            "related_prompts",
            "--value",
            "[]",
            "--expect-value",
            '["[]"]',
            "--apply",
            "--backup-to",
            str(backup),
            "--json",
        )
        assert result.returncode == 2
        assert json.loads(result.stderr)["code"] == "CATALOG_BUSY"
        assert result.stdout == ""
        assert not backup.exists()
        assert _stored(db, prompt_id)[0] == before[0]
