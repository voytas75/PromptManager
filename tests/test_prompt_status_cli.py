"""Provider-free process and persistence contracts for prompt status transitions."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, cast

import pytest

from core.repository import PromptRepository
from models.prompt_model import Prompt

ROOT = Path(__file__).resolve().parents[1]


def _invoke(
    tmp_path: Path, config: Path, *args: str, module: bool
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
    )
    executable = (
        [sys.executable, "-m", "main"] if module else [str(ROOT / ".venv/bin/prompt-manager")]
    )
    return subprocess.run(
        [*executable, "prompt-status", *args],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=35,
        check=False,
    )


@pytest.mark.parametrize("module", [False, True])
def test_status_cli_cas_preserves_asset_and_relations(tmp_path: Path, module: bool) -> None:
    db = tmp_path / "catalog.db"
    config = tmp_path / "settings.json"
    config.write_text(
        json.dumps(
            {
                "database_path": str(db),
                "chroma_path": str(tmp_path / "chroma"),
                "embedding_backend": "deterministic",
                "redis_dsn": None,
            }
        ),
        encoding="utf-8",
    )
    repo = PromptRepository(str(db))
    parent = Prompt(
        id=uuid.uuid4(),
        name="Parent",
        description="desc",
        category="General",
        context="PRIVATE_BODY",
    )
    child = Prompt(
        id=uuid.uuid4(),
        name="Child",
        description="desc",
        category="General",
        context="child",
        related_prompts=[str(parent.id)],
    )
    repo.add(parent)
    repo.add(child)
    initial = repo.get(parent.id)
    before_versions = repo.get_prompt_latest_version(parent.id)

    def call(*args: str) -> subprocess.CompletedProcess[str]:
        return _invoke(tmp_path, config, str(parent.id), *args, module=module)

    changed = call("deactivate", "--expect-active", "true", "--json")
    assert changed.returncode == 0, changed.stderr
    assert changed.stderr == ""
    assert json.loads(changed.stdout)["active"] is False
    stored = repo.get(parent.id)
    assert stored.is_active is False and stored.context == initial.context
    assert stored.last_modified >= initial.last_modified
    assert repo.get(child.id).related_prompts == [str(parent.id)]
    assert repo.get_prompt_latest_version(parent.id) == before_versions
    assert not (tmp_path / "chroma").exists()
    same = call("deactivate", "--expect-active", "false", "--json")
    assert same.returncode == 0 and json.loads(same.stdout)["changed"] is False
    assert repo.get(parent.id).last_modified == stored.last_modified
    stale = call("activate", "--expect-active", "true", "--json")
    assert stale.returncode != 0 and stale.stdout == ""
    assert json.loads(stale.stderr)["error"]["code"] == "STATUS_CONFLICT"
    assert "PRIVATE_BODY" not in stale.stderr
    assert repo.get(parent.id).is_active is False
    active = call("activate", "--expect-active", "false", "--json")
    assert active.returncode == 0 and json.loads(active.stdout)["active"] is True
    assert repo.get(parent.id).id == parent.id


@pytest.mark.parametrize("module", [False, True])
def test_status_cli_rejects_missing_catalog_and_invalid_id_without_creation(
    tmp_path: Path, module: bool
) -> None:
    db = tmp_path / "missing.db"
    config = tmp_path / "settings.json"
    config.write_text(json.dumps({"database_path": str(db)}), encoding="utf-8")
    for identifier in (str(uuid.uuid4()), "NOT_A_UUID"):
        result = _invoke(
            tmp_path,
            config,
            identifier,
            "deactivate",
            "--expect-active",
            "true",
            "--json",
            module=module,
        )
        assert result.returncode != 0 and result.stdout == ""
        assert json.loads(result.stderr)["ok"] is False
        assert not db.exists()


def test_manager_status_transition_evicts_cache_without_embedding(tmp_path: Path) -> None:
    from core.prompt_manager import PromptManager

    db = tmp_path / "catalog.db"
    repo = PromptRepository(str(db))
    item = Prompt(
        id=uuid.uuid4(), name="Cache test", description="desc", category="General", context="Body"
    )
    repo.add(item)
    manager = PromptManager(
        chroma_path=str(tmp_path / "chroma"),
        db_path=str(db),
        repository=repo,
    )
    evicted: list[uuid.UUID] = []
    try:
        manager._evict_cached_prompt = evicted.append  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]
        changed, _ = manager.set_prompt_active(item.id, active=False, expect_active=True)
        assert changed is True
        assert evicted == [item.id]
        assert repo.get(item.id).is_active is False
        assert repo.get_prompt_latest_version(item.id) is None
        assert cast("Any", manager.collection).get(ids=[str(item.id)])["ids"] == []
    finally:
        manager.close()


@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize(
    "extra", [("--expect-active", "SECRET_VALUE"), ("--expect-active", "true", "PRIVATE_EXTRA")]
)
def test_status_cli_parse_errors_are_bounded_json(
    tmp_path: Path, module: bool, extra: tuple[str, ...]
) -> None:
    config = tmp_path / "settings.json"
    config.write_text(json.dumps({"database_path": str(tmp_path / "missing.db")}), encoding="utf-8")
    result = _invoke(
        tmp_path, config, str(uuid.uuid4()), "deactivate", *extra, "--json", module=module
    )
    assert result.returncode == 2 and result.stdout == ""
    assert json.loads(result.stderr)["error"]["code"] == "INVALID_USAGE"
    assert "SECRET_VALUE" not in result.stderr and "PRIVATE_EXTRA" not in result.stderr


@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize("stamp", ["NOT_A_DATE", "9999-12-31T23:59:59.999999+00:00"])
def test_status_cli_rejects_bad_or_unrepresentable_timestamp(
    tmp_path: Path, module: bool, stamp: str
) -> None:
    db = tmp_path / "catalog.db"
    config = tmp_path / "settings.json"
    config.write_text(json.dumps({"database_path": str(db)}), encoding="utf-8")
    repo = PromptRepository(str(db))
    item = Prompt(
        id=uuid.uuid4(), name="Timestamp", description="desc", category="General", context="BODY"
    )
    repo.add(item)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE prompts SET last_modified=? WHERE id=?", (stamp, str(item.id)))
    for action, expected in (("deactivate", "true"), ("activate", "true")):
        result = _invoke(
            tmp_path,
            config,
            str(item.id),
            action,
            "--expect-active",
            expected,
            "--json",
            module=module,
        )
        assert result.returncode != 0 and result.stdout == ""
        assert json.loads(result.stderr)["error"]["code"] == "CATALOG_INVALID"
        assert "Traceback" not in result.stderr and "BODY" not in result.stderr
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT is_active, last_modified FROM prompts WHERE id=?", (str(item.id),)
        ).fetchone() == (1, stamp)


@pytest.mark.parametrize("module", [False, True])
def test_status_cli_rejects_incomplete_schema_without_modification(
    tmp_path: Path, module: bool
) -> None:
    db = tmp_path / "incomplete.db"
    config = tmp_path / "settings.json"
    config.write_text(json.dumps({"database_path": str(db)}), encoding="utf-8")
    identifier = str(uuid.uuid4())
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE prompts(id TEXT PRIMARY KEY, is_active INTEGER, last_modified TEXT)"
        )
        conn.execute(
            "INSERT INTO prompts VALUES (?, 1, ?)", (identifier, "2026-09-27T00:00:00+00:00")
        )
    result = _invoke(
        tmp_path,
        config,
        identifier,
        "deactivate",
        "--expect-active",
        "true",
        "--json",
        module=module,
    )
    assert result.returncode != 0 and result.stdout == ""
    assert json.loads(result.stderr)["error"]["code"] == "CATALOG_INVALID"
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT is_active FROM prompts WHERE id=?", (identifier,)
        ).fetchone() == (1,)


@pytest.mark.parametrize("module", [False, True])
def test_status_cli_rejects_lookalike_schema_with_lifecycle_tables(
    tmp_path: Path, module: bool
) -> None:
    db = tmp_path / "lookalike.db"
    config = tmp_path / "settings.json"
    config.write_text(json.dumps({"database_path": str(db)}), encoding="utf-8")
    identifier = str(uuid.uuid4())
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE prompts(id TEXT PRIMARY KEY, name TEXT, context TEXT, "
            "last_modified TEXT, is_active INTEGER, related_prompts TEXT)"
        )
        for table in ("prompt_versions", "prompt_forks", "prompt_chain_steps"):
            conn.execute(f"CREATE TABLE {table} (id TEXT)")
        conn.execute(
            "INSERT INTO prompts VALUES (?, 'looks valid', 'SECRET_BODY', ?, 1, '[]')",
            (identifier, "2026-09-27T00:00:00+00:00"),
        )
    result = _invoke(
        tmp_path,
        config,
        identifier,
        "deactivate",
        "--expect-active",
        "true",
        "--json",
        module=module,
    )
    assert result.returncode != 0 and result.stdout == ""
    assert json.loads(result.stderr)["error"]["code"] == "CATALOG_INVALID"
    assert "SECRET_BODY" not in result.stderr
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT is_active FROM prompts WHERE id=?", (identifier,)
        ).fetchone() == (1,)
