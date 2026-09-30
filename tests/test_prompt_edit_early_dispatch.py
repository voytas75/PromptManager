"""Real-entrypoint parity with fail-on-import guards and disposable catalogs."""

from __future__ import annotations

import json
import os
import runpy
import sqlite3
import subprocess
import sys
from argparse import Namespace
from contextlib import closing
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from core.repository import PromptRepository
from models.prompt_model import Prompt

ROOT = Path(__file__).resolve().parents[1]
PROMPT_ID = "00000000-0000-0000-0000-000000000995"
GUARD = """
import atexit, json, os, socket, sys
from pathlib import Path
blocked = []
network = []
forbidden = ('core', 'gui', 'cli.commands', 'cli.gui_launcher', 'cli.runtime',
             'litellm', 'PySide6', 'chromadb', 'redis')
def audit(event, args):
    if event == 'import' and any(args[0] == p or args[0].startswith(p + '.')
                                 for p in forbidden):
        blocked.append(args[0])
        raise RuntimeError('FORBIDDEN_IMPORT: ' + args[0])
sys.addaudithook(audit)
def deny(*args, **kwargs):
    network.append('attempt')
    raise RuntimeError('NETWORK_FORBIDDEN')
socket.socket.connect = deny
socket.socket.connect_ex = deny
socket.create_connection = deny
@atexit.register
def receipt():
    loaded = sorted(n for n in sys.modules if any(n == p or n.startswith(p + '.')
                                                  for p in forbidden))
    Path(os.environ['QW5_RECEIPT']).write_text(json.dumps({
        'blocked': blocked, 'network': network, 'loaded': loaded}), encoding='utf-8')
"""


def _catalog(tmp_path: Path) -> tuple[Path, Path]:
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
    PromptRepository(str(db)).add(
        Prompt(
            id=UUID(PROMPT_ID),
            name="QW5 fixture",
            description="preserve description",
            category="Test",
            context="preserve body",
            related_prompts=["[]"],
        )
    )
    return db, config


def _rows(db: Path) -> tuple[str, str, str, int, int]:
    with closing(sqlite3.connect(db)) as conn:
        row = conn.execute(
            "SELECT related_prompts, description, context FROM prompts WHERE id=?",
            (PROMPT_ID,),
        ).fetchone()
        assert row is not None
        versions = conn.execute("SELECT COUNT(*) FROM prompt_versions").fetchone()[0]
        activity = conn.execute("SELECT COUNT(*) FROM prompt_activity_events").fetchone()[0]
    return (*row, versions, activity)


def _run(
    tmp_path: Path,
    config: Path,
    argv: list[str],
    *,
    module: bool,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    guard = tmp_path / "guard"
    guard.mkdir(exist_ok=True)
    (guard / "sitecustomize.py").write_text(GUARD, encoding="utf-8")
    receipt = tmp_path / "import-receipt.json"
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(
            (
                "PROMPT_MANAGER_",
                "AZURE_",
                "OPENAI_",
                "LITELLM_",
                "ANTHROPIC_",
                "GOOGLE_",
                "EXA_",
                "TAVILY_",
                "SERPER_",
                "SERPAPI_",
                "QW5_",
            )
        )
    }
    env.update(
        HOME=str(tmp_path),
        TMPDIR=str(tmp_path),
        PYTHONPATH=f"{guard}{os.pathsep}{ROOT}",
        PROMPT_MANAGER_CONFIG_JSON=str(config),
        PROMPT_MANAGER_ENV_FILE="",
        PYTHONDONTWRITEBYTECODE="1",
        LITELLM_LOCAL_MODEL_COST_MAP="True",
        CHROMA_ANONYMIZED_TELEMETRY="0",
        QW5_RECEIPT=str(receipt),
    )
    receipt.unlink(missing_ok=True)
    command = [sys.executable, "-m", "main"] if module else [str(ROOT / ".venv/bin/prompt-manager")]
    result = subprocess.run(
        [*command, *argv],
        cwd=tmp_path,
        env=env,
        input="",
        text=True,
        capture_output=True,
        timeout=40,
        check=False,
    )
    assert receipt.is_file(), result.stderr
    return result, json.loads(receipt.read_text(encoding="utf-8"))


def _args(json_mode: bool) -> list[str]:
    args = ["prompt-edit", PROMPT_ID, "set", "--attr", "related_prompts", "--value", "[]"]
    return [*args, "--json"] if json_mode else args


def _quiet_receipt(receipt: dict[str, Any]) -> None:
    assert receipt == {"blocked": [], "network": [], "loaded": []}


@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize("apply", [False, True])
def test_edit_preview_and_apply_bypass_heavy_imports(
    tmp_path: Path,
    module: bool,
    json_mode: bool,
    apply: bool,
) -> None:
    db, config = _catalog(tmp_path)
    before = _rows(db)
    image = db.read_bytes()
    backup = tmp_path / "backup.db"
    args = _args(json_mode)
    if apply:
        args.extend(["--apply", "--expect-value", '["[]"]', "--backup-to", str(backup)])
    result, receipt = _run(tmp_path, config, args, module=module)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    _quiet_receipt(receipt)
    if json_mode:
        assert json.loads(result.stdout) == {
            "command": "prompt-edit",
            "prompt_id": PROMPT_ID,
            "attr": "related_prompts",
            "before": ["[]"],
            "after": [],
            "changed": True,
            "applied": apply,
        }
    else:
        assert ("Applied:" if apply else "Preview:") in result.stdout
        assert "Before:" in result.stdout and "After:" in result.stdout
    if apply:
        assert _rows(db) == ("[]", *before[1:])
        assert _rows(backup) == before
        with closing(sqlite3.connect(backup)) as conn:
            assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    else:
        assert db.read_bytes() == image
        assert _rows(db) == before
        assert not backup.exists()
    assert not (tmp_path / "chroma").exists()


@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize(
    "case,code",
    [
        ("invalid_value", "INVALID_VALUE"),
        ("unsupported", "UNSUPPORTED_ATTRIBUTE"),
        ("missing_precondition", "MISSING_PRECONDITION"),
        ("stale", "STALE_VALUE"),
        ("missing_prompt", "PROMPT_NOT_FOUND"),
        ("missing_catalog", "CATALOG_UNAVAILABLE"),
    ],
)
def test_edit_rejections_bypass_heavy_imports_without_writing(
    tmp_path: Path,
    module: bool,
    json_mode: bool,
    case: str,
    code: str,
) -> None:
    db, config = _catalog(tmp_path)
    image = db.read_bytes()
    args = _args(json_mode)
    backup = tmp_path / "rejected.db"
    if case == "invalid_value":
        args[args.index("--value") + 1] = '"QW5_PRIVATE_VALUE"'
    elif case == "unsupported":
        args[args.index("--attr") + 1] = "context"
    elif case == "missing_precondition":
        args.append("--apply")
    elif case == "stale":
        args.extend(["--apply", "--expect-value", "[]", "--backup-to", str(backup)])
    elif case == "missing_prompt":
        args[1] = "00000000-0000-0000-0000-000000000996"
    elif case == "missing_catalog":
        db.unlink()
    result, receipt = _run(tmp_path, config, args, module=module)
    assert result.returncode == 2, result.stderr
    assert result.stdout == ""
    assert "QW5_PRIVATE_VALUE" not in result.stderr
    assert "Traceback" not in result.stderr
    if json_mode:
        payload = json.loads(result.stderr)
        assert payload["command"] == "prompt-edit" and payload["ok"] is False
        assert payload["code"] == code
    else:
        assert f"FAIL ({code})" in result.stderr
    _quiet_receipt(receipt)
    assert not backup.exists()
    assert not (tmp_path / "chroma").exists()
    if case == "missing_catalog":
        assert not db.exists()
    else:
        assert db.read_bytes() == image


@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize("leaf", [False, True])
def test_edit_help_remains_lightweight(tmp_path: Path, module: bool, leaf: bool) -> None:
    argv = ["prompt-edit", PROMPT_ID, "set", "--help"] if leaf else ["prompt-edit", "--help"]
    result, receipt = _run(tmp_path, tmp_path / "absent.json", argv, module=module)
    assert result.returncode == 0 and result.stderr == ""
    assert "usage:" in result.stdout
    _quiet_receipt(receipt)
    assert not (tmp_path / "absent.json").exists()


@pytest.mark.parametrize("module", [False, True])
def test_other_commands_still_reach_existing_runtime(tmp_path: Path, module: bool) -> None:
    result, receipt = _run(
        tmp_path, tmp_path / "absent.json", ["tag-list", "--json"], module=module
    )
    assert result.returncode != 0
    assert receipt == {"blocked": ["cli.commands"], "loaded": [], "network": []}


@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize("case", ["backup_collision", "pending_wal", "noop", "valid_reference"])
def test_guarded_edit_additional_backup_and_value_branches(
    tmp_path: Path,
    module: bool,
    json_mode: bool,
    case: str,
) -> None:
    db, config = _catalog(tmp_path)
    backup = tmp_path / "extra-backup.db"
    target_id = "00000000-0000-0000-0000-000000000997"
    if case == "valid_reference":
        PromptRepository(str(db)).add(
            Prompt(
                id=UUID(target_id),
                name="target",
                description="target",
                category="Test",
            )
        )
    if case == "backup_collision":
        backup.write_bytes(b"PRESERVE_EXISTING_BACKUP")
    args = _args(json_mode)
    if case == "valid_reference":
        args[args.index("--value") + 1] = json.dumps([target_id])
    with closing(sqlite3.connect(db)) as writer:
        if case == "noop":
            writer.execute("UPDATE prompts SET related_prompts='[]' WHERE id=?", (PROMPT_ID,))
            writer.commit()
            writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        elif case == "pending_wal":
            writer.execute("PRAGMA journal_mode=WAL")
            writer.execute("PRAGMA wal_autocheckpoint=0")
            writer.execute("UPDATE prompts SET description='pending' WHERE id=?", (PROMPT_ID,))
            writer.commit()
            assert Path(f"{db}-wal").stat().st_size > 0
        before = _rows(db)
        image = db.read_bytes()
        wal = Path(f"{db}-wal")
        wal_image = wal.read_bytes() if wal.exists() else None
        expected = "[]" if case == "noop" else '["[]"]'
        args.extend(["--apply", "--expect-value", expected, "--backup-to", str(backup)])
        result, receipt = _run(tmp_path, config, args, module=module)
        _quiet_receipt(receipt)
        if case in {"backup_collision", "pending_wal"}:
            code = "BACKUP_UNAVAILABLE" if case == "backup_collision" else "CATALOG_BUSY"
            assert result.returncode == 2 and result.stdout == ""
            if json_mode:
                assert json.loads(result.stderr)["code"] == code
            else:
                assert f"FAIL ({code})" in result.stderr
        else:
            assert result.returncode == 0 and result.stderr == ""
            changed = case == "valid_reference"
            if json_mode:
                payload = json.loads(result.stdout)
                assert payload["changed"] is changed and payload["applied"] is changed
            else:
                assert ("Applied:" if changed else "No change:") in result.stdout
        if case == "valid_reference":
            assert _rows(db) == (json.dumps([target_id]), *before[1:])
            assert _rows(backup) == before
        else:
            assert _rows(db) == before and db.read_bytes() == image
            if wal_image is not None:
                assert wal.read_bytes() == wal_image
            if case == "backup_collision":
                assert backup.read_bytes() == b"PRESERVE_EXISTING_BACKUP"
            else:
                assert not backup.exists()
    assert not (tmp_path / "chroma").exists()


@pytest.mark.parametrize("outcome", [0, 2])
def test_module_early_dispatch_passes_args_and_exit_exactly(
    monkeypatch: pytest.MonkeyPatch,
    outcome: int,
) -> None:
    import cli.parser
    import cli.prompt_edit

    args = Namespace(command="prompt-edit")
    calls: list[Namespace] = []

    def handler(parsed: Namespace) -> int:
        calls.append(parsed)
        return outcome

    monkeypatch.setattr(cli.parser, "parse_args", lambda: args)
    monkeypatch.setattr(cli.prompt_edit, "run_prompt_edit", handler)
    with pytest.raises(SystemExit) as raised:
        runpy.run_module("main", run_name="__main__")
    assert raised.value.code == outcome
    assert calls == [args]
