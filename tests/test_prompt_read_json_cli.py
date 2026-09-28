"""Real-process JSON outcome checks for selected asset-read commands."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import subprocess
import sys
import uuid
from argparse import Namespace
from contextlib import closing
from pathlib import Path
from typing import Any, cast

import pytest

from cli.commands import run_catalog_import, run_prompt_history
from core.catalog_importer import CatalogImportResult
from core.repository import PromptRepository
from models.prompt_model import Prompt

ROOT = Path(__file__).resolve().parents[1]


def _run(
    tmp_path: Path, config: Path, installed: bool, *args: str, stdin_text: str | None = None
) -> subprocess.CompletedProcess[str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PROMPT_MANAGER_", "AZURE_", "OPENAI_", "LITELLM_"))
        and key not in {"OPENAI_API_KEY", "LITELLM_API_KEY"}
    }
    env.update(
        HOME=str(tmp_path),
        TMPDIR=str(tmp_path),
        PYTHONPATH=str(ROOT),
        PROMPT_MANAGER_CONFIG_JSON=str(config),
        PROMPT_MANAGER_ENV_FILE="",
        PYTHONDONTWRITEBYTECODE="1",
        CHROMA_ANONYMIZED_TELEMETRY="0",
    )
    front = (
        [str(ROOT / ".venv/bin/prompt-manager")] if installed else [sys.executable, "-m", "main"]
    )
    return subprocess.run(
        [*front, *args],
        cwd=tmp_path,
        env=env,
        stdin=subprocess.DEVNULL if stdin_text is None else None,
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=35,
        check=False,
    )


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize(
    ("argv", "private_marker"),
    [
        (["prompt-add", "--json", "{PRIVATE_JSON_123", "--result-json"], "PRIVATE_JSON_123"),
        (
            ["prompt-add", "--json", "{PRIVATE_JSON_123", "--from-stdin", "--result-json"],
            "PRIVATE_JSON_123",
        ),
        (["prompt-list", "--json", "--limit", "PRIVATE_LIMIT_123"], "PRIVATE_LIMIT_123"),
        (["--bogus", "prompt-list", "--json", "--limit", "PRIVATE_LIMIT_123"], "PRIVATE_LIMIT_123"),
        (["-x", "prompt-list", "--json", "--limit", "PRIVATE_LIMIT_123"], "PRIVATE_LIMIT_123"),
        (
            ["--bogus=value", "prompt-list", "--j", "--limit", "PRIVATE_LIMIT_123"],
            "PRIVATE_LIMIT_123",
        ),
        (["--", "prompt-list", "--json", "--limit", "PRIVATE_LIMIT_123"], "PRIVATE_LIMIT_123"),
        (["prompt-list", "--json", "--", "PRIVATE_LIMIT_123"], "PRIVATE_LIMIT_123"),
        (
            ["prompt-add", "--result-json", "--", "PRIVATE_INPUT_123", "EXTRA"],
            "PRIVATE_INPUT_123",
        ),
        (
            [
                "--logging-conf",
                "unused.ini",
                "prompt-list",
                "--json",
                "--limit",
                "PRIVATE_LIMIT_123",
            ],
            "PRIVATE_LIMIT_123",
        ),
        (
            ["--logging-conf=unused.ini", "prompt-list", "--j", "--limit", "PRIVATE_LIMIT_123"],
            "PRIVATE_LIMIT_123",
        ),
        (["prompt-list", "--j", "--limit", "PRIVATE_LIMIT_123"], "PRIVATE_LIMIT_123"),
        (
            ["prompt-add", "--json", "{PRIVATE_JSON_123", "--result-j"],
            "PRIVATE_JSON_123",
        ),
        (["prompt-find", "query", "--json", "--limit", "PRIVATE_LIMIT_123"], "PRIVATE_LIMIT_123"),
        (["prompt-find", "query", "--j", "--limit", "PRIVATE_LIMIT_123"], "PRIVATE_LIMIT_123"),
        (["prompt-show", "--j"], "PRIVATE_ID_123"),
        (["prompt-history", "--j"], "PRIVATE_ID_123"),
        (["prompt-show", "--json"], "PRIVATE_ID_123"),
        (["prompt-history", "--json"], "PRIVATE_ID_123"),
    ],
)
def test_agent_json_parser_errors_are_bounded(
    tmp_path: Path, installed: bool, argv: list[str], private_marker: str
) -> None:
    result = _run(tmp_path, tmp_path / "unused.json", installed, *argv)
    assert result.returncode == 2
    assert result.stdout == ""
    assert json.loads(result.stderr) == {
        "ok": False,
        "error": {"code": "INVALID_USAGE", "message": "Invalid command or arguments."},
    }
    assert private_marker not in result.stderr
    assert not (tmp_path / "catalog.db").exists()


@pytest.mark.parametrize("installed", [False, True])
def test_agent_json_parser_controls_keep_legacy_text_and_help(
    tmp_path: Path, installed: bool
) -> None:
    config = tmp_path / "unused.json"
    for argv in (
        ("prompt-add", "--json", "{PRIVATE_JSON_123"),
        ("prompt-add", "--j", "{PRIVATE_JSON_123"),
        ("prompt-add", "--", "--result-json", "INVALID_POSITIONAL"),
        ("prompt-add", "--", "--result-j", "INVALID_POSITIONAL"),
        ("prompt-show", "--", "--json", "INVALID_POSITIONAL"),
        ("prompt-show", "--", "--j", "INVALID_POSITIONAL"),
        ("prompt-list", "--limit", "PRIVATE_LIMIT_123"),
        ("--bogus", "prompt-list", "--limit", "PRIVATE_LIMIT_123"),
    ):
        result = _run(tmp_path, config, installed, *argv)
        assert result.returncode == 2 and result.stdout == ""
        assert "usage:" in result.stderr
        with pytest.raises(json.JSONDecodeError):
            json.loads(result.stderr)
    help_result = _run(tmp_path, config, installed, "prompt-add", "--help")
    assert help_result.returncode == 0
    assert "--result-json" in help_result.stdout and help_result.stderr == ""


@pytest.mark.parametrize("installed", [False, True])
def test_agent_json_parser_does_not_select_command_from_global_value(
    tmp_path: Path, installed: bool
) -> None:
    result = _run(
        tmp_path,
        tmp_path / "unused.json",
        installed,
        "--logging-config",
        "prompt-add",
        "prompt-list",
        "--limit",
        "PRIVATE_LIMIT_123",
        "--json",
    )
    assert result.returncode == 2 and result.stdout == ""
    assert json.loads(result.stderr)["error"]["code"] == "INVALID_USAGE"
    assert "PRIVATE_LIMIT_123" not in result.stderr

    abbreviated = _run(
        tmp_path,
        tmp_path / "unused.json",
        installed,
        "--logging-conf",
        "prompt-add",
        "prompt-list",
        "--j",
        "--limit",
        "PRIVATE_LIMIT_123",
    )
    assert abbreviated.returncode == 2 and abbreviated.stdout == ""
    assert json.loads(abbreviated.stderr)["error"]["code"] == "INVALID_USAGE"
    assert "PRIVATE_LIMIT_123" not in abbreviated.stderr


@pytest.mark.parametrize("installed", [False, True])
def test_prompt_find_explain_requires_json_before_startup(tmp_path: Path, installed: bool) -> None:
    result = _run(
        tmp_path, tmp_path / "unused.json", installed, "prompt-find", "query", "--explain"
    )
    assert result.returncode == 2 and result.stdout == ""
    assert "--explain requires --json" in result.stderr
    assert not (tmp_path / "catalog.db").exists()


@pytest.mark.parametrize("installed", [False, True])
def test_prompt_find_explain_real_process_qualifies_empty_candidate_set(
    tmp_path: Path, installed: bool
) -> None:
    config = tmp_path / "settings.json"
    config.write_text(
        json.dumps(
            {
                "database_path": str(tmp_path / "catalog.db"),
                "chroma_path": str(tmp_path / "chroma"),
                "embedding_backend": "deterministic",
                "redis_dsn": None,
                "litellm_model": None,
                "litellm_inference_model": None,
            }
        ),
        encoding="utf-8",
    )
    repository = PromptRepository(str(tmp_path / "catalog.db"))
    stored = Prompt(
        id=uuid.uuid4(),
        name="Triage",
        description="Triage",
        category="Operations",
        context="Diagnose workflows",
        source="local",
    )
    repository.add(stored)
    result = _run(
        tmp_path,
        config,
        installed,
        "prompt-find",
        "workflow triage",
        "--limit",
        "1",
        "--category",
        "Writing",
        "--json",
        "--explain",
    )
    assert result.returncode == 0 and result.stderr == "", result
    payload = json.loads(result.stdout)
    assert payload == {
        "results": [],
        "retrieval": {
            "scope": "retrieved_candidates",
            "requested_limit": 1,
            "retrieved_count": 0,
            "matched_count": 0,
        },
    }
    assert repository.get(stored.id).category == "Operations"


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize(
    "machine_args",
    [
        ("prompt-list", "--json"),
        ("prompt-find", "query", "--json"),
        ("prompt-show", "missing", "--json"),
        ("prompt-history", "missing", "--json"),
        (
            "prompt-add",
            "--name",
            "Test",
            "--description",
            "Test",
            "--prompt-text",
            "Body",
            "--result-json",
            "--dry-run",
        ),
    ],
)
@pytest.mark.parametrize(
    ("failure_kind", "exit_code", "error_code"),
    [("config", 2, "CONFIG_UNAVAILABLE"), ("manager", 3, "MANAGER_INIT_FAILED")],
)
def test_agent_json_startup_errors_are_bounded(
    tmp_path: Path,
    installed: bool,
    machine_args: tuple[str, ...],
    failure_kind: str,
    exit_code: int,
    error_code: str,
) -> None:
    config = tmp_path / "bad.json"
    if failure_kind == "config":
        config.write_text("{PRIVATE_CONFIG_MARKER", encoding="utf-8")
    else:
        db_directory = tmp_path / "catalog-directory"
        db_directory.mkdir()
        config.write_text(
            json.dumps(
                {
                    "database_path": str(db_directory),
                    "chroma_path": str(tmp_path / "chroma"),
                    "embedding_backend": "deterministic",
                    "redis_dsn": None,
                    "litellm_model": None,
                    "litellm_inference_model": None,
                }
            ),
            encoding="utf-8",
        )
    result = _run(tmp_path, config, installed, *machine_args)
    assert result.returncode == exit_code and result.stdout == "", result
    assert json.loads(result.stderr) == {
        "ok": False,
        "error": {
            "code": error_code,
            "message": (
                "Unable to load configuration."
                if failure_kind == "config"
                else "Unable to initialise services."
            ),
        },
    }
    assert "PRIVATE_CONFIG_MARKER" not in result.stderr
    assert str(tmp_path) not in result.stderr


@pytest.mark.parametrize("installed", [False, True])
def test_agent_json_startup_legacy_error_is_still_text(tmp_path: Path, installed: bool) -> None:
    config = tmp_path / "bad.json"
    config.write_text("{bad", encoding="utf-8")
    result = _run(tmp_path, config, installed, "prompt-list")
    assert result.returncode == 2 and result.stdout == ""
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.stderr)


@pytest.mark.parametrize("installed", [False, True])
def test_agent_json_startup_missing_default_config_never_prompts_or_creates(
    tmp_path: Path, installed: bool
) -> None:
    config = Path("config/config.json")
    result = _run(tmp_path, config, installed, "prompt-list", "--json")
    assert result.returncode == 2 and result.stdout == "", result
    assert json.loads(result.stderr) == {
        "ok": False,
        "error": {
            "code": "CONFIG_UNAVAILABLE",
            "message": "Unable to load configuration.",
        },
    }
    assert not (tmp_path / config).exists()


@pytest.mark.parametrize("installed", [False, True])
def test_prompt_read_json_process_outcomes(tmp_path: Path, installed: bool) -> None:
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
    repository = PromptRepository(str(db))
    missing = str(uuid.uuid4())
    show_missing = _run(tmp_path, config, installed, "prompt-show", missing, "--json")
    assert show_missing.returncode == 4 and show_missing.stdout == "", show_missing
    assert json.loads(show_missing.stderr)["error"]["code"] == "PROMPT_NOT_FOUND"

    empty = _run(tmp_path, config, installed, "prompt-find", "unmatched", "--json")
    assert empty.returncode == 0 and json.loads(empty.stdout) == [], empty
    assert empty.stderr == ""

    invalid = _run(
        tmp_path, config, installed, "prompt-find", "query", "--active", "invalid", "--json"
    )
    assert invalid.returncode == 5 and invalid.stdout == "", invalid
    assert json.loads(invalid.stderr)["error"]["code"] == "INVALID_ACTIVE"

    text = _run(tmp_path, config, installed, "prompt-show", missing)
    assert text.returncode == 4 and "Prompt not found" in text.stdout

    prompt = Prompt(
        id=uuid.uuid4(),
        name="Readback Prompt",
        description="Local readback.",
        category="Testing",
        context="Prompt body",
        source="local",
    )
    repository.add(prompt)
    shown = _run(tmp_path, config, installed, "prompt-show", str(prompt.id), "--json")
    assert shown.returncode == 0 and json.loads(shown.stdout)["id"] == str(prompt.id), shown
    assert shown.stderr == ""
    assert repository.get(prompt.id).context == "Prompt body"


@pytest.mark.parametrize("installed", [False, True])
def test_prompt_history_json_process_channels(tmp_path: Path, installed: bool) -> None:
    """History machine results and failures have unambiguous process channels."""
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
                "web_search_provider": None,
            }
        ),
        encoding="utf-8",
    )
    repository = PromptRepository(str(db))
    prompt = Prompt(
        id=uuid.uuid4(), name="History probe", description="Local test", category="Test"
    )
    repository.add(prompt)
    cases: list[tuple[str, list[str], int, str | None]] = [
        (str(prompt.id), [], 0, None),
        (str(uuid.uuid4()), [], 4, "PROMPT_NOT_FOUND"),
        (str(prompt.id), ["--status", "private-status"], 5, "INVALID_STATUS"),
    ]
    for identifier, extra, expected_code, expected_error in cases:
        result = _run(tmp_path, config, installed, "prompt-history", identifier, *extra, "--json")
        assert result.returncode == expected_code, result
        if expected_error is None:
            payload = json.loads(result.stdout)
            assert payload["prompt"]["id"] == identifier
            assert set(payload) == {"prompt", "analytics", "executions"}
            assert result.stderr == ""
        else:
            assert result.stdout == ""
            assert json.loads(result.stderr) == {
                "ok": False,
                "error": {
                    "code": expected_error,
                    "message": (
                        "Prompt not found."
                        if expected_code == 4
                        else "Use success or failed for --status."
                    ),
                },
            }
            assert "private-status" not in result.stderr


def test_prompt_history_json_read_failure_redacts_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Injected history storage exceptions do not leak raw provider or prompt data."""

    class FailingHistory:
        def __init__(self) -> None:
            self.repository = self

        def get(self, _id: uuid.UUID) -> Prompt:
            return prompt

        def list_executions_for_prompt(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("PRIVATE_HISTORY_DETAIL")

    prompt = Prompt(
        id=uuid.uuid4(), name="History probe", description="Local test", category="Test"
    )
    manager = FailingHistory()
    args = Namespace(prompt_id=str(prompt.id), limit=5, status=None, window_days=0, json=True)
    assert (
        run_prompt_history(cast("Any", manager), args, logging.getLogger("test_history_fail")) == 7
    )
    channels = capsys.readouterr()
    assert channels.out == ""
    assert json.loads(channels.err) == {
        "ok": False,
        "error": {"code": "HISTORY_LOAD_FAILED", "message": "Unable to load prompt history."},
    }
    assert "PRIVATE_HISTORY_DETAIL" not in channels.err


@pytest.mark.parametrize("installed", [False, True])
def test_prompt_add_result_json_process_contract(tmp_path: Path, installed: bool) -> None:
    """Preview and apply keep legacy input JSON but emit a machine receipt on opt-in."""
    config = tmp_path / "settings.json"
    config.write_text(
        json.dumps(
            {
                "database_path": str(tmp_path / "catalog.db"),
                "chroma_path": str(tmp_path / "chroma"),
                "embedding_backend": "deterministic",
                "redis_dsn": None,
                "litellm_model": None,
                "litellm_inference_model": None,
                "web_search_provider": None,
            }
        ),
        encoding="utf-8",
    )
    payload = json.dumps(
        {"name": "Added by agent", "description": "Test record", "context": "Body"}
    )
    for preview in (True, False):
        args = ["prompt-add", "--json", payload, "--result-json"]
        if preview:
            args.append("--dry-run")
        result = _run(tmp_path, config, installed, *args)
        assert result.returncode == 0, result
        assert result.stderr == ""
        receipt = json.loads(result.stdout)
        expected: dict[str, Any] = {
            "ok": True,
            "command": "prompt-add",
            "mode": "preview" if preview else "apply",
            "counts": (
                {"added": 1, "updated": 0, "skipped": 0, "unchanged": 0}
                if preview
                else {"added": 1, "updated": 0, "skipped": 0, "errors": 0}
            ),
        }
        if not preview:
            with closing(sqlite3.connect(tmp_path / "catalog.db")) as conn:
                stored_id = conn.execute(
                    "SELECT id FROM prompts WHERE name='Added by agent'"
                ).fetchone()[0]
            expected["records"] = [{"id": stored_id, "action": "created"}]
        assert receipt == expected
        with closing(sqlite3.connect(tmp_path / "catalog.db")) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM prompts WHERE name='Added by agent'"
            ).fetchone()[0]
        assert count == (0 if preview else 1)
        assert not list(tmp_path.glob("prompt-add-inline-*.json"))


@pytest.mark.parametrize("installed", [False, True])
def test_prompt_add_result_json_update_and_skip_keep_canonical_id(
    tmp_path: Path, installed: bool
) -> None:
    config = tmp_path / "settings.json"
    config.write_text(
        json.dumps(
            {
                "database_path": str(tmp_path / "catalog.db"),
                "chroma_path": str(tmp_path / "chroma"),
                "embedding_backend": "deterministic",
                "redis_dsn": None,
                "litellm_model": None,
                "litellm_inference_model": None,
            }
        ),
        encoding="utf-8",
    )
    original = json.dumps({"name": "Stable identity", "description": "Before", "context": "Body"})
    changed = json.dumps({"name": "Stable identity", "description": "After", "context": "Body"})
    first = _run(tmp_path, config, installed, "prompt-add", "--json", original, "--result-json")
    assert first.returncode == 0 and first.stderr == "", first
    created_id = json.loads(first.stdout)["records"][0]["id"]
    updated = _run(tmp_path, config, installed, "prompt-add", "--json", changed, "--result-json")
    assert updated.returncode == 0 and updated.stderr == "", updated
    assert json.loads(updated.stdout)["records"] == [{"id": created_id, "action": "updated"}]
    skipped = _run(
        tmp_path,
        config,
        installed,
        "prompt-add",
        "--json",
        original,
        "--no-overwrite",
        "--result-json",
    )
    assert skipped.returncode == 0 and skipped.stderr == "", skipped
    assert json.loads(skipped.stdout)["records"] == []
    with closing(sqlite3.connect(tmp_path / "catalog.db")) as conn:
        rows = conn.execute(
            "SELECT id, description FROM prompts WHERE name='Stable identity'"
        ).fetchall()
    assert rows == [(created_id, "After")]


@pytest.mark.parametrize("installed", [False, True])
def test_prompt_add_result_json_preview_failure_is_sanitized(
    tmp_path: Path, installed: bool
) -> None:
    config = tmp_path / "settings.json"
    config.write_text(
        json.dumps(
            {
                "database_path": str(tmp_path / "catalog.db"),
                "chroma_path": str(tmp_path / "chroma"),
                "embedding_backend": "deterministic",
                "redis_dsn": None,
                "litellm_model": None,
                "litellm_inference_model": None,
            }
        ),
        encoding="utf-8",
    )
    result = _run(
        tmp_path,
        config,
        installed,
        "prompt-add",
        "PRIVATE_ABSENT.json",
        "--dry-run",
        "--result-json",
    )
    assert result.returncode == 6, result
    assert result.stdout == ""
    assert json.loads(result.stderr) == {
        "ok": False,
        "command": "prompt-add",
        "error": {
            "code": "IMPORT_PREVIEW_FAILED",
            "message": "Unable to preview prompt import.",
        },
    }
    assert "PRIVATE_ABSENT" not in result.stderr


@pytest.mark.parametrize("installed", [False, True])
def test_prompt_add_result_json_bad_file_fails_without_raw_log(
    tmp_path: Path, installed: bool
) -> None:
    config = tmp_path / "settings.json"
    config.write_text(
        json.dumps(
            {
                "database_path": str(tmp_path / "catalog.db"),
                "chroma_path": str(tmp_path / "chroma"),
                "embedding_backend": "deterministic",
                "redis_dsn": None,
                "litellm_model": None,
                "litellm_inference_model": None,
            }
        ),
        encoding="utf-8",
    )
    bad = tmp_path / "PRIVATE_INVALID.json"
    bad.write_text("{not-json", encoding="utf-8")
    result = _run(tmp_path, config, installed, "prompt-add", str(bad), "--result-json")
    assert result.returncode == 6, result
    assert result.stdout == ""
    assert json.loads(result.stderr) == {
        "ok": False,
        "command": "prompt-add",
        "error": {"code": "IMPORT_INPUT_FAILED", "message": "Unable to prepare prompt import."},
    }
    assert "PRIVATE_INVALID" not in result.stderr


def test_prompt_add_result_json_partial_import_receipt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A non-atomic import never presents an unsuccessful batch as safe to retry."""

    stored_id = str(uuid.uuid4())

    def partial(*_args: object, **_kwargs: object) -> CatalogImportResult:
        return CatalogImportResult(
            added=1,
            errors=1,
            records=[{"id": stored_id, "action": "created"}],
        )

    monkeypatch.setattr(
        "main.import_prompt_catalog",
        partial,
    )
    args = Namespace(
        command="prompt-add",
        path="ignored.json",
        dry_run=False,
        no_overwrite=False,
        result_json=True,
    )
    assert run_catalog_import(cast("Any", object()), args, logging.getLogger("test_add")) == 6
    channels = capsys.readouterr()
    assert channels.out == ""
    assert json.loads(channels.err) == {
        "ok": False,
        "command": "prompt-add",
        "partial": True,
        "counts": {"added": 1, "updated": 0, "skipped": 0, "errors": 1},
        "records": [{"id": stored_id, "action": "created"}],
        "error": {
            "code": "IMPORT_PARTIAL",
            "message": "Import had errors; some records may have been written.",
        },
    }


def test_prompt_add_result_json_apply_exception_warns_of_uncertain_state(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An exception after an unknown number of writes is not retry-safe."""

    def interrupted(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("PRIVATE_IMPORT_EXCEPTION")

    monkeypatch.setattr("main.import_prompt_catalog", interrupted)
    args = Namespace(
        command="prompt-add",
        path="ignored.json",
        dry_run=False,
        no_overwrite=False,
        result_json=True,
    )
    assert run_catalog_import(cast("Any", object()), args, logging.getLogger("test_add")) == 6
    channels = capsys.readouterr()
    assert channels.out == ""
    assert json.loads(channels.err) == {
        "ok": False,
        "command": "prompt-add",
        "partial": True,
        "error": {"code": "IMPORT_FAILED", "message": "Unable to import prompts."},
    }
    assert "PRIVATE_IMPORT_EXCEPTION" not in channels.err


@pytest.mark.parametrize("installed", [False, True])
def test_prompt_add_result_json_invalid_entry_is_not_empty_success(
    tmp_path: Path, installed: bool
) -> None:
    config = tmp_path / "settings.json"
    config.write_text(
        json.dumps(
            {
                "database_path": str(tmp_path / "catalog.db"),
                "chroma_path": str(tmp_path / "chroma"),
                "embedding_backend": "deterministic",
                "redis_dsn": None,
                "litellm_model": None,
                "litellm_inference_model": None,
            }
        ),
        encoding="utf-8",
    )
    bad = tmp_path / "entry.json"
    bad.write_text(
        json.dumps(
            [
                {"name": "Valid before invalid", "description": "Would be stored"},
                {"name": "PRIVATE_ENTRY"},
            ]
        ),
        encoding="utf-8",
    )
    result = _run(tmp_path, config, installed, "prompt-add", str(bad), "--result-json")
    assert result.returncode == 6, result
    assert result.stdout == ""
    assert json.loads(result.stderr)["error"]["code"] == "IMPORT_INPUT_FAILED"
    assert "partial" not in json.loads(result.stderr)
    assert "PRIVATE_ENTRY" not in result.stderr
    with closing(sqlite3.connect(tmp_path / "catalog.db")) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM prompts WHERE name='Valid before invalid'"
        ).fetchone()[0]
    assert count == 0


@pytest.mark.parametrize("installed", [False, True])
def test_prompt_add_result_json_stdin_and_mixed_invalid_are_safe(
    tmp_path: Path, installed: bool
) -> None:
    config = tmp_path / "settings.json"
    config.write_text(
        json.dumps(
            {
                "database_path": str(tmp_path / "catalog.db"),
                "chroma_path": str(tmp_path / "chroma"),
                "embedding_backend": "deterministic",
                "redis_dsn": None,
                "litellm_model": None,
                "litellm_inference_model": None,
            }
        ),
        encoding="utf-8",
    )
    valid = {"name": "Stdin asset", "description": "Private local example"}
    malformed = {"name": "PRIVATE_BROKEN_ENTRY"}
    rejected = _run(
        tmp_path,
        config,
        installed,
        "prompt-add",
        "--from-stdin",
        "--result-json",
        stdin_text=json.dumps([valid, malformed]),
    )
    # Inline parser rejects malformed entries before manager initialization or write.
    assert rejected.returncode != 0 and rejected.stdout == ""
    assert not list(tmp_path.glob("prompt-add-inline-*.json"))
    result = _run(
        tmp_path,
        config,
        installed,
        "prompt-add",
        "--from-stdin",
        "--result-json",
        stdin_text=json.dumps(valid),
    )
    assert result.returncode == 0 and result.stderr == "", result
    assert json.loads(result.stdout)["counts"]["added"] == 1
    with closing(sqlite3.connect(tmp_path / "catalog.db")) as conn:
        count = conn.execute("SELECT COUNT(*) FROM prompts WHERE name='Stdin asset'").fetchone()[0]
    assert count == 1
    assert not list(tmp_path.glob("prompt-add-inline-*.json"))
