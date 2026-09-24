"""Real-process contract for the read-only, provider-free doctor landing command."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _environment(root: Path, config: Path | None) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not (
            key.startswith(("PROMPT_MANAGER_", "AZURE_OPENAI_"))
            or key in {"LITELLM_API_KEY", "OPENAI_API_KEY"}
        )
    }
    env.update(
        HOME=str(root),
        PYTHONPATH=str(REPO_ROOT),
        PROMPT_MANAGER_ENV_FILE="",
        PYTHONDONTWRITEBYTECODE="1",
    )
    for key in (
        "DB_PATH",
        "DATABASE_PATH",
        "CHROMA_PATH",
        "LITELLM_MODEL",
        "LITELLM_INFERENCE_MODEL",
        "LITELLM_API_BASE",
        "AZURE_OPENAI_BASE_URL",
    ):
        env.pop(key, None)
    if config is not None:
        env["PROMPT_MANAGER_CONFIG_JSON"] = str(config)
    return env


def _invoke(root: Path, config: Path | None, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "main", "doctor", *arguments],
        cwd=root,
        env=_environment(root, config),
        input="",
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )


def _console_invoke(
    root: Path, config: Path | None, *arguments: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(REPO_ROOT / ".venv" / "bin" / "prompt-manager"), "doctor", *arguments],
        cwd=root,
        env=_environment(root, config),
        input="",
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )


def _config(root: Path) -> Path:
    config = root / "settings.json"
    config.write_text(
        json.dumps(
            {
                "database_path": str(root / "catalog.db"),
                "chroma_path": str(root / "chroma"),
                "embedding_backend": "deterministic",
                "redis_dsn": None,
                "litellm_model": None,
                "litellm_inference_model": None,
            }
        ),
        encoding="utf-8",
    )
    return config


def test_doctor_offline_existing_database_is_read_only(tmp_path: Path) -> None:
    config = _config(tmp_path)
    db = tmp_path / "catalog.db"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE prompts (id TEXT PRIMARY KEY)")
    original_config = config.read_bytes()
    original_db = db.read_bytes()
    before = set(tmp_path.iterdir())

    completed = _invoke(tmp_path, config, "--json")

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    report = json.loads(completed.stdout)
    assert report["schema_version"] == 1
    assert report["command"] == "doctor"
    assert report["ok"] is True
    assert report["status"] == "WARN"
    assert [item["id"] for item in report["checks"]] == [
        "config",
        "local_catalog",
        "search_embeddings",
        "model_execution",
    ]
    checks = {item["id"]: item for item in report["checks"]}
    assert checks["local_catalog"]["status"] == "OK"
    assert checks["model_execution"]["status"] == "WARN"
    assert set(tmp_path.iterdir()) == before
    assert config.read_bytes() == original_config
    assert db.read_bytes() == original_db
    assert "chroma" not in {item.name for item in tmp_path.iterdir()}


def test_doctor_first_run_does_not_create_default_config_or_database(tmp_path: Path) -> None:
    completed = _invoke(tmp_path, None, "--json")

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    report = json.loads(completed.stdout)
    assert report["status"] == "WARN"
    checks = {item["id"]: item for item in report["checks"]}
    assert checks["config"]["status"] == "WARN"
    assert checks["local_catalog"]["status"] == "WARN"
    assert list(tmp_path.iterdir()) == []


def test_doctor_console_entrypoint_is_read_only_with_ambient_dotenv(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dotenv = tmp_path / ".env"
    dotenv.write_text("PROMPT_MANAGER_CONFIG_JSON=missing.json\n", encoding="utf-8")
    before = {path.name: path.read_bytes() for path in (config, dotenv)}

    result = _console_invoke(tmp_path, config, "--json")

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert json.loads(result.stdout)["command"] == "doctor"
    assert {path.name: path.read_bytes() for path in (config, dotenv)} == before
    assert set(tmp_path.iterdir()) == {config, dotenv}


def test_doctor_console_first_run_ignores_provider_dotenv(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("PROMPT_MANAGER_CONFIG_JSON=missing.json\n", encoding="utf-8")

    result = _console_invoke(tmp_path, None, "--json")

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    report = json.loads(result.stdout)
    assert report["checks"][0]["code"] == "DEFAULT_CONFIG_ABSENT"
    assert list(tmp_path.iterdir()) == [dotenv]


def test_doctor_does_not_print_synthetic_secret_or_dsn(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with config.open(encoding="utf-8") as source:
        payload = json.load(source)
    payload["redis_dsn"] = "redis://user:synthetic-secret@localhost:6379/0?token=synthetic-secret"
    config.write_text(json.dumps(payload), encoding="utf-8")
    env = _environment(tmp_path, config)
    env["PROMPT_MANAGER_LITELLM_API_KEY"] = "synthetic-secret"

    result = subprocess.run(
        [sys.executable, "-m", "main", "doctor", "--json"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert "synthetic-secret" not in result.stdout
    assert "redis://" not in result.stdout


def test_doctor_json_is_clean_when_settings_ignores_json_credentials(tmp_path: Path) -> None:
    config = _config(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["litellm_api_key"] = "synthetic-secret"
    config.write_text(json.dumps(payload), encoding="utf-8")

    result = _invoke(tmp_path, config, "--json")

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert json.loads(result.stdout)["command"] == "doctor"
    assert "synthetic-secret" not in result.stdout


def test_doctor_existing_sqlite_with_pending_wal_is_not_claimed_readable(tmp_path: Path) -> None:
    config = _config(tmp_path)
    db = tmp_path / "catalog.db"
    with sqlite3.connect(db) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE prompts (id TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO prompts VALUES ('first')")
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    with sqlite3.connect(db) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("INSERT INTO prompts VALUES ('second')")
        connection.commit()
        before = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
        result = _invoke(tmp_path, config, "--json")
        assert result.returncode == 1, result.stderr
        checks = {item["id"]: item for item in json.loads(result.stdout)["checks"]}
        assert checks["local_catalog"]["code"] == "DB_WAL_UNVERIFIED"
        assert {
            path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
        } == before


@pytest.mark.parametrize("invalid_content", [None, "{not-json", '{"cache_ttl_seconds": 0}'])
def test_doctor_reports_explicit_config_failure_without_creating_files(
    tmp_path: Path, invalid_content: str | None
) -> None:
    config = tmp_path / "settings.json"
    if invalid_content is not None:
        config.write_text(invalid_content, encoding="utf-8")
    before = set(tmp_path.iterdir())

    completed = _invoke(tmp_path, config, "--json")

    assert completed.returncode == 1, completed.stderr
    assert completed.stderr == ""
    report = json.loads(completed.stdout)
    assert report["ok"] is False
    assert report["status"] == "FAIL"
    assert report["checks"][0]["id"] == "config"
    assert report["checks"][0]["status"] == "FAIL"
    assert set(tmp_path.iterdir()) == before


def test_doctor_reports_corrupt_existing_sqlite_without_mutation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    db = tmp_path / "catalog.db"
    db.write_bytes(b"not a sqlite database")
    before = set(tmp_path.iterdir())

    completed = _invoke(tmp_path, config, "--json")

    assert completed.returncode == 1, completed.stderr
    assert completed.stderr == ""
    checks = {item["id"]: item for item in json.loads(completed.stdout)["checks"]}
    assert checks["local_catalog"]["status"] == "FAIL"
    assert db.read_bytes() == b"not a sqlite database"
    assert set(tmp_path.iterdir()) == before


def test_doctor_does_not_call_empty_sqlite_file_a_catalog(tmp_path: Path) -> None:
    config = _config(tmp_path)
    db = tmp_path / "catalog.db"
    with sqlite3.connect(db):
        pass
    original = db.read_bytes()

    completed = _invoke(tmp_path, config, "--json")

    assert completed.returncode == 1, completed.stderr
    checks = {item["id"]: item for item in json.loads(completed.stdout)["checks"]}
    assert checks["local_catalog"]["status"] == "FAIL"
    assert db.read_bytes() == original


def test_doctor_text_and_help_are_actionable(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = _invoke(tmp_path, config)
    help_result = _invoke(tmp_path, config, "--help")

    assert result.returncode == 0
    assert result.stderr == ""
    assert "Local catalog" in result.stdout
    assert "Model execution" in result.stdout
    assert "Next:" in result.stdout
    assert help_result.returncode == 0
    assert "--json" in help_result.stdout
    assert "provider" in help_result.stdout.lower()
    assert help_result.stderr == ""
    assert not (tmp_path / "catalog.db").exists()
    assert not (tmp_path / "chroma").exists()
