"""Real-process contract for the read-only, provider-free doctor landing command."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest

from core.repository import PromptRepository
from models.prompt_model import Prompt

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


def test_doctor_catalog_reports_issues_without_exposing_prompt_text(tmp_path: Path) -> None:
    config = _config(tmp_path)
    db = tmp_path / "catalog.db"
    repository = PromptRepository(str(db))
    missing = uuid4()
    repository.add(
        Prompt(
            id=uuid4(),
            name="synthetic-private-title",
            description="example",
            category="Test",
            context="{% if synthetic-private-body %}",
            related_prompts=[str(missing)],
        )
    )
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}

    result = _invoke(tmp_path, config, "catalog", "--json")

    assert result.returncode == 1, result.stderr
    assert result.stderr == ""
    report = json.loads(result.stdout)
    assert report["schema_version"] == 1
    assert report["command"] == "doctor catalog"
    assert report["report"]["summary"]["prompts"] == 1
    assert {issue["code"] for issue in report["report"]["issues"]} >= {"CAT003", "CAT004", "CAT005"}
    assert "synthetic-private" not in result.stdout
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()} == before
    assert not (tmp_path / "chroma").exists()


def test_doctor_catalog_missing_database_and_help_are_read_only(tmp_path: Path) -> None:
    config = _config(tmp_path)
    missing = _console_invoke(tmp_path, config, "catalog", "--json")
    help_result = _invoke(tmp_path, config, "catalog", "--help")
    assert missing.returncode == 0, missing.stderr
    assert missing.stderr == ""
    assert json.loads(missing.stdout)["report"]["summary"]["prompts"] == 0
    assert help_result.returncode == 0
    assert help_result.stderr == ""
    assert "--json" in help_result.stdout
    assert set(tmp_path.iterdir()) == {config}


def test_doctor_catalog_first_run_does_not_import_provider_dotenv(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("PROMPT_MANAGER_CONFIG_JSON=missing.json\n", encoding="utf-8")
    result = _console_invoke(tmp_path, None, "catalog", "--json")
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert json.loads(result.stdout)["code"] == "DB_NOT_CREATED"
    assert list(tmp_path.iterdir()) == [dotenv]


def test_doctor_catalog_json_flag_before_or_after_subcommand(tmp_path: Path) -> None:
    config = _config(tmp_path)
    for arguments in (("--json", "catalog"), ("catalog", "--json")):
        result = _invoke(tmp_path, config, *arguments)
        assert result.returncode == 0, result.stderr
        assert result.stderr == ""
        assert json.loads(result.stdout)["command"] == "doctor catalog"


def test_doctor_catalog_preserves_legacy_issue_codes_and_counts(tmp_path: Path) -> None:
    config = _config(tmp_path)
    repository = PromptRepository(str(tmp_path / "catalog.db"))
    missing = uuid4()
    repository.add(
        Prompt(
            id=uuid4(),
            name="synthetic-private-duplicate",
            description="example",
            category="Test",
            context="Same body",
            related_prompts=[str(missing)],
        )
    )
    repository.add(
        Prompt(
            id=uuid4(),
            name="synthetic-private-duplicate",
            description="example",
            category="Test",
            context="Same body",
        )
    )
    doctor = _invoke(tmp_path, config, "catalog", "--json")
    legacy = subprocess.run(
        [sys.executable, "-m", "main", "catalog-check", "--json"],
        cwd=tmp_path,
        env=_environment(tmp_path, config),
        input="",
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )

    assert doctor.returncode == 1, doctor.stderr
    assert legacy.returncode == 5, legacy.stderr
    doctor_report = json.loads(doctor.stdout)["report"]
    legacy_report = json.loads(legacy.stdout)
    assert doctor_report["summary"] == legacy_report["summary"]
    assert [issue["code"] for issue in doctor_report["issues"]] == [
        issue["code"] for issue in legacy_report["issues"]
    ]
    assert "synthetic-private" not in doctor.stdout


def test_doctor_catalog_corrupt_db_fails_without_mutation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    db = tmp_path / "catalog.db"
    db.write_bytes(b"corrupt database")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    result = _invoke(tmp_path, config, "catalog", "--json")
    assert result.returncode == 1
    assert result.stderr == ""
    assert json.loads(result.stdout)["code"] == "DB_UNREADABLE"
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_doctor_catalog_nonempty_wal_fails_closed_without_sidecar_changes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    db = tmp_path / "catalog.db"
    PromptRepository(str(db))
    connection = sqlite3.connect(db)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA wal_autocheckpoint=0")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "INSERT INTO prompts "
            "(id,name,description,category,context,last_modified,version,created_at) "
            "VALUES (?,?,?,?,?,CURRENT_TIMESTAMP,1,CURRENT_TIMESTAMP)",
            (str(uuid4()), "synthetic-private-wal", "example", "Test", "body"),
        )
        connection.commit()
        wal = Path(f"{db}-wal")
        assert wal.stat().st_size > 0
        before = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
        result = _invoke(tmp_path, config, "catalog", "--json")
        assert result.returncode == 1, result.stderr
        assert result.stderr == ""
        assert json.loads(result.stdout)["code"] == "DB_UNREADABLE"
        assert {
            path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
        } == before
    finally:
        connection.close()


def test_doctor_catalog_incomplete_schema_fails_closed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    db = tmp_path / "catalog.db"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE prompts (id TEXT PRIMARY KEY)")
    before = db.read_bytes()

    result = _invoke(tmp_path, config, "catalog", "--json")

    assert result.returncode == 1, result.stderr
    assert result.stderr == ""
    assert json.loads(result.stdout)["code"] == "DB_UNREADABLE"
    assert db.read_bytes() == before
    assert set(tmp_path.iterdir()) == {config, db}


def test_doctor_catalog_checks_missing_chain_prompt_step(tmp_path: Path) -> None:
    config = _config(tmp_path)
    db = tmp_path / "catalog.db"
    PromptRepository(str(db))
    chain_id, step_id, missing_id = str(uuid4()), str(uuid4()), str(uuid4())
    with sqlite3.connect(db) as connection:
        connection.execute(
            "INSERT INTO prompt_chains (id,name,description,created_at,updated_at) "
            "VALUES (?,?,?,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)",
            (chain_id, "synthetic-private-chain", "example"),
        )
        connection.execute(
            "INSERT INTO prompt_chain_steps "
            "(id,chain_id,prompt_id,order_index,input_template,output_variable) "
            "VALUES (?,?,?,?,?,?)",
            (step_id, chain_id, missing_id, 1, "", "result"),
        )
    connection.close()
    before = db.read_bytes()
    result = _invoke(tmp_path, config, "catalog", "--json")

    assert result.returncode == 1, result.stderr
    assert result.stderr == ""
    report = json.loads(result.stdout)["report"]
    assert report["summary"]["chains"] == 1
    assert [item["code"] for item in report["issues"]] == ["CAT006"]
    assert "synthetic-private" not in result.stdout
    assert db.read_bytes() == before


def test_doctor_catalog_does_not_claim_a_locked_rollback_journal_healthy(tmp_path: Path) -> None:
    config = _config(tmp_path)
    db = tmp_path / "catalog.db"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE prompts (id TEXT PRIMARY KEY)")
        connection.execute("CREATE TABLE prompt_chains (id TEXT PRIMARY KEY)")
    connection = sqlite3.connect(db)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("BEGIN EXCLUSIVE")
        connection.execute("INSERT INTO prompts VALUES ('pending')")
        before = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
        result = _invoke(tmp_path, config, "catalog", "--json")
        assert result.returncode == 1, result.stderr
        assert json.loads(result.stdout)["code"] == "DB_UNREADABLE"
        assert {
            path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
        } == before
    finally:
        connection.rollback()
        connection.close()


def test_doctor_config_details_never_echo_credentials_or_dsn(tmp_path: Path) -> None:
    config = _config(tmp_path)
    marker = "synthetic-private-credential"
    config.write_text(
        json.dumps(
            {
                "DB_PATH": str(tmp_path / "catalog.db"),
                "CHROMA_PATH": str(tmp_path / "chroma"),
                "LITELLM_MODEL": "azure/test",
                "LITELLM_API_KEY": marker,
                "LITELLM_API_BASE": f"https://user:{marker}@example.invalid/?key={marker}",
                "REDIS_DSN": f"redis://user:{marker}@example.invalid/0",
            }
        ),
        encoding="utf-8",
    )
    before = config.read_bytes()
    for arguments in (("config", "--json", "--details"), ("--json", "config", "--details")):
        result = _invoke(tmp_path, config, *arguments)
        assert result.returncode == 0, result.stderr
        assert result.stderr == ""
        report = json.loads(result.stdout)
        assert report["command"] == "doctor config"
        assert report["details"]["credential_present"] is False
        assert marker not in result.stdout
    assert config.read_bytes() == before
    assert set(tmp_path.iterdir()) == {config}


def test_doctor_embeddings_offline_does_not_probe_or_create_state(tmp_path: Path) -> None:
    config = _config(tmp_path)
    for invoke in (_invoke, _console_invoke):
        result = invoke(tmp_path, config, "embeddings", "--json")
        assert result.returncode == 0, result.stderr
        assert result.stderr == ""
        report = json.loads(result.stdout)
        assert report["command"] == "doctor embeddings"
        assert report["checks"][-1]["id"] == "search_embeddings"
        assert report["checks"][-1]["status"] in {"WARN", "SKIP"}
    assert set(tmp_path.iterdir()) == {config}


def test_doctor_embeddings_live_probe_is_bounded_and_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from types import SimpleNamespace

    from cli import doctor
    from core import litellm_adapter

    marker = "private-provider-response"
    requests: list[dict[str, object]] = []

    def fake_embedding(**kwargs: object) -> dict[str, object]:
        requests.append(kwargs)
        return {"data": [{"embedding": [0.1, 0.2, 0.3]}], "id": marker}

    monkeypatch.setattr(litellm_adapter, "get_embedding", lambda: (fake_embedding, Exception))
    settings = SimpleNamespace(
        embedding_backend="litellm",
        embedding_model="azure/synthetic-embedding",
        litellm_api_key="private-key",
        litellm_api_base="https://example.openai.azure.com/",
        litellm_api_version="2024-01-01",
        embedding_device=None,
    )
    monkeypatch.setattr(doctor, "load_settings", lambda: settings)
    monkeypatch.setattr(doctor, "_config_source", lambda: (True, True))

    assert doctor.run_doctor(command="embeddings", json_output=True) == 0
    offline_output = capsys.readouterr()
    assert offline_output.err == ""
    assert json.loads(offline_output.out)["checks"][-1]["code"] == "EMBEDDING_NOT_PROBED"
    assert requests == []
    assert doctor.run_doctor(command="embeddings", json_output=True, live=True) == 0
    live_output = capsys.readouterr()
    assert live_output.err == ""
    live = json.loads(live_output.out)
    assert live["checks"][-1]["code"] == "EMBEDDING_BACKEND_REACHABLE"
    assert live["probe"] == {
        "performed": True,
        "backend_dimension": 3,
        "vector_index": "not_inspected",
    }
    assert len(requests) == 1
    assert requests[0]["input"] == ["Prompt Manager diagnostics probe"]
    assert requests[0]["timeout"] == 15
    assert requests[0]["num_retries"] == 0
    assert requests[0]["api_version"] == "2024-01-01"
    assert marker not in live_output.out
    assert "private-key" not in live_output.out


def test_doctor_embeddings_live_failure_is_sanitized_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from types import SimpleNamespace

    from cli import doctor
    from core import litellm_adapter

    requests: list[dict[str, object]] = []

    def failing_embedding(**kwargs: object) -> object:
        requests.append(kwargs)
        raise RuntimeError("private-key https://private-host.example/failure")

    monkeypatch.setattr(litellm_adapter, "get_embedding", lambda: (failing_embedding, Exception))
    settings = SimpleNamespace(
        embedding_backend="litellm",
        embedding_model="azure/synthetic-embedding",
        litellm_api_key="private-key",
        litellm_api_base="https://example.openai.azure.com/",
        litellm_api_version="2024-01-01",
        embedding_device=None,
    )
    monkeypatch.setattr(doctor, "load_settings", lambda: settings)
    monkeypatch.setattr(doctor, "_config_source", lambda: (True, True))
    assert doctor.run_doctor(command="embeddings", json_output=True, live=True) == 1
    output = capsys.readouterr()
    assert output.err == ""
    assert json.loads(output.out)["checks"][-1]["code"] == "EMBEDDING_PROBE_FAILED"
    assert json.loads(output.out)["next_step"] == (
        "Check provider configuration and availability before a new approved probe"
    )
    assert "private-" not in output.out
    assert len(requests) == 1


def test_doctor_embeddings_live_help_warns_of_provider_cost(tmp_path: Path) -> None:
    for invoke in (_invoke, _console_invoke):
        result = invoke(tmp_path, None, "embeddings", "--help")
        assert result.returncode == 0, result.stderr
        assert "--live" in result.stdout
        assert "cost" in result.stdout.lower()


def test_doctor_config_invalid_explicit_source_is_diagnosed(tmp_path: Path) -> None:
    missing = tmp_path / "absent.json"
    result = _invoke(tmp_path, missing, "config", "--json")
    assert result.returncode == 1, result.stderr
    assert result.stderr == ""
    assert json.loads(result.stdout)["checks"][0]["code"] == "CONFIG_INVALID"
    assert set(tmp_path.iterdir()) == set()


def test_focused_doctor_checks_do_not_inherit_unrelated_catalog_failure(tmp_path: Path) -> None:
    config = _config(tmp_path)
    (tmp_path / "catalog.db").write_bytes(b"bad sqlite")
    for command in ("config", "embeddings"):
        result = _invoke(tmp_path, config, command, "--json")
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        assert report["ok"] is True
        assert "DB_UNREADABLE" not in result.stdout


def test_doctor_config_detail_source_precedence_without_secret_values(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dotenv = tmp_path / "provider.env"
    marker = "synthetic-private-dotenv-key"
    dotenv.write_text(f"PROMPT_MANAGER_LITELLM_API_KEY={marker}\n", encoding="utf-8")
    env = _environment(tmp_path, config)
    env["PROMPT_MANAGER_ENV_FILE"] = str(dotenv)
    result = subprocess.run(
        [sys.executable, "-m", "main", "doctor", "config", "--json", "--details"],
        cwd=tmp_path,
        env=env,
        input="",
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    report = json.loads(result.stdout)
    assert report["details"]["credential_present"] is True
    assert marker not in result.stdout
    assert set(tmp_path.iterdir()) == {config, dotenv}


def test_doctor_analytics_reads_aggregate_without_probe_or_writes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    db = tmp_path / "catalog.db"
    PromptRepository(str(db))
    with sqlite3.connect(db) as connection:
        connection.execute(
            "INSERT INTO prompt_executions "
            "(id,prompt_id,request_text,status,executed_at,input_hash) "
            "VALUES (?,?,?,'success',CURRENT_TIMESTAMP,?)",
            (str(uuid4()), str(uuid4()), "synthetic-private-request", "synthetic-private-hash"),
        )
        connection.execute(
            "INSERT INTO prompt_executions "
            "(id,prompt_id,request_text,status,executed_at,input_hash) "
            "VALUES (?,?,?,'error',CURRENT_TIMESTAMP,?)",
            (str(uuid4()), str(uuid4()), "synthetic-private-request", "synthetic-private-hash"),
        )
    connection.close()
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    for invoke in (_invoke, _console_invoke):
        result = invoke(tmp_path, config, "analytics", "--json")
        assert result.returncode == 0, result.stderr
        assert result.stderr == ""
        report = json.loads(result.stdout)
        assert report["command"] == "doctor analytics"
        assert report["kind"] == "report_not_health"
        assert report["report"]["total_runs"] == 2
        assert report["report"]["success_runs"] == 1
        assert "synthetic-private" not in result.stdout
        assert {
            path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
        } == before


def test_doctor_analytics_export_is_explicit_and_non_overwriting(tmp_path: Path) -> None:
    config = _config(tmp_path)
    PromptRepository(str(tmp_path / "catalog.db"))
    destination = tmp_path / "report.csv"
    result = _invoke(tmp_path, config, "analytics", "--json", "--export-csv", str(destination))
    assert result.returncode == 0, result.stderr
    assert destination.read_text(encoding="utf-8").startswith("total_runs,success_runs")
    assert json.loads(result.stdout)["exported"] is True
    before = destination.read_bytes()
    second = _invoke(tmp_path, config, "analytics", "--json", "--export-csv", str(destination))
    assert second.returncode == 1, second.stderr
    assert json.loads(second.stdout)["code"] == "EXPORT_EXISTS"
    assert destination.read_bytes() == before


def test_doctor_analytics_first_run_does_not_create_database(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = _invoke(tmp_path, config, "analytics", "--json")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["code"] == "DB_NOT_CREATED"
    assert set(tmp_path.iterdir()) == {config}


def test_doctor_analytics_unreadable_db_never_exports(tmp_path: Path) -> None:
    config = _config(tmp_path)
    db = tmp_path / "catalog.db"
    db.write_bytes(b"not sqlite")
    destination = tmp_path / "report.csv"
    before = db.read_bytes()
    result = _invoke(tmp_path, config, "analytics", "--json", "--export-csv", str(destination))
    assert result.returncode == 1, result.stderr
    assert result.stderr == ""
    report = json.loads(result.stdout)
    assert report["code"] == "DB_UNREADABLE"
    assert report["report"] is None
    assert not destination.exists()
    assert db.read_bytes() == before


def test_doctor_analytics_pending_wal_is_not_reported_as_current(tmp_path: Path) -> None:
    config = _config(tmp_path)
    db = tmp_path / "catalog.db"
    PromptRepository(str(db))
    connection = sqlite3.connect(db)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA wal_autocheckpoint=0")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "INSERT INTO prompt_executions "
            "(id,prompt_id,request_text,status,executed_at,input_hash) "
            "VALUES (?,?,?,'success',CURRENT_TIMESTAMP,?)",
            (str(uuid4()), str(uuid4()), "private-request", "hash"),
        )
        connection.commit()
        wal = Path(f"{db}-wal")
        assert wal.stat().st_size > 0
        before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
        result = _console_invoke(tmp_path, config, "analytics", "--json")
        assert result.returncode == 1, result.stderr
        assert result.stderr == ""
        assert json.loads(result.stdout)["code"] == "DB_UNREADABLE"
        assert {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()} == before
    finally:
        connection.close()


def test_doctor_analytics_invalid_config_does_not_create_export(tmp_path: Path) -> None:
    destination = tmp_path / "report.csv"
    result = _invoke(
        tmp_path, tmp_path / "missing.json", "--json", "analytics", "--export-csv", str(destination)
    )
    assert result.returncode == 1, result.stderr
    assert result.stderr == ""
    assert json.loads(result.stdout)["code"] == "CONFIG_INVALID"
    assert set(tmp_path.iterdir()) == set()


def test_doctor_prompt_validate_and_lint_are_private_and_read_only(tmp_path: Path) -> None:
    config = _config(tmp_path)
    db = tmp_path / "catalog.db"
    repository = PromptRepository(str(db))
    prompt_id = uuid4()
    repository.add(
        Prompt(
            id=prompt_id,
            name="private-title",
            description="short",
            category="Test",
            context="{% if private-body %}",
        )
    )
    before = db.read_bytes()
    for check, code in (("validate", "VAL004"), ("lint", "LINT001")):
        result = _invoke(tmp_path, config, "prompt", str(prompt_id), check, "--json")
        assert result.returncode == (1 if check == "validate" else 0), result.stderr
        assert result.stderr == ""
        report = json.loads(result.stdout)
        assert report["command"] == f"doctor prompt {check}"
        assert code in [issue["code"] for issue in report["report"]["issues"]]
        assert "private-" not in result.stdout
        assert db.read_bytes() == before


def test_doctor_prompt_name_ambiguity_and_uuid_resolution(tmp_path: Path) -> None:
    config = _config(tmp_path)
    repository = PromptRepository(str(tmp_path / "catalog.db"))
    identifiers = [uuid4(), uuid4()]
    for prompt_id in identifiers:
        repository.add(
            Prompt(
                id=prompt_id,
                name="same-private-name",
                description="Description",
                category="Test",
                context="Review task",
            )
        )
    ambiguous = _invoke(tmp_path, config, "prompt", "same-private-name", "validate", "--json")
    assert ambiguous.returncode == 1, ambiguous.stderr
    assert json.loads(ambiguous.stdout)["code"] == "PROMPT_AMBIGUOUS"
    assert "same-private-name" not in ambiguous.stdout
    selected = _invoke(tmp_path, config, "prompt", str(identifiers[0]), "validate", "--json")
    assert selected.returncode == 0, selected.stderr
    assert json.loads(selected.stdout)["report"]["prompt_id"] == str(identifiers[0])


def test_doctor_prompt_test_suite_is_provider_free_and_sanitized(tmp_path: Path) -> None:
    config = _config(tmp_path)
    repository = PromptRepository(str(tmp_path / "catalog.db"))
    prompt_id = uuid4()
    repository.add(
        Prompt(
            id=prompt_id,
            name="private-name",
            description="example",
            category="Test",
            context="Review {{ value }}",
        )
    )
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "case1",
                        "variables": {"value": "private-input"},
                        "expected": "Review private-input",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    result = _console_invoke(
        tmp_path, config, "prompt", str(prompt_id), "test", "--suite", str(suite), "--json"
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    report = json.loads(result.stdout)
    assert report["report"]["summary"] == {"total": 1, "passed": 1, "failed": 0}
    assert "private-" not in result.stdout
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()} == before


@pytest.mark.parametrize("attempt", ["environment", "filesystem"])
def test_doctor_prompt_test_cannot_access_process_or_write_files(
    tmp_path: Path, attempt: str
) -> None:
    config = _config(tmp_path)
    repository = PromptRepository(str(tmp_path / "catalog.db"))
    prompt_id = uuid4()
    marker = tmp_path / "injected-write"
    context = (
        "{{ cycler.__init__.__globals__.os.environ['DOCTOR_SYNTHETIC_SECRET'] }}"
        if attempt == "environment"
        else "{{ cycler.__init__.__globals__.os.system(" + json.dumps(f"touch {marker}") + ") }}"
    )
    repository.add(
        Prompt(
            id=prompt_id,
            name="safe-check",
            description="example",
            category="Test",
            context=context,
        )
    )
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "safe",
                        "variables": {},
                        "expected": "synthetic-private" if attempt == "environment" else "0",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    environment = _environment(tmp_path, config)
    environment["DOCTOR_SYNTHETIC_SECRET"] = "synthetic-private"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "main",
            "doctor",
            "prompt",
            str(prompt_id),
            "test",
            "--suite",
            str(suite),
            "--json",
        ],
        cwd=tmp_path,
        env=environment,
        input="",
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )
    assert result.returncode == 1, result.stderr
    assert result.stderr == ""
    assert json.loads(result.stdout)["report"]["summary"] == {"total": 1, "passed": 0, "failed": 1}
    assert "synthetic-private" not in result.stdout
    assert not marker.exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ("--json", "prompt", "private-reference", "private-action"),
        ("prompt", "private-reference", "private-action", "--json"),
        ("prompt", "private-reference", "validate", "--json", "private-extra"),
    ],
)
def test_doctor_json_parser_error_is_bounded(tmp_path: Path, arguments: tuple[str, ...]) -> None:
    result = _invoke(tmp_path, None, *arguments)
    assert result.returncode == 2
    assert result.stdout == ""
    assert json.loads(result.stderr) == {
        "ok": False,
        "error": {"code": "INVALID_USAGE", "message": "Invalid doctor command or arguments"},
    }
    assert "private-" not in result.stderr


@pytest.mark.parametrize(
    "template",
    [
        '{{ "x" * 8000000 }}',
        "{% for item in range(8000000) %}x{% endfor %}",
        "{{ value.upper() }}",
    ],
)
def test_doctor_prompt_test_rejects_unbounded_expressions(tmp_path: Path, template: str) -> None:
    config = _config(tmp_path)
    repository = PromptRepository(str(tmp_path / "catalog.db"))
    prompt_id = uuid4()
    repository.add(
        Prompt(
            id=prompt_id, name="bounded", description="example", category="Test", context=template
        )
    )
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps({"cases": [{"id": "case", "variables": {"value": "x"}, "expected": "x"}]}),
        encoding="utf-8",
    )
    result = _invoke(
        tmp_path, config, "prompt", str(prompt_id), "test", "--suite", str(suite), "--json"
    )
    assert result.returncode == 1, result.stderr
    assert result.stderr == ""
    report = json.loads(result.stdout)
    assert report["report"]["summary"] == {"total": 1, "passed": 0, "failed": 1}


def test_doctor_chain_validate_reports_without_disclosing_definition(tmp_path: Path) -> None:
    config = _config(tmp_path)
    definition = tmp_path / "chain.json"
    definition.write_text(
        json.dumps(
            {
                "name": "private-chain",
                "description": "example",
                "steps": [{"prompt_id": str(uuid4()), "input_template": "private-legacy"}],
            }
        ),
        encoding="utf-8",
    )
    before = definition.read_bytes()
    result = _invoke(tmp_path, config, "chain", str(definition), "validate", "--json")
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    report = json.loads(result.stdout)
    assert report["command"] == "doctor chain validate"
    assert report["report"]["summary"]["step_count"] == 1
    assert "private-" not in result.stdout
    assert definition.read_bytes() == before


def test_doctor_chain_invalid_definition_and_prompt_invalid_suite_are_bounded(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    definition = tmp_path / "chain.json"
    definition.write_text('{"name": "private-chain", "steps": []}', encoding="utf-8")
    chain = _console_invoke(tmp_path, config, "chain", str(definition), "validate", "--json")
    assert chain.returncode == 1, chain.stderr
    assert chain.stderr == ""
    assert json.loads(chain.stdout)["code"] == "CHAIN_INVALID"
    assert "private-chain" not in chain.stdout

    repository = PromptRepository(str(tmp_path / "catalog.db"))
    prompt_id = uuid4()
    repository.add(
        Prompt(
            id=prompt_id,
            name="private-prompt",
            description="desc",
            category="Test",
            context="Review task",
        )
    )
    suite = tmp_path / "suite.json"
    suite.write_text('{"cases": [{"id": "private-identifier"}]}', encoding="utf-8")
    test = _invoke(
        tmp_path, config, "prompt", str(prompt_id), "test", "--suite", str(suite), "--json"
    )
    assert test.returncode == 1, test.stderr
    assert test.stderr == ""
    assert json.loads(test.stdout)["code"] == "SUITE_INVALID"
    assert "private-" not in test.stdout


def test_doctor_prompt_missing_ref_and_invalid_action_help(tmp_path: Path) -> None:
    config = _config(tmp_path)
    PromptRepository(str(tmp_path / "catalog.db"))
    missing = _invoke(tmp_path, config, "prompt", str(uuid4()), "validate", "--json")
    assert missing.returncode == 1, missing.stderr
    assert json.loads(missing.stdout)["code"] == "PROMPT_NOT_FOUND"
    help_result = _invoke(tmp_path, config, "prompt", "example", "test", "--help")
    assert help_result.returncode == 0
    assert help_result.stderr == ""
    assert "--suite" in help_result.stdout
    invalid = _invoke(tmp_path, config, "prompt", "example", "unknown", "--json")
    assert invalid.returncode == 2


def test_doctor_prompt_issue_codes_match_legacy_without_copying_messages(tmp_path: Path) -> None:
    config = _config(tmp_path)
    repository = PromptRepository(str(tmp_path / "catalog.db"))
    prompt_id = uuid4()
    repository.add(
        Prompt(
            id=prompt_id,
            name="private-name",
            description="short",
            category="Test",
            context="{% if private-body %}",
        )
    )
    doctor = _invoke(tmp_path, config, "prompt", str(prompt_id), "validate", "--json")
    legacy = subprocess.run(
        [sys.executable, "-m", "main", "prompt-validate", str(prompt_id), "--json"],
        cwd=tmp_path,
        env=_environment(tmp_path, config),
        input="",
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )
    assert doctor.returncode == 1, doctor.stderr
    assert legacy.returncode == 5, legacy.stderr
    doctor_codes = [item["code"] for item in json.loads(doctor.stdout)["report"]["issues"]]
    legacy_codes = [item["code"] for item in json.loads(legacy.stdout)["issues"]]
    assert doctor_codes == legacy_codes
    assert "private-" not in doctor.stdout


def test_doctor_prompt_and_chain_validate_help_discovery(tmp_path: Path) -> None:
    config = _config(tmp_path)
    for arguments, expected in (
        (("prompt", "--help"), "exact name"),
        (("prompt", "ref", "validate", "--help"), "--json"),
        (("chain", "--help"), "definition"),
        (("chain", "input.json", "validate", "--help"), "--json"),
    ):
        result = _console_invoke(tmp_path, config, *arguments)
        assert result.returncode == 0, result.stderr
        assert result.stderr == ""
        assert expected.lower() in result.stdout.lower()
    assert set(tmp_path.iterdir()) == {config}


def test_doctor_prompt_suite_mismatch_is_completed_failure_without_body(tmp_path: Path) -> None:
    config = _config(tmp_path)
    repository = PromptRepository(str(tmp_path / "catalog.db"))
    prompt_id = uuid4()
    repository.add(
        Prompt(
            id=prompt_id,
            name="private-name",
            description="example",
            category="Test",
            context="Review {{ value }}",
        )
    )
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "private-case",
                        "variables": {"value": "private-data"},
                        "expected": "private-incorrect",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    result = _invoke(
        tmp_path, config, "prompt", str(prompt_id), "test", "--suite", str(suite), "--json"
    )
    assert result.returncode == 1, result.stderr
    assert result.stderr == ""
    report = json.loads(result.stdout)
    assert report["report"]["summary"]["failed"] == 1
    assert "private-" not in result.stdout


def test_doctor_prompt_lint_codes_match_legacy_alias(tmp_path: Path) -> None:
    config = _config(tmp_path)
    repository = PromptRepository(str(tmp_path / "catalog.db"))
    prompt_id = uuid4()
    repository.add(
        Prompt(
            id=prompt_id,
            name="private-name",
            description="short",
            category="Test",
            context="Unstructured instruction",
        )
    )
    doctor = _invoke(tmp_path, config, "prompt", str(prompt_id), "lint", "--json")
    legacy = subprocess.run(
        [sys.executable, "-m", "main", "prompt-lint", str(prompt_id), "--json"],
        cwd=tmp_path,
        env=_environment(tmp_path, config),
        input="",
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )
    assert doctor.returncode == legacy.returncode == 0
    assert doctor.stderr == ""
    assert [item["code"] for item in json.loads(doctor.stdout)["report"]["issues"]] == [
        item["code"] for item in json.loads(legacy.stdout)["issues"]
    ]


def test_doctor_chain_validity_matches_legacy_alias(tmp_path: Path) -> None:
    config = _config(tmp_path)
    definition = tmp_path / "chain.json"
    definition.write_text(
        json.dumps(
            {
                "name": "private-chain",
                "description": "example",
                "steps": [{"prompt_id": str(uuid4())}],
            }
        ),
        encoding="utf-8",
    )
    doctor = _invoke(tmp_path, config, "chain", str(definition), "validate", "--json")
    legacy = subprocess.run(
        [sys.executable, "-m", "main", "prompt-chain-validate", str(definition), "--json"],
        cwd=tmp_path,
        env=_environment(tmp_path, config),
        input="",
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )
    assert doctor.returncode == legacy.returncode == 0
    assert doctor.stderr == legacy.stderr == ""
    assert (
        json.loads(doctor.stdout)["report"]["summary"]["step_count"]
        == (json.loads(legacy.stdout)["step_count"])
    )


def test_doctor_help_mentions_explicit_csv_exception(tmp_path: Path) -> None:
    result = _console_invoke(tmp_path, None, "--help")
    assert result.returncode == 0
    assert result.stderr == ""
    assert "--export-csv" in result.stdout
    assert set(tmp_path.iterdir()) == set()
