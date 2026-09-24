"""Bounded, provider-free CLI diagnosis before normal application startup."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from config import SettingsError, load_settings
from config.settings import _read_dotenv_values  # pyright: ignore[reportPrivateUsage]

if TYPE_CHECKING:
    from config import PromptManagerSettings

Status = Literal["OK", "WARN", "FAIL", "SKIP"]


@dataclass(frozen=True)
class Check:
    """One bounded diagnostic result without user data or secrets."""

    id: str
    status: Status
    code: str
    message: str
    next_step: str | None = None


def _config_source() -> tuple[bool, bool]:
    """Return (explicit config selected, effective config file exists)."""
    explicit = os.getenv("PROMPT_MANAGER_CONFIG_JSON") or _read_dotenv_values().get(
        "PROMPT_MANAGER_CONFIG_JSON"
    )
    if explicit:
        return True, Path(explicit).expanduser().is_file()
    return False, (Path("config") / "config.json").is_file()


def _catalog_check(db_path: Path) -> Check:
    """Read existing SQLite metadata without opening any writing connection."""
    if not db_path.exists():
        return Check(
            "local_catalog",
            "WARN",
            "DB_NOT_CREATED",
            "Local catalog has not been created yet",
            "Start PromptManager to create the local catalog",
        )
    if not db_path.is_file():
        return Check(
            "local_catalog",
            "FAIL",
            "DB_NOT_FILE",
            "Catalog path is not a file",
            "Choose a valid database path in your configuration",
        )
    # immutable=1 prevents creation of -shm and -wal, but deliberately ignores
    # uncheckpointed WAL content; do not claim a current snapshot in that case.
    wal = Path(f"{db_path}-wal")
    if wal.exists() and wal.stat().st_size:
        return Check(
            "local_catalog",
            "FAIL",
            "DB_WAL_UNVERIFIED",
            "Existing catalog has uncheckpointed WAL data; read-only check deferred",
            "Close catalog writers and run doctor again after a checkpoint",
        )
    try:
        uri = db_path.resolve().as_uri() + "?mode=ro&immutable=1"
        with sqlite3.connect(uri, uri=True, timeout=1) as connection:
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='prompts'"
                )
            }
        if "prompts" not in tables:
            return Check(
                "local_catalog",
                "FAIL",
                "DB_SCHEMA_MISSING",
                "Existing SQLite catalog is missing the prompts table",
                "Inspect the database schema or select the correct catalog",
            )
    except (sqlite3.Error, OSError, ValueError):
        return Check(
            "local_catalog",
            "FAIL",
            "DB_UNREADABLE",
            "Existing catalog metadata cannot be read",
            "Check the database file and run doctor again",
        )
    return Check(
        "local_catalog",
        "OK",
        "DB_METADATA_READABLE",
        "Existing SQLite catalog metadata readable (records not audited)",
    )


def _embedding_check(settings: PromptManagerSettings) -> Check:
    backend = (settings.embedding_backend or "deterministic").strip().lower()
    azure_embedding = str(settings.embedding_model or "").lower().startswith("azure/")
    ready = not (
        backend in {"litellm", "openai"}
        and (
            not settings.embedding_model
            or not settings.litellm_api_key
            or (
                azure_embedding
                and (not settings.litellm_api_base or not settings.litellm_api_version)
            )
        )
    )
    if not ready:
        return Check(
            "search_embeddings",
            "WARN",
            "EMBEDDING_NOT_CONFIGURED",
            "Semantic embedding backend not configured; no probe was run",
            "Configure the embedding backend if semantic search is needed",
        )
    return Check(
        "search_embeddings",
        "SKIP",
        "EMBEDDING_NOT_PROBED",
        "Embedding backend and vector index not probed",
        "Inspect the embedding backend separately before relying on semantic search",
    )


def _model_check(settings: PromptManagerSettings) -> Check:
    models = (settings.litellm_model, settings.litellm_inference_model)
    azure_model = any(str(model).lower().startswith("azure/") for model in models if model)
    ready = bool(
        any(models)
        and settings.litellm_api_key
        and (not azure_model or (settings.litellm_api_base and settings.litellm_api_version))
    )
    if not ready:
        return Check(
            "model_execution",
            "WARN",
            "LLM_NOT_CONFIGURED",
            "Model execution not configured",
            "Configure LiteLLM only if you need model runs",
        )
    return Check(
        "model_execution",
        "SKIP",
        "LLM_NOT_PROBED",
        "Model execution configured but not probed",
        "Run a separately approved model acceptance check if execution is needed",
    )


def _diagnose() -> tuple[dict[str, object], list[Check]]:
    explicit, exists = _config_source()
    try:
        settings_logger = logging.getLogger("prompt_manager.settings")
        previous_level = settings_logger.level
        settings_logger.setLevel(logging.ERROR)
        try:
            settings = load_settings()
        finally:
            settings_logger.setLevel(previous_level)
    except SettingsError:
        config_check = Check(
            "config",
            "FAIL",
            "CONFIG_INVALID",
            "Effective configuration missing or invalid",
            "Check the selected configuration file and settings",
        )
        checks = [
            config_check,
            Check("local_catalog", "SKIP", "CONFIG_REQUIRED", "Catalog location unknown"),
            Check("search_embeddings", "SKIP", "CONFIG_REQUIRED", "Embedding settings unknown"),
            Check("model_execution", "SKIP", "CONFIG_REQUIRED", "Model settings unknown"),
        ]
    else:
        if not exists and not explicit:
            config_check = Check(
                "config",
                "WARN",
                "DEFAULT_CONFIG_ABSENT",
                "Using valid effective defaults without a config file",
            )
        else:
            config_check = Check("config", "OK", "CONFIG_VALID", "Effective settings valid")
        checks = [
            config_check,
            _catalog_check(settings.db_path),
            _embedding_check(settings),
            _model_check(settings),
        ]

    required_failed = any(
        check.status == "FAIL" for check in checks if check.id in {"config", "local_catalog"}
    )
    status: Status = (
        "FAIL"
        if required_failed
        else "WARN"
        if any(check.status in {"WARN", "SKIP"} for check in checks)
        else "OK"
    )
    next_step = next(
        (check.next_step for check in checks if check.status == "FAIL" and check.next_step),
        None,
    )
    if next_step is None:
        next_step = next(
            (check.next_step for check in checks if check.status == "WARN" and check.next_step),
            None,
        )
    if next_step is None:
        next_step = "Use the local catalog; run specific checks for deeper diagnostics"
    return {
        "schema_version": 1,
        "command": "doctor",
        "ok": not required_failed,
        "status": status,
        "checks": [asdict(check) for check in checks],
        "next_step": next_step,
    }, checks


def run_doctor(*, json_output: bool = False) -> int:
    """Emit a sanitized report or bounded unexpected error without creating state."""
    try:
        report, checks = _diagnose()
    except Exception:  # unexpected failures must not expose sensitive exception text
        import sys

        if json_output:
            print(
                json.dumps(
                    {"code": "DOCTOR_INSPECTION_ERROR", "message": "Inspection could not finish"}
                ),
                file=sys.stderr,
            )
        else:
            print("Doctor: inspection could not finish", file=sys.stderr)
        return 3
    if json_output:
        print(json.dumps(report, ensure_ascii=False))
    else:
        catalog = next(check for check in checks if check.id == "local_catalog")
        model = next(check for check in checks if check.id == "model_execution")
        print(f"Doctor: Local catalog {catalog.status}; Model execution {model.status}")
        for check in checks:
            print(f"{check.status:4} {check.id} — {check.message}")
        print(f"Next: {report['next_step']}")
        print("Details: prompt-manager doctor --help")
    return 0 if report["ok"] else 1
