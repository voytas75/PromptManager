"""Bounded, provider-free CLI diagnosis before normal application startup."""

from __future__ import annotations

import csv
import json
import logging
import os
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

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


def _diagnose_catalog() -> dict[str, object]:
    """Run the existing integrity rules on a separate read-only catalog adapter."""
    from core.catalog_check import run_catalog_check
    from core.repository.read_only_catalog import CatalogReadError, read_existing_catalog

    try:
        settings_logger = logging.getLogger("prompt_manager.settings")
        previous_level = settings_logger.level
        settings_logger.setLevel(logging.ERROR)
        try:
            settings = load_settings()
        finally:
            settings_logger.setLevel(previous_level)
    except SettingsError:
        return {
            "schema_version": 1,
            "command": "doctor catalog",
            "ok": False,
            "status": "FAIL",
            "report": None,
            "code": "CONFIG_INVALID",
            "next_step": "Check the selected configuration file",
        }
    if not settings.db_path.exists():
        summary = {"prompts": 0, "chains": 0, "errors": 0, "warnings": 0}
        return {
            "schema_version": 1,
            "command": "doctor catalog",
            "ok": True,
            "status": "WARN",
            "report": {"summary": summary, "issues": []},
            "code": "DB_NOT_CREATED",
            "next_step": "Start PromptManager to create the local catalog",
        }
    try:
        prompts, chains = read_existing_catalog(settings.db_path)
        result = run_catalog_check(prompts, chains)
    except CatalogReadError:
        return {
            "schema_version": 1,
            "command": "doctor catalog",
            "ok": False,
            "status": "FAIL",
            "report": None,
            "code": "DB_UNREADABLE",
            "next_step": "Check the existing catalog and retry after closing writers",
        }
    # The legacy checker may quote names or Jinja lines in its messages. Keep
    # issue identifiers and severities, but never echo source text in doctor.
    report = {
        "summary": result.to_record()["summary"],
        "issues": [
            {
                "code": issue.code,
                "severity": issue.severity,
                "prompt_ids": issue.prompt_ids,
                "chain_id": issue.chain_id,
            }
            for issue in result.issues
        ],
    }
    return {
        "schema_version": 1,
        "command": "doctor catalog",
        "ok": result.error_count == 0,
        "status": "FAIL" if result.error_count else "WARN" if result.warning_count else "OK",
        "report": report,
        "code": "CATALOG_ISSUES" if result.error_count else "CATALOG_CHECKED",
        "next_step": "Review reported catalog issue codes"
        if result.issues
        else "Catalog records passed",
    }


def _live_embedding_check(settings: PromptManagerSettings) -> tuple[Check, int | None]:
    """Probe only the configured remote backend with one synthetic embedding request."""
    import math

    from core.embedding import LiteLLMEmbeddingFunction
    from core.litellm_adapter import get_embedding

    backend = (settings.embedding_backend or "deterministic").strip().lower()
    if backend not in {"litellm", "openai"}:
        return (
            Check(
                "search_embeddings",
                "SKIP",
                "EMBEDDING_LIVE_UNSUPPORTED",
                "Live probe supports LiteLLM only; vector index not inspected",
                "Use the offline readiness check for this backend",
            ),
            None,
        )
    if _embedding_check(settings).code == "EMBEDDING_NOT_CONFIGURED":
        return (
            Check(
                "search_embeddings",
                "WARN",
                "EMBEDDING_NOT_CONFIGURED",
                "Embedding backend configuration incomplete; no probe was run",
                "Configure embedding credentials and model before retrying",
            ),
            None,
        )
    try:
        embedding, _ = get_embedding()
        request: dict[str, object] = {
            "model": settings.embedding_model,
            "input": ["Prompt Manager diagnostics probe"],
            "api_key": settings.litellm_api_key,
            "timeout": 15,
            "num_retries": 0,
        }
        if settings.litellm_api_base:
            request["api_base"] = settings.litellm_api_base
        if settings.litellm_api_version:
            request["api_version"] = settings.litellm_api_version
        response = embedding(**request)
        payload = LiteLLMEmbeddingFunction._extract_payload(response)  # pyright: ignore[reportPrivateUsage]
        data = LiteLLMEmbeddingFunction._extract_data_array(payload)  # pyright: ignore[reportPrivateUsage]
        if len(data) != 1:
            raise ValueError("Unexpected embedding count")
        vector = LiteLLMEmbeddingFunction._extract_embedding_vector(data[0], 0)  # pyright: ignore[reportPrivateUsage]
        if not vector or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in vector
        ):
            raise ValueError("Invalid embedding vector")
    except Exception:  # provider exceptions can include credentials and raw response text
        return (
            Check(
                "search_embeddings",
                "FAIL",
                "EMBEDDING_PROBE_FAILED",
                "Embedding probe failed; vector index not inspected",
                "Check provider configuration and availability before a new approved probe",
            ),
            None,
        )
    return (
        Check(
            "search_embeddings",
            "OK",
            "EMBEDDING_BACKEND_REACHABLE",
            "Embedding backend returned a usable vector; vector index not inspected",
        ),
        len(vector),
    )


def _diagnose_focused(command: str, *, details: bool, live: bool = False) -> dict[str, object]:
    """Diagnose only the requested capability; never open the catalog here."""
    explicit, exists = _config_source()
    settings_logger = logging.getLogger("prompt_manager.settings")
    previous_level = settings_logger.level
    settings_logger.setLevel(logging.ERROR)
    try:
        settings = load_settings()
    except SettingsError:
        settings = None
    finally:
        settings_logger.setLevel(previous_level)
    if settings is None:
        config = Check(
            "config",
            "FAIL",
            "CONFIG_INVALID",
            "Effective configuration missing or invalid",
            "Check the selected configuration file and settings",
        )
    elif not explicit and not exists:
        config = Check(
            "config",
            "WARN",
            "DEFAULT_CONFIG_ABSENT",
            "Using valid effective defaults without a config file",
        )
    else:
        config = Check("config", "OK", "CONFIG_VALID", "Effective settings valid")
    live_check, dimension = (
        _live_embedding_check(settings)
        if live and command == "embeddings" and settings is not None
        else (None, None)
    )
    selected = (
        config
        if command == "config"
        else (
            live_check or _embedding_check(settings)
            if settings is not None
            else Check("search_embeddings", "SKIP", "CONFIG_REQUIRED", "Embedding settings unknown")
        )
    )
    required_failed = config.status == "FAIL" or selected.status == "FAIL"
    status: Status = (
        "FAIL"
        if required_failed
        else "WARN"
        if selected.status in {"WARN", "SKIP"} or config.status == "WARN"
        else "OK"
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "command": f"doctor {command}",
        "ok": not required_failed,
        "status": status,
        "checks": [asdict(config), asdict(selected)]
        if command == "embeddings"
        else [asdict(config)],
        "next_step": config.next_step if config.status == "FAIL" else selected.next_step,
    }
    if command == "config" and details and settings is not None:
        # Only allowlisted constant labels and booleans: config paths, DSNs,
        # model names, sources and free-form validator text can carry secrets.
        report["details"] = {
            "source": "explicit" if explicit else "default",
            "config_file_present": exists,
            "credential_present": bool(settings.litellm_api_key),
            "database_path_configured": bool(settings.db_path),
            "vector_path_configured": bool(settings.chroma_path),
        }
    if command == "embeddings" and live:
        report["probe"] = {
            "performed": dimension is not None or selected.code == "EMBEDDING_PROBE_FAILED",
            "backend_dimension": dimension,
            "vector_index": "not_inspected",
        }
    return report


def _diagnose_analytics(export_csv: Path | None, *, live: bool = False) -> dict[str, object]:
    """Summarize stored counts; only an explicit live request probes the backend."""
    from core.repository.read_only_analytics import read_execution_counts
    from core.repository.read_only_catalog import CatalogReadError

    settings_logger = logging.getLogger("prompt_manager.settings")
    previous_level = settings_logger.level
    settings_logger.setLevel(logging.ERROR)
    try:
        settings = load_settings()
    except SettingsError:
        settings = None
    finally:
        settings_logger.setLevel(previous_level)
    code, status, summary = "ANALYTICS_REPORTED", "OK", None
    if settings is None:
        code, status = "CONFIG_INVALID", "FAIL"
    elif not settings.db_path.exists():
        code, status = "DB_NOT_CREATED", "WARN"
        summary = {"total_runs": 0, "success_runs": 0}
    else:
        try:
            summary = read_execution_counts(settings.db_path)
        except CatalogReadError:
            code, status = "DB_UNREADABLE", "FAIL"

    report: dict[str, object] = {
        "schema_version": 1,
        "command": "doctor analytics",
        "kind": "report_not_health",
        "ok": status != "FAIL",
        "status": status,
        "code": code,
        "report": summary,
        "exported": False,
        "next_step": "Review stored execution counts"
        if summary is not None
        else "Check local catalog",
    }
    if export_csv is not None and summary is not None:
        try:
            with export_csv.open("x", newline="", encoding="utf-8") as destination:
                writer = csv.DictWriter(destination, fieldnames=("total_runs", "success_runs"))
                writer.writeheader()
                writer.writerow(
                    {"total_runs": summary["total_runs"], "success_runs": summary["success_runs"]}
                )
        except FileExistsError:
            report.update(
                ok=False, status="FAIL", code="EXPORT_EXISTS", next_step="Choose a new CSV path"
            )
        except OSError:
            report.update(
                ok=False, status="FAIL", code="EXPORT_FAILED", next_step="Check CSV destination"
            )
        else:
            report["exported"] = True
    if live and report["ok"] and code == "ANALYTICS_REPORTED":
        assert settings is not None
        check, dimension = _live_embedding_check(settings)
        report["probe"] = {
            "performed": dimension is not None or check.code == "EMBEDDING_PROBE_FAILED",
            "backend_dimension": dimension,
            "vector_index": "not_inspected",
            "code": check.code,
            "status": check.status,
        }
        if check.status == "FAIL":
            report.update(ok=False, status="FAIL", code=check.code, next_step=check.next_step)
        elif check.status in {"WARN", "SKIP"}:
            report.update(status="WARN", code=check.code, next_step=check.next_step)
    return report


def _asset_report(
    command: str,
    code: str,
    *,
    summary: dict[str, object] | None = None,
    issues: list[dict[str, str]] | None = None,
    prompt_id: str | None = None,
    ok: bool = True,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "command": command,
        "ok": ok,
        "status": "OK" if ok and not issues else "WARN" if ok else "FAIL",
        "code": code,
        "report": None
        if summary is None
        else {"prompt_id": prompt_id, "summary": summary, "issues": issues or []},
        "next_step": (
            "Review finding codes"
            if issues
            else "Check input and retry"
            if not ok
            else "No action needed"
        ),
    }


def _diagnose_prompt(reference: str, action: str, suite: Path | None) -> dict[str, object]:
    """Run only deterministic prompt checks on a read-only catalog snapshot."""
    from core.prompt_linting import lint_prompt
    from core.prompt_testing import parse_prompt_test_suite, run_prompt_test_suite
    from core.prompt_validation import validate_prompt
    from core.repository.read_only_catalog import CatalogReadError, read_existing_catalog

    command = f"doctor prompt {action}"
    settings_logger = logging.getLogger("prompt_manager.settings")
    previous_level = settings_logger.level
    settings_logger.setLevel(logging.ERROR)
    try:
        settings = load_settings()
    except SettingsError:
        return _asset_report(command, "CONFIG_INVALID", ok=False)
    finally:
        settings_logger.setLevel(previous_level)
    try:
        prompts, _ = read_existing_catalog(settings.db_path)
    except CatalogReadError:
        return _asset_report(command, "DB_UNREADABLE", ok=False)
    try:
        prompt_id = uuid.UUID(reference)
    except (ValueError, TypeError):
        prompt_id = None
    if prompt_id is not None:
        matches = [prompt for prompt in prompts if prompt.id == prompt_id]
        if not matches:
            matches = [prompt for prompt in prompts if prompt.name == reference]
    else:
        matches = [prompt for prompt in prompts if prompt.name == reference]
    if not matches:
        return _asset_report(command, "PROMPT_NOT_FOUND", ok=False)
    if len(matches) != 1:
        return _asset_report(command, "PROMPT_AMBIGUOUS", ok=False)
    prompt = matches[0]
    if action == "validate":
        validation = validate_prompt(prompt, (item.id for item in prompts))
        issues = [{"code": issue.code, "severity": issue.severity} for issue in validation.issues]
        return _asset_report(
            command,
            "PROMPT_VALIDATED",
            summary={"errors": validation.error_count, "warnings": validation.warning_count},
            issues=issues,
            prompt_id=str(prompt.id),
            ok=validation.valid,
        )
    if action == "lint":
        lint = lint_prompt(prompt)
        issues = [{"code": issue.code, "severity": issue.severity} for issue in lint.issues]
        return _asset_report(
            command,
            "PROMPT_LINTED",
            summary={"warnings": lint.warning_count},
            issues=issues,
            prompt_id=str(prompt.id),
        )
    if action == "test" and suite is not None:
        try:
            with suite.open("r", encoding="utf-8") as source:
                source_text = source.read(65537)
            if len(source_text) > 65536:
                return _asset_report(command, "SUITE_INVALID", ok=False)
            raw = json.loads(source_text)
            cases = parse_prompt_test_suite(raw)
            if len(cases) > 64 or any(len(case.expected) > 16384 for case in cases):
                return _asset_report(command, "SUITE_INVALID", ok=False)
        except (OSError, ValueError, UnicodeError):
            return _asset_report(command, "SUITE_INVALID", ok=False)
        from core.templating import DoctorFixtureRenderer

        tested = run_prompt_test_suite(
            prompt, cases, suite_path="(local)", renderer=DoctorFixtureRenderer()
        )
        issues = [
            {"code": "TEST_FAILED", "severity": "error"}
            for case in tested.cases
            if case.status == "failed"
        ]
        return _asset_report(
            command,
            "PROMPT_TESTED",
            summary={
                "total": tested.case_count,
                "passed": tested.passed_count,
                "failed": tested.failed_count,
            },
            issues=issues,
            prompt_id=str(prompt.id),
            ok=tested.ok,
        )
    return _asset_report(command, "INVALID_ACTION", ok=False)


def _diagnose_chain(path: Path) -> dict[str, object]:
    """Validate a chain definition using the production pure model parser."""
    from models.prompt_chain_model import chain_from_payload

    command = "doctor chain validate"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Not a mapping")
        payload = cast("dict[str, object]", raw)
        chain = chain_from_payload(payload)
        if not chain.steps:
            raise ValueError("No steps")
    except (OSError, UnicodeError, ValueError, TypeError, KeyError):
        return _asset_report(command, "CHAIN_INVALID", ok=False)
    return _asset_report(command, "CHAIN_VALID", summary={"step_count": len(chain.steps)})


def run_doctor(
    *,
    json_output: bool = False,
    command: str | None = None,
    details: bool = False,
    live: bool = False,
    export_csv: Path | None = None,
    action: str | None = None,
    reference: str | None = None,
    suite: Path | None = None,
    path: Path | None = None,
) -> int:
    """Emit a sanitized report or bounded unexpected error without creating state."""
    try:
        if command == "catalog":
            catalog_report = _diagnose_catalog()
            if json_output:
                print(json.dumps(catalog_report, ensure_ascii=False))
            else:
                print(f"Doctor catalog: {catalog_report['status']} ({catalog_report['code']})")
                if catalog_report["report"] is not None:
                    print(json.dumps(catalog_report["report"], ensure_ascii=False))
                print(f"Next: {catalog_report['next_step']}")
            return 0 if catalog_report["ok"] else 1
        if command == "analytics":
            analytics_report = _diagnose_analytics(export_csv, live=live)
            if json_output:
                print(json.dumps(analytics_report, ensure_ascii=False))
            else:
                print(f"Doctor analytics (report, not health): {analytics_report['status']}")
                if analytics_report["report"] is not None:
                    print(json.dumps(analytics_report["report"], ensure_ascii=False))
                if "probe" in analytics_report:
                    probe = analytics_report["probe"]
                    assert isinstance(probe, dict)
                    print(f"Embedding probe: {probe['status']} ({probe['code']})")
                print(f"Next: {analytics_report['next_step']}")
            return 0 if analytics_report["ok"] else 1
        if command in {"prompt", "chain"}:
            asset = (
                _diagnose_prompt(reference or "", action or "", suite)
                if command == "prompt"
                else _diagnose_chain(path or Path(""))
            )
            if json_output:
                print(json.dumps(asset, ensure_ascii=False))
            else:
                print(f"{asset['command']}: {asset['status']} ({asset['code']})")
                if asset["report"] is not None:
                    print(json.dumps(asset["report"], ensure_ascii=False))
                print(f"Next: {asset['next_step']}")
            return 0 if asset["ok"] else 1
        if command in {"config", "embeddings"}:
            focused = _diagnose_focused(command, details=details, live=live)
            if json_output:
                print(json.dumps(focused, ensure_ascii=False))
            else:
                print(f"Doctor {command}: {focused['status']}")
                raw_checks = focused["checks"]
                if isinstance(raw_checks, list):
                    for item in cast("list[object]", raw_checks):
                        if isinstance(item, dict):
                            print(f"{item['status']} {item['id']}: {item['message']}")
                if "details" in focused:
                    print(json.dumps(focused["details"], ensure_ascii=False))
                print(f"Next: {focused['next_step'] or 'No action needed'}")
            return 0 if focused["ok"] else 1
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
