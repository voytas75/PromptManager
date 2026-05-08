"""CLI command handlers for Prompt Manager.

Updates:
  v0.33.9 - 2026-04-29 - Add prompt-history JSON output for structured execution evidence reads.
  v0.33.8 - 2026-04-29 - Add prompt-history CLI command for per-prompt execution evidence.
  v0.33.7 - 2026-04-29 - Add prompt-find source and active filters
    for deterministic console discovery.
  v0.33.6 - 2026-04-29 - Add prompt-find category and tag filters
    for deterministic prompt discovery.
  v0.33.5 - 2026-04-29 - Add prompt-find JSON output for structured console lists.
  v0.33.4 - 2026-04-29 - Add prompt-show JSON output for structured console reads.
  v0.33.3 - 2026-04-29 - Add prompt-find CLI command for deterministic prompt listing.
  v0.33.2 - 2026-04-29 - Add prompt-show CLI command for deterministic single-prompt reads.
  v0.33.1 - 2025-12-08 - Surface token usage totals in history analytics output.
  v0.33.0 - 2025-12-06 - Switch prompt chain CLI to plain-text inputs.
  v0.32.2 - 2025-12-05 - Add prompt chain web search toggle wiring for CLI runs.
  v0.32.1 - 2025-12-05 - Sort imports for lint compliance.
  v0.32.0 - 2025-12-04 - Reuse shared chain_from_payload helper for JSON imports.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
import uuid
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core import (
    PromptChainError,
    PromptChainExecutionError,
    PromptHistoryError,
    PromptManagerError,
    RepositoryNotFoundError,
    TokenUsageTotals,
    build_analytics_snapshot,
    diff_prompt_catalog,
    export_prompt_catalog,
    import_prompt_catalog,
    snapshot_dataset_rows,
)
from models.prompt_chain_model import (
    chain_from_payload,
)

from .utils import (
    format_metric,
    print_and_log,
    resolve_export_format,
    write_csv_rows,
)

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from core.prompt_manager import PromptManager
else:  # pragma: no cover - runtime placeholders for type-only imports
    PromptManager = object

CommandHandler = Callable[[PromptManager | None, argparse.Namespace, logging.Logger], int]


@dataclass(frozen=True)
class CommandSpec:
    """Metadata for dispatching CLI command handlers."""

    handler: CommandHandler
    requires_manager: bool = True


def _coerce_mapping(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): value[key] for key in value}
    return None


def _load_json_file(path: Path) -> Any:
    try:
        content = path.expanduser().read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - IO error
        raise ValueError(f"Unable to read {path}: {exc}") from exc
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _resolve_chain_input(
    inline_text: str | None,
    file_path: Path | None,
) -> str:
    if inline_text and file_path:
        raise ValueError("Specify only one of --input or --input-file.")
    raw_text = inline_text or ""
    if file_path:
        try:
            raw_text = file_path.expanduser().read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"Unable to read {file_path}: {exc}") from exc
    trimmed = raw_text.strip()
    if not trimmed:
        raise ValueError("Chain input must not be empty.")
    return raw_text


def _get_main_callable(name: str, fallback: Callable[..., Any]) -> Callable[..., Any]:
    module = sys.modules.get("main")
    attr = getattr(module, name, None) if module else None
    return attr if callable(attr) else fallback


def run_catalog_export(
    manager: PromptManager | None,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> int:
    if manager is None:
        raise ValueError("Prompt Manager is required for catalog export.")
    output_path = Path(args.path).expanduser()
    fmt = resolve_export_format(output_path, getattr(args, "format", None))
    export_fn = _get_main_callable("export_prompt_catalog", export_prompt_catalog)
    try:
        resolved = export_fn(
            manager,
            output_path,
            fmt=fmt,
            include_inactive=getattr(args, "include_inactive", False),
        )
    except Exception as exc:  # pragma: no cover - surfaced to CLI
        message = f"Failed to export catalogue: {exc}"
        print_and_log(logger, logging.ERROR, message)
        return 6
    message = f"Prompt catalogue exported to {resolved} ({fmt})"
    print_and_log(logger, logging.INFO, message)
    return 0


def run_catalog_import(
    manager: PromptManager | None,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> int:
    if manager is None:
        raise ValueError("Prompt Manager is required for catalog import.")
    input_path = Path(args.path).expanduser()
    overwrite = not bool(getattr(args, "no_overwrite", False))
    if getattr(args, "dry_run", False):
        diff_fn = _get_main_callable("diff_prompt_catalog", diff_prompt_catalog)
        try:
            diff = diff_fn(manager, input_path, overwrite=overwrite)
        except Exception as exc:  # pragma: no cover - surfaced to CLI
            message = f"Failed to preview catalogue import: {exc}"
            print_and_log(logger, logging.ERROR, message)
            return 6
        print_and_log(
            logger,
            logging.INFO,
            "Catalog import preview: "
            f"added={diff.added} updated={diff.updated} "
            f"skipped={diff.skipped} unchanged={diff.unchanged}",
        )
        return 0

    import_fn = _get_main_callable("import_prompt_catalog", import_prompt_catalog)
    try:
        result = import_fn(manager, input_path, overwrite=overwrite)
    except Exception as exc:  # pragma: no cover - surfaced to CLI
        message = f"Failed to import catalogue: {exc}"
        print_and_log(logger, logging.ERROR, message)
        return 6
    print_and_log(
        logger,
        logging.INFO,
        "Catalog import applied: "
        f"added={result.added} updated={result.updated} "
        f"skipped={result.skipped} errors={result.errors}",
    )
    return 0 if result.errors == 0 else 6


def run_usage_report(
    manager: PromptManager | None,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> int:
    del manager
    path_value = getattr(args, "path", None)
    log_path = Path(path_value or Path("data") / "logs" / "intent_usage.jsonl").expanduser()
    if not log_path.exists():
        logger.info("Usage log not found at %s", log_path)
        return 0

    events = []
    try:
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping invalid log line: %s", line[:80])
    except OSError as exc:
        logger.error("Unable to read usage log: %s", exc)
        return 5

    if not events:
        print(f"No intent workspace events recorded in {log_path}")
        return 0

    total_events = len(events)
    by_event = Counter(event.get("event", "unknown") for event in events)
    labels = Counter(
        event.get("label", "unknown")
        for event in events
        if event.get("event") in {"detect", "suggest"}
    )
    top_prompts = Counter()
    for event in events:
        if event.get("event") == "suggest":
            for name in event.get("top_prompts", []):
                if name:
                    top_prompts[name] += 1

    print(f"Usage summary from {log_path}:\n")
    print(f"Total events: {total_events}")
    for event, count in sorted(by_event.items()):
        print(f"  {event}: {count}")

    if labels:
        print("\nTop inferred intents:")
        for label, count in labels.most_common(5):
            print(f"  {label}: {count}")

    if top_prompts:
        print("\nTop recommended prompts:")
        for name, count in top_prompts.most_common(5):
            print(f"  {name}: {count}")

    return 0


def run_benchmark(
    manager: PromptManager | None,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> int:
    if manager is None:
        raise ValueError("Prompt Manager is required for benchmarking.")
    prompt_values: list[str] = list(getattr(args, "prompt_ids", []) or [])
    if not prompt_values:
        logger.error("At least one --prompt value is required for benchmarking.")
        return 5
    try:
        prompt_ids = [uuid.UUID(value) for value in prompt_values]
    except (ValueError, TypeError) as exc:
        logger.error("Invalid prompt identifier: %s", exc)
        return 5

    request_text = getattr(args, "request", None)
    request_file = getattr(args, "request_file", None)
    if request_file is not None:
        try:
            request_text = Path(request_file).expanduser().read_text(encoding="utf-8")
        except OSError as exc:
            logger.error("Unable to read request file: %s", exc)
            return 5
    if not request_text or not request_text.strip():
        logger.error("Benchmark input must be supplied via --request or --request-file.")
        return 5

    history_window = getattr(args, "history_window", None)
    if history_window is not None and history_window <= 0:
        history_window = None
    trend_window = max(1, int(getattr(args, "trend_window", 5) or 5))
    models = getattr(args, "models", None)

    try:
        report = manager.benchmark_prompts(
            prompt_ids,
            request_text,
            models=models,
            persist_history=getattr(args, "persist_history", False),
            history_window_days=history_window,
            trend_window=trend_window,
        )
    except PromptManagerError as exc:
        logger.error("Benchmark failed: %s", exc)
        return 5

    if not report.runs:
        logger.info("No benchmark runs were executed.")
        return 0

    print("\nBenchmark results\n-----------------")
    for run in report.runs:
        status = "ERROR" if run.error else "OK"
        usage_parts = []
        usage_map = run.usage if isinstance(run.usage, dict) else {}
        prompt_tokens = usage_map.get("prompt_tokens")
        completion_tokens = usage_map.get("completion_tokens")
        total_tokens = usage_map.get("total_tokens")
        if prompt_tokens is not None:
            usage_parts.append(f"prompt={prompt_tokens}")
        if completion_tokens is not None:
            usage_parts.append(f"completion={completion_tokens}")
        if total_tokens is not None:
            usage_parts.append(f"total={total_tokens}")
        usage_text = f"tokens({', '.join(usage_parts)})" if usage_parts else "tokens(n/a)"

        if run.error:
            print(f"- {run.prompt_name} [{run.model}] -> {status}: {run.error}")
        else:
            duration_text = f"{run.duration_ms} ms" if run.duration_ms is not None else "n/a"
            print(f"- {run.prompt_name} [{run.model}] -> {status}: {duration_text}, {usage_text}")
            preview = run.response_preview.replace("\n", " ") if run.response_preview else ""
            if preview:
                if len(preview) > 120:
                    preview = preview[:117].rstrip() + "..."
                print(f"  preview: {preview}")
        if run.history:
            history = run.history
            success_rate = f"{history.success_rate * 100:.1f}%" if history.success_rate else "0%"
            avg_duration = (
                f"{history.average_duration_ms:.0f} ms" if history.average_duration_ms else "n/a"
            )
            avg_rating = (
                f"{history.average_rating:.1f}" if history.average_rating is not None else "n/a"
            )
            print(
                f"  history: runs={history.total_runs}, success_rate={success_rate}, "
                f"avg_duration={avg_duration}, avg_rating={avg_rating}"
            )
        if run.error is None and not run.response_preview:
            print("  preview: (empty response)")

    return 0


def run_refresh_scenarios(
    manager: PromptManager | None,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> int:
    if manager is None:
        raise ValueError("Prompt Manager is required for scenario refresh.")
    try:
        prompt_id = uuid.UUID(str(args.prompt_id))
    except (ValueError, TypeError) as exc:
        logger.error("Invalid prompt id: %s", exc)
        return 5

    max_scenarios = max(1, int(getattr(args, "max_scenarios", 3) or 3))
    try:
        prompt = manager.refresh_prompt_scenarios(prompt_id, max_scenarios=max_scenarios)
    except PromptManagerError as exc:
        logger.error("Failed to refresh scenarios: %s", exc)
        return 5

    print(f"Updated scenarios for {prompt.name}:")
    if prompt.scenarios:
        for scenario in prompt.scenarios:
            print(f" - {scenario}")
    else:
        print(" - (none)")
    return 0


def run_diagnostics(
    manager: PromptManager | None,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> int:
    if manager is None:
        raise ValueError("Prompt Manager is required for diagnostics.")
    target = getattr(args, "target", None)
    if target == "embeddings":
        return _run_embedding_diagnostics(manager, args, logger)
    if target == "analytics":
        return _run_analytics_diagnostics(manager, args, logger)
    logger.error("Unknown diagnostics target: %s", target)
    return 5


def _run_embedding_diagnostics(
    manager: PromptManager,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> int:
    sample_text = getattr(args, "sample_text", "Prompt Manager diagnostics probe")
    try:
        report = manager.diagnose_embeddings(sample_text=sample_text)
    except PromptManagerError as exc:
        logger.error("Embedding diagnostics failed: %s", exc)
        return 5

    print("\nEmbedding diagnostics\n---------------------")
    dimension = report.backend_dimension or report.inferred_dimension
    dimension_text = str(dimension) if dimension is not None else "unknown"
    backend_status = "OK" if report.backend_ok else "ERROR"
    print(f"Backend: {backend_status} (dimension={dimension_text}) - {report.backend_message}")

    chroma_status = "OK" if report.chroma_ok else "ERROR"
    chroma_count_text = str(report.chroma_count) if report.chroma_count is not None else "unknown"
    print(f"Chroma: {chroma_status} (documents={chroma_count_text}) - {report.chroma_message}")

    missing_count = len(report.missing_prompts)
    print(
        f"Repository: {report.repository_total} prompts "
        f"({report.prompts_with_embeddings} with embeddings, missing={missing_count})"
    )

    if report.consistent_counts is None:
        print("Vector store consistency: unknown (Chroma document count unavailable)")
    elif report.consistent_counts:
        print("Vector store consistency: OK (counts match)")
    else:
        chroma_value = report.chroma_count if report.chroma_count is not None else "unknown"
        print(
            f"Vector store consistency: MISMATCH (Chroma={chroma_value}, "
            f"stored embeddings={report.prompts_with_embeddings})"
        )

    if missing_count:
        print(f"\nPrompts missing embeddings ({missing_count}):")
        for issue in report.missing_prompts[:10]:
            name = issue.prompt_name or "Unnamed prompt"
            print(f" - {name} ({issue.prompt_id})")
        if missing_count > 10:
            print(f"   ... {missing_count - 10} more")

    mismatch_count = len(report.mismatched_prompts)
    if mismatch_count:
        print(f"\nDimension mismatches ({mismatch_count} prompts):")
        for mismatch in report.mismatched_prompts[:10]:
            name = mismatch.prompt_name or "Unnamed prompt"
            print(f" - {name} ({mismatch.prompt_id}) stored={mismatch.stored_dimension}")
        if mismatch_count > 10:
            print(f"   ... {mismatch_count - 10} more")

    issues: list[str] = []
    if not report.backend_ok:
        issues.append("backend")
    if not report.chroma_ok:
        issues.append("chroma")
    if mismatch_count:
        issues.append("dimension mismatches")
    if report.consistent_counts is False:
        issues.append("vector store count mismatch")

    if issues:
        logger.warning(
            "Embedding diagnostics completed with issues: %s",
            ", ".join(issues),
        )
        return 6
    logger.info("Embedding diagnostics completed successfully.")
    return 0


def _run_analytics_diagnostics(
    manager: PromptManager,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> int:
    window_days = max(0, int(getattr(args, "window_days", 30) or 0))
    prompt_limit = max(1, int(getattr(args, "prompt_limit", 5) or 1))
    usage_log_path = getattr(args, "usage_log", None)

    snapshot_builder = _get_main_callable("build_analytics_snapshot", build_analytics_snapshot)
    snapshot = snapshot_builder(
        manager,
        window_days=window_days if window_days > 0 else 0,
        prompt_limit=prompt_limit,
        usage_log_path=usage_log_path,
    )

    def _pct(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value * 100:.1f}%"

    print("\nAnalytics dashboard\n-------------------")

    execution = snapshot.execution
    if execution is None:
        print("No execution history analytics available.")
    else:
        avg_duration = (
            f"{execution.average_duration_ms:.0f} ms"
            if execution.average_duration_ms is not None
            else "n/a"
        )
        avg_rating = (
            f"{execution.average_rating:.2f}" if execution.average_rating is not None else "n/a"
        )
        print(
            textwrap.dedent(
                f"""
                Execution summary (last {window_days or "all"} days)
                  runs: {execution.total_runs}
                  success rate: {_pct(execution.success_rate)}
                  average duration: {avg_duration}
                  average rating: {avg_rating}
                """
            ).strip()
        )
        if execution.prompt_breakdown:
            print("\nTop prompt trends:")
            for prompt_stats in execution.prompt_breakdown:
                avg_duration_prompt = (
                    f"{prompt_stats.average_duration_ms:.0f} ms"
                    if prompt_stats.average_duration_ms is not None
                    else "n/a"
                )
                avg_rating_prompt = (
                    f"{prompt_stats.average_rating:.2f}"
                    if prompt_stats.average_rating is not None
                    else "n/a"
                )
                print(
                    f"  - {prompt_stats.name}: runs={prompt_stats.total_runs}, "
                    f"success={_pct(prompt_stats.success_rate)}, "
                    f"duration={avg_duration_prompt}, rating={avg_rating_prompt}"
                )

    if snapshot.usage_frequency:
        print("\nCatalogue usage frequency:")
        for entry in snapshot.usage_frequency:
            last_used = entry.last_executed_at.isoformat() if entry.last_executed_at else "n/a"
            print(
                f"  - {entry.name}: counter={entry.usage_count}, "
                f"success={_pct(entry.success_rate)}, last_used={last_used}"
            )

    if snapshot.model_costs:
        print("\nModel cost breakdown (tokens):")
        for entry in snapshot.model_costs:
            print(
                f"  - {entry.model}: runs={entry.run_count}, prompt={entry.prompt_tokens}, "
                f"completion={entry.completion_tokens}, total={entry.total_tokens}"
            )

    if snapshot.benchmark_stats:
        print("\nBenchmark success by model:")
        for entry in snapshot.benchmark_stats:
            avg_duration = (
                f"{entry.average_duration_ms:.0f} ms"
                if entry.average_duration_ms is not None
                else "n/a"
            )
            print(
                f"  - {entry.model}: runs={entry.run_count}, "
                f"success={_pct(entry.success_rate)}, duration={avg_duration}, "
                f"tokens={entry.total_tokens}"
            )

    if snapshot.intent_success:
        print("\nIntent workspace execution success:")
        for point in snapshot.intent_success[-10:]:
            print(
                f"  - {point.bucket.date().isoformat()}: {_pct(point.success_rate)} "
                f"({point.success}/{point.total})"
            )

    embedding = snapshot.embedding
    if embedding is not None:
        print("\nEmbedding diagnostics summary:")
        print(
            f"  backend: {'ok' if embedding.backend_ok else 'error'} ({embedding.backend_message})"
        )
        print(
            "  dimension: {value}".format(
                value=embedding.backend_dimension or embedding.inferred_dimension or "n/a",
            )
        )
        print(f"  chroma: {'ok' if embedding.chroma_ok else 'error'} ({embedding.chroma_message})")
        if embedding.consistent_counts is not None:
            status = "matched" if embedding.consistent_counts else "mismatch"
            print(
                f"  vectors stored: {embedding.prompts_with_embeddings} / "
                f"repository {embedding.repository_total} (chroma {status})"
            )

    export_path = getattr(args, "export_csv", None)
    if export_path:
        dataset = getattr(args, "dataset", "usage")
        dataset_rows_fn = _get_main_callable("snapshot_dataset_rows", snapshot_dataset_rows)
        try:
            rows = dataset_rows_fn(snapshot, dataset)
        except ValueError as exc:
            logger.error("%s", exc)
            return 5
        if not rows:
            logger.info("Dataset '%s' is empty; skipping export.", dataset)
        else:
            try:
                resolved = write_csv_rows(Path(export_path), rows)
            except (OSError, ValueError) as exc:
                logger.error("Unable to export analytics dataset: %s", exc)
                return 5
            logger.info("Analytics dataset '%s' exported to %s", dataset, resolved)

    return 0


def run_prompt_chain_list(
    manager: PromptManager | None,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> int:
    if manager is None:
        raise ValueError("Prompt Manager is required for prompt chain listing.")
    include_inactive = bool(getattr(args, "include_inactive", False))
    try:
        chains = manager.list_prompt_chains(include_inactive=include_inactive)
    except PromptChainError as exc:
        logger.error("Unable to list prompt chains: %s", exc)
        return 5
    if not chains:
        print("No prompt chains defined.")
        return 0
    print("\nPrompt Chains\n-------------")
    for chain in chains:
        status = "active" if chain.is_active else "inactive"
        print(f"- {chain.name} ({chain.id}) [{status}] steps={len(chain.steps)}")
    return 0


def _format_chain_legacy_semantics(chain: Any) -> dict[str, Any]:
    metadata = _coerce_mapping(getattr(chain, "metadata", None)) or {}
    legacy = metadata.get("legacy_chain_fields")
    if isinstance(legacy, Mapping):
        return {
            str(key): dict(value) for key, value in legacy.items() if isinstance(value, Mapping)
        }
    return {}


def _format_step_legacy_semantics(step: Any) -> dict[str, Any]:
    metadata = _coerce_mapping(getattr(step, "metadata", None)) or {}
    legacy_fields = metadata.get("legacy_runtime_fields")
    if not isinstance(legacy_fields, Mapping):
        return {}
    compatibility_note = "Compatibility-only field preserved for import/export boundaries."
    semantics: dict[str, Any] = {}
    if "input_template" in legacy_fields:
        semantics["input_template"] = {
            "status": "inactive",
            "note": compatibility_note,
        }
    if "condition" in legacy_fields:
        semantics["condition"] = {
            "status": "inactive",
            "note": compatibility_note,
        }
    if "output_variable" in legacy_fields:
        semantics["output_variable"] = {
            "status": "inactive_alias",
            "note": (
                "Legacy alias field preserved for import/export boundaries; "
                "active runtime uses canonical step_output_key."
            ),
        }
    return semantics


def run_prompt_chain_show(
    manager: PromptManager | None,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> int:
    if manager is None:
        raise ValueError("Prompt Manager is required for prompt chain inspection.")
    try:
        chain_id = uuid.UUID(str(args.chain_id))
    except (TypeError, ValueError) as exc:
        logger.error("Invalid chain id: %s", exc)
        return 5
    try:
        chain = manager.get_prompt_chain(chain_id)
    except PromptChainError as exc:
        logger.error("Unable to load prompt chain: %s", exc)
        return 5
    history_limit = max(1, int(getattr(args, "history_limit", 3) or 3))
    recent_runs = [
        record
        for record in manager.list_recent_prompt_chain_runs(limit=history_limit * 5)
        if str(record.get("chain_id") or "") == str(chain.id)
    ][:history_limit]
    status = "active" if chain.is_active else "inactive"
    chain_legacy_semantics = _format_chain_legacy_semantics(chain)
    if bool(getattr(args, "json", False)):
        payload = {
            "id": str(chain.id),
            "name": chain.name,
            "description": chain.description,
            "status": status,
            "is_active": chain.is_active,
            "summarize_last_response": bool(chain.summarize_last_response),
            "legacy_semantics": chain_legacy_semantics,
            "recent_runs": recent_runs,
            "steps": [
                {
                    "id": str(step.id),
                    "order_index": step.order_index,
                    "prompt_id": str(step.prompt_id),
                    "step_label": step.output_variable or f"step_{step.order_index}",
                    "stop_on_failure": bool(step.stop_on_failure),
                    "legacy_semantics": _format_step_legacy_semantics(step),
                }
                for step in chain.steps
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print(f"\nChain: {chain.name} ({chain.id}) [{status}]")
    if chain.description:
        print(f"Description: {chain.description}")
    if chain_legacy_semantics:
        print("Legacy semantics:")
        for field_name, _details in chain_legacy_semantics.items():
            print(f"  - {field_name}: inactive compatibility-only field")
    if recent_runs:
        print("\nRecent runs:")
        for record in recent_runs:
            timestamp = str(record.get("run_timestamp") or "unknown time")
            run_status = str(record.get("status") or "unknown")
            input_preview = str(record.get("input_preview") or "(empty)")
            output_preview = str(record.get("final_output_preview") or "(empty)")
            print(f"  - {timestamp} [{run_status}] input: {input_preview}")
            print(f"    output: {output_preview}")
    if not chain.steps:
        print("Steps: (none)")
        return 0
    print("\nSteps:")
    for step in chain.steps:
        step_label = step.output_variable or str(step.prompt_id)
        failure_label = "Stop chain on failure" if step.stop_on_failure else "Continue on failure"
        print(f"  {step.order_index}. {step_label}")
        print(f"     Prompt: {step.prompt_id}")
        print(f"     Failure: {failure_label}")
        step_legacy_semantics = _format_step_legacy_semantics(step)
        if step_legacy_semantics:
            print(
                "     Legacy fields: input_template, output_variable, condition "
                "(inactive semantics)"
            )
            if "output_variable" in step_legacy_semantics:
                print(
                    "     - output_variable: compatibility alias only; "
                    "runtime uses canonical step_output_key"
                )
            if "input_template" in step_legacy_semantics:
                print("     - input_template: compatibility-only field; inactive in active runtime")
            if "condition" in step_legacy_semantics:
                print("     - condition: compatibility-only field; inactive in active runtime")
    return 0


def run_prompt_chain_history(
    manager: PromptManager | None,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> int:
    if manager is None:
        raise ValueError("Prompt Manager is required for prompt chain history inspection.")
    history_limit = max(1, int(getattr(args, "limit", 10) or 10))
    records = manager.list_recent_prompt_chain_runs(limit=history_limit)
    chain_id_text = str(getattr(args, "chain_id", "") or "").strip()
    if chain_id_text:
        try:
            chain_id = str(uuid.UUID(chain_id_text))
        except (TypeError, ValueError) as exc:
            logger.error("Invalid chain id: %s", exc)
            return 5
        records = [record for record in records if str(record.get("chain_id") or "") == chain_id]
    if bool(getattr(args, "json", False)):
        print(json.dumps(records, ensure_ascii=False, indent=2))
        return 0
    if not records:
        print("No recent prompt chain runs found.")
        return 0
    print("Recent prompt chain runs:")
    for record in records:
        timestamp = str(record.get("run_timestamp") or "unknown time")
        chain_name = str(record.get("chain_name") or "Unknown chain")
        run_status = str(record.get("status") or "unknown")
        input_preview = str(record.get("input_preview") or "(empty)")
        output_preview = str(record.get("final_output_preview") or "(empty)")
        print(f"  - {timestamp} [{run_status}] {chain_name}")
        print(f"    input: {input_preview}")
        print(f"    output: {output_preview}")
    return 0


def run_prompt_chain_export(
    manager: PromptManager | None,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> int:
    if manager is None:
        raise ValueError("Prompt Manager is required for prompt chain export.")
    try:
        chain_id = uuid.UUID(str(args.chain_id))
    except (TypeError, ValueError) as exc:
        logger.error("Invalid chain id: %s", exc)
        return 5
    try:
        chain = manager.get_prompt_chain(chain_id)
    except PromptChainError as exc:
        logger.error("Unable to load prompt chain: %s", exc)
        return 5
    payload = {
        "id": str(chain.id),
        "name": chain.name,
        "description": chain.description,
        "is_active": bool(chain.is_active),
        "summarize_last_response": bool(chain.summarize_last_response),
        "steps": [
            {
                "id": str(step.id),
                "prompt_id": str(step.prompt_id),
                "order_index": step.order_index,
                "stop_on_failure": bool(step.stop_on_failure),
            }
            for step in chain.steps
        ],
    }
    path = Path(args.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Exported prompt chain '{chain.name}' ({chain.id}) to {path}.")
    return 0


def _validate_prompt_chain_payload(path: Path) -> tuple[dict[str, Any], Any]:
    payload = _load_json_file(path)
    if not isinstance(payload, Mapping):
        raise ValueError("Chain definition must be a JSON object.")
    materialized_payload = dict(payload)
    chain = chain_from_payload(materialized_payload)
    if not chain.steps:
        raise ValueError("Prompt chain definition must include at least one step.")
    return materialized_payload, chain


def _build_prompt_chain_validate_report(
    payload: Mapping[str, Any],
    chain: Any,
) -> dict[str, Any]:
    warnings: list[str] = []
    chain_legacy_fields: list[str] = []
    steps_legacy_fields: dict[str, list[str]] = {}

    if payload.get("variables_schema") is not None:
        chain_legacy_fields.append("variables_schema")
        warnings.append("Chain uses compatibility-only legacy field: variables_schema")

    for step in getattr(chain, "steps", []):
        step_metadata = _coerce_mapping(getattr(step, "metadata", None)) or {}
        legacy_runtime_fields = step_metadata.get("legacy_runtime_fields")
        if not isinstance(legacy_runtime_fields, Mapping):
            continue
        step_fields: list[str] = []
        if legacy_runtime_fields.get("input_template") not in (None, ""):
            step_fields.append("input_template")
            warnings.append(f"Step {step.order_index} uses inactive legacy field: input_template")
        if legacy_runtime_fields.get("output_variable") not in (None, ""):
            step_fields.append("output_variable")
            warnings.append(
                f"Step {step.order_index} uses inactive legacy alias field: output_variable"
            )
        if legacy_runtime_fields.get("condition") not in (None, ""):
            step_fields.append("condition")
            warnings.append(f"Step {step.order_index} uses inactive legacy field: condition")
        if step_fields:
            steps_legacy_fields[str(step.order_index)] = step_fields

    return {
        "valid": True,
        "warnings": warnings,
        "step_count": len(getattr(chain, "steps", [])),
        "legacy_fields_detected": {
            "chain": chain_legacy_fields,
            "steps": steps_legacy_fields,
        },
    }


def run_prompt_chain_validate(
    manager: PromptManager | None,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> int:
    del manager
    path: Path = args.path
    json_output = bool(getattr(args, "json", False))
    try:
        payload, chain = _validate_prompt_chain_payload(path)
    except ValueError as exc:
        if json_output:
            report = {
                "valid": False,
                "warnings": [],
                "step_count": 0,
                "legacy_fields_detected": {"chain": [], "steps": {}},
                "error": str(exc),
            }
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            logger.error("Invalid chain definition: %s", exc)
        return 5
    if json_output:
        report = _build_prompt_chain_validate_report(payload, chain)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    is_update = bool(payload.get("id"))
    mode = "update" if is_update else "create"
    print(
        "Valid prompt chain definition: "
        f"{path} -> mode={mode} name='{chain.name}' steps={len(chain.steps)}"
    )
    return 0


def run_prompt_chain_apply(
    manager: PromptManager | None,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> int:
    if manager is None:
        raise ValueError("Prompt Manager is required for prompt chain apply.")
    path: Path = args.path
    try:
        payload, chain = _validate_prompt_chain_payload(path)
    except ValueError as exc:
        logger.error("Invalid chain definition: %s", exc)
        return 5
    is_update = bool(payload.get("id"))
    if bool(getattr(args, "dry_run", False)):
        report = _build_prompt_chain_validate_report(payload, chain)
        preview = {
            "dry_run": True,
            "valid": True,
            "mode": "update" if is_update else "create",
            "name": chain.name,
            "step_count": report["step_count"],
            "warnings": report["warnings"],
            "legacy_fields_detected": report["legacy_fields_detected"],
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0
    try:
        saved = manager.save_prompt_chain(chain)
    except PromptChainError as exc:
        logger.error("Failed to persist prompt chain: %s", exc)
        return 5
    action = "Updated" if is_update else "Created"
    print(f"{action} prompt chain '{saved.name}' ({saved.id}) with {len(saved.steps)} steps.")
    return 0


def _build_prompt_chain_run_json_payload(result: Any) -> dict[str, Any]:
    return {
        "chain": {
            "id": str(result.chain.id),
            "name": result.chain.name,
        },
        "chain_input": result.chain_input,
        "final_output_text": result.final_output_text,
        "final_summary_text": result.final_summary_text,
        "run_status": result.run_status,
        "final_step_id": str(result.final_step_id) if result.final_step_id is not None else None,
        "final_step_output_key": result.final_step_output_key,
        "final_step_label": result.final_step_label,
        "terminal_step_id": (
            str(result.terminal_step_id) if result.terminal_step_id is not None else None
        ),
        "terminal_step_output_key": result.terminal_step_output_key,
        "terminal_step_label": result.terminal_step_label,
        "terminal_step_status": result.terminal_step_status,
        "step_aliases": result.step_aliases or {},
        "step_outputs": result.step_outputs,
        "steps": [
            {
                "step_id": str(step_run.step.id),
                "order_index": step_run.step.order_index,
                "prompt_id": str(step_run.step.prompt_id),
                "step_label": step_run.step_label
                or step_run.step.output_variable
                or f"step_{step_run.step.order_index}",
                "step_output_key": step_run.step_output_key or f"step_{step_run.step.order_index}",
                "output_variable": step_run.step.output_variable,
                "prompt_name": step_run.prompt_name,
                "status": step_run.status,
                "duration_ms": step_run.duration_ms,
                "error": step_run.error,
                "request_text": step_run.request_text,
                "response_text": step_run.response_text,
                "web_search_requested": step_run.web_search_requested,
                "web_search_applied": step_run.web_search_applied,
                "skip_reason": step_run.skip_reason,
            }
            for step_run in result.steps
        ],
    }


def _build_prompt_chain_run_text_output(result: Any, final_status: str) -> str:
    text_lines = [
        f"\nChain '{result.chain.name}'",
        "Input to chain:",
    ]
    chain_input_text = result.chain_input or ""
    text_lines.append(textwrap.indent(chain_input_text or "(empty input)", "  "))
    if result.final_output_text:
        text_lines.extend(
            [
                "\nFinal output:",
                textwrap.indent(result.final_output_text, "  "),
            ]
        )
    if result.final_summary_text:
        text_lines.extend(
            [
                "\nFinal summary:",
                textwrap.indent(result.final_summary_text, "  "),
            ]
        )
    text_lines.append("\nStep outputs:")
    if not result.step_outputs:
        text_lines.append("  (no step outputs captured)")
    else:
        for key, value in result.step_outputs.items():
            text_lines.append(f"  {key}: {value}")
    text_lines.append("\nStep summary:")
    for step_run in result.steps:
        step_label = (
            step_run.step_label
            or step_run.step.output_variable
            or f"step_{step_run.step.order_index}"
        )
        step_output_key = step_run.step_output_key or f"step_{step_run.step.order_index}"
        text_lines.append(
            f"  Step {step_run.step.order_index} "
            f"({step_label} | output key: {step_output_key}): {step_run.status}"
        )
        if step_run.prompt_name:
            text_lines.append(f"    Prompt: {step_run.prompt_name}")
        if step_run.duration_ms is not None:
            text_lines.append(f"    Duration: {step_run.duration_ms} ms")
        text_lines.append(
            "    Web search: "
            f"requested={'yes' if step_run.web_search_requested else 'no'}, "
            f"applied={'yes' if step_run.web_search_applied else 'no'}"
        )
        if step_run.skip_reason:
            text_lines.append(f"    Skip reason: {step_run.skip_reason}")
        if step_run.error:
            text_lines.append(f"    Error: {step_run.error}")
        if step_run.request_text:
            text_lines.extend(
                [
                    "    Request text:",
                    textwrap.indent(step_run.request_text, "      "),
                ]
            )
        if step_run.response_text:
            text_lines.extend(
                [
                    "    Response text:",
                    textwrap.indent(step_run.response_text, "      "),
                ]
            )
    return "\n".join(text_lines)


def _build_prompt_chain_compact_output(result: Any, final_status: str) -> str:
    lines = [
        f"Chain: {result.chain.name}",
        f"Status: {result.run_status or final_status}",
        f"Final output preview: {result.final_output_text or '(empty)'}",
        f"Summary preview: {result.final_summary_text or '(empty)'}",
    ]
    return "\n".join(lines)


def run_prompt_chain_run(
    manager: PromptManager | None,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> int:
    if manager is None:
        raise ValueError("Prompt Manager is required for prompt chain execution.")
    try:
        chain_id = uuid.UUID(str(args.chain_id))
    except (TypeError, ValueError) as exc:
        logger.error("Invalid chain id: %s", exc)
        return 5
    try:
        chain_input = _resolve_chain_input(
            getattr(args, "chain_input", None),
            getattr(args, "chain_input_file", None),
        )
    except ValueError as exc:
        logger.error("Invalid chain input: %s", exc)
        return 5
    use_web_search = not bool(getattr(args, "no_web_search", False))
    try:
        result = manager.run_prompt_chain(
            chain_id,
            chain_input=chain_input,
            use_web_search=use_web_search,
        )
    except PromptChainExecutionError as exc:
        logger.error("Chain execution failed: %s", exc)
        return 5
    except PromptChainError as exc:
        logger.error("Unable to execute prompt chain: %s", exc)
        return 5
    final_status = result.run_status or "success"

    selective_flags_enabled = sum(
        1
        for enabled in (
            bool(getattr(args, "json", False)),
            bool(getattr(args, "final_output_only", False)),
            bool(getattr(args, "summary_only", False)),
            bool(getattr(args, "status_only", False)),
            bool(getattr(args, "step_output", None)),
            bool(getattr(args, "step_alias", None)),
            bool(getattr(args, "final_step_meta", False)),
            bool(getattr(args, "compact", False)),
        )
        if enabled
    )
    if selective_flags_enabled > 1:
        logger.error(
            "Conflicting output modes: choose only one of --json, --final-output-only, "
            "--summary-only, --status-only, --step-output, --step-alias, "
            "--final-step-meta, or --compact."
        )
        return 5

    if bool(getattr(args, "json", False)):
        payload_text = json.dumps(
            _build_prompt_chain_run_json_payload(result),
            indent=2,
            ensure_ascii=False,
        )
        output_file = getattr(args, "output_file", None)
        if output_file is not None:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(payload_text + "\n", encoding="utf-8")
            print(f"Saved prompt chain run artifact to {output_path}.")
            return 0
        print(payload_text)
        return 0

    text_output = _build_prompt_chain_run_text_output(result, final_status)
    if bool(getattr(args, "final_output_only", False)):
        text_output = result.final_output_text or ""
    elif bool(getattr(args, "summary_only", False)):
        text_output = result.final_summary_text or ""
    elif bool(getattr(args, "status_only", False)):
        text_output = final_status
    elif getattr(args, "step_output", None):
        requested_key = str(args.step_output)
        if requested_key not in result.step_outputs:
            logger.error("Unknown step output key: %s", requested_key)
            return 5
        text_output = result.step_outputs[requested_key] or ""
    elif getattr(args, "step_alias", None):
        requested_alias = str(args.step_alias)
        step_aliases = result.step_aliases or {}
        resolved_key = step_aliases.get(requested_alias)
        if not resolved_key:
            logger.error("Unknown step alias: %s", requested_alias)
            return 5
        text_output = result.step_outputs.get(resolved_key, "") or ""
    elif bool(getattr(args, "final_step_meta", False)):
        text_output = json.dumps(
            {
                "final_step_id": (
                    str(result.final_step_id) if result.final_step_id is not None else None
                ),
                "final_step_output_key": result.final_step_output_key,
                "final_step_label": result.final_step_label,
                "terminal_step_id": (
                    str(result.terminal_step_id) if result.terminal_step_id is not None else None
                ),
                "terminal_step_output_key": result.terminal_step_output_key,
                "terminal_step_label": result.terminal_step_label,
                "terminal_step_status": result.terminal_step_status,
                "run_status": result.run_status,
            },
            indent=2,
            ensure_ascii=False,
        )
    elif bool(getattr(args, "compact", False)):
        text_output = _build_prompt_chain_compact_output(result, final_status)
    output_file = getattr(args, "output_file", None)
    if output_file is not None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text_output + "\n", encoding="utf-8")
        print(f"Saved prompt chain run artifact to {output_path}.")
        return 0
    print(text_output)
    return 0


def run_history_analytics(
    manager: PromptManager | None,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> int:
    if manager is None:
        raise ValueError("Prompt Manager is required for history analytics.")
    window_days = getattr(args, "window_days", None)
    if window_days is not None and window_days <= 0:
        window_days = None
    prompt_limit = max(1, int(getattr(args, "limit", 5) or 5))
    trend_window = max(1, int(getattr(args, "trend_window", 5) or 5))
    try:
        analytics = manager.get_execution_analytics(
            window_days=window_days,
            prompt_limit=prompt_limit,
            trend_window=trend_window,
        )
    except PromptHistoryError as exc:
        message = f"Unable to compute execution analytics: {exc}"
        print_and_log(logger, logging.ERROR, message)
        return 7

    if analytics is None or analytics.total_runs == 0:
        print_and_log(
            logger,
            logging.INFO,
            "No execution history available for the requested window.",
        )
        return 0

    window_label = (
        analytics.window_start.isoformat(timespec="seconds")
        if analytics.window_start is not None
        else "Full history"
    )

    def _format_totals(label: str, totals: TokenUsageTotals | None) -> str:
        if totals is None:
            return f"{label}: unavailable"
        return (
            f"{label}: prompt={totals.prompt_tokens} "
            f"completion={totals.completion_tokens} total={totals.total_tokens}"
        )

    window_totals: TokenUsageTotals | None
    overall_totals: TokenUsageTotals | None
    try:
        window_totals = manager.get_token_usage_totals(since=analytics.window_start)
        overall_totals = manager.get_token_usage_totals()
    except PromptHistoryError:
        logger.warning("Unable to compute token usage totals for CLI output", exc_info=True)
        window_totals = None
        overall_totals = None

    lines = [
        "Execution analytics",
        "-------------------",
        f"Window start: {window_label}",
        f"Total runs: {analytics.total_runs}",
        f"Success rate: {analytics.success_rate * 100:.1f}%",
        f"Average latency: {format_metric(analytics.average_duration_ms, suffix=' ms')}",
        f"Average rating: {format_metric(analytics.average_rating)}",
        "",
    ]
    if window_totals is not None:
        lines.append(_format_totals("Tokens (window)", window_totals))
    if overall_totals is not None:
        lines.append(_format_totals("Tokens (overall)", overall_totals))
    if window_totals is not None or overall_totals is not None:
        lines.append("")

    if not analytics.prompt_breakdown:
        lines.append("No prompts have execution history within this window.")
    else:
        lines.append("Top prompts:")
        for index, stats in enumerate(analytics.prompt_breakdown, start=1):
            avg_rating = format_metric(stats.average_rating)
            latency = format_metric(stats.average_duration_ms, suffix=" ms")
            trend = format_metric(stats.rating_trend)
            last_run = (
                stats.last_executed_at.isoformat(timespec="seconds")
                if stats.last_executed_at is not None
                else "n/a"
            )
            lines.append(
                f"{index}. {stats.name} — runs:{stats.total_runs} "
                f"success:{stats.success_rate * 100:.1f}% "
                f"avg_rating:{avg_rating} trend:{trend} latency:{latency} "
                f"last:{last_run} tokens:{stats.total_tokens}"
            )
            if stats.decision_summary:
                lines.append(f"   decision: {stats.decision_summary}")
            if stats.next_action_summary:
                lines.append(f"   next: {stats.next_action_summary}")
            if stats.freshness_summary:
                lines.append(f"   freshness: {stats.freshness_summary}")

    print("\n".join(lines))
    return 0


def run_reembed(
    manager: PromptManager | None,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> int:
    del args
    if manager is None:
        raise ValueError("Prompt Manager is required for embedding rebuild.")
    try:
        successes, failures = manager.rebuild_embeddings(reset_store=True)
    except PromptManagerError as exc:
        print_and_log(logger, logging.ERROR, f"Failed to rebuild embeddings: {exc}")
        return 7

    if failures:
        print_and_log(
            logger,
            logging.ERROR,
            f"Embedding rebuild skipped {failures} prompt(s).",
        )
        return 7

    print_and_log(logger, logging.INFO, f"Rebuilt embeddings for {successes} prompt(s).")
    return 0


def run_suggest(
    manager: PromptManager | None,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> int:
    if manager is None:
        raise ValueError("Prompt Manager is required for suggestions.")
    query = getattr(args, "query", "") or ""
    if not query.strip():
        logger.error("Suggestion query must be provided.")
        return 5

    limit = max(1, int(getattr(args, "limit", 5) or 5))
    suggestions = manager.suggest_prompts(query, limit=limit)

    prediction = suggestions.prediction
    label = prediction.label.value.replace("_", " ").title()
    logger.info(
        "Intent: %s (confidence=%0.2f, hints=%s, tags=%s, languages=%s, fallback=%s)",
        label,
        prediction.confidence,
        ", ".join(prediction.category_hints) or "-",
        ", ".join(prediction.tag_hints) or "-",
        ", ".join(prediction.language_hints) or "-",
        suggestions.fallback_used,
    )

    print(f"\nTop {len(suggestions.prompts)} suggestions for: {query!r}\n")
    for index, prompt in enumerate(suggestions.prompts, start=1):
        quality = f"{prompt.quality_score:.1f}" if prompt.quality_score is not None else "n/a"
        tags = ", ".join(prompt.tags) if prompt.tags else "-"
        print(
            textwrap.dedent(
                f"""\
                {index}. {prompt.name} [{prompt.category or "Uncategorised"}]
                   Quality: {quality}  Tags: {tags}
                   Description: {prompt.description}
                """
            )
        )
    return 0


def run_prompt_show(
    manager: PromptManager | None,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> int:
    if manager is None:
        raise ValueError("Prompt Manager is required for prompt display.")
    raw_prompt_id = str(getattr(args, "prompt_id", "") or "").strip()
    prompt = None

    try:
        prompt_id = uuid.UUID(raw_prompt_id)
    except (ValueError, TypeError):
        prompt_id = None
    if prompt_id is not None:
        try:
            prompt = manager.repository.get(prompt_id)
        except RepositoryNotFoundError:
            prompt = None
        except KeyError:
            prompt = None
        except Exception as exc:  # pragma: no cover - surfaced to CLI
            print_and_log(logger, logging.ERROR, f"Failed to load prompt: {exc}")
            return 6

    if prompt is None and raw_prompt_id:
        try:
            for candidate in manager.repository.list():
                if candidate.name == raw_prompt_id:
                    prompt = candidate
                    break
        except Exception as exc:  # pragma: no cover - surfaced to CLI
            print_and_log(logger, logging.ERROR, f"Failed to search prompt by name: {exc}")
            return 6

    if prompt is None:
        if prompt_id is None and raw_prompt_id:
            print_and_log(logger, logging.ERROR, f"Prompt not found: {raw_prompt_id}")
            return 4
        missing_ref = prompt_id if prompt_id is not None else raw_prompt_id
        print_and_log(logger, logging.ERROR, f"Prompt not found: {missing_ref}")
        return 4

    if bool(getattr(args, "json", False)):
        print(json.dumps(prompt.to_record(), ensure_ascii=False, indent=2))
        return 0

    tags = ", ".join(prompt.tags) if prompt.tags else "-"
    context = prompt.context.strip() if isinstance(prompt.context, str) else ""
    lines = [
        f"id: {prompt.id}",
        f"name: {prompt.name}",
        f"description: {prompt.description}",
        f"category: {prompt.category}",
        f"tags: {tags}",
        f"source: {prompt.source}",
        f"active: {'yes' if prompt.is_active else 'no'}",
    ]
    if context:
        lines.append("context:")
        lines.append(context)

    print("\n".join(lines))
    return 0


def run_prompt_find(
    manager: PromptManager | None,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> int:
    if manager is None:
        raise ValueError("Prompt Manager is required for prompt search.")
    query = str(getattr(args, "query", "") or "").strip().lower()
    if not query:
        logger.error("Prompt search query must be provided.")
        return 5
    limit = max(1, int(getattr(args, "limit", 10) or 10))
    category_filter = str(getattr(args, "category", "") or "").strip().lower()
    tag_filter = str(getattr(args, "tag", "") or "").strip().lower()
    source_filter = str(getattr(args, "source", "") or "").strip().lower()
    active_raw = str(getattr(args, "active", "") or "").strip().lower()
    active_filter: bool | None = None
    if active_raw:
        if active_raw in {"1", "true", "yes", "y", "active"}:
            active_filter = True
        elif active_raw in {"0", "false", "no", "n", "inactive"}:
            active_filter = False
        else:
            print_and_log(
                logger,
                logging.ERROR,
                f"Invalid --active value: {getattr(args, 'active', '')}. Use true/false.",
            )
            return 5

    try:
        prompts = manager.repository.list()
    except Exception as exc:  # pragma: no cover - surfaced to CLI
        print_and_log(logger, logging.ERROR, f"Failed to list prompts: {exc}")
        return 6

    matches = []
    for prompt in prompts:
        haystacks = [
            prompt.name,
            prompt.description,
            prompt.category,
            " ".join(prompt.tags or []),
        ]
        if not any(query in str(value).lower() for value in haystacks if value):
            continue
        if category_filter and category_filter != str(prompt.category or "").strip().lower():
            continue
        if tag_filter and tag_filter not in {
            str(tag).strip().lower() for tag in (prompt.tags or []) if str(tag).strip()
        }:
            continue
        if source_filter and source_filter != str(prompt.source or "").strip().lower():
            continue
        if active_filter is not None and bool(prompt.is_active) is not active_filter:
            continue
        matches.append(prompt)
        if len(matches) >= limit:
            break

    if not matches:
        print(f"No prompts matched: {getattr(args, 'query', '')}")
        return 0

    if bool(getattr(args, "json", False)):
        payload = [prompt.to_record() for prompt in matches]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    lines = []
    for prompt in matches:
        tags = ", ".join(prompt.tags) if prompt.tags else "-"
        lines.append(f"{prompt.id} | {prompt.name} | [{prompt.category}] | {tags}")
    print("\n".join(lines))
    return 0


def run_prompt_history(
    manager: PromptManager | None,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> int:
    if manager is None:
        raise ValueError("Prompt Manager is required for prompt history.")
    raw_prompt_id = str(getattr(args, "prompt_id", "") or "").strip()
    limit = max(1, int(getattr(args, "limit", 5) or 5))
    status_raw = str(getattr(args, "status", "") or "").strip().lower()
    status_filter: str | None = None
    if status_raw:
        if status_raw in {"success", "successful", "ok"}:
            status_filter = "success"
        elif status_raw in {"failed", "failure", "error"}:
            status_filter = "failed"
        else:
            print_and_log(
                logger,
                logging.ERROR,
                f"Invalid --status value: {getattr(args, 'status', '')}. Use success or failed.",
            )
            return 5
    window_days = max(0, int(getattr(args, "window_days", 0) or 0))
    prompt = None

    try:
        prompt_id = uuid.UUID(raw_prompt_id)
    except (ValueError, TypeError):
        prompt_id = None
    if prompt_id is not None:
        try:
            prompt = manager.repository.get(prompt_id)
        except RepositoryNotFoundError:
            prompt = None
        except KeyError:
            prompt = None
        except Exception as exc:  # pragma: no cover - surfaced to CLI
            print_and_log(logger, logging.ERROR, f"Failed to load prompt: {exc}")
            return 6

    if prompt is None and raw_prompt_id:
        try:
            for candidate in manager.repository.list():
                if candidate.name == raw_prompt_id:
                    prompt = candidate
                    break
        except Exception as exc:  # pragma: no cover - surfaced to CLI
            print_and_log(logger, logging.ERROR, f"Failed to search prompt by name: {exc}")
            return 6

    if prompt is None:
        missing_ref = prompt_id if prompt_id is not None else raw_prompt_id
        print_and_log(logger, logging.ERROR, f"Prompt not found: {missing_ref}")
        return 4

    analytics = None
    try:
        get_prompt_execution_analytics = getattr(manager, "get_prompt_execution_analytics", None)
        if callable(get_prompt_execution_analytics):
            analytics = get_prompt_execution_analytics(prompt.id)
        executions = manager.list_executions_for_prompt(prompt.id, limit=limit)
    except PromptHistoryError as exc:
        print_and_log(logger, logging.ERROR, f"Unable to load prompt history: {exc}")
        return 7
    except Exception as exc:  # pragma: no cover - surfaced to CLI
        print_and_log(logger, logging.ERROR, f"Unable to load prompt history: {exc}")
        return 7

    if status_filter is not None:
        executions = [
            execution for execution in executions if execution.status.value == status_filter
        ]
    if window_days > 0:
        cutoff = datetime.now(UTC) - timedelta(days=window_days)
        executions = [execution for execution in executions if execution.executed_at >= cutoff]
    executions = executions[:limit]

    def _format_metric(value: float | int | None, *, suffix: str = "") -> str:
        if value is None:
            return "n/a"
        if isinstance(value, int):
            return f"{value}{suffix}"
        return f"{value:.1f}{suffix}"

    if bool(getattr(args, "json", False)):
        payload = {
            "prompt": prompt.to_record(),
            "analytics": {
                "prompt_id": str(analytics.prompt_id),
                "name": analytics.name,
                "total_runs": analytics.total_runs,
                "success_rate": analytics.success_rate,
                "average_duration_ms": analytics.average_duration_ms,
                "average_rating": analytics.average_rating,
                "rating_trend": analytics.rating_trend,
                "last_executed_at": (
                    analytics.last_executed_at.isoformat()
                    if analytics.last_executed_at is not None
                    else None
                ),
                "prompt_tokens": analytics.prompt_tokens,
                "completion_tokens": analytics.completion_tokens,
                "total_tokens": analytics.total_tokens,
                "decision_summary": analytics.decision_summary,
                "next_action_summary": analytics.next_action_summary,
                "freshness_summary": analytics.freshness_summary,
            }
            if analytics is not None
            else None,
            "executions": [execution.to_record() for execution in executions],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    lines = [
        f"id: {prompt.id}",
        f"name: {prompt.name}",
    ]
    if analytics is not None:
        last_run = (
            analytics.last_executed_at.isoformat(timespec="seconds")
            if analytics.last_executed_at is not None
            else "n/a"
        )
        lines.extend(
            [
                f"runs: {analytics.total_runs}",
                f"success: {analytics.success_rate * 100:.1f}%",
                f"avg_rating: {_format_metric(analytics.average_rating)}",
                f"avg_latency: {_format_metric(analytics.average_duration_ms, suffix=' ms')}",
                f"last_run: {last_run}",
                f"tokens: {analytics.total_tokens}",
            ]
        )
        if analytics.decision_summary:
            lines.append(f"decision: {analytics.decision_summary}")
        if analytics.next_action_summary:
            lines.append(f"next: {analytics.next_action_summary}")
        if analytics.freshness_summary:
            lines.append(f"freshness: {analytics.freshness_summary}")
    else:
        lines.append("runs: 0")

    lines.append("")
    lines.append("recent executions:")
    if not executions:
        lines.append("(none)")
    else:
        for index, execution in enumerate(executions, start=1):
            metadata = execution.metadata if isinstance(execution.metadata, dict) else {}
            model = metadata.get("model") or "n/a"
            total_tokens = metadata.get("total_tokens")
            token_label = total_tokens if total_tokens not in (None, "") else "n/a"
            duration = f"{execution.duration_ms} ms" if execution.duration_ms is not None else "n/a"
            rating = _format_metric(execution.rating)
            executed_at = execution.executed_at.isoformat(timespec="seconds")
            lines.append(
                f"[{index}] {executed_at} | {execution.status.value} | {duration} | "
                f"rating: {rating} | model: {model} | tokens: {token_label}"
            )
            lines.append(f"request: {execution.request_text}")
            if execution.response_text:
                lines.append(f"response: {execution.response_text}")
            if execution.error_message:
                lines.append(f"error: {execution.error_message}")

    print("\n".join(lines))
    return 0


COMMAND_SPECS: dict[str | None, CommandSpec] = {
    "catalog-export": CommandSpec(run_catalog_export),
    "catalog-import": CommandSpec(run_catalog_import),
    "prompt-add": CommandSpec(run_catalog_import),
    "prompt-show": CommandSpec(run_prompt_show),
    "prompt-find": CommandSpec(run_prompt_find),
    "prompt-history": CommandSpec(run_prompt_history),
    "suggest": CommandSpec(run_suggest),
    "usage-report": CommandSpec(run_usage_report),
    "history-analytics": CommandSpec(run_history_analytics),
    "reembed": CommandSpec(run_reembed),
    "benchmark": CommandSpec(run_benchmark),
    "refresh-scenarios": CommandSpec(run_refresh_scenarios),
    "diagnostics": CommandSpec(run_diagnostics),
    "prompt-chain-list": CommandSpec(run_prompt_chain_list),
    "prompt-chain-show": CommandSpec(run_prompt_chain_show),
    "prompt-chain-history": CommandSpec(run_prompt_chain_history),
    "prompt-chain-export": CommandSpec(run_prompt_chain_export),
    "prompt-chain-validate": CommandSpec(run_prompt_chain_validate, requires_manager=False),
    "prompt-chain-apply": CommandSpec(run_prompt_chain_apply),
    "prompt-chain-run": CommandSpec(run_prompt_chain_run),
}


__all__ = ["CommandSpec", "COMMAND_SPECS"]
