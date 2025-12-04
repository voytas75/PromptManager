"""CLI command handlers for Prompt Manager.

Updates:
  v0.32.0 - 2025-12-04 - Reuse shared chain_from_payload helper for JSON imports.
"""

from __future__ import annotations

import argparse
import json
import logging
import textwrap
import uuid
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
import sys

from core import (
    PromptChainError,
    PromptChainExecutionError,
    PromptHistoryError,
    PromptManagerError,
    build_analytics_snapshot,
    export_prompt_catalog,
    snapshot_dataset_rows,
)
from models.prompt_chain_model import chain_from_payload

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


def _load_chain_variables(
    vars_file: Path | None,
    vars_json: str | None,
) -> dict[str, Any]:
    data: Any = {}
    if vars_file:
        data = _load_json_file(vars_file)
    elif vars_json:
        try:
            data = json.loads(vars_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid inline JSON: {exc}") from exc
    if data in (None, ""):
        return {}
    if not isinstance(data, Mapping):
        raise ValueError("Chain variables must be a JSON object.")
    return {str(key): value for key, value in data.items()}


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
    status = "active" if chain.is_active else "inactive"
    print(f"\nChain: {chain.name} ({chain.id}) [{status}]")
    if chain.description:
        print(f"Description: {chain.description}")
    if chain.variables_schema:
        print("Variables schema keys:", ", ".join(chain.variables_schema.keys()))
    if not chain.steps:
        print("Steps: (none)")
        return 0
    print("\nSteps:")
    for step in chain.steps:
        condition = f" when {step.condition}" if step.condition else ""
        print(
            f"  {step.order_index}. prompt={step.prompt_id} -> ${step.output_variable}{condition}"
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
        payload = _load_json_file(path)
    except ValueError as exc:
        logger.error(str(exc))
        return 5
    if not isinstance(payload, Mapping):
        logger.error("Chain definition must be a JSON object.")
        return 5
    is_update = bool(payload.get("id"))
    try:
        chain = chain_from_payload(payload)
    except ValueError as exc:
        logger.error("Invalid chain definition: %s", exc)
        return 5
    try:
        saved = manager.save_prompt_chain(chain)
    except PromptChainError as exc:
        logger.error("Failed to persist prompt chain: %s", exc)
        return 5
    action = "Updated" if is_update else "Created"
    print(f"{action} prompt chain '{saved.name}' ({saved.id}) with {len(saved.steps)} steps.")
    return 0


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
    vars_file: Path | None = getattr(args, "vars_file", None)
    vars_json: str | None = getattr(args, "vars_json", None)
    try:
        variables = _load_chain_variables(vars_file, vars_json)
    except ValueError as exc:
        logger.error("Invalid chain variables: %s", exc)
        return 5
    try:
        result = manager.run_prompt_chain(chain_id, variables=variables)
    except PromptChainExecutionError as exc:
        logger.error("Chain execution failed: %s", exc)
        return 5
    except PromptChainError as exc:
        logger.error("Unable to execute prompt chain: %s", exc)
        return 5
    print(f"\nChain '{result.chain.name}' outputs:")
    if not result.outputs:
        print("  (no outputs captured)")
    else:
        for key, value in result.outputs.items():
            print(f"  {key}: {value}")
    print("\nStep summary:")
    for step_run in result.steps:
        label = f"{step_run.step.order_index}. {step_run.step.output_variable}"
        if step_run.status == "success":
            preview = step_run.outcome.result.response_text if step_run.outcome else ""
            if preview and len(preview) > 80:
                preview = preview[:77].rstrip() + "..."
            print(f"  [OK] {label} -> {preview or '(empty response)'}")
        elif step_run.status == "skipped":
            print(f"  [SKIP] {label}")
        else:
            print(f"  [ERR] {label}: {step_run.error}")
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
                f"last:{last_run}"
            )

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


COMMAND_SPECS: dict[str | None, CommandSpec] = {
    "catalog-export": CommandSpec(run_catalog_export),
    "suggest": CommandSpec(run_suggest),
    "usage-report": CommandSpec(run_usage_report),
    "history-analytics": CommandSpec(run_history_analytics),
    "reembed": CommandSpec(run_reembed),
    "benchmark": CommandSpec(run_benchmark),
    "refresh-scenarios": CommandSpec(run_refresh_scenarios),
    "diagnostics": CommandSpec(run_diagnostics),
    "prompt-chain-list": CommandSpec(run_prompt_chain_list),
    "prompt-chain-show": CommandSpec(run_prompt_chain_show),
    "prompt-chain-apply": CommandSpec(run_prompt_chain_apply),
    "prompt-chain-run": CommandSpec(run_prompt_chain_run),
}


__all__ = ["CommandSpec", "COMMAND_SPECS"]
