"""Application entry point for Prompt Manager.

Updates:
  v0.8.3 - 2025-11-30 - Restore stdout messaging for CLI commands and relax history imports.
  v0.8.2 - 2025-11-29 - Reformat CLI summary output and modernise type hints.
  v0.8.1 - 2025-11-28 - Add analytics diagnostics target with dashboard export.
  v0.8.0 - 2025-02-14 - Add embedding diagnostics CLI command.
  v0.7.9 - 2025-11-28 - Add benchmark and scenario refresh CLI commands.
  v0.7.8 - 2025-12-07 - Add CLI command to rebuild embeddings from scratch.
  v0.7.7 - 2025-11-05 - Surface LiteLLM workflow routing details in CLI summaries.
  v0.7.6 - 2025-11-05 - Expand CLI settings summary to list fast and inference models.
  v0.7.5 - 2025-11-30 - Remove catalogue import command and startup messaging.
  v0.7.4 - 2025-11-26 - Surface LiteLLM streaming configuration in CLI summaries.
  v0.7.3 - 2025-11-17 - Require explicit catalogue paths; skip built-in seeding.
  v0.7.2 - 2025-11-15 - Extend --print-settings with health checks and masked output.
  v0.7.1 - 2025-11-14 - Simplify GUI dependency guidance for unified installs.
  v0.7.0 - 2025-11-07 - Add semantic suggestion CLI to verify embedding backends.
  v0.4.1 - 2025-11-05 - Launch GUI by default and add --no-gui flag.
  v0.4.0 - 2025-11-05 - Ensure manager shutdown occurs on exit and update GUI guidance.
  v0.3.0 - 2025-11-05 - Gracefully handle missing GUI dependencies.
  v0.2.0 - 2025-11-04 - Add optional PySide6 GUI launcher toggle.
  v0.1.0 - 2025-10-30 - Initial CLI bootstrap loading settings and building services.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import logging.config
import textwrap
import uuid
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, cast

import core as core_services
from config import (
    DEFAULT_EMBEDDING_BACKEND,
    DEFAULT_EMBEDDING_MODEL,
    LITELLM_ROUTED_WORKFLOWS,
    PromptManagerSettings,
    load_settings,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Mapping, Sequence

PromptManagerError = core_services.PromptManagerError
build_prompt_manager = core_services.build_prompt_manager
build_analytics_snapshot = core_services.build_analytics_snapshot
export_prompt_catalog = core_services.export_prompt_catalog
snapshot_dataset_rows = core_services.snapshot_dataset_rows
PromptHistoryError = getattr(core_services, "PromptHistoryError", PromptManagerError)


def _print_and_log(logger: logging.Logger, level: int, message: str) -> None:
    """Log a message and mirror it to stdout for CLI expectations."""
    logger.log(level, message)
    print(message)


def _mask_secret(value: str | None) -> str:
    """Return masked representation of secret values."""
    if not value:
        return "not set"
    secret = value.strip()
    if len(secret) <= 6:
        return "set (****)"
    prefix = secret[:4]
    suffix = secret[-4:]
    return f"set ({prefix}...{suffix})"


def _describe_path(
    path_value: object,
    *,
    expect_directory: bool,
    allow_missing_file: bool = False,
) -> str:
    """Return a human-readable description of a configured filesystem path."""
    try:
        path = Path(path_value) if path_value is not None else None
    except TypeError:
        path = None
    if path is None:
        return "not set"

    resolved = path.expanduser()
    if resolved.exists():
        if expect_directory and not resolved.is_dir():
            return f"{resolved} (exists but is not a directory)"
        if not expect_directory and resolved.is_dir():
            return f"{resolved} (exists but is a directory)"
        return f"{resolved} (exists)"

    message = f"{resolved} (missing)"
    if not expect_directory and allow_missing_file:
        message = f"{resolved} (missing - created on demand)"
    parent = resolved.parent
    if not parent.exists():
        message += f", parent missing: {parent}"
    return message


def _write_csv_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    """Persist dictionaries to CSV and return the resolved path."""
    if not rows:
        raise ValueError("No rows available for export")
    headers: list[str] = []
    seen_keys: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen_keys:
                continue
            seen_keys.add(key)
            headers.append(str(key))

    resolved = path.expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})
    return resolved


def _print_settings_summary(settings: PromptManagerSettings) -> None:
    """Emit a readable summary of core configuration and health checks."""
    redis_dsn = getattr(settings, "redis_dsn", None)
    litellm_model = getattr(settings, "litellm_model", None)
    litellm_inference_model = getattr(settings, "litellm_inference_model", None)
    litellm_api_key = getattr(settings, "litellm_api_key", None)
    litellm_api_base = getattr(settings, "litellm_api_base", None)
    litellm_api_version = getattr(settings, "litellm_api_version", None)
    litellm_reasoning_effort = getattr(settings, "litellm_reasoning_effort", None)
    litellm_tts_model = getattr(settings, "litellm_tts_model", None)
    litellm_stream = getattr(settings, "litellm_stream", False)
    litellm_workflow_models = getattr(settings, "litellm_workflow_models", None) or {}
    embedding_backend = getattr(settings, "embedding_backend", None)
    embedding_model = getattr(settings, "embedding_model", None)

    db_path_desc = _describe_path(
        settings.db_path,
        expect_directory=False,
        allow_missing_file=True,
    )
    chroma_path_desc = _describe_path(settings.chroma_path, expect_directory=True)
    default_model = (
        "(auto)" if embedding_backend == "litellm" and litellm_model else DEFAULT_EMBEDDING_MODEL
    )
    resolved_model = embedding_model or default_model

    def _format_tier(value: str) -> str:
        return "Inference" if value == "inference" else "Fast"

    lines = [
        "Prompt Manager configuration summary",
        "------------------------------------",
        f"Database path: {db_path_desc}",
        f"Chroma directory: {chroma_path_desc}",
        f"Redis DSN: {redis_dsn or 'not set'}",
        f"Cache TTL (seconds): {getattr(settings, 'cache_ttl_seconds', 'n/a')}",
        "",
        "LiteLLM configuration",
        "---------------------",
        f"Fast model: {litellm_model or 'not set'}",
        f"Inference model: {litellm_inference_model or 'not set'}",
        f"TTS model: {litellm_tts_model or 'not set'}",
        f"LiteLLM API key: {_mask_secret(litellm_api_key)}",
        f"LiteLLM API base: {litellm_api_base or 'not set'}",
        f"LiteLLM API version: {litellm_api_version or 'not set'}",
        f"Reasoning effort: {litellm_reasoning_effort or 'not set'}",
        f"Streaming enabled: {'yes' if litellm_stream else 'no'}",
        "",
        "LiteLLM routing",
        "----------------",
    ]

    for workflow_key, workflow_label in LITELLM_ROUTED_WORKFLOWS.items():
        tier = litellm_workflow_models.get(workflow_key, "fast")
        lines.append(f"{workflow_label}: {_format_tier(tier)}")

    lines.extend(
        [
            "",
            "Embedding configuration",
            "-----------------------",
            f"Backend: {embedding_backend or DEFAULT_EMBEDDING_BACKEND}",
            f"Model: {resolved_model}",
        ]
    )
    print("\n".join(lines))


def _setup_logging(logging_conf_path: Path | None) -> None:
    """Configure logging from a dictConfig file when present."""
    # Default to config/logging.conf if not provided
    path = logging_conf_path or Path("config/logging.conf")
    if path.exists():
        try:
            logging.config.fileConfig(path, disable_existing_loggers=False)
            return
        except Exception:  # pragma: no cover - logging config errors are non-critical
            pass
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


def _resolve_export_format(path: Path, explicit_format: str | None) -> str:
    if explicit_format:
        return explicit_format.lower()
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return "yaml"
    return "json"


def _run_catalog_export(manager, args: argparse.Namespace, logger: logging.Logger) -> int:
    output_path = Path(args.path).expanduser()
    fmt = _resolve_export_format(output_path, getattr(args, "format", None))
    try:
        resolved = export_prompt_catalog(
            manager,
            output_path,
            fmt=fmt,
            include_inactive=getattr(args, "include_inactive", False),
        )
    except Exception as exc:
        message = f"Failed to export catalogue: {exc}"
        _print_and_log(logger, logging.ERROR, message)
        return 6
    message = f"Prompt catalogue exported to {resolved} ({fmt})"
    _print_and_log(logger, logging.INFO, message)
    return 0


def _run_usage_report(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Summarise intent workspace usage analytics."""
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


def _run_benchmark(manager, args: argparse.Namespace, logger: logging.Logger) -> int:
    prompt_values: Sequence[str] = getattr(args, "prompt_ids", []) or []
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


def _run_refresh_scenarios(manager, args: argparse.Namespace, logger: logging.Logger) -> int:
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


def _run_diagnostics(manager, args: argparse.Namespace, logger: logging.Logger) -> int:
    target = getattr(args, "target", None)
    if target == "embeddings":
        return _run_embedding_diagnostics(manager, args, logger)
    if target == "analytics":
        return _run_analytics_diagnostics(manager, args, logger)
    logger.error("Unknown diagnostics target: %s", target)
    return 5


def _run_embedding_diagnostics(manager, args: argparse.Namespace, logger: logging.Logger) -> int:
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


def _run_analytics_diagnostics(manager, args: argparse.Namespace, logger: logging.Logger) -> int:
    window_days = max(0, int(getattr(args, "window_days", 30) or 0))
    prompt_limit = max(1, int(getattr(args, "prompt_limit", 5) or 1))
    usage_log_path = getattr(args, "usage_log", None)

    snapshot = build_analytics_snapshot(
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
        try:
            rows = snapshot_dataset_rows(snapshot, dataset)
        except ValueError as exc:
            logger.error("%s", exc)
            return 5
        if not rows:
            logger.info("Dataset '%s' is empty; skipping export.", dataset)
        else:
            try:
                resolved = _write_csv_rows(Path(export_path), rows)
            except (OSError, ValueError) as exc:
                logger.error("Unable to export analytics dataset: %s", exc)
                return 5
            logger.info("Analytics dataset '%s' exported to %s", dataset, resolved)

    return 0


def _format_metric(value: float | None, *, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    formatted = f"{value:.2f}" if abs(value) < 1000 else f"{value:.0f}"
    return f"{formatted}{suffix}" if suffix else formatted


def _run_history_analytics(manager, args: argparse.Namespace, logger: logging.Logger) -> int:
    """Render aggregate execution analytics for CLI consumers."""
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
        _print_and_log(logger, logging.ERROR, message)
        return 7

    if analytics is None or analytics.total_runs == 0:
        _print_and_log(
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
        f"Average latency: {_format_metric(analytics.average_duration_ms, suffix=' ms')}",
        f"Average rating: {_format_metric(analytics.average_rating)}",
        "",
    ]

    if not analytics.prompt_breakdown:
        lines.append("No prompts have execution history within this window.")
    else:
        lines.append("Top prompts:")
        for index, stats in enumerate(analytics.prompt_breakdown, start=1):
            avg_rating = _format_metric(stats.average_rating)
            latency = _format_metric(stats.average_duration_ms, suffix=" ms")
            trend = _format_metric(stats.rating_trend)
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


def _run_reembed(manager, logger: logging.Logger) -> int:
    """Reset the Chroma vector store and regenerate embeddings for all prompts."""
    try:
        successes, failures = manager.rebuild_embeddings(reset_store=True)
    except PromptManagerError as exc:
        _print_and_log(logger, logging.ERROR, f"Failed to rebuild embeddings: {exc}")
        return 7

    if failures:
        _print_and_log(
            logger,
            logging.ERROR,
            f"Embedding rebuild skipped {failures} prompt(s).",
        )
        return 7

    _print_and_log(logger, logging.INFO, f"Rebuilt embeddings for {successes} prompt(s).")
    return 0


def _run_suggest(manager, args: argparse.Namespace, logger: logging.Logger) -> int:
    """Render semantic suggestions for manual verification."""
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


def parse_args() -> argparse.Namespace:
    """Return parsed CLI arguments for the Prompt Manager launcher."""
    parser = argparse.ArgumentParser(description="Prompt Manager launcher")
    parser.add_argument(
        "--logging-config",
        type=Path,
        default=None,
        help="Path to logging configuration file (INI format)",
    )
    parser.add_argument(
        "--print-settings",
        action="store_true",
        help="Print resolved settings and exit",
    )
    parser.add_argument(
        "--gui",
        dest="gui",
        action="store_true",
        default=None,
        help="Launch the PySide6 interface after services are initialised (default behaviour).",
    )
    parser.add_argument(
        "--no-gui",
        dest="gui",
        action="store_false",
        help="Skip launching the GUI and exit once services are initialised.",
    )

    subparsers = parser.add_subparsers(dest="command")

    export_parser = subparsers.add_parser(
        "catalog-export",
        help="Export the current prompt catalogue to JSON or YAML.",
    )
    export_parser.add_argument("path", type=Path, help="Destination file path (.json or .yaml)")
    export_parser.add_argument(
        "--format",
        choices=("json", "yaml"),
        default=None,
        help="Explicit output format (defaults based on file extension).",
    )
    export_parser.add_argument(
        "--include-inactive",
        action="store_true",
        help="Include inactive prompts in the export payload.",
    )

    suggest_parser = subparsers.add_parser(
        "suggest",
        help="Run semantic suggestions for a given query using the configured embedding backend.",
    )
    suggest_parser.add_argument(
        "query",
        type=str,
        help="Freeform query, code, or text used to retrieve prompts.",
    )
    suggest_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of prompt suggestions to display (default: 5).",
    )

    usage_parser = subparsers.add_parser(
        "usage-report",
        help="Summarise GUI intent workspace analytics from the usage log.",
    )
    usage_parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Path to the usage log (defaults to data/logs/intent_usage.jsonl).",
    )

    analytics_parser = subparsers.add_parser(
        "history-analytics",
        help="Display aggregated execution analytics for recorded prompts.",
    )
    analytics_parser.add_argument(
        "--window-days",
        type=int,
        default=30,
        help="Look-back window in days (<=0 includes full history).",
    )
    analytics_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of prompts to display (default: 5).",
    )
    analytics_parser.add_argument(
        "--trend-window",
        type=int,
        default=5,
        help="Executions considered when computing rating trends (default: 5).",
    )

    subparsers.add_parser(
        "reembed",
        help="Delete the current ChromaDB directory and regenerate embeddings for all prompts.",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run one or more prompts against configured models for side-by-side comparison.",
    )
    benchmark_parser.add_argument(
        "--prompt",
        dest="prompt_ids",
        action="append",
        required=True,
        help="Prompt UUID to benchmark (repeat for multiple prompts).",
    )
    benchmark_parser.add_argument(
        "--request",
        type=str,
        default=None,
        help="Inline benchmark input text.",
    )
    benchmark_parser.add_argument(
        "--request-file",
        type=Path,
        default=None,
        help="Path to a file containing the benchmark input text.",
    )
    benchmark_parser.add_argument(
        "--model",
        dest="models",
        action="append",
        help=(
            "Model identifier to benchmark (repeatable). "
            "Defaults to configured fast/inference models."
        ),
    )
    benchmark_parser.add_argument(
        "--history-window",
        type=int,
        default=30,
        help="Days of execution history to summarise (set to 0 for full history).",
    )
    benchmark_parser.add_argument(
        "--trend-window",
        type=int,
        default=5,
        help="Executions considered when computing rating trend (default: 5).",
    )
    benchmark_parser.add_argument(
        "--persist-history",
        action="store_true",
        help="Persist benchmark runs to execution history for future analytics.",
    )

    refresh_scenarios_parser = subparsers.add_parser(
        "refresh-scenarios",
        help="Regenerate and persist usage scenarios for a prompt.",
    )
    refresh_scenarios_parser.add_argument(
        "prompt_id",
        type=str,
        help="Prompt UUID to refresh.",
    )
    refresh_scenarios_parser.add_argument(
        "--max-scenarios",
        type=int,
        default=3,
        help="Number of scenarios to request from the generator (default: 3).",
    )

    diagnostics_parser = subparsers.add_parser(
        "diagnostics",
        help="Run backend diagnostics such as embedding health checks.",
    )
    diagnostics_parser.add_argument(
        "target",
        choices=("embeddings", "analytics"),
        help="Diagnostics target to execute.",
    )
    diagnostics_parser.add_argument(
        "--sample-text",
        type=str,
        default="Prompt Manager diagnostics probe",
        help="Sample text used when probing the embedding backend (default provided).",
    )
    diagnostics_parser.add_argument(
        "--window-days",
        type=int,
        default=30,
        help="Analytics look-back window in days (analytics target only).",
    )
    diagnostics_parser.add_argument(
        "--prompt-limit",
        type=int,
        default=5,
        help="Number of prompts to summarise in analytics outputs (analytics target).",
    )
    diagnostics_parser.add_argument(
        "--usage-log",
        type=Path,
        default=None,
        help=(
            "Path to the intent usage log for analytics exports "
            "(defaults to data/logs/intent_usage.jsonl)."
        ),
    )
    diagnostics_parser.add_argument(
        "--dataset",
        choices=("usage", "model_costs", "benchmark", "intent", "embedding"),
        default="usage",
        help="Analytics dataset exported when --export-csv is provided.",
    )
    diagnostics_parser.add_argument(
        "--export-csv",
        type=Path,
        default=None,
        help="Optional CSV path for analytics dataset export.",
    )

    return parser.parse_args()


def main() -> int:
    """Entrypoint that wires settings, services, and CLI commands."""
    args = parse_args()
    _setup_logging(args.logging_config)

    logger = logging.getLogger("prompt_manager.main")
    try:
        settings = load_settings()
    except Exception as exc:
        logger.error("Failed to load settings: %s", exc)
        return 2

    if args.print_settings:
        _print_settings_summary(settings)
        return 0

    # Build core services; GUI wiring will attach here in later milestones
    try:
        manager = build_prompt_manager(settings)
    except Exception as exc:
        logger.error("Failed to initialise services: %s", exc)
        return 3

    command = getattr(args, "command", None)
    try:
        if command == "catalog-export":
            return _run_catalog_export(manager, args, logger)

        if command == "suggest":
            try:
                result = _run_suggest(manager, args, logger)
            finally:
                manager.close()
            return result

        if command == "usage-report":
            try:
                result = _run_usage_report(args, logger)
            finally:
                manager.close()
            return result

        if command == "history-analytics":
            try:
                result = _run_history_analytics(manager, args, logger)
            finally:
                manager.close()
            return result

        if command == "reembed":
            try:
                result = _run_reembed(manager, logger)
            finally:
                manager.close()
            return result

        if command == "benchmark":
            try:
                result = _run_benchmark(manager, args, logger)
            finally:
                manager.close()
            return result

        if command == "refresh-scenarios":
            try:
                result = _run_refresh_scenarios(manager, args, logger)
            finally:
                manager.close()
            return result

        if command == "diagnostics":
            try:
                result = _run_diagnostics(manager, args, logger)
            finally:
                manager.close()
            return result

        # Minimal interactive stub to verify bootstrap until GUI arrives
        _print_and_log(
            logger,
            logging.INFO,
            f"Prompt Manager ready. Database at {settings.db_path}",
        )
        _print_and_log(
            logger,
            logging.INFO,
            f"ChromaDB at {settings.chroma_path}",
        )
        launch_requested = args.gui if args.gui is not None else True
        if launch_requested:
            try:
                gui_module = importlib.import_module("gui")
            except ModuleNotFoundError as exc:
                logger.error(
                    "GUI launch requested but dependency %s is missing. Install requirements with "
                    "`pip install -r requirements.txt` or rerun without --gui.",
                    exc.name,
                )
                return 4
            try:
                launch_gui_callable = gui_module.launch_prompt_manager
            except AttributeError:
                logger.error(
                    "GUI module is missing the launch_prompt_manager entry point. "
                    "Reinstall dependencies with `pip install -r requirements.txt` "
                    "or launch without --gui."
                )
                return 4
            if not callable(launch_gui_callable):
                logger.error(
                    "GUI module launch_prompt_manager entry point is not callable. "
                    "Reinstall dependencies with `pip install -r requirements.txt` "
                    "or launch without --gui."
                )
                return 4
            launch_callable = cast("Callable[[object, object | None], int]", launch_gui_callable)

            dependency_error_type = getattr(gui_module, "GuiDependencyError", RuntimeError)
            if not isinstance(dependency_error_type, type) or not issubclass(
                dependency_error_type, BaseException
            ):
                dependency_error_type = RuntimeError

            try:
                return launch_callable(manager, settings)
            except Exception as exc:
                if isinstance(exc, dependency_error_type):
                    logger.error("Unable to start GUI: %s", exc)
                    return 4
                logger.error("Unexpected error while starting GUI: %s", exc)
                return 4

        return 0
    finally:
        manager.close()


if __name__ == "__main__":
    raise SystemExit(main())
