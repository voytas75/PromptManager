"""Argument parser for Prompt Manager CLI.

Updates:
  v0.3.0 - 2025-12-05 - Add prompt chain web search toggle flags.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast


def _build_inline_prompt_payload(args: argparse.Namespace) -> dict[str, object]:
    tags_raw = getattr(args, "tags", None)
    tags = []
    if tags_raw:
        tags = [item.strip() for item in str(tags_raw).split(",") if item.strip()]

    payload: dict[str, object] = {
        "name": args.name,
        "description": args.description,
        "context": args.prompt_text,
        "category": args.category or "General",
        "tags": tags,
    }
    if getattr(args, "language", None):
        payload["language"] = args.language
    if getattr(args, "scenario", None):
        payload["ext5"] = {"scenarios": [args.scenario]}
    return payload


def _write_temp_prompt_payload(payload: object) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="prompt-add-inline-",
        delete=False,
        encoding="utf-8",
    )
    temp_path = Path(handle.name)
    with handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return temp_path


def _validate_prompt_payload(
    payload: object,
    parser: argparse.ArgumentParser,
) -> dict[str, Any] | list[dict[str, Any]]:
    entries: list[dict[str, Any]]
    if isinstance(payload, Mapping):
        payload_mapping = cast("Mapping[str, Any]", payload)
        prompts_value = payload_mapping.get("prompts")
        if prompts_value is not None:
            if not isinstance(prompts_value, list):
                parser.error(
                    "prompt-add payload field 'prompts' must be a JSON list of prompt objects."
                )
            prompt_entries = cast("list[object]", prompts_value)
            entries = []
            for index, item in enumerate(prompt_entries, start=1):
                if not isinstance(item, Mapping):
                    parser.error(f"prompt-add entry #{index} must be a JSON object.")
                entries.append(dict(cast("Mapping[str, Any]", item)))
        else:
            entries = [dict(payload_mapping)]
    elif isinstance(payload, list):
        payload_entries = cast("list[object]", payload)
        entries = []
        for index, item in enumerate(payload_entries, start=1):
            if not isinstance(item, Mapping):
                parser.error(f"prompt-add entry #{index} must be a JSON object.")
            entries.append(dict(cast("Mapping[str, Any]", item)))
    else:
        parser.error("prompt-add payload must be a JSON object or a list of prompt objects.")

    if not entries:
        parser.error("prompt-add payload must contain at least one prompt object.")

    missing_messages: list[str] = []
    for index, entry in enumerate(entries, start=1):
        missing_fields = [field for field in ("name", "description") if not entry.get(field)]
        if missing_fields:
            missing_messages.append(
                f"entry #{index} is missing required field(s): {', '.join(missing_fields)}"
            )
    if missing_messages:
        parser.error("prompt-add payload validation failed: " + "; ".join(missing_messages))

    if isinstance(payload, Mapping):
        return dict(cast("Mapping[str, Any]", payload))
    return entries


def _parse_json_string_payload(raw_json: str, parser: argparse.ArgumentParser) -> object:
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        parser.error(f"prompt-add received invalid JSON: {exc}")
    return _validate_prompt_payload(payload, parser)


def _read_stdin_json_payload(parser: argparse.ArgumentParser) -> object:
    raw_stdin = sys.stdin.read()
    if not raw_stdin.strip():
        parser.error("prompt-add --from-stdin requires a non-empty JSON payload on stdin.")
    return _parse_json_string_payload(raw_stdin, parser)


def _normalise_prompt_add_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if getattr(args, "command", None) != "prompt-add":
        return

    inline_fields = [
        getattr(args, "name", None),
        getattr(args, "description", None),
        getattr(args, "prompt_text", None),
        getattr(args, "category", None),
        getattr(args, "tags", None),
        getattr(args, "language", None),
        getattr(args, "scenario", None),
    ]
    has_inline = any(value not in (None, "") for value in inline_fields)
    path_value = getattr(args, "path", None)
    json_value = getattr(args, "json_payload", None)
    input_file_value = getattr(args, "input_file", None)
    use_stdin = bool(getattr(args, "from_stdin", False))
    sources = [
        ("path", path_value is not None),
        ("inline", has_inline),
        ("json", json_value not in (None, "")),
        ("input-file", input_file_value is not None),
        ("stdin", use_stdin),
    ]
    selected_sources = [name for name, enabled in sources if enabled]

    if len(selected_sources) == 0:
        parser.error(
            "prompt-add requires exactly one input source: PATH, inline fields, "
            "--json, --input-file, or --from-stdin."
        )
    if len(selected_sources) > 1:
        parser.error(
            "prompt-add accepts exactly one input source: PATH, inline fields, "
            "--json, --input-file, or --from-stdin."
        )
    if has_inline:
        if not getattr(args, "name", None):
            parser.error("prompt-add inline mode requires --name.")
        if not getattr(args, "description", None):
            parser.error("prompt-add inline mode requires --description.")
        if not getattr(args, "prompt_text", None):
            parser.error("prompt-add inline mode requires --prompt-text.")

        payload = _build_inline_prompt_payload(args)
        args.path = _write_temp_prompt_payload(payload)
        return
    if json_value not in (None, ""):
        payload = _parse_json_string_payload(str(json_value), parser)
        args.path = _write_temp_prompt_payload(payload)
        return
    if input_file_value is not None:
        args.path = Path(input_file_value).expanduser()
        return
    if use_stdin:
        payload = _read_stdin_json_payload(parser)
        args.path = _write_temp_prompt_payload(payload)


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

    import_parser = subparsers.add_parser(
        "catalog-import",
        help="Create or update prompts from a JSON catalogue file or directory.",
    )
    import_parser.add_argument(
        "path",
        type=Path,
        help="Source catalogue path (.json file or directory of JSON files).",
    )
    import_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the import summary without writing any changes.",
    )
    import_parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip updates when a prompt with the same name already exists.",
    )

    prompt_add_parser = subparsers.add_parser(
        "prompt-add",
        help="Add or update a prompt from a JSON file or directory of JSON files.",
    )
    prompt_add_parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        help="Source prompt path (.json file or directory of JSON files).",
    )
    prompt_add_parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Prompt name when adding a single prompt inline.",
    )
    prompt_add_parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Prompt description when adding a single prompt inline.",
    )
    prompt_add_parser.add_argument(
        "--prompt-text",
        type=str,
        default=None,
        help="Main prompt body/context when adding a single prompt inline.",
    )
    prompt_add_parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Optional category label for inline prompt creation.",
    )
    prompt_add_parser.add_argument(
        "--tags",
        type=str,
        default=None,
        help="Comma-separated tags for inline prompt creation.",
    )
    prompt_add_parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional language code for inline prompt creation.",
    )
    prompt_add_parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Optional scenario note for inline prompt creation.",
    )
    prompt_add_parser.add_argument(
        "--json",
        dest="json_payload",
        type=str,
        default=None,
        help="Prompt JSON payload passed directly on the command line.",
    )
    prompt_add_parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Read a prompt JSON payload from a file path without using positional PATH.",
    )
    prompt_add_parser.add_argument(
        "--from-stdin",
        action="store_true",
        help="Read a prompt JSON payload from standard input.",
    )
    prompt_add_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the add/update summary without writing any changes.",
    )
    prompt_add_parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip updates when a prompt with the same name already exists.",
    )

    prompt_show_parser = subparsers.add_parser(
        "prompt-show",
        help="Display a prompt by UUID.",
    )
    prompt_show_parser.add_argument(
        "prompt_id",
        type=str,
        help="Prompt UUID to display.",
    )

    prompt_show_parser.add_argument(
        "--json",
        action="store_true",
        help="Render the prompt as structured JSON.",
    )

    prompt_find_parser = subparsers.add_parser(
        "prompt-find",
        help="List prompts matching a text query.",
    )
    prompt_find_parser.add_argument(
        "query",
        type=str,
        help="Case-insensitive text query matched against prompt fields.",
    )
    prompt_find_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of matching prompts to display (default: 10).",
    )
    prompt_find_parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Require an exact category match (case-insensitive).",
    )
    prompt_find_parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Require a matching tag (case-insensitive).",
    )
    prompt_find_parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Require an exact source match (case-insensitive).",
    )
    prompt_find_parser.add_argument(
        "--active",
        type=str,
        default=None,
        help="Require active state: true/false/yes/no/1/0.",
    )
    prompt_find_parser.add_argument(
        "--json",
        action="store_true",
        help="Render matching prompts as structured JSON.",
    )

    prompt_history_parser = subparsers.add_parser(
        "prompt-history",
        help="Display read-only execution history for a specific prompt.",
    )
    prompt_history_parser.add_argument(
        "prompt_id",
        type=str,
        help="Prompt UUID or exact prompt name to inspect.",
    )
    prompt_history_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of recent executions to display (default: 5).",
    )
    prompt_history_parser.add_argument(
        "--status",
        type=str,
        default=None,
        help="Filter executions by status (success or failed).",
    )
    prompt_history_parser.add_argument(
        "--window-days",
        type=int,
        default=0,
        help="Look-back window in days for recent executions (0 keeps full history).",
    )
    prompt_history_parser.add_argument(
        "--json",
        action="store_true",
        help="Render prompt history as structured JSON.",
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

    chain_list_parser = subparsers.add_parser(
        "prompt-chain-list",
        help="List configured prompt chains.",
    )
    chain_list_parser.add_argument(
        "--include-inactive",
        action="store_true",
        help="Include inactive chains in the listing.",
    )

    chain_show_parser = subparsers.add_parser(
        "prompt-chain-show",
        help="Display a prompt chain and its steps.",
    )
    chain_show_parser.add_argument(
        "chain_id",
        type=str,
        help="Prompt chain UUID.",
    )
    chain_show_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the prompt chain as deterministic JSON.",
    )
    chain_show_parser.add_argument(
        "--history-limit",
        type=int,
        default=3,
        help="Show up to this many recent persisted runs for the selected chain.",
    )

    chain_history_parser = subparsers.add_parser(
        "prompt-chain-history",
        help="Display bounded recent prompt chain run history.",
    )
    chain_history_parser.add_argument(
        "--chain-id",
        type=str,
        default=None,
        help="Optional prompt chain UUID to filter recent history.",
    )
    chain_history_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Show up to this many recent prompt chain runs.",
    )
    chain_history_parser.add_argument(
        "--json",
        action="store_true",
        help="Print recent prompt chain history as deterministic JSON.",
    )

    chain_export_parser = subparsers.add_parser(
        "prompt-chain-export",
        help="Export a prompt chain to a JSON file.",
    )
    chain_export_parser.add_argument(
        "chain_id",
        type=str,
        help="Prompt chain UUID.",
    )
    chain_export_parser.add_argument(
        "path",
        type=Path,
        help="Destination JSON file path.",
    )

    chain_validate_parser = subparsers.add_parser(
        "prompt-chain-validate",
        help="Validate a prompt chain JSON definition without persisting it.",
    )
    chain_validate_parser.add_argument(
        "path",
        type=Path,
        help="Path to the JSON file containing the prompt chain definition.",
    )
    chain_validate_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit deterministic JSON output for prompt chain validation.",
    )

    chain_apply_parser = subparsers.add_parser(
        "prompt-chain-apply",
        help="Create or update a prompt chain from a JSON definition.",
    )
    chain_apply_parser.add_argument(
        "path",
        type=Path,
        help="Path to the JSON file containing the prompt chain definition.",
    )
    chain_apply_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the parsed prompt chain without persisting it.",
    )

    chain_run_parser = subparsers.add_parser(
        "prompt-chain-run",
        help="Execute a prompt chain with plain-text input.",
    )
    chain_run_parser.add_argument(
        "chain_id",
        type=str,
        help="Prompt chain UUID to run.",
    )
    chain_run_parser.add_argument(
        "--input",
        dest="chain_input",
        type=str,
        default=None,
        help="Plain-text input sent to the first step (omit to use --input-file).",
    )
    chain_run_parser.add_argument(
        "--input-file",
        dest="chain_input_file",
        type=Path,
        default=None,
        help="Path to a UTF-8 text file whose contents feed the first step.",
    )
    chain_run_parser.add_argument(
        "--no-web-search",
        action="store_true",
        help="Disable live web search enrichment for prompt chain runs.",
    )
    chain_run_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit deterministic JSON output for prompt chain execution.",
    )
    chain_run_parser.add_argument(
        "--final-output-only",
        action="store_true",
        help="Print only the final raw output text.",
    )
    chain_run_parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only the final summary text.",
    )
    chain_run_parser.add_argument(
        "--status-only",
        action="store_true",
        help="Print only the final chain status.",
    )
    chain_run_parser.add_argument(
        "--step-output",
        type=str,
        default=None,
        help="Print only the canonical output text for one step output key.",
    )
    chain_run_parser.add_argument(
        "--step-alias",
        type=str,
        default=None,
        help="Print only the output text resolved from one step alias.",
    )
    chain_run_parser.add_argument(
        "--final-step-meta",
        action="store_true",
        help="Print only bounded terminal metadata for the final step.",
    )
    chain_run_parser.add_argument(
        "--compact",
        action="store_true",
        help="Print a compact operator-facing run summary.",
    )
    chain_run_parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Save the run artifact to a file instead of only printing to stdout.",
    )

    args = parser.parse_args()
    _normalise_prompt_add_args(args, parser)
    return args
