"""Argument parser for Prompt Manager CLI.

Updates:
  v0.3.0 - 2025-12-05 - Add prompt chain web search toggle flags.
"""

from __future__ import annotations

import argparse
from pathlib import Path


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

    chain_apply_parser = subparsers.add_parser(
        "prompt-chain-apply",
        help="Create or update a prompt chain from a JSON definition.",
    )
    chain_apply_parser.add_argument(
        "path",
        type=Path,
        help="Path to the JSON file containing the prompt chain definition.",
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

    return parser.parse_args()
