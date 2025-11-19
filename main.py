"""Application entry point for Prompt Manager.

Updates: v0.7.7 - 2025-11-05 - Surface LiteLLM workflow routing details in CLI summaries.
Updates: v0.7.6 - 2025-11-05 - Expand CLI settings summary to list fast and inference LiteLLM models.
Updates: v0.7.5 - 2025-11-30 - Remove catalogue import command and startup messaging.
Updates: v0.7.4 - 2025-11-26 - Surface LiteLLM streaming configuration in CLI summaries.
Updates: v0.7.3 - 2025-11-17 - Require explicit catalogue paths; skip built-in seeding on startup.
Updates: v0.7.2 - 2025-11-15 - Extend --print-settings with health checks and masked secret output.
Updates: v0.7.1 - 2025-11-14 - Simplify GUI dependency guidance for unified installs.
Updates: v0.7.0 - 2025-11-07 - Add semantic suggestion CLI to verify embedding backends.
Updates: v0.4.1 - 2025-11-05 - Launch GUI by default and add --no-gui flag.
Updates: v0.4.0 - 2025-11-05 - Ensure manager shutdown occurs on exit and update GUI guidance.
Updates: v0.3.0 - 2025-11-05 - Gracefully handle missing GUI dependencies.
Updates: v0.2.0 - 2025-11-04 - Add optional PySide6 GUI launcher toggle.
Updates: v0.1.0 - 2025-10-30 - Initial CLI bootstrap loading settings and building services.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import logging.config
import textwrap
from collections import Counter
from pathlib import Path
from typing import Callable, Optional, cast

from config import (
    DEFAULT_EMBEDDING_BACKEND,
    DEFAULT_EMBEDDING_MODEL,
    LITELLM_ROUTED_WORKFLOWS,
    PromptManagerSettings,
    load_settings,
)
from core import build_prompt_manager, export_prompt_catalog


def _mask_secret(value: Optional[str]) -> str:
    """Return masked representation of secret values."""

    if not value:
        return "not set"
    secret = value.strip()
    if len(secret) <= 6:
        return "set (****)"
    prefix = secret[:4]
    suffix = secret[-4:]
    return f"set ({prefix}...{suffix})"


def _describe_path(path_value: object, *, expect_directory: bool, allow_missing_file: bool = False) -> str:
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


def _print_settings_summary(settings: PromptManagerSettings) -> None:
    """Emit a readable summary of core configuration and health checks."""

    redis_dsn = getattr(settings, "redis_dsn", None)
    litellm_model = getattr(settings, "litellm_model", None)
    litellm_inference_model = getattr(settings, "litellm_inference_model", None)
    litellm_api_key = getattr(settings, "litellm_api_key", None)
    litellm_api_base = getattr(settings, "litellm_api_base", None)
    litellm_api_version = getattr(settings, "litellm_api_version", None)
    litellm_reasoning_effort = getattr(settings, "litellm_reasoning_effort", None)
    litellm_stream = getattr(settings, "litellm_stream", False)
    litellm_workflow_models = getattr(settings, "litellm_workflow_models", None) or {}
    embedding_backend = getattr(settings, "embedding_backend", None)
    embedding_model = getattr(settings, "embedding_model", None)

    def _format_tier(value: str) -> str:
        return "Inference" if value == "inference" else "Fast"

    lines = [
        "Prompt Manager configuration summary",
        "------------------------------------",
        f"Database path: {_describe_path(settings.db_path, expect_directory=False, allow_missing_file=True)}",
        f"Chroma directory: {_describe_path(settings.chroma_path, expect_directory=True)}",
        f"Redis DSN: {redis_dsn or 'not set'}",
        f"Cache TTL (seconds): {getattr(settings, 'cache_ttl_seconds', 'n/a')}",
        "",
        "LiteLLM configuration",
        "---------------------",
        f"Fast model: {litellm_model or 'not set'}",
        f"Inference model: {litellm_inference_model or 'not set'}",
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
        f"Model: {embedding_model or ('(auto)' if embedding_backend == 'litellm' and litellm_model else DEFAULT_EMBEDDING_MODEL)}",
        ]
    )
    print("\n".join(lines))


def _setup_logging(logging_conf_path: Optional[Path]) -> None:
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


def _resolve_export_format(path: Path, explicit_format: Optional[str]) -> str:
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
        logger.error("Failed to export catalogue: %s", exc)
        return 6
    logger.info("Prompt catalogue exported to %s (%s)", resolved, fmt)
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
                {index}. {prompt.name} [{prompt.category or 'Uncategorised'}]
                   Quality: {quality}  Tags: {tags}
                   Description: {prompt.description}
                """
            )
        )
    return 0


def parse_args() -> argparse.Namespace:
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

    return parser.parse_args()


def main() -> int:
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

        # Minimal interactive stub to verify bootstrap until GUI arrives
        logger.info("Prompt Manager ready. Database at %s", settings.db_path)
        logger.info("ChromaDB at %s", settings.chroma_path)
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
                launch_gui_callable = getattr(gui_module, "launch_prompt_manager")
            except AttributeError:
                logger.error(
                    "GUI module is missing the launch_prompt_manager entry point. "
                    "Reinstall dependencies with `pip install -r requirements.txt` or launch without --gui."
                )
                return 4
            if not callable(launch_gui_callable):
                logger.error(
                    "GUI module launch_prompt_manager entry point is not callable. "
                    "Reinstall dependencies with `pip install -r requirements.txt` or launch without --gui."
                )
                return 4
            launch_callable = cast(Callable[[object, Optional[object]], int], launch_gui_callable)

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
