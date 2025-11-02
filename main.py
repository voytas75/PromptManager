"""Application entry point for Prompt Manager.

Updates: v0.7.1 - 2025-11-14 - Simplify GUI dependency guidance for unified installs.
Updates: v0.7.0 - 2025-11-07 - Add semantic suggestion CLI to verify embedding backends.
Updates: v0.6.0 - 2025-11-06 - Add CLI catalogue import/export commands with diff previews.
Updates: v0.5.0 - 2025-11-05 - Seed prompt repository from packaged catalogue before launch.
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

from config import load_settings
from core import (
    CatalogDiff,
    build_prompt_manager,
    diff_prompt_catalog,
    export_prompt_catalog,
    import_prompt_catalog,
)


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


def _print_catalog_preview(diff: CatalogDiff) -> None:
    header = (
        f"Catalogue preview ({diff.source or 'builtin'}): "
        f"added={diff.added} updated={diff.updated} "
        f"skipped={diff.skipped} unchanged={diff.unchanged}"
    )
    print(header)
    for entry in diff.entries:
        print(f"\n[{entry.change_type.value.upper()}] {entry.name} ({entry.prompt_id})")
        if entry.diff:
            print(textwrap.indent(entry.diff, "  "))
        else:
            print("  (no diff)")


def _resolve_export_format(path: Path, explicit_format: Optional[str]) -> str:
    if explicit_format:
        return explicit_format.lower()
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return "yaml"
    return "json"


def _run_catalog_import(manager, args: argparse.Namespace, logger: logging.Logger) -> int:
    catalog_path = Path(args.path).expanduser()
    overwrite = not getattr(args, "no_overwrite", False)
    try:
        preview = diff_prompt_catalog(manager, catalog_path, overwrite=overwrite)
    except Exception as exc:
        logger.error("Unable to preview catalogue: %s", exc)
        return 5
    _print_catalog_preview(preview)
    if getattr(args, "dry_run", False):
        logger.info("Dry-run complete. No changes applied.")
        return 0
    if not preview.has_changes():
        logger.info("No catalogue changes detected; nothing to apply.")
        return 0
    result = import_prompt_catalog(manager, catalog_path, overwrite=overwrite)
    logger.info(
        "Catalogue import applied: added=%d updated=%d skipped=%d errors=%d",
        result.added,
        result.updated,
        result.skipped,
        result.errors,
    )
    return 0 if result.errors == 0 else 5


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

    import_parser = subparsers.add_parser(
        "catalog-import",
        help="Import prompts from a JSON file or directory, showing a diff preview first.",
    )
    import_parser.add_argument("path", type=Path, help="Path to a catalogue JSON file or directory")
    import_parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip updates for existing prompts (new prompts will still be added).",
    )
    import_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview catalogue changes without applying them.",
    )

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
        logger.info(
            "Resolved settings: db=%s chroma=%s redis=%s ttl=%s catalog=%s litellm_model=%s",
            settings.db_path,
            settings.chroma_path,
            settings.redis_dsn,
            settings.cache_ttl_seconds,
            settings.catalog_path,
            settings.litellm_model,
        )
        return 0

    # Build core services; GUI wiring will attach here in later milestones
    try:
        manager = build_prompt_manager(settings)
    except Exception as exc:
        logger.error("Failed to initialise services: %s", exc)
        return 3

    command = getattr(args, "command", None)
    try:
        if command == "catalog-import":
            return _run_catalog_import(manager, args, logger)

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

        catalog_result = None
        try:
            catalog_result = import_prompt_catalog(manager, settings.catalog_path)
            if catalog_result.added or catalog_result.updated:
                logger.info(
                    "Prompt catalogue synced: added=%d updated=%d skipped=%d",
                    catalog_result.added,
                    catalog_result.updated,
                    catalog_result.skipped,
                )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to import prompt catalogue: %s", exc)

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
