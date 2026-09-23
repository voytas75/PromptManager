"""Argument parser for Prompt Manager CLI.

Updates:
  v0.3.0 - 2025-12-05 - Add prompt chain web search toggle flags.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

ROOT_COMMAND_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Prompt catalog",
        (
            "catalog-export",
            "catalog-import",
            "catalog-check",
            "prompt-add",
            "prompt-show",
            "prompt-random",
            "prompt-find",
            "tag-list",
            "tag-show",
            "prompt-tag",
        ),
    ),
    (
        "Prompt lifecycle and versions",
        (
            "prompt-history",
            "prompt-lineage",
            "prompt-fork",
            "prompt-restore-version",
            "prompt-version-diff",
            "prompt-version-list",
            "prompt-compare",
            "prompt-render",
            "prompt-validate",
            "prompt-lint",
            "prompt-test",
            "prompt-template-list",
        ),
    ),
    (
        "Search and recommendations",
        (
            "suggest",
            "usage-report",
            "history-analytics",
            "refresh-scenarios",
        ),
    ),
    (
        "Prompt chains",
        (
            "prompt-chain-list",
            "prompt-chain-show",
            "prompt-chain-history",
            "prompt-chain-export",
            "prompt-chain-validate",
            "prompt-chain-apply",
            "prompt-chain-run",
        ),
    ),
    (
        "Operations and diagnostics",
        (
            "reembed",
            "benchmark",
            "diagnostics",
        ),
    ),
)

_ROOT_HELP_WIDTH = 110
_ROOT_HELP_COMMAND_COLUMN = 34
_ROOT_HELP_OPTION_COLUMN = 34


class _RootHelpParser(argparse.ArgumentParser):
    """Render a grouped root help card without changing subcommand parsers."""

    subparsers_action: argparse._SubParsersAction[Any] | None = None  # pyright: ignore[reportPrivateUsage]

    def format_help(self) -> str:
        if self.subparsers_action is None:
            return super().format_help()

        choices_actions = cast(
            "list[argparse.Action]",
            getattr(self.subparsers_action, "_choices_actions", []),
        )
        command_help = {action.dest: action.help or "" for action in choices_actions}
        grouped_commands = [command for _, commands in ROOT_COMMAND_GROUPS for command in commands]
        missing = set(command_help).difference(grouped_commands)
        unknown = set(grouped_commands).difference(command_help)
        if missing or unknown:
            return super().format_help()

        lines = [
            f"usage: {self.prog} [GLOBAL OPTIONS] COMMAND [COMMAND OPTIONS]",
            "",
            "Prompt Manager",
            "Manage, search, version, and run reusable prompts.",
            "",
            "Commands",
        ]
        for heading, commands in ROOT_COMMAND_GROUPS:
            lines.extend(["", f"  {heading}"])
            for command in commands:
                summary = " ".join(command_help[command].split())
                available_width = _ROOT_HELP_WIDTH - _ROOT_HELP_COMMAND_COLUMN - 4
                wrapped_summary = textwrap.wrap(summary, width=available_width) or [""]
                lines.append(f"    {command:<{_ROOT_HELP_COMMAND_COLUMN}}{wrapped_summary[0]}")
                lines.extend(
                    f"    {'':<{_ROOT_HELP_COMMAND_COLUMN}}{continuation}"
                    for continuation in wrapped_summary[1:]
                )

        lines.extend(
            [
                "",
                "Global options",
                f"  {'--logging-config PATH':<{_ROOT_HELP_OPTION_COLUMN}}"
                "Load logging configuration from an INI file.",
                f"  {'--print-settings':<{_ROOT_HELP_OPTION_COLUMN}}"
                "Print resolved settings and exit.",
                f"  {'--no-gui':<{_ROOT_HELP_OPTION_COLUMN}}"
                "Initialise services without the desktop app.",
                f"  {'-h, --help':<{_ROOT_HELP_OPTION_COLUMN}}Show this help card.",
                "",
                "Run `main.py COMMAND --help` for command-specific options.",
                "The desktop app is the default when no command is supplied.",
            ]
        )
        return "\n".join(lines) + "\n"


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
        ext5 = entry.get("ext5")
        if ext5 is not None and not isinstance(ext5, Mapping):
            missing_messages.append(f"entry #{index} field 'ext5' must be a JSON object")
        elif isinstance(ext5, Mapping):
            ext5_mapping = cast("Mapping[str, Any]", ext5)
            scenarios = ext5_mapping.get("scenarios")
            if scenarios is not None and not isinstance(scenarios, (list, str)):
                missing_messages.append(
                    f"entry #{index} field 'ext5.scenarios' must be a JSON list or string"
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
    parser = _RootHelpParser(
        description="Prompt Manager launcher",
        formatter_class=lambda prog: argparse.HelpFormatter(
            prog,
            width=_ROOT_HELP_WIDTH,
            max_help_position=_ROOT_HELP_COMMAND_COLUMN,
        ),
    )
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
    parser.subparsers_action = subparsers

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

    catalog_check_parser = subparsers.add_parser(
        "catalog-check",
        help="Run deterministic read-only integrity checks across prompts and chains.",
    )
    catalog_check_parser.add_argument(
        "--json",
        action="store_true",
        help="Render catalog-check results as structured JSON.",
    )

    prompt_show_parser = subparsers.add_parser(
        "prompt-show",
        help="Display a prompt by UUID or exact name.",
        description=(
            "Display a readable prompt record. Default text output wraps its description and "
            "encloses the prompt context in <prompt_body> tags for easy copy/paste."
        ),
    )
    prompt_show_parser.add_argument(
        "prompt_id",
        type=str,
        help="Prompt UUID or exact prompt name to display.",
    )

    prompt_show_parser.add_argument(
        "--json",
        action="store_true",
        help="Render the prompt as structured JSON.",
    )
    prompt_show_parser.add_argument(
        "--full",
        action="store_true",
        help="Include the full persisted record, including the embedding vector (requires --json).",
    )

    _ = subparsers.add_parser(
        "prompt-random",
        help="Display one randomly selected local prompt.",
    )
    prompt_find_parser = subparsers.add_parser(
        "prompt-find",
        help="Find prompts by raw semantic rank for a natural-language query.",
    )
    prompt_find_parser.add_argument(
        "query",
        type=str,
        help="Natural-language query used for raw semantic prompt retrieval.",
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
    prompt_find_parser.add_argument(
        "--full",
        action="store_true",
        help="Include complete stored records, including embedding vectors (requires --json).",
    )

    tag_list_parser = subparsers.add_parser(
        "tag-list",
        help="List distinct prompt tags with prompt and active-prompt counts.",
    )
    tag_list_parser.add_argument(
        "--json",
        action="store_true",
        help="Render tag aggregates as structured JSON.",
    )

    tag_show_parser = subparsers.add_parser(
        "tag-show",
        help="Show prompts having one exact tag (case-insensitive).",
    )
    tag_show_parser.add_argument(
        "tag",
        type=str,
        help="Tag to inspect (case-insensitive exact match).",
    )
    tag_show_parser.add_argument(
        "--json",
        action="store_true",
        help="Render the tag and matching prompts as structured JSON.",
    )

    prompt_tag_parser = subparsers.add_parser(
        "prompt-tag",
        help="Add or remove one tag on a prompt by UUID or exact name.",
    )
    prompt_tag_parser.add_argument(
        "prompt_id",
        type=str,
        help="Prompt UUID or exact prompt name to update.",
    )
    prompt_tag_parser.add_argument(
        "action",
        choices=("add", "remove"),
        help="Add a missing tag or remove an existing tag.",
    )
    prompt_tag_parser.add_argument(
        "tag",
        type=str,
        help="Single non-blank tag to add or remove (case-insensitive match).",
    )
    prompt_tag_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the tag change without writing it.",
    )
    prompt_tag_parser.add_argument(
        "--json",
        action="store_true",
        help="Render the tag change summary as structured JSON.",
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

    prompt_lineage_parser = subparsers.add_parser(
        "prompt-lineage",
        help="Inspect parent and child fork lineage for a prompt.",
    )
    prompt_lineage_parser.add_argument(
        "prompt_id",
        type=str,
        help="Prompt UUID or exact prompt name to inspect.",
    )
    prompt_lineage_parser.add_argument(
        "--json",
        action="store_true",
        help="Render prompt lineage as structured JSON.",
    )

    prompt_fork_parser = subparsers.add_parser(
        "prompt-fork",
        help="Create a named prompt variant while preserving source lineage.",
    )
    prompt_fork_parser.add_argument(
        "prompt_id",
        type=str,
        help="Source prompt UUID or exact prompt name.",
    )
    prompt_fork_parser.add_argument(
        "--name",
        required=True,
        type=str,
        help="Required name for the new forked prompt.",
    )
    prompt_fork_parser.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Optional version message recorded for the new fork.",
    )
    prompt_fork_parser.add_argument(
        "--json",
        action="store_true",
        help="Render the created fork and its lineage as structured JSON.",
    )

    prompt_restore_parser = subparsers.add_parser(
        "prompt-restore-version",
        help="Restore a prompt snapshot as a new recorded version.",
    )
    prompt_restore_parser.add_argument(
        "version_id",
        type=int,
        help="Version snapshot ID to restore.",
    )
    prompt_restore_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required acknowledgement that this updates the live prompt.",
    )
    prompt_restore_parser.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Optional message recorded for the restored version.",
    )
    prompt_restore_parser.add_argument(
        "--json",
        action="store_true",
        help="Render the restored prompt as structured JSON.",
    )

    prompt_version_diff_parser = subparsers.add_parser(
        "prompt-version-diff",
        help="Compare two recorded versions of the same prompt.",
    )
    prompt_version_diff_parser.add_argument(
        "base_version_id",
        type=int,
        help="First/base version snapshot ID; output preserves this comparison order.",
    )
    prompt_version_diff_parser.add_argument(
        "target_version_id",
        type=int,
        help="Second/target version snapshot ID; output preserves this comparison order.",
    )
    prompt_version_diff_parser.add_argument(
        "--json",
        action="store_true",
        help="Render the version diff as structured JSON.",
    )

    prompt_version_list_parser = subparsers.add_parser(
        "prompt-version-list",
        help="List recorded versions for a specific prompt.",
    )
    prompt_version_list_parser.add_argument(
        "prompt_id",
        type=str,
        help="Prompt UUID or exact prompt name to inspect.",
    )
    prompt_version_list_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of versions to display (default: 20).",
    )
    prompt_version_list_parser.add_argument(
        "--json",
        action="store_true",
        help="Render prompt version history as structured JSON.",
    )

    prompt_compare_parser = subparsers.add_parser(
        "prompt-compare",
        help="Compare two current prompts without rendering or calling a model.",
    )
    prompt_compare_parser.add_argument(
        "left_prompt_id",
        type=str,
        help="Left prompt UUID or exact prompt name.",
    )
    prompt_compare_parser.add_argument(
        "right_prompt_id",
        type=str,
        help="Right prompt UUID or exact prompt name.",
    )
    prompt_compare_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit deterministic structured comparison data.",
    )

    prompt_render_parser = subparsers.add_parser(
        "prompt-render",
        help="Render and validate a prompt template without calling a model.",
    )
    prompt_render_parser.add_argument(
        "prompt_id",
        type=str,
        help="Prompt UUID or exact prompt name to render.",
    )
    variables_group = prompt_render_parser.add_mutually_exclusive_group()
    variables_group.add_argument(
        "--variables-json",
        type=str,
        default=None,
        help="JSON object containing template variable values.",
    )
    variables_group.add_argument(
        "--variables-file",
        type=Path,
        default=None,
        help="UTF-8 JSON file containing a template variable object.",
    )
    prompt_render_parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate template variables without emitting rendered text.",
    )
    prompt_render_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a deterministic JSON render or validation result.",
    )

    prompt_validate_parser = subparsers.add_parser(
        "prompt-validate",
        help="Validate one stored prompt without rendering or calling a model.",
    )
    prompt_validate_parser.add_argument(
        "prompt_id",
        type=str,
        help="Prompt UUID or exact prompt name to validate.",
    )
    prompt_validate_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit deterministic structured validation evidence.",
    )

    prompt_lint_parser = subparsers.add_parser(
        "prompt-lint",
        help="Lint one stored prompt with deterministic advisory checks.",
    )
    prompt_lint_parser.add_argument(
        "prompt_id",
        type=str,
        help="Prompt UUID or exact prompt name to lint.",
    )
    prompt_lint_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit deterministic structured lint evidence.",
    )

    prompt_test_parser = subparsers.add_parser(
        "prompt-test",
        help="Run deterministic local template fixtures for one stored prompt.",
    )
    prompt_test_parser.add_argument(
        "prompt_id",
        type=str,
        help="Prompt UUID or exact prompt name to test.",
    )
    prompt_test_parser.add_argument(
        "--suite",
        type=Path,
        required=True,
        help="UTF-8 JSON fixture suite with cases containing id, variables, and expected.",
    )
    prompt_test_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit deterministic structured test results.",
    )

    prompt_template_list_parser = subparsers.add_parser(
        "prompt-template-list",
        help="Show all effective built-in workflow templates without running a model.",
    )
    prompt_template_list_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit complete effective template records as structured JSON.",
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
    is_compact_prompt_read = getattr(args, "command", None) in {"prompt-find", "prompt-show"}
    if is_compact_prompt_read and args.full and not args.json:
        parser.error(f"{args.command} --full requires --json.")
    _normalise_prompt_add_args(args, parser)
    return args
