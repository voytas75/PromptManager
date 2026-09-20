"""Regression checks for the public CLI help and documentation contract."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from core.catalog_importer import _entry_to_prompt  # pyright: ignore[reportPrivateUsage]

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _parser_command_names() -> set[str]:
    """Return public command names declared by the argparse parser."""
    parser_path = REPOSITORY_ROOT / "cli" / "parser.py"
    tree = ast.parse(parser_path.read_text(encoding="utf-8"))
    commands: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_parser":
            continue
        if not node.args:
            continue
        first_argument = node.args[0]
        if isinstance(first_argument, ast.Constant) and isinstance(first_argument.value, str):
            commands.add(first_argument.value)
    return commands


def test_prompt_add_sample_uses_importer_body_field() -> None:
    """The checked-in sample must preserve its body through the catalog importer."""
    sample_path = REPOSITORY_ROOT / "examples" / "prompt-import-example.json"
    payload = json.loads(sample_path.read_text(encoding="utf-8"))

    assert "context" in payload
    assert "prompt_text" not in payload
    assert _entry_to_prompt(payload).context == payload["context"]


def test_developer_cli_index_covers_every_public_command() -> None:
    """The developer command index must not silently omit public CLI commands."""
    guide = (REPOSITORY_ROOT / "docs" / "README-DEV.md").read_text(encoding="utf-8")
    cli_section = guide.split("## CLI Utilities", maxsplit=1)[1].split(
        "### GUI Prompt Chain Manager", maxsplit=1
    )[0]

    missing = [
        command
        for command in sorted(_parser_command_names())
        if f"`python -m main {command}" not in cli_section
    ]

    assert missing == []
