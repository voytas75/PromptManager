"""Regression checks for the public CLI help and documentation contract."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

from cli.parser import ROOT_COMMAND_GROUPS, parse_args
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


def test_root_help_card_groups_every_public_command(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Root help must be compact, grouped, and cover every public command once."""
    monkeypatch.setattr(sys, "argv", ["prompt-manager", "--help"])

    with pytest.raises(SystemExit) as excinfo:
        parse_args()

    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert output.startswith("usage: prompt-manager [GLOBAL OPTIONS] COMMAND [COMMAND OPTIONS]\n")
    assert "{catalog-export," not in output
    assert "Prompt catalog" in output
    assert "Prompt lifecycle and versions" in output
    assert "Search and recommendations" in output
    assert "Prompt chains" in output
    assert "Operations and diagnostics" in output
    assert "Global options" in output
    assert "--gui" not in output
    assert "Run `main.py COMMAND --help` for command-specific options." in output
    assert "The desktop app is the default when no command is supplied." in output

    public_commands = _parser_command_names()
    grouped_commands = [command for _, commands in ROOT_COMMAND_GROUPS for command in commands]
    assert set(grouped_commands) == public_commands
    for command in public_commands:
        assert sum(line.startswith(f"    {command}") for line in output.splitlines()) == 1

    for command in ("prompt-restore-version", "prompt-chain-validate"):
        line = next(line for line in output.splitlines() if command in line)
        assert line.index(command) < _ROOT_HELP_DESCRIPTION_COLUMN
        assert len(line) <= 110
    assert max(map(len, output.splitlines())) <= 110


_ROOT_HELP_DESCRIPTION_COLUMN = 36


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
