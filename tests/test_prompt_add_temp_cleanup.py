"""Process-level cleanup contract for generated prompt-add JSON payloads."""

from __future__ import annotations

import builtins
import json
import os
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from cli import parser as prompt_parser

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import TextIO

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("installed", [False, True])
def test_prompt_add_cleans_payload_when_application_import_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, installed: bool
) -> None:
    temp = tmp_path / "temp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prompt-manager",
            "prompt-add",
            "--name",
            "Private title",
            "--description",
            "Private description",
            "--prompt-text",
            "PRIVATE_BODY",
            "--dry-run",
        ],
    )
    original_import = builtins.__import__

    def fail_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = None,
        level: int = 0,
    ) -> object:
        if name == ("main" if installed else "config"):
            raise ImportError("controlled startup failure")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_import)
    with pytest.raises(ImportError, match="controlled startup failure"):
        if installed:
            from cli.entrypoint import main as console_main

            console_main()
        else:
            runpy.run_path(str(ROOT / "main.py"), run_name="__main__")
    assert list(temp.glob("prompt-add-inline-*.json")) == []


def test_prompt_add_cleans_partially_written_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temp = tmp_path / "temp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))

    def fail_dump(payload: object, handle: TextIO, **kwargs: object) -> None:
        handle.write("PRIVATE_BODY")
        raise OSError("controlled write failure")

    monkeypatch.setattr(prompt_parser.json, "dump", fail_dump)
    with pytest.raises(OSError, match="controlled write failure"):
        prompt_parser._write_temp_prompt_payload({"context": "PRIVATE_BODY"})  # pyright: ignore[reportPrivateUsage]
    assert list(temp.glob("prompt-add-inline-*.json")) == []


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("source", ["inline", "json", "stdin", "file"])
@pytest.mark.parametrize("outcome", ["preview", "apply", "startup_error"])
def test_prompt_add_generated_payload_is_removed(
    tmp_path: Path, installed: bool, source: str, outcome: str
) -> None:
    temp = tmp_path / "temp"
    temp.mkdir()
    config = tmp_path / "settings.json"
    config.write_text(
        json.dumps(
            {
                "database_path": str(tmp_path / "catalog.db"),
                "chroma_path": str(tmp_path / "chroma"),
                "embedding_backend": "deterministic",
                "redis_dsn": None,
                "litellm_model": None,
                "litellm_inference_model": None,
            }
        )
    )
    payload = {
        "name": "Private audit prompt",
        "description": "Private description",
        "context": "secret body",
    }
    supplied = tmp_path / "supplied.json"
    supplied.write_text(json.dumps(payload))
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PROMPT_MANAGER_", "AZURE_OPENAI_"))
        and key not in {"LITELLM_API_KEY", "OPENAI_API_KEY"}
    }
    env.update(
        HOME=str(tmp_path),
        TMPDIR=str(temp),
        PYTHONPATH=str(ROOT),
        PROMPT_MANAGER_CONFIG_JSON=str(
            config if outcome != "startup_error" else tmp_path / "absent.json"
        ),
        PROMPT_MANAGER_ENV_FILE="",
        PYTHONDONTWRITEBYTECODE="1",
    )
    input_text = ""
    if source == "inline":
        arguments = [
            "--name",
            payload["name"],
            "--description",
            payload["description"],
            "--prompt-text",
            payload["context"],
        ]
    elif source == "json":
        arguments = ["--json", json.dumps(payload)]
    elif source == "stdin":
        arguments = ["--from-stdin"]
        input_text = json.dumps(payload)
    else:
        arguments = [str(supplied)]
    if outcome == "preview":
        arguments.append("--dry-run")
    command = (
        [str(ROOT / ".venv/bin/prompt-manager")] if installed else [sys.executable, "-m", "main"]
    )
    result = subprocess.run(
        [*command, "prompt-add", *arguments],
        cwd=tmp_path,
        env=env,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )
    assert result.returncode == (2 if outcome == "startup_error" else 0), result.stderr
    assert list(temp.glob("prompt-add-inline-*.json")) == []
    assert supplied.read_text() == json.dumps(payload)
    if outcome == "apply":
        assert (tmp_path / "catalog.db").exists()
