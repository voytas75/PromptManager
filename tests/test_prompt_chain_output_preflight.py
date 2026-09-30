"""Provider-free process regressions for chain output preparation.

Updates:
  v0.1.0 - 2026-09-30 - Cover output failures before and after chain execution.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from cli.commands import run_prompt_chain_run
from cli.utils import prepare_output_file

ROOT = Path(__file__).resolve().parents[1]
CHAIN_ID = "00000000-0000-0000-0000-000000000991"
PRIVATE_MARKER = "PRIVATE_SYNTHETIC_OUTPUT_123"
BOOTSTRAP = """
import json, os, runpy, socket, sys
from pathlib import Path
from types import SimpleNamespace as NS

def deny_network(*args, **kwargs):
    raise AssertionError('NETWORK_FORBIDDEN')
socket.socket.connect = deny_network
socket.socket.connect_ex = deny_network
socket.create_connection = deny_network
import core

receipt = Path(os.environ['TEST_RECEIPT'])
target = Path(os.environ['TEST_OUTPUT'])
mode = os.environ['TEST_MODE']
calls = []
original_write_text = Path.write_text
class Manager:
    def run_prompt_chain(self, *args, **kwargs):
        calls.append('chain')
        if mode in {'late_write_failure', 'late_encoding_failure'}:
            def fail_write(path, *args, **kwargs):
                if path == target:
                    if mode == 'late_encoding_failure':
                        raise UnicodeEncodeError('utf-8', 'Synthetic result', 0, 1,
                            'PRIVATE_SYNTHETIC_OUTPUT_123')
                    raise OSError('PRIVATE_SYNTHETIC_OUTPUT_123')
                return original_write_text(path, *args, **kwargs)
            Path.write_text = fail_write
        fields = {key: None for key in (
            'final_step_id', 'final_step_output_key', 'final_step_label',
            'terminal_step_id', 'terminal_step_output_key', 'terminal_step_label',
            'terminal_step_status')}
        return NS(**fields, chain=NS(id='00000000-0000-0000-0000-000000000991',
            name='Synthetic chain'), chain_input='Synthetic input',
            final_output_text='Synthetic result', final_summary_text='',
            run_status='success', step_aliases={}, step_outputs={}, steps=[])
    def close(self):
        original_write_text(receipt, json.dumps(calls))
core.build_prompt_manager = lambda *args, **kwargs: Manager()
if mode == 'prepare_failure':
    original_mkdir = Path.mkdir
    def fail_mkdir(path, *args, **kwargs):
        if path == target.parent:
            raise PermissionError('PRIVATE_SYNTHETIC_OUTPUT_123')
        return original_mkdir(path, *args, **kwargs)
    Path.mkdir = fail_mkdir
if mode in {'unwritable_target', 'unwritable_parent'}:
    original_access = os.access
    def fail_access(path, mode, *args, **kwargs):
        if Path(path) == target or Path(path) == target.parent:
            return False
        return original_access(path, mode, *args, **kwargs)
    os.access = fail_access
if mode == 'stat_failure':
    original_stat = Path.stat
    def fail_stat(path, *args, **kwargs):
        if path == target:
            raise PermissionError('PRIVATE_SYNTHETIC_OUTPUT_123')
        return original_stat(path, *args, **kwargs)
    Path.stat = fail_stat
"""


def _run(
    tmp_path: Path, *, installed: bool, json_mode: bool, mode: str, target: Path | None
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    config = tmp_path / "settings.json"
    config.write_text(
        json.dumps(
            {
                "embedding_backend": "deterministic",
                "redis_dsn": None,
                "litellm_model": None,
                "litellm_inference_model": None,
            }
        ),
        encoding="utf-8",
    )
    receipt = tmp_path / "receipt.json"
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(
            (
                "PROMPT_MANAGER_",
                "AZURE_",
                "OPENAI_",
                "LITELLM_",
                "ANTHROPIC_",
                "EXA_",
                "TAVILY_",
                "SERPER_",
                "SERPAPI_",
                "GOOGLE_",
                "TEST_",
            )
        )
    }
    env.update(
        HOME=str(tmp_path),
        TMPDIR=str(tmp_path),
        PYTHONPATH=str(ROOT),
        PROMPT_MANAGER_CONFIG_JSON=str(config),
        PROMPT_MANAGER_ENV_FILE="",
        PYTHONDONTWRITEBYTECODE="1",
        LITELLM_LOCAL_MODEL_COST_MAP="True",
        CHROMA_ANONYMIZED_TELEMETRY="0",
        TEST_MODE=mode,
        TEST_RECEIPT=str(receipt),
        TEST_OUTPUT=str(target or tmp_path / "unused"),
    )
    entrypoint = str(ROOT / ".venv/bin/prompt-manager") if installed else "main"
    dispatch = (
        "runpy.run_path(sys.argv.pop(1), run_name='__main__')"
        if installed
        else "sys.argv.pop(1); runpy.run_module('main', run_name='__main__')"
    )
    args = ["prompt-chain-run", CHAIN_ID, "--input", "Synthetic input", "--no-web-search"]
    if json_mode:
        args.append("--json")
    if target is not None:
        args.extend(("--output-file", str(target)))
    result = subprocess.run(
        [sys.executable, "-c", BOOTSTRAP + "\n" + dispatch, entrypoint, *args],
        cwd=tmp_path,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert receipt.exists(), (result.stdout, result.stderr)
    return result, json.loads(receipt.read_text(encoding="utf-8"))


def _assert_safe_error(
    result: subprocess.CompletedProcess[str], *, json_mode: bool, code: str
) -> None:
    assert result.returncode == 5
    assert result.stdout == ""
    assert "Traceback" not in result.stderr
    assert PRIVATE_MARKER not in result.stderr
    assert "Synthetic input" not in result.stderr
    assert "Synthetic result" not in result.stderr
    message = (
        "Unable to prepare chain output file."
        if code == "OUTPUT_UNAVAILABLE"
        else "Unable to write chain output file."
    )
    if json_mode:
        payload = json.loads(result.stderr)
        assert payload == {
            "ok": False,
            "command": "prompt-chain-run",
            "error": {"code": code, "message": message},
        }
    else:
        assert result.stderr == message + "\n"


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize(
    "mode",
    [
        "parent_file",
        "target_directory",
        "prepare_failure",
        "unwritable_target",
        "unwritable_parent",
        "stat_failure",
    ],
)
def test_output_preflight_rejects_before_execution(
    tmp_path: Path, installed: bool, json_mode: bool, mode: str
) -> None:
    target = tmp_path / PRIVATE_MARKER / "artifact.json"
    if mode == "parent_file":
        target.parent.write_text("Synthetic blocker", encoding="utf-8")
    elif mode == "target_directory":
        target.mkdir(parents=True)
    elif mode == "unwritable_target":
        target.parent.mkdir()
        target.write_text("Original artifact", encoding="utf-8")
    result, calls = _run(
        tmp_path, installed=installed, json_mode=json_mode, mode=mode, target=target
    )
    assert calls == [], (result.returncode, result.stderr)
    _assert_safe_error(result, json_mode=json_mode, code="OUTPUT_UNAVAILABLE")
    if mode == "parent_file":
        assert target.parent.read_text(encoding="utf-8") == "Synthetic blocker"
    elif mode == "target_directory":
        assert target.is_dir()
    elif mode == "unwritable_target":
        assert target.read_text(encoding="utf-8") == "Original artifact"
    else:
        assert not target.exists()


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize("mode", ["late_write_failure", "late_encoding_failure"])
def test_late_output_failure_does_not_repeat_execution(
    tmp_path: Path, installed: bool, json_mode: bool, mode: str
) -> None:
    target = tmp_path / PRIVATE_MARKER / "artifact.json"
    result, calls = _run(
        tmp_path, installed=installed, json_mode=json_mode, mode=mode, target=target
    )
    assert calls == ["chain"]
    _assert_safe_error(result, json_mode=json_mode, code="OUTPUT_WRITE_FAILED")


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize("existing", [False, True])
def test_output_file_preserves_creation_and_overwrite(
    tmp_path: Path, installed: bool, json_mode: bool, existing: bool
) -> None:
    target = tmp_path / "new-parent" / "artifact.json"
    if existing:
        target.parent.mkdir()
        target.write_text("Original artifact", encoding="utf-8")
    result, calls = _run(
        tmp_path, installed=installed, json_mode=json_mode, mode="success", target=target
    )
    assert calls == ["chain"]
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout == f"Saved prompt chain run artifact to {target}.\n"
    artifact = target.read_text(encoding="utf-8")
    if json_mode:
        assert json.loads(artifact)["final_output_text"] == "Synthetic result"
    else:
        assert "Synthetic result" in artifact
    assert sorted(path.name for path in target.parent.iterdir()) == [target.name]


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("json_mode", [False, True])
def test_no_output_file_does_not_prepare_unused_output(
    tmp_path: Path, installed: bool, json_mode: bool
) -> None:
    result, calls = _run(
        tmp_path, installed=installed, json_mode=json_mode, mode="prepare_failure", target=None
    )
    assert calls == ["chain"]
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    if json_mode:
        assert json.loads(result.stdout)["final_output_text"] == "Synthetic result"
    else:
        assert "Synthetic result" in result.stdout
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize(
    "mode", ["new", "overwrite", "directory", "unwritable", "parent_unwritable"]
)
def test_preflight_helper_preserves_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    target = tmp_path / "parent" / "artifact.txt"
    if mode == "directory":
        target.mkdir(parents=True)
    elif mode in {"overwrite", "unwritable"}:
        target.parent.mkdir()
        target.write_text("Original artifact", encoding="utf-8")
    if mode in {"unwritable", "parent_unwritable"}:

        def deny_access(*_args: Any) -> bool:
            return False

        monkeypatch.setattr(os, "access", deny_access)
    if mode in {"directory", "unwritable", "parent_unwritable"}:
        with pytest.raises(OSError):
            prepare_output_file(target)
    else:
        prepare_output_file(target)
    if mode in {"overwrite", "unwritable"}:
        assert target.read_text(encoding="utf-8") == "Original artifact"
    elif mode != "directory":
        assert not target.exists()


@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize("late_failure", [False, True])
def test_handler_sanitizes_output_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    json_mode: bool,
    late_failure: bool,
) -> None:
    target = tmp_path / "artifact"
    calls: list[str] = []

    def run_chain(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("chain")

        def fail_write(*_args: Any, **_kwargs: Any) -> Any:
            raise UnicodeError(PRIVATE_MARKER)

        monkeypatch.setattr(Path, "write_text", fail_write)
        return SimpleNamespace(
            chain=SimpleNamespace(id=CHAIN_ID, name="Synthetic"),
            chain_input="",
            final_output_text="",
            final_summary_text="",
            run_status="success",
            steps=[],
            step_aliases={},
            step_outputs={},
            final_step_id=None,
            final_step_output_key=None,
            final_step_label=None,
            terminal_step_id=None,
            terminal_step_output_key=None,
            terminal_step_label=None,
            terminal_step_status=None,
        )

    if not late_failure:
        target.mkdir()
    args = Namespace(
        chain_id=CHAIN_ID, chain_input="Synthetic input", json=json_mode, output_file=target
    )
    exit_code = run_prompt_chain_run(
        cast("Any", SimpleNamespace(run_prompt_chain=run_chain)), args, logging.getLogger(__name__)
    )
    captured = capsys.readouterr()
    result = subprocess.CompletedProcess([], exit_code, captured.out, captured.err)
    assert calls == (["chain"] if late_failure else [])
    _assert_safe_error(
        result,
        json_mode=json_mode,
        code="OUTPUT_WRITE_FAILED" if late_failure else "OUTPUT_UNAVAILABLE",
    )
