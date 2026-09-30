"""Provider-free process regressions for chain output preparation.

Updates:
  v0.1.2 - 2026-09-30 - Align chain/benchmark process exits with domain outcomes.
  v0.1.1 - 2026-09-30 - Cover JSON chain receipts and sanitized runtime errors.
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
from core import PromptChainError, PromptChainExecutionError

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
from core import PromptChainError, PromptChainExecutionError

receipt = Path(os.environ['TEST_RECEIPT'])
target = Path(os.environ['TEST_OUTPUT'])
mode = os.environ['TEST_MODE']
calls = []
original_write_text = Path.write_text
class Manager:
    def benchmark_prompts(self, *args, **kwargs):
        calls.append('benchmark')
        errors = {
            'benchmark_success': [None, None],
            'benchmark_mixed': [None, 'Synthetic failure'],
            'benchmark_failed': ['Synthetic failure', 'Second failure'],
            'benchmark_empty_error': [''],
            'benchmark_empty': [],
        }[mode]
        return NS(runs=[NS(prompt_name=f'Synthetic prompt {i}', model='offline-stub',
            error=error, usage={}, duration_ms=1, response_preview='Synthetic result',
            history=None) for i, error in enumerate(errors)])
    def run_prompt_chain(self, *args, **kwargs):
        calls.append('chain')
        if mode == 'execution_error':
            raise PromptChainExecutionError('PRIVATE_SYNTHETIC_OUTPUT_123')
        if mode == 'chain_error':
            raise PromptChainError('PRIVATE_SYNTHETIC_OUTPUT_123')
        if mode == 'unexpected_error':
            raise RuntimeError('PRIVATE_SYNTHETIC_OUTPUT_123')
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
            run_status={'empty_status': '', 'unknown_status': 'future_status'}.get(mode,
                mode if mode in {'failed', 'partial_success', 'skipped'} else 'success'),
            step_aliases={'final': 'step_1'},
            step_outputs={'step_1': 'Synthetic result'}, steps=[])
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
    tmp_path: Path,
    *,
    installed: bool,
    json_mode: bool,
    mode: str,
    target: Path | None,
    command: str = "prompt-chain-run",
    extra_args: tuple[str, ...] = (),
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
    args = (
        ["benchmark", "--prompt", CHAIN_ID, "--request", "Synthetic input"]
        if command == "benchmark"
        else ["prompt-chain-run", CHAIN_ID, "--input", "Synthetic input", "--no-web-search"]
    )
    args.extend(extra_args)
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
    if json_mode:
        assert json.loads(result.stdout) == {
            "command": "prompt-chain-run",
            "artifact_path": str(target),
            "run_status": "success",
        }
    else:
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


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("existing", [None, False, True])
@pytest.mark.parametrize("mode", ["execution_error", "chain_error", "unexpected_error"])
def test_json_runner_error_preserves_artifact(
    tmp_path: Path, installed: bool, existing: bool | None, mode: str
) -> None:
    target = tmp_path / "artifact.json" if existing is not None else None
    if existing and target is not None:
        target.write_text("Original artifact", encoding="utf-8")
    result, calls = _run(tmp_path, installed=installed, json_mode=True, mode=mode, target=target)
    assert calls == ["chain"]
    assert result.returncode == 5
    assert result.stdout == ""
    code = "CHAIN_EXECUTION_FAILED" if mode == "execution_error" else "CHAIN_RUN_FAILED"
    assert json.loads(result.stderr) == {
        "ok": False,
        "command": "prompt-chain-run",
        "error": {"code": code, "message": "Unable to execute prompt chain."},
    }
    assert PRIVATE_MARKER not in result.stderr
    assert "Synthetic input" not in result.stderr
    assert "Synthetic result" not in result.stderr
    if existing and target is not None:
        assert target.read_text(encoding="utf-8") == "Original artifact"
    elif target is not None:
        assert not target.exists()
    else:
        assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("file_mode", [False, True])
@pytest.mark.parametrize(
    "status", ["success", "partial_success", "failed", "skipped", "empty_status", "unknown_status"]
)
def test_json_receipt_preserves_outcome_before_domain_exit(
    tmp_path: Path, installed: bool, file_mode: bool, status: str
) -> None:
    target = tmp_path / "artifact.json" if file_mode else None
    result, calls = _run(tmp_path, installed=installed, json_mode=True, mode=status, target=target)
    assert calls == ["chain"]
    assert result.returncode == (0 if status == "success" else 5)
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    if target is not None:
        assert payload == {
            "command": "prompt-chain-run",
            "artifact_path": str(target),
            "run_status": {"empty_status": "unknown", "unknown_status": "future_status"}.get(
                status, status
            ),
        }
        payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["run_status"] == {"empty_status": "", "unknown_status": "future_status"}.get(
        status, status
    )
    assert payload["final_output_text"] == "Synthetic result"
    assert payload["chain_input"] == "Synthetic input"


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("file_mode", [False, True])
@pytest.mark.parametrize("status", ["success", "partial_success", "failed"])
@pytest.mark.parametrize(
    "selector",
    [
        (),
        ("--compact",),
        ("--final-output-only",),
        ("--summary-only",),
        ("--status-only",),
        ("--step-output", "step_1"),
        ("--step-alias", "final"),
        ("--final-step-meta",),
    ],
)
def test_text_selectors_preserve_output_before_domain_exit(
    tmp_path: Path, installed: bool, file_mode: bool, status: str, selector: tuple[str, ...]
) -> None:
    target = tmp_path / "artifact.txt" if file_mode else None
    result, calls = _run(
        tmp_path,
        installed=installed,
        json_mode=False,
        mode=status,
        target=target,
        extra_args=selector,
    )
    assert calls == ["chain"]
    assert result.returncode == (0 if status == "success" else 5)
    assert result.stderr == ""
    text = result.stdout
    if target is not None:
        assert text == f"Saved prompt chain run artifact to {target}.\n"
        text = target.read_text(encoding="utf-8")
    if selector == ("--status-only",):
        assert text == status + "\n"
    elif selector == ("--final-step-meta",):
        assert json.loads(text)["run_status"] == status
    elif selector == ("--summary-only",):
        assert text == "\n"
    else:
        assert "Synthetic result" in text


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("file_mode", [False, True])
@pytest.mark.parametrize("status", ["empty_status", "unknown_status"])
@pytest.mark.parametrize("selector", ["--status-only", "--compact"])
def test_text_status_views_do_not_present_unknown_as_success(
    tmp_path: Path, installed: bool, file_mode: bool, status: str, selector: str
) -> None:
    target = tmp_path / "artifact.txt" if file_mode else None
    result, calls = _run(
        tmp_path,
        installed=installed,
        json_mode=False,
        mode=status,
        target=target,
        extra_args=(selector,),
    )
    assert calls == ["chain"]
    assert result.returncode == 5
    assert result.stderr == ""
    text = result.stdout
    if target is not None:
        assert text == f"Saved prompt chain run artifact to {target}.\n"
        text = target.read_text(encoding="utf-8")
    expected_status = "unknown" if status == "empty_status" else "future_status"
    if selector == "--status-only":
        assert text == expected_status + "\n"
    else:
        assert f"Status: {expected_status}" in text
        assert "Synthetic result" in text


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize(
    "mode",
    [
        "benchmark_success",
        "benchmark_mixed",
        "benchmark_failed",
        "benchmark_empty_error",
        "benchmark_empty",
    ],
)
def test_benchmark_process_exit_preserves_report(
    tmp_path: Path, installed: bool, mode: str
) -> None:
    result, calls = _run(
        tmp_path,
        installed=installed,
        json_mode=False,
        mode=mode,
        target=None,
        command="benchmark",
    )
    assert calls == ["benchmark"]
    assert result.returncode == (0 if mode == "benchmark_success" else 5)
    assert "Traceback" not in result.stderr
    if mode == "benchmark_empty":
        assert result.stdout == ""
        assert "No benchmark runs were executed." in result.stderr
    else:
        assert result.stderr == ""
        assert result.stdout.startswith("\nBenchmark results\n-----------------\n")
        if mode in {"benchmark_success", "benchmark_mixed"}:
            assert "-> OK:" in result.stdout
            assert "preview: Synthetic result" in result.stdout
        if mode != "benchmark_success":
            assert "-> ERROR:" in result.stdout
        if mode in {"benchmark_failed", "benchmark_empty_error"}:
            assert "-> OK:" not in result.stdout


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("mode", ["execution_error", "chain_error"])
def test_text_runner_error_preserves_legacy_diagnostic(
    tmp_path: Path, installed: bool, mode: str
) -> None:
    result, calls = _run(tmp_path, installed=installed, json_mode=False, mode=mode, target=None)
    assert calls == ["chain"]
    assert result.returncode == 5
    assert result.stdout == ""
    assert PRIVATE_MARKER in result.stderr


@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize("error_type", [PromptChainExecutionError, PromptChainError, RuntimeError])
def test_handler_runner_exception_boundary(
    capsys: pytest.CaptureFixture[str], json_mode: bool, error_type: type[Exception]
) -> None:
    calls: list[str] = []

    def fail_run(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("chain")
        raise error_type(PRIVATE_MARKER)

    manager = cast("Any", SimpleNamespace(run_prompt_chain=fail_run))
    args = Namespace(chain_id=CHAIN_ID, chain_input="Synthetic input", json=json_mode)
    if not json_mode and error_type is RuntimeError:
        with pytest.raises(RuntimeError, match=PRIVATE_MARKER):
            run_prompt_chain_run(manager, args, logging.getLogger(__name__))
    else:
        assert run_prompt_chain_run(manager, args, logging.getLogger(__name__)) == 5
    captured = capsys.readouterr()
    assert calls == ["chain"]
    assert captured.out == ""
    if json_mode:
        code = (
            "CHAIN_EXECUTION_FAILED"
            if error_type is PromptChainExecutionError
            else "CHAIN_RUN_FAILED"
        )
        assert json.loads(captured.err) == {
            "ok": False,
            "command": "prompt-chain-run",
            "error": {"code": code, "message": "Unable to execute prompt chain."},
        }


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("relative", [False, True])
def test_json_receipt_preserves_unicode_artifact_path(
    tmp_path: Path, installed: bool, relative: bool
) -> None:
    target = Path("wyniki ze spacją") / "Łańcuch.json"
    if not relative:
        target = tmp_path / target
    result, calls = _run(
        tmp_path, installed=installed, json_mode=True, mode="success", target=target
    )
    assert calls == ["chain"]
    assert result.returncode == 0
    assert result.stderr == ""
    assert json.loads(result.stdout) == {
        "command": "prompt-chain-run",
        "artifact_path": str(target),
        "run_status": "success",
    }
    artifact = tmp_path / target if relative else target
    assert (
        json.loads(artifact.read_text(encoding="utf-8"))["final_output_text"] == "Synthetic result"
    )
