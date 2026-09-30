"""Provider-free process checks for JSON-only offline announcement suppression."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import main
from cli.parser import parse_args
from config.settings import PromptManagerSettings

ROOT = Path(__file__).resolve().parents[1]
PROMPT_ID = "00000000-0000-0000-0000-000000000994"
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
import core.factory as factory
from models.prompt_model import Prompt
from core.prompt_manager import PromptManager as RealManager

calls = []
status = None
prompt = Prompt(id='00000000-0000-0000-0000-000000000994', name='Synthetic',
    description='Synthetic fixture', category='Analysis', tags=['offline'],
    context='Hello {{ subject }}')
class Repository:
    def get(self, *args, **kwargs):
        calls.append('get')
        return prompt
    def list(self, *args, **kwargs):
        calls.append('list')
        return [prompt]
class Manager:
    _initialise_llm_status = RealManager._initialise_llm_status
    set_llm_status = RealManager.set_llm_status
    def __init__(self, **kwargs):
        self.repository = Repository()
        self._notification_center = NS(publish=lambda *args: calls.append('notification'))
    def set_redis_status(self, *args, **kwargs):
        pass
    def close(self):
        Path(os.environ['TEST_RECEIPT']).write_text(json.dumps({
            'calls': calls, 'available': self._llm_available,
            'reason': self._llm_unavailable_reason}), encoding='utf-8')
factory.PromptManager = Manager
original_build = factory.build_prompt_manager
def build(settings, **kwargs):
    if os.environ['TEST_MODE'] == 'init_error':
        raise RuntimeError('Synthetic service failure')
    if os.environ['TEST_MODE'] == 'other_warning':
        factory.factory_logger.warning('OTHER_WARNING_VISIBLE')
    calls.append('build')
    return original_build(settings, repository=Repository(), enable_background_sync=False,
        **kwargs)
core.build_prompt_manager = build
"""


def _run(
    tmp_path: Path,
    *,
    installed: bool,
    example_logging: bool,
    command: str,
    json_mode: bool,
    mode: str = "success",
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any] | None]:
    config = tmp_path / "settings.json"
    config.write_text(
        json.dumps(
            {
                "embedding_backend": "deterministic",
                "redis_dsn": None,
                "litellm_model": None,
                "litellm_inference_model": None,
                "db_path": str(tmp_path / "unused.db"),
                "chroma_path": str(tmp_path / "unused-chroma"),
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
        TEST_RECEIPT=str(receipt),
        TEST_MODE=mode,
    )
    args: list[str] = []
    if example_logging:
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "logging.conf.example").write_bytes(
            (ROOT / "config/logging.conf.example").read_bytes()
        )
    args.append(command)
    if command == "prompt-render":
        variables = {} if mode == "render_error" else {"subject": "world"}
        args.extend((PROMPT_ID, "--variables-json", json.dumps(variables)))
        if mode == "validate_only":
            args.append("--validate-only")
    if json_mode:
        args.append("--json")
    dispatch = (
        "runpy.run_path(sys.argv.pop(1), run_name='__main__')"
        if installed
        else "sys.argv.pop(1); runpy.run_module('main', run_name='__main__')"
    )
    entrypoint = str(ROOT / ".venv/bin/prompt-manager") if installed else "main"
    result = subprocess.run(
        [sys.executable, "-c", BOOTSTRAP + "\n" + dispatch, entrypoint, *args],
        cwd=tmp_path,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "NETWORK_FORBIDDEN" not in result.stdout + result.stderr
    assert not (tmp_path / "unused.db").exists()
    assert not (tmp_path / "unused-chroma").exists()
    state = json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else None
    return result, state


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("example_logging", [False, True])
@pytest.mark.parametrize("command", ["prompt-render", "tag-list"])
@pytest.mark.parametrize("json_mode", [False, True])
def test_offline_success_keeps_json_clean_and_text_announcements(
    tmp_path: Path, installed: bool, example_logging: bool, command: str, json_mode: bool
) -> None:
    result, state = _run(
        tmp_path,
        installed=installed,
        example_logging=example_logging,
        command=command,
        json_mode=json_mode,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert state is not None
    assert state["available"] is False
    assert "LLM-backed features are offline" in state["reason"]
    assert state["calls"] == (
        ["build", "get" if command == "prompt-render" else "list"]
        if json_mode
        else ["build", "notification", "get" if command == "prompt-render" else "list"]
    )
    if json_mode:
        assert result.stderr == ""
        payload = json.loads(result.stdout)
        if command == "prompt-render":
            assert payload["ok"] is True
            assert payload["rendered_text"] == "Hello world"
        else:
            assert payload == {
                "tags": [{"tag": "offline", "prompt_count": 1, "active_prompt_count": 1}]
            }
    else:
        logs = result.stdout if example_logging else result.stderr
        assert "LiteLLM configuration incomplete" in logs
        assert "LLM features disabled" in logs
        assert (
            "Hello world" in result.stdout
            if command == "prompt-render"
            else "Tags: 1" in result.stdout
        )


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("example_logging", [False, True])
@pytest.mark.parametrize("command", ["prompt-render", "tag-list"])
@pytest.mark.parametrize("mode", ["other_warning", "init_error"])
def test_json_does_not_suppress_other_warnings_or_service_failures(
    tmp_path: Path, installed: bool, example_logging: bool, command: str, mode: str
) -> None:
    result, state = _run(
        tmp_path,
        installed=installed,
        example_logging=example_logging,
        command=command,
        json_mode=True,
        mode=mode,
    )
    logs = result.stdout if example_logging else result.stderr
    if mode == "init_error":
        assert result.returncode == 3
        assert "Failed to initialise services: Synthetic service failure" in logs
        assert state is None
    else:
        assert result.returncode == 0
        assert "OTHER_WARNING_VISIBLE" in logs
        assert "LiteLLM configuration incomplete" not in result.stdout + result.stderr
        assert state is not None


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("example_logging", [False, True])
def test_validate_only_json_uses_same_quiet_offline_boundary(
    tmp_path: Path, installed: bool, example_logging: bool
) -> None:
    result, state = _run(
        tmp_path,
        installed=installed,
        example_logging=example_logging,
        command="prompt-render",
        json_mode=True,
        mode="validate_only",
    )
    assert result.returncode == 0
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["rendered_text"] is None
    assert state is not None
    assert state["calls"] == ["build", "get"]


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("example_logging", [False, True])
def test_render_failure_is_still_observable_in_json(
    tmp_path: Path, installed: bool, example_logging: bool
) -> None:
    result, state = _run(
        tmp_path,
        installed=installed,
        example_logging=example_logging,
        command="prompt-render",
        json_mode=True,
        mode="render_error",
    )
    assert result.returncode == 5
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["missing_variables"] == ["subject"]
    assert payload["errors"]
    assert state is not None


@pytest.mark.parametrize("command", ["prompt-render", "tag-list", "prompt-validate"])
@pytest.mark.parametrize("json_mode", [False, True])
def test_main_scopes_announcement_option_to_selected_json_commands(
    monkeypatch: pytest.MonkeyPatch, command: str, json_mode: bool
) -> None:
    argv = ["prompt-manager", command]
    if command != "tag-list":
        argv.extend([PROMPT_ID])
    if json_mode:
        argv.append("--json")
    monkeypatch.setattr(sys, "argv", argv)
    args = parse_args()
    options: dict[str, Any] = {}
    closed: list[bool] = []
    manager = SimpleNamespace(close=lambda: closed.append(True))

    def build(*_args: Any, **kwargs: Any) -> Any:
        options.update(kwargs)
        return manager

    def setup(*_args: Any) -> None:
        pass

    def handler(*_args: Any) -> int:
        return 0

    # This bootstrap-only control must never consult environment, dotenv or JSON sources.
    settings = PromptManagerSettings.model_construct(litellm_logging_enabled=False)
    monkeypatch.setattr(main, "load_settings", lambda: settings)
    monkeypatch.setattr(main, "_runtime_setup_logging", setup)
    monkeypatch.setattr(main, "build_prompt_manager", build)
    spec = main.COMMAND_SPECS[command]
    replacement = SimpleNamespace(
        requires_manager=spec.requires_manager,
        announce_offline_llm=spec.announce_offline_llm,
        handler=handler,
    )
    monkeypatch.setitem(main.COMMAND_SPECS, command, cast("Any", replacement))
    assert main.main(args) == 0
    assert options["announce_offline_llm"] is not (json_mode and command != "prompt-validate")
    assert closed == [True]
