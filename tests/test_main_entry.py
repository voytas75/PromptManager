"""Lightweight integration checks for main module."""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import main


class _DummySettings(SimpleNamespace):
    def __init__(self) -> None:
        super().__init__(
            db_path="/tmp/db",
            chroma_path="/tmp/chroma",
            redis_dsn="redis://localhost:6379/0",
            cache_ttl_seconds=120,
        )


def test_main_print_settings_logs_and_exits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager", "--print-settings"])
    monkeypatch.setattr(main, "load_settings", lambda: _DummySettings())
    monkeypatch.setattr(main, "build_prompt_manager", lambda settings: object())

    exit_code = main.main()

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Resolved settings" in captured.out


def test_main_returns_error_when_settings_fail(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager"])

    def _raise() -> None:
        raise ValueError("cannot load")

    monkeypatch.setattr(main, "load_settings", _raise)

    exit_code = main.main()

    assert exit_code == 2
    assert "Failed to load settings" in capsys.readouterr().out


def test_main_returns_error_when_manager_init_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager"])
    monkeypatch.setattr(main, "load_settings", lambda: _DummySettings())

    def _boom(_: _DummySettings) -> None:
        raise RuntimeError("init failed")

    monkeypatch.setattr(main, "build_prompt_manager", _boom)

    exit_code = main.main()

    assert exit_code == 3
    assert "Failed to initialise services" in capsys.readouterr().out


def test_main_logs_ready_message_on_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager"])
    monkeypatch.setattr(main, "load_settings", lambda: _DummySettings())
    monkeypatch.setattr(main, "build_prompt_manager", lambda settings: object())

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Prompt Manager ready" in output
    assert "ChromaDB at" in output


def test_setup_logging_basic_config_fallback(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    missing_path = tmp_path / "absent.ini"
    with caplog.at_level(logging.INFO):
        main._setup_logging(missing_path)
        logging.getLogger("prompt_manager.testing").info("hello from fallback")

    assert "hello from fallback" in caplog.text


def test_main_entrypoint_guard_executes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager"])

    config_stub = types.ModuleType("config")
    config_stub.load_settings = lambda: _DummySettings()
    core_stub = types.ModuleType("core")
    core_stub.build_prompt_manager = lambda settings: object()

    monkeypatch.setitem(sys.modules, "config", config_stub)
    monkeypatch.setitem(sys.modules, "core", core_stub)

    main_path = Path(main.__file__)
    code = compile(main_path.read_text(encoding="utf-8"), str(main_path), "exec")

    with pytest.raises(SystemExit) as excinfo:
        exec(code, {"__name__": "__main__"})

    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "ChromaDB at" in output
