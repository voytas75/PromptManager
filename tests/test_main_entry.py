"""Lightweight integration checks for main module.

Updates: v0.3.0 - 2025-11-15 - Cover enhanced --print-settings summary and masked API keys.
Updates: v0.2.0 - 2025-11-05 - Add coverage for GUI dependency fallback.
"""

from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import main
from core.intent_classifier import IntentLabel, IntentPrediction


class _DummySettings(SimpleNamespace):
    def __init__(self) -> None:
        super().__init__(
            db_path="/tmp/db",
            chroma_path="/tmp/chroma",
            redis_dsn="redis://localhost:6379/0",
            cache_ttl_seconds=120,
            litellm_model=None,
            litellm_api_key=None,
            litellm_api_base=None,
            litellm_api_version=None,
            litellm_drop_params=None,
            litellm_reasoning_effort=None,
            litellm_stream=False,
            embedding_backend="deterministic",
            embedding_model=None,
        )


class _DummyManager:
    def __init__(self) -> None:
        self.closed = False
        self.suggestion_response: object | None = None

    def close(self) -> None:
        self.closed = True

    def generate_prompt_name(self, context: str) -> str:
        return f"Suggested {context[:10]}".strip()

    def suggest_prompts(self, query: str, limit: int = 5):
        if self.suggestion_response is None:
            raise AssertionError("suggest_prompts called unexpectedly")
        return self.suggestion_response


class _CatalogResult:
    def __init__(self) -> None:
        self.added = 0
        self.updated = 0
        self.skipped = 0
        self.errors = 0


def test_main_print_settings_logs_and_exits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager", "--print-settings"])
    monkeypatch.setattr(main, "load_settings", lambda: _DummySettings())
    manager = _DummyManager()
    monkeypatch.setattr(main, "build_prompt_manager", lambda settings: manager)
    monkeypatch.setattr(main, "import_prompt_catalog", lambda *_: _CatalogResult())

    exit_code = main.main()

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Prompt Manager configuration summary" in captured.out
    assert "LiteLLM API key: not set" in captured.out
    assert "Streaming enabled:" in captured.out


def test_main_print_settings_masks_api_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager", "--print-settings"])
    settings = _DummySettings()
    settings.litellm_model = "azure/gpt-4o"
    settings.litellm_api_key = "sk-1234567890abcd"
    monkeypatch.setattr(main, "load_settings", lambda: settings)

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "LiteLLM API key: set (sk-1...abcd)" in output
    assert "Model: azure/gpt-4o" in output


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
    monkeypatch.setattr("sys.argv", ["prompt-manager", "--no-gui"])
    monkeypatch.setattr(main, "load_settings", lambda: _DummySettings())
    manager = _DummyManager()
    monkeypatch.setattr(main, "build_prompt_manager", lambda settings: manager)
    monkeypatch.setattr(main, "import_prompt_catalog", lambda *_: _CatalogResult())

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Prompt Manager ready" in output
    assert "ChromaDB at" in output
    assert manager.closed is True


def test_main_launches_gui_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager"])
    dummy_manager = _DummyManager()
    monkeypatch.setattr(main, "load_settings", lambda: _DummySettings())
    monkeypatch.setattr(main, "build_prompt_manager", lambda settings: dummy_manager)
    monkeypatch.setattr(main, "import_prompt_catalog", lambda *_: _CatalogResult())

    called = {}

    def _fake_launch(manager: object, settings: object | None = None) -> int:
        called["manager"] = manager
        called["settings"] = settings
        return 5

    gui_stub = types.SimpleNamespace(launch_prompt_manager=_fake_launch)
    monkeypatch.setitem(sys.modules, "gui", gui_stub)

    exit_code = main.main()

    assert exit_code == 5
    assert called["manager"] is dummy_manager
    assert called["settings"] is not None
    assert dummy_manager.closed is True


def test_main_returns_error_when_gui_dependency_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager", "--gui"])
    manager = _DummyManager()
    monkeypatch.setattr(main, "load_settings", lambda: _DummySettings())
    monkeypatch.setattr(main, "build_prompt_manager", lambda settings: manager)
    monkeypatch.setattr(main, "import_prompt_catalog", lambda *_: _CatalogResult())

    class _GuiError(RuntimeError):
        pass

    def _raise(_: object, __: object | None = None) -> int:
        raise _GuiError("PySide6 is not installed")

    gui_stub = types.SimpleNamespace(
        launch_prompt_manager=_raise,
        GuiDependencyError=_GuiError,
    )
    monkeypatch.setitem(sys.modules, "gui", gui_stub)

    exit_code = main.main()

    assert exit_code == 4
    output = capsys.readouterr().out
    assert "Unable to start GUI" in output
    assert manager.closed is True


def test_suggest_command_outputs_results(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager", "suggest", "Find failing test"])
    settings = _DummySettings()
    monkeypatch.setattr(main, "load_settings", lambda: settings)
    manager = _DummyManager()
    prediction = IntentPrediction(
        label=IntentLabel.DEBUG,
        confidence=0.75,
        rationale=None,
        category_hints=["Debugging"],
        tag_hints=["debug"],
        language_hints=["python"],
    )
    suggestion_payload = SimpleNamespace(
        prediction=prediction,
        prompts=[
            SimpleNamespace(
                name="Debug Sentinel",
                category="Code Analysis",
                quality_score=9.1,
                tags=["debug"],
                description="Guide to diagnose failures",
            )
        ],
        fallback_used=False,
    )
    manager.suggestion_response = suggestion_payload
    monkeypatch.setattr(main, "build_prompt_manager", lambda _: manager)

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Top 1 suggestions" in output
    assert "Debug Sentinel" in output
    assert manager.closed is True


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
    config_stub.PromptManagerSettings = type("PromptManagerSettings", (), {})
    core_stub = types.ModuleType("core")
    dummy_manager = _DummyManager()
    core_stub.build_prompt_manager = lambda settings: dummy_manager
    core_stub.CatalogDiff = main.CatalogDiff
    core_stub.diff_prompt_catalog = lambda *args, **kwargs: core_stub.CatalogDiff()
    core_stub.export_prompt_catalog = lambda *args, **kwargs: Path("export.json")
    core_stub.import_prompt_catalog = lambda *args, **kwargs: _CatalogResult()

    monkeypatch.setitem(sys.modules, "config", config_stub)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    gui_stub = types.SimpleNamespace(
        launch_prompt_manager=lambda manager, settings=None: 0,
        GuiDependencyError=RuntimeError,
    )
    monkeypatch.setitem(sys.modules, "gui", gui_stub)
    monkeypatch.setattr(main, "import_prompt_catalog", lambda *_: _CatalogResult())

    main_path = Path(main.__file__)
    code = compile(main_path.read_text(encoding="utf-8"), str(main_path), "exec")

    with pytest.raises(SystemExit) as excinfo:
        exec(code, {"__name__": "__main__"})

    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "ChromaDB at" in output
    assert dummy_manager.closed is True


def test_catalog_import_dry_run(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["prompt-manager", "catalog-import", "catalog.json", "--dry-run"],
    )
    manager = _DummyManager()
    monkeypatch.setattr(main, "load_settings", lambda: _DummySettings())
    monkeypatch.setattr(main, "build_prompt_manager", lambda settings: manager)

    diff = main.CatalogDiff(added=1)
    monkeypatch.setattr(main, "diff_prompt_catalog", lambda *_, **__: diff)

    applied = {}

    def _import_stub(*_: object, **__: object) -> _CatalogResult:
        applied["called"] = True
        return _CatalogResult()

    monkeypatch.setattr(main, "import_prompt_catalog", _import_stub)

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Catalogue preview" in output
    assert "Dry-run complete" in output
    assert "called" not in applied
    assert manager.closed is True


def test_usage_report_command(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    log_path = tmp_path / "intent_usage.jsonl"
    entries = [
        {
            "timestamp": "2025-11-07T12:00:00Z",
            "event": "detect",
            "label": "analysis",
        },
        {
            "timestamp": "2025-11-07T12:01:00Z",
            "event": "suggest",
            "label": "analysis",
            "top_prompts": ["Refactor Helper"],
        },
        {
            "timestamp": "2025-11-07T12:02:00Z",
            "event": "copy",
            "prompt_name": "Refactor Helper",
            "has_body": True,
        },
    ]
    log_path.write_text("\n".join(json.dumps(entry) for entry in entries), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        ["prompt-manager", "usage-report", "--path", str(log_path)],
    )
    monkeypatch.setattr(main, "load_settings", lambda: _DummySettings())
    manager = _DummyManager()
    monkeypatch.setattr(main, "build_prompt_manager", lambda settings: manager)
    monkeypatch.setattr(main, "import_prompt_catalog", lambda *_: _CatalogResult())

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Total events: 3" in output
    assert "Top recommended prompts" in output
    assert manager.closed is True


def test_catalog_export_command(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["prompt-manager", "catalog-export", "out.json"],
    )
    manager = _DummyManager()
    monkeypatch.setattr(main, "load_settings", lambda: _DummySettings())
    monkeypatch.setattr(main, "build_prompt_manager", lambda settings: manager)

    exported = {}

    def _export_stub(*args: object, **kwargs: object) -> Path:
        exported["args"] = (args, kwargs)
        return Path("out.json")

    monkeypatch.setattr(main, "export_prompt_catalog", _export_stub)

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out.lower()
    assert "exported" in output
    assert manager.closed is True
    assert exported
