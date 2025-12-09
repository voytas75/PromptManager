"""Lightweight integration checks for main module.

Updates:
  v0.5.4 - 2025-12-09 - Offer to create config/config.json during first-run tests.
  v0.5.3 - 2025-12-08 - Include token usage aggregates in execution analytics helper.
  v0.5.2 - 2025-12-08 - Route monkeypatches through helper to satisfy Pyright.
  v0.5.1 - 2025-11-29 - Extend entrypoint guard stub for analytics helpers.
  v0.5.0 - 2025-11-28 - Cover analytics diagnostics CLI path and export flags.
  v0.4.0 - 2025-11-30 - Remove catalogue import command coverage.
  v0.3.0 - 2025-11-15 - Cover enhanced --print-settings summary and masked API keys.
  v0.2.0 - 2025-11-05 - Add coverage for GUI dependency fallback.
"""

from __future__ import annotations

import json
import logging
import sys
import types
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import main
from config import SettingsError
from config.settings import DEFAULT_EMBEDDING_BACKEND, DEFAULT_EMBEDDING_MODEL
from core.history_tracker import (
    ExecutionAnalytics,
    PromptExecutionAnalytics,
    TokenUsageTotals,
)
from core.intent_classifier import IntentLabel, IntentPrediction
from core.prompt_manager import PromptManagerError


def _patch_main(monkeypatch: pytest.MonkeyPatch, name: str, value: object) -> None:
    monkeypatch.setattr(cast(Any, main), name, value)


def _mock_module(name: str) -> Any:
    return cast(Any, types.ModuleType(name))


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
            litellm_tts_model=None,
            litellm_tts_stream=True,
            litellm_stream=False,
            embedding_backend=DEFAULT_EMBEDDING_BACKEND,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
        )


class _DummyManager:
    def __init__(self) -> None:
        self.closed = False
        self.suggestion_response: object | None = None
        self.reembed_result: tuple[int, int] = (0, 0)
        self.reembed_error: Exception | None = None
        self.reembed_called = False
        self.reembed_reset = False
        self.execution_analytics: ExecutionAnalytics | None = None
        self.embedding_diagnostics = SimpleNamespace(
            backend_ok=True,
            backend_message="Backend reachable",
            backend_dimension=32,
            inferred_dimension=None,
            chroma_ok=True,
            chroma_message="Chroma reachable",
            chroma_count=1,
            repository_total=1,
            prompts_with_embeddings=1,
            missing_prompts=[],
            mismatched_prompts=[],
            consistent_counts=True,
        )
        self.diagnostics_sample_text: str | None = None
        self.token_usage_totals_window = TokenUsageTotals(25, 50, 75)
        self.token_usage_totals_all = TokenUsageTotals(125, 250, 375)

    def close(self) -> None:
        self.closed = True

    def generate_prompt_name(self, context: str) -> str:
        return f"Suggested {context[:10]}".strip()

    def suggest_prompts(self, query: str, limit: int = 5):
        if self.suggestion_response is None:
            raise AssertionError("suggest_prompts called unexpectedly")
        return self.suggestion_response

    def rebuild_embeddings(self, *, reset_store: bool = False) -> tuple[int, int]:
        self.reembed_called = True
        self.reembed_reset = reset_store
        if self.reembed_error is not None:
            raise self.reembed_error
        return self.reembed_result

    def get_execution_analytics(
        self,
        *,
        window_days: int | None = None,
        prompt_limit: int = 5,
        trend_window: int = 5,
    ) -> ExecutionAnalytics | None:
        return self.execution_analytics

    def diagnose_embeddings(self, *, sample_text: str = "Prompt Manager diagnostics probe"):
        self.diagnostics_sample_text = sample_text
        return self.embedding_diagnostics

    def get_token_usage_totals(self, *, since: datetime | None = None) -> TokenUsageTotals:
        if since is None:
            return self.token_usage_totals_all
        return self.token_usage_totals_window


def _build_execution_analytics(total_runs: int = 5) -> ExecutionAnalytics:
    now = datetime.now(UTC)
    prompt_stats = PromptExecutionAnalytics(
        prompt_id=uuid.uuid4(),
        name="Prompt Alpha",
        total_runs=total_runs,
        success_rate=1.0,
        average_duration_ms=150.0,
        average_rating=4.8,
        rating_trend=0.4,
        last_executed_at=now,
        prompt_tokens=25,
        completion_tokens=50,
        total_tokens=75,
    )
    return ExecutionAnalytics(
        total_runs=total_runs,
        success_rate=0.9,
        average_duration_ms=200.0,
        average_rating=4.5,
        prompt_breakdown=[prompt_stats],
        window_start=now,
        prompt_tokens=25,
        completion_tokens=50,
        total_tokens=75,
    )


def _build_dummy_snapshot() -> SimpleNamespace:
    now = datetime.now(UTC)
    return SimpleNamespace(
        execution=_build_execution_analytics(),
        usage_frequency=[
            SimpleNamespace(
                name="Prompt Delta",
                usage_count=4,
                success_rate=0.75,
                last_executed_at=now,
            )
        ],
        model_costs=[
            SimpleNamespace(
                model="gpt-fast",
                run_count=2,
                prompt_tokens=10,
                completion_tokens=8,
                total_tokens=18,
            )
        ],
        benchmark_stats=[
            SimpleNamespace(
                model="gpt-bench",
                run_count=1,
                success_rate=1.0,
                average_duration_ms=110.0,
                total_tokens=25,
            )
        ],
        intent_success=[
            SimpleNamespace(
                bucket=now,
                success_rate=1.0,
                success=2,
                total=2,
            )
        ],
        embedding=SimpleNamespace(
            backend_ok=True,
            backend_message="ok",
            backend_dimension=32,
            inferred_dimension=None,
            chroma_ok=True,
            chroma_message="ok",
            chroma_count=2,
            repository_total=2,
            prompts_with_embeddings=2,
            missing_prompts=[],
            mismatched_prompts=[],
            consistent_counts=True,
        ),
    )


def test_main_print_settings_logs_and_exits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager", "--print-settings"])
    _patch_main(monkeypatch, "load_settings", lambda: _DummySettings())
    manager = _DummyManager()
    _patch_main(monkeypatch, "build_prompt_manager", lambda settings: manager)

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
    _patch_main(monkeypatch, "load_settings", lambda: settings)

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "LiteLLM API key: set (sk-1...abcd)" in output
    assert "Fast model: azure/gpt-4o" in output


def test_main_returns_error_when_settings_fail(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager"])

    def _raise() -> None:
        raise ValueError("cannot load")

    _patch_main(monkeypatch, "load_settings", _raise)

    exit_code = main.main()

    assert exit_code == 2
    assert "Failed to load settings" in capsys.readouterr().out


def test_main_offers_config_creation_on_missing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager", "--print-settings"])
    monkeypatch.chdir(tmp_path)
    template_path = tmp_path / "config" / "config.template.json"
    template_path.parent.mkdir(parents=True)
    payload = {"litellm_model": "template-model"}
    template_path.write_text(json.dumps(payload), encoding="utf-8")

    config_path = template_path.parent / "config.json"
    attempts = {"count": 0}

    def _load() -> _DummySettings:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise SettingsError(f"Configuration file not found: {config_path}")
        return _DummySettings()

    _patch_main(monkeypatch, "load_settings", _load)
    monkeypatch.setattr(main.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda _: "y")

    exit_code = main.main()

    assert exit_code == 0
    assert attempts["count"] == 2
    assert config_path.exists()
    assert json.loads(config_path.read_text(encoding="utf-8")) == payload
    output = capsys.readouterr().out
    assert "Created configuration at" in output


def test_main_returns_error_when_manager_init_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager"])
    _patch_main(monkeypatch, "load_settings", lambda: _DummySettings())

    def _boom(_: _DummySettings) -> None:
        raise RuntimeError("init failed")

    _patch_main(monkeypatch, "build_prompt_manager", _boom)

    exit_code = main.main()

    assert exit_code == 3
    assert "Failed to initialise services" in capsys.readouterr().out


def test_main_logs_ready_message_on_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager", "--no-gui"])
    _patch_main(monkeypatch, "load_settings", lambda: _DummySettings())
    manager = _DummyManager()
    _patch_main(monkeypatch, "build_prompt_manager", lambda settings: manager)

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Prompt Manager ready" in output
    assert "ChromaDB at" in output
    assert manager.closed is True


def test_main_launches_gui_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager"])
    dummy_manager = _DummyManager()
    _patch_main(monkeypatch, "load_settings", lambda: _DummySettings())
    _patch_main(monkeypatch, "build_prompt_manager", lambda settings: dummy_manager)

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
    _patch_main(monkeypatch, "load_settings", lambda: _DummySettings())
    _patch_main(monkeypatch, "build_prompt_manager", lambda settings: manager)

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


def test_main_runs_embedding_diagnostics(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager", "diagnostics", "embeddings"])
    _patch_main(monkeypatch, "load_settings", lambda: _DummySettings())
    manager = _DummyManager()
    _patch_main(monkeypatch, "build_prompt_manager", lambda settings: manager)

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Embedding diagnostics" in output
    assert manager.diagnostics_sample_text == "Prompt Manager diagnostics probe"
    assert manager.closed is True


def test_main_embedding_diagnostics_returns_failure_on_issue(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager", "diagnostics", "embeddings"])
    _patch_main(monkeypatch, "load_settings", lambda: _DummySettings())
    manager = _DummyManager()
    mismatch = SimpleNamespace(prompt_name="Mismatch", prompt_id=uuid.uuid4(), stored_dimension=8)
    missing = SimpleNamespace(prompt_name="Missing", prompt_id=uuid.uuid4())
    manager.embedding_diagnostics = SimpleNamespace(
        backend_ok=False,
        backend_message="Backend unreachable",
        backend_dimension=None,
        inferred_dimension=32,
        chroma_ok=True,
        chroma_message="ok",
        chroma_count=2,
        repository_total=2,
        prompts_with_embeddings=1,
        missing_prompts=[missing],
        mismatched_prompts=[mismatch],
        consistent_counts=False,
    )
    _patch_main(monkeypatch, "build_prompt_manager", lambda settings: manager)

    exit_code = main.main()

    assert exit_code == 6
    output = capsys.readouterr().out
    assert "Backend: ERROR" in output
    assert "Dimension mismatches" in output
    assert manager.closed is True


def test_main_runs_analytics_diagnostics(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prompt-manager",
            "diagnostics",
            "analytics",
            "--window-days",
            "14",
            "--prompt-limit",
            "7",
            "--dataset",
            "usage",
        ],
    )
    _patch_main(monkeypatch, "load_settings", lambda: _DummySettings())
    manager = _DummyManager()
    captured_args: dict[str, object] = {}

    def _snapshot_stub(
        _manager: _DummyManager,
        *,
        window_days: int,
        prompt_limit: int,
        usage_log_path: Path | None,
    ) -> SimpleNamespace:
        captured_args["window_days"] = window_days
        captured_args["prompt_limit"] = prompt_limit
        captured_args["usage_log_path"] = usage_log_path
        return _build_dummy_snapshot()

    _patch_main(monkeypatch, "build_prompt_manager", lambda settings: manager)
    _patch_main(monkeypatch, "build_analytics_snapshot", _snapshot_stub)
    _patch_main(monkeypatch, "snapshot_dataset_rows", lambda *_: [])

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Analytics dashboard" in output
    assert captured_args["window_days"] == 14
    assert captured_args["prompt_limit"] == 7
    assert manager.closed is True


def test_main_analytics_diagnostics_exports_csv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys,
) -> None:
    export_path = tmp_path / "analytics.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prompt-manager",
            "diagnostics",
            "analytics",
            "--export-csv",
            str(export_path),
            "--dataset",
            "usage",
        ],
    )
    _patch_main(monkeypatch, "load_settings", lambda: _DummySettings())
    manager = _DummyManager()
    snapshot = _build_dummy_snapshot()
    _patch_main(monkeypatch, "build_prompt_manager", lambda settings: manager)
    _patch_main(monkeypatch, "build_analytics_snapshot", lambda *_, **__: snapshot)

    def _dataset_rows(_, dataset: str) -> list[dict[str, object]]:
        assert dataset == "usage"
        return [{"prompt_name": "Prompt Delta", "usage_count": 4}]

    _patch_main(monkeypatch, "snapshot_dataset_rows", _dataset_rows)

    exit_code = main.main()

    assert exit_code == 0
    assert export_path.exists()
    contents = export_path.read_text(encoding="utf-8")
    assert "prompt_name" in contents
    assert "Prompt Delta" in contents
    assert manager.closed is True


def test_suggest_command_outputs_results(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager", "suggest", "Find failing test"])
    settings = _DummySettings()
    _patch_main(monkeypatch, "load_settings", lambda: settings)
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
    _patch_main(monkeypatch, "build_prompt_manager", lambda _: manager)

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

    config_stub = _mock_module("config")
    config_stub.load_settings = lambda: _DummySettings()
    config_stub.PromptManagerSettings = type("PromptManagerSettings", (), {})
    config_stub.LITELLM_ROUTED_WORKFLOWS = {"prompt_execution": "Prompt execution"}
    config_stub.DEFAULT_EMBEDDING_BACKEND = DEFAULT_EMBEDDING_BACKEND
    config_stub.DEFAULT_EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL
    core_stub = _mock_module("core")
    dummy_manager = _DummyManager()
    core_stub.build_prompt_manager = lambda settings: dummy_manager
    core_stub.export_prompt_catalog = lambda *args, **kwargs: Path("export.json")
    core_stub.PromptManagerError = RuntimeError
    core_stub.build_analytics_snapshot = lambda *args, **kwargs: SimpleNamespace()
    core_stub.snapshot_dataset_rows = lambda *args, **kwargs: []

    monkeypatch.setitem(sys.modules, "config", config_stub)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    gui_stub = types.SimpleNamespace(
        launch_prompt_manager=lambda manager, settings=None: 0,
        GuiDependencyError=RuntimeError,
    )
    monkeypatch.setitem(sys.modules, "gui", gui_stub)

    main_path = Path(main.__file__)
    code = compile(main_path.read_text(encoding="utf-8"), str(main_path), "exec")

    with pytest.raises(SystemExit) as excinfo:
        exec(code, {"__name__": "__main__"})

    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "ChromaDB at" in output
    assert dummy_manager.closed is True


def test_usage_report_command(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
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
    _patch_main(monkeypatch, "load_settings", lambda: _DummySettings())
    manager = _DummyManager()
    _patch_main(monkeypatch, "build_prompt_manager", lambda settings: manager)

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Total events: 3" in output
    assert "Top recommended prompts" in output
    assert manager.closed is True


def test_catalog_export_command(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["prompt-manager", "catalog-export", "out.json"],
    )
    manager = _DummyManager()
    _patch_main(monkeypatch, "load_settings", lambda: _DummySettings())
    _patch_main(monkeypatch, "build_prompt_manager", lambda settings: manager)

    exported = {}

    def _export_stub(*args: object, **kwargs: object) -> Path:
        exported["args"] = (args, kwargs)
        return Path("out.json")

    _patch_main(monkeypatch, "export_prompt_catalog", _export_stub)

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out.lower()
    assert "exported" in output
    assert manager.closed is True
    assert exported


def test_reembed_command_succeeds(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager", "reembed"])
    manager = _DummyManager()
    manager.reembed_result = (4, 0)
    _patch_main(monkeypatch, "load_settings", lambda: _DummySettings())
    _patch_main(monkeypatch, "build_prompt_manager", lambda _: manager)

    exit_code = main.main()

    assert exit_code == 0
    assert manager.reembed_called is True
    assert manager.reembed_reset is True
    assert "Rebuilt embeddings for 4 prompt(s)." in capsys.readouterr().out


def test_reembed_command_reports_failures(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager", "reembed"])
    manager = _DummyManager()
    manager.reembed_result = (2, 1)
    _patch_main(monkeypatch, "load_settings", lambda: _DummySettings())
    _patch_main(monkeypatch, "build_prompt_manager", lambda _: manager)

    exit_code = main.main()

    assert exit_code == 7
    assert manager.reembed_called is True
    output = capsys.readouterr().out
    assert "Embedding rebuild skipped 1 prompt(s)." in output


def test_reembed_command_handles_manager_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager", "reembed"])
    manager = _DummyManager()
    manager.reembed_error = PromptManagerError("failed")
    _patch_main(monkeypatch, "load_settings", lambda: _DummySettings())
    _patch_main(monkeypatch, "build_prompt_manager", lambda _: manager)

    exit_code = main.main()

    assert exit_code == 7
    output = capsys.readouterr().out
    assert "Failed to rebuild embeddings" in output


def test_history_analytics_command_renders_summary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["prompt-manager", "history-analytics", "--window-days", "7", "--limit", "2"],
    )
    manager = _DummyManager()
    manager.execution_analytics = _build_execution_analytics()
    _patch_main(monkeypatch, "load_settings", lambda: _DummySettings())
    _patch_main(monkeypatch, "build_prompt_manager", lambda _: manager)

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Execution analytics" in output
    assert "Prompt Alpha" in output
    assert "Tokens (window): prompt=25 completion=50 total=75" in output
    assert manager.closed is True


def test_history_analytics_handles_empty_results(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager", "history-analytics", "--window-days", "0"])
    manager = _DummyManager()
    manager.execution_analytics = ExecutionAnalytics(
        total_runs=0,
        success_rate=0.0,
        average_duration_ms=None,
        average_rating=None,
        prompt_breakdown=[],
        window_start=None,
    )
    _patch_main(monkeypatch, "load_settings", lambda: _DummySettings())
    _patch_main(monkeypatch, "build_prompt_manager", lambda _: manager)

    exit_code = main.main()

    assert exit_code == 0
    assert "No execution history" in capsys.readouterr().out
    assert manager.closed is True
