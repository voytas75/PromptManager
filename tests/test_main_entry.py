"""Lightweight integration checks for main module.

Updates:
  v0.5.5 - 2025-12-10 - Cover LiteLLM logging toggle helper.
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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture

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
from models.prompt_model import ExecutionStatus, Prompt, PromptExecution


def _patch_main(monkeypatch: pytest.MonkeyPatch, name: str, value: object) -> None:
    monkeypatch.setattr(cast("Any", main), name, value)


def _mock_module(name: str) -> Any:
    return cast("Any", types.ModuleType(name))


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


class _DummyRepository:
    def __init__(self) -> None:
        self.store: list[object] = []

    def list(self, limit: int | None = None) -> list[object]:
        values = list(self.store)
        return values if limit is None else values[:limit]

    def get(self, prompt_id: uuid.UUID) -> object:
        for prompt in self.store:
            if getattr(prompt, "id", None) == prompt_id:
                return prompt
        raise KeyError(prompt_id)


class _DummyManager:
    def __init__(self) -> None:
        self.closed = False
        self.repository = _DummyRepository()
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
        self.prompt_execution_analytics: PromptExecutionAnalytics | None = None
        self.prompt_executions: list[PromptExecution] = []

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

    def get_prompt_execution_analytics(
        self,
        prompt_id: uuid.UUID,
        *,
        window_days: int | None = None,
        trend_window: int = 5,
    ) -> PromptExecutionAnalytics | None:
        del prompt_id, window_days, trend_window
        return self.prompt_execution_analytics

    def list_executions_for_prompt(
        self,
        prompt_id: uuid.UUID,
        *,
        limit: int = 20,
    ) -> list[PromptExecution]:
        del prompt_id
        return list(self.prompt_executions[:limit])

    def diagnose_embeddings(self, *, sample_text: str = "Prompt Manager diagnostics probe"):
        self.diagnostics_sample_text = sample_text
        return self.embedding_diagnostics

    def get_token_usage_totals(self, *, since: datetime | None = None) -> TokenUsageTotals:
        if since is None:
            return self.token_usage_totals_all
        return self.token_usage_totals_window

    def create_prompt(self, prompt: object, embedding: object | None = None) -> object:
        del embedding
        self.repository.store.append(prompt)
        return prompt

    def update_prompt(self, prompt: object, embedding: object | None = None) -> object:
        del embedding
        for index, existing in enumerate(self.repository.store):
            if getattr(existing, "id", None) == getattr(prompt, "id", None):
                self.repository.store[index] = prompt
                return prompt
        raise KeyError(getattr(prompt, "id", None))


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
        decision_summary="Keep baseline",
        next_action_summary="Prefer baseline before reuse",
        freshness_summary="Validation freshness: recent",
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


def _build_manager_with(manager: _DummyManager):
    def _builder(_settings: object) -> _DummyManager:
        return manager

    return _builder


def _build_snapshot(
    *_args: object,
    snapshot: SimpleNamespace | None = None,
    **_kwargs: object,
) -> SimpleNamespace:
    if snapshot is not None:
        return snapshot
    return _build_dummy_snapshot()


def _yes_input(_prompt: object) -> str:
    return "y"


def _empty_dataset_rows(*_args: object) -> list[object]:
    return []


def _snapshot_builder_with(snapshot: SimpleNamespace) -> Any:
    def _builder(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return _build_snapshot(snapshot=snapshot)

    return _builder


def _dataset_rows_for_usage(_snapshot: object, dataset: str) -> list[dict[str, object]]:
    assert dataset == "usage"
    return [{"prompt_name": "Prompt Delta", "usage_count": 4}]


def _load_entrypoint_settings() -> _DummySettings:
    settings = _DummySettings()
    settings.litellm_model = "azure/gpt-4o-mini"
    settings.litellm_api_key = "secret-key"
    return settings


def _export_catalog_stub(*_args: object, **_kwargs: object) -> Path:
    return Path("export.json")


def _diff_catalog_stub(*_args: object, **_kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(
        added=0,
        updated=0,
        skipped=0,
        unchanged=0,
    )


def _import_catalog_stub(*_args: object, **_kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(
        added=0,
        updated=0,
        skipped=0,
        errors=0,
    )


def _analytics_snapshot_stub(*_args: object, **_kwargs: object) -> SimpleNamespace:
    return SimpleNamespace()


def _snapshot_rows_stub(*_args: object, **_kwargs: object) -> list[object]:
    return []


def _launch_prompt_manager_stub(_manager: object, settings: object | None = None) -> int:
    del settings
    return 0


def test_main_print_settings_logs_and_exits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager", "--print-settings"])
    settings = _DummySettings()
    settings.litellm_drop_params = ["max_tokens", "temperature"]
    _patch_main(monkeypatch, "load_settings", lambda: settings)
    manager = _DummyManager()
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))

    exit_code = main.main()

    assert exit_code == 0
    captured = capsys.readouterr()
    output = captured.out
    assert "Prompt Manager configuration summary" in captured.out
    diagnostics_start = output.index("Diagnostics\n-----------")
    next_steps_start = output.index("Next steps:\n")
    database_start = output.index("Database path:")
    assert diagnostics_start < next_steps_start < database_start
    assert "LiteLLM API key: not set" in captured.out
    assert "Drop params: max_tokens, temperature" in captured.out
    assert "Overall status: FAIL" in output
    assert "FAIL | Fast model: missing [default; default value in effect]" in output
    assert "FAIL | API key: missing [default; default value in effect]" in output
    assert "- Set a LiteLLM fast model to unlock chat, prompt runs, and derived defaults." in output
    assert "- Add a LiteLLM API key so model calls can authenticate successfully." in output
    assert f"Model: {DEFAULT_EMBEDDING_MODEL} (default)" in output
    assert "Prompt execution: Fast (default)" in captured.out
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
    assert "OK | Fast model: azure/gpt-4o [config]" in output


def test_main_diagnostics_analytics_uses_shared_snapshot_sections(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Analytics diagnostics CLI should surface the shared snapshot sections for operator parity."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "prompt-manager",
            "diagnostics",
            "analytics",
            "--window-days",
            "14",
            "--prompt-limit",
            "3",
        ],
    )
    settings = _DummySettings()
    manager = _DummyManager()
    _patch_main(monkeypatch, "load_settings", lambda: settings)
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))
    _patch_main(monkeypatch, "build_analytics_snapshot", _build_snapshot)

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Analytics dashboard" in output
    assert "Execution summary (last 14 days)" in output
    assert "Model cost breakdown (tokens):" in output
    assert "Benchmark success by model:" in output
    assert "Intent workspace execution success:" in output
    assert "Embedding diagnostics summary:" in output


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
    monkeypatch.setattr("builtins.input", _yes_input)

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
    settings = _DummySettings()
    settings.litellm_model = "azure/gpt-4o-mini"
    settings.litellm_api_key = "secret-key"
    _patch_main(monkeypatch, "load_settings", lambda: settings)

    def _boom(_: _DummySettings) -> None:
        raise RuntimeError("init failed")

    _patch_main(monkeypatch, "build_prompt_manager", _boom)

    exit_code = main.main()

    assert exit_code == 3
    assert "Failed to initialise services" in capsys.readouterr().out


def test_main_blocks_default_startup_when_critical_runtime_state_is_invalid(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager", "--no-gui"])
    settings = _DummySettings()
    _patch_main(monkeypatch, "load_settings", lambda: settings)

    build_calls = {"count": 0}

    def _should_not_run(_: _DummySettings) -> _DummyManager:
        build_calls["count"] += 1
        return _DummyManager()

    _patch_main(monkeypatch, "build_prompt_manager", _should_not_run)

    exit_code = main.main()

    assert exit_code == 2
    output = capsys.readouterr().out
    assert "Blocking configuration issues:" in output
    assert "- LiteLLM fast model is missing." in output
    assert "- LiteLLM API key is missing." in output
    assert build_calls["count"] == 0


def test_main_allows_print_settings_even_when_runtime_state_is_blocking(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager", "--print-settings"])
    settings = _DummySettings()
    _patch_main(monkeypatch, "load_settings", lambda: settings)

    build_calls = {"count": 0}

    def _should_not_run(_: _DummySettings) -> _DummyManager:
        build_calls["count"] += 1
        return _DummyManager()

    _patch_main(monkeypatch, "build_prompt_manager", _should_not_run)

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Prompt Manager configuration summary" in output
    assert "Overall status: FAIL" in output
    assert build_calls["count"] == 0


def test_main_logs_ready_message_on_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager", "--no-gui"])
    settings = _DummySettings()
    settings.litellm_model = "azure/gpt-4o-mini"
    settings.litellm_api_key = "secret-key"
    _patch_main(monkeypatch, "load_settings", lambda: settings)
    manager = _DummyManager()
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Prompt Manager ready" in output
    assert "ChromaDB at" in output
    assert manager.closed is True


def test_main_reexports_core_headless_surfaces_for_cli_parity() -> None:
    assert main.export_prompt_catalog.__name__ == "export_prompt_catalog"
    assert main.build_analytics_snapshot.__name__ == "build_analytics_snapshot"
    assert main.snapshot_dataset_rows.__name__ == "snapshot_dataset_rows"


def test_main_launches_gui_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager"])
    dummy_manager = _DummyManager()
    settings = _DummySettings()
    settings.litellm_model = "azure/gpt-4o-mini"
    settings.litellm_api_key = "secret-key"
    _patch_main(monkeypatch, "load_settings", lambda: settings)
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(dummy_manager))

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
    settings = _DummySettings()
    settings.litellm_model = "azure/gpt-4o-mini"
    settings.litellm_api_key = "secret-key"
    _patch_main(monkeypatch, "load_settings", lambda: settings)
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))

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


def test_main_runs_embedding_diagnostics(
    monkeypatch: pytest.MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager", "diagnostics", "embeddings"])
    _patch_main(monkeypatch, "load_settings", lambda: _DummySettings())
    manager = _DummyManager()
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Embedding diagnostics" in output
    assert manager.diagnostics_sample_text == "Prompt Manager diagnostics probe"
    assert manager.closed is True


def test_main_embedding_diagnostics_returns_failure_on_issue(
    monkeypatch: pytest.MonkeyPatch, capsys: CaptureFixture[str]
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
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))

    exit_code = main.main()

    assert exit_code == 6
    output = capsys.readouterr().out
    assert "Backend: ERROR" in output
    assert "Dimension mismatches" in output
    assert manager.closed is True


def test_main_analytics_diagnostics_runs(
    monkeypatch: pytest.MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
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

    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))
    _patch_main(monkeypatch, "build_analytics_snapshot", _snapshot_stub)
    _patch_main(monkeypatch, "snapshot_dataset_rows", _empty_dataset_rows)

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Analytics dashboard" in output
    assert captured_args["window_days"] == 14
    assert captured_args["prompt_limit"] == 7
    assert "Execution summary" in output
    assert "runs: 5" in output
    assert "Top prompt trends:" in output
    assert "Prompt Alpha" in output
    assert "Model cost breakdown (tokens):" in output
    assert "gpt-fast: runs=2, prompt=10, completion=8, total=18" in output
    assert "Benchmark success by model:" in output
    assert "gpt-bench: runs=1, success=100.0%, duration=110 ms, tokens=25" in output
    assert manager.closed is True


def test_main_analytics_diagnostics_exports_csv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: CaptureFixture[str],
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
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))
    _patch_main(
        monkeypatch,
        "build_analytics_snapshot",
        _snapshot_builder_with(snapshot),
    )

    _patch_main(monkeypatch, "snapshot_dataset_rows", _dataset_rows_for_usage)

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
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Top 1 suggestions" in output
    assert "Debug Sentinel" in output
    assert manager.closed is True


def test_prompt_show_command_outputs_prompt_details(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompt_id = uuid.uuid4()
    monkeypatch.setattr("sys.argv", ["prompt-manager", "prompt-show", str(prompt_id)])
    settings = _DummySettings()
    _patch_main(monkeypatch, "load_settings", lambda: settings)
    manager = _DummyManager()
    manager.repository.store.append(
        Prompt(
            id=prompt_id,
            name="CI Failure Triage",
            description="Summarise the first-pass diagnosis for a failing workflow.",
            category="Debugging",
            tags=["ci", "triage"],
            context="Inspect logs, isolate the first failing step, and propose next checks.",
            is_active=True,
            source="catalog",
        )
    )
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert f"id: {prompt_id}" in output
    assert "name: CI Failure Triage" in output
    assert "description: Summarise the first-pass diagnosis for a failing workflow." in output
    assert "category: Debugging" in output
    assert "tags: ci, triage" in output
    assert "source: catalog" in output
    assert "active: yes" in output
    assert (
        "context:\nInspect logs, isolate the first failing step, and propose next checks." in output
    )
    assert manager.closed is True


def test_prompt_show_command_falls_back_to_exact_name(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager", "prompt-show", "CI Failure Triage"])
    settings = _DummySettings()
    _patch_main(monkeypatch, "load_settings", lambda: settings)
    manager = _DummyManager()
    prompt_id = uuid.uuid4()
    manager.repository.store.append(
        Prompt(
            id=prompt_id,
            name="CI Failure Triage",
            description="Summarise the first-pass diagnosis for a failing workflow.",
            category="Debugging",
            tags=["ci", "triage"],
            context="Inspect logs, isolate the first failing step, and propose next checks.",
            is_active=True,
            source="catalog",
        )
    )
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert f"id: {prompt_id}" in output
    assert "name: CI Failure Triage" in output
    assert manager.closed is True


def test_prompt_find_command_lists_matching_prompts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager", "prompt-find", "triage"])
    settings = _DummySettings()
    _patch_main(monkeypatch, "load_settings", lambda: settings)
    manager = _DummyManager()
    matching_id = uuid.uuid4()
    manager.repository.store.extend(
        [
            Prompt(
                id=matching_id,
                name="CI Failure Triage",
                description="Summarise the first-pass diagnosis for a failing workflow.",
                category="Debugging",
                tags=["ci", "triage"],
                context="Inspect logs, isolate the first failing step, and propose next checks.",
                is_active=True,
                source="catalog",
            ),
            Prompt(
                id=uuid.uuid4(),
                name="Release Notes Writer",
                description="Draft release notes from merged pull requests.",
                category="Writing",
                tags=["release"],
                context="Summarise user-visible changes only.",
                is_active=True,
                source="catalog",
            ),
        ]
    )
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert f"{matching_id} | CI Failure Triage | [Debugging] | ci, triage" in output
    assert "Release Notes Writer" not in output
    assert manager.closed is True


def test_prompt_find_command_outputs_json_payload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager", "prompt-find", "triage", "--json"])
    settings = _DummySettings()
    _patch_main(monkeypatch, "load_settings", lambda: settings)
    manager = _DummyManager()
    matching_id = uuid.uuid4()
    manager.repository.store.extend(
        [
            Prompt(
                id=matching_id,
                name="CI Failure Triage",
                description="Summarise the first-pass diagnosis for a failing workflow.",
                category="Debugging",
                tags=["ci", "triage"],
                context="Inspect logs, isolate the first failing step, and propose next checks.",
                is_active=True,
                source="catalog",
            ),
            Prompt(
                id=uuid.uuid4(),
                name="Release Notes Writer",
                description="Draft release notes from merged pull requests.",
                category="Writing",
                tags=["release"],
                context="Summarise user-visible changes only.",
                is_active=True,
                source="catalog",
            ),
        ]
    )
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))

    exit_code = main.main()

    assert exit_code == 0
    output = cast("list[dict[str, object]]", json.loads(capsys.readouterr().out))
    assert isinstance(output, list)
    assert len(output) == 1
    assert output[0]["id"] == str(matching_id)
    assert output[0]["name"] == "CI Failure Triage"
    assert output[0]["tags"] == ["ci", "triage"]
    assert output[0]["source"] == "catalog"
    assert manager.closed is True


def test_prompt_find_command_filters_by_category_and_tag(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prompt-manager",
            "prompt-find",
            "prompt",
            "--category",
            "Debugging",
            "--tag",
            "triage",
        ],
    )
    settings = _DummySettings()
    _patch_main(monkeypatch, "load_settings", lambda: settings)
    manager = _DummyManager()
    matching_id = uuid.uuid4()
    manager.repository.store.extend(
        [
            Prompt(
                id=matching_id,
                name="CI Failure Triage Prompt",
                description="Summarise the first-pass diagnosis for a failing workflow.",
                category="Debugging",
                tags=["ci", "triage"],
                context="Inspect logs, isolate the first failing step, and propose next checks.",
                is_active=True,
                source="catalog",
            ),
            Prompt(
                id=uuid.uuid4(),
                name="CI Failure Prompt Without Triage",
                description="Another debugging helper.",
                category="Debugging",
                tags=["ci"],
                context="Focus on failed jobs only.",
                is_active=True,
                source="catalog",
            ),
            Prompt(
                id=uuid.uuid4(),
                name="Writing Prompt With Triage Tag",
                description="Draft comms for incidents.",
                category="Writing",
                tags=["triage"],
                context="Write a short update.",
                is_active=True,
                source="catalog",
            ),
        ]
    )
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert f"{matching_id} | CI Failure Triage Prompt | [Debugging] | ci, triage" in output
    assert "CI Failure Prompt Without Triage" not in output
    assert "Writing Prompt With Triage Tag" not in output
    assert manager.closed is True


def test_prompt_find_command_filters_by_source_and_active_state(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prompt-manager",
            "prompt-find",
            "prompt",
            "--source",
            "catalog",
            "--active",
            "true",
        ],
    )
    settings = _DummySettings()
    _patch_main(monkeypatch, "load_settings", lambda: settings)
    manager = _DummyManager()
    matching_id = uuid.uuid4()
    manager.repository.store.extend(
        [
            Prompt(
                id=matching_id,
                name="Catalog Active Prompt",
                description="Prompt from the catalog.",
                category="Debugging",
                tags=["triage"],
                context="Inspect the active catalog entry.",
                is_active=True,
                source="catalog",
            ),
            Prompt(
                id=uuid.uuid4(),
                name="Catalog Inactive Prompt",
                description="Inactive catalog entry.",
                category="Debugging",
                tags=["triage"],
                context="Should be filtered out by active state.",
                is_active=False,
                source="catalog",
            ),
            Prompt(
                id=uuid.uuid4(),
                name="User Active Prompt",
                description="User-created active entry.",
                category="Debugging",
                tags=["triage"],
                context="Should be filtered out by source.",
                is_active=True,
                source="user",
            ),
        ]
    )
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert f"{matching_id} | Catalog Active Prompt | [Debugging] | triage" in output
    assert "Catalog Inactive Prompt" not in output
    assert "User Active Prompt" not in output
    assert manager.closed is True


def test_prompt_history_command_outputs_recent_execution_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompt_id = uuid.uuid4()
    monkeypatch.setattr(
        "sys.argv",
        ["prompt-manager", "prompt-history", str(prompt_id), "--limit", "2"],
    )
    settings = _DummySettings()
    _patch_main(monkeypatch, "load_settings", lambda: settings)
    manager = _DummyManager()
    manager.repository.store.append(
        Prompt(
            id=prompt_id,
            name="CI Failure Triage",
            description="Summarise the first-pass diagnosis for a failing workflow.",
            category="Debugging",
            tags=["ci", "triage"],
            context="Inspect logs, isolate the first failing step, and propose next checks.",
            is_active=True,
            source="catalog",
        )
    )
    manager.prompt_executions = [
        PromptExecution(
            id=uuid.uuid4(),
            prompt_id=prompt_id,
            request_text="First request payload",
            response_text="First response payload",
            status=ExecutionStatus.SUCCESS,
            duration_ms=210,
            executed_at=datetime(2026, 4, 29, 10, 30, tzinfo=UTC),
            rating=4.5,
            metadata={"model": "gpt-fast", "total_tokens": 42},
        ),
        PromptExecution(
            id=uuid.uuid4(),
            prompt_id=prompt_id,
            request_text="Second request payload",
            response_text=None,
            status=ExecutionStatus.FAILED,
            error_message="Timeout while calling model",
            duration_ms=900,
            executed_at=datetime(2026, 4, 29, 9, 15, tzinfo=UTC),
            rating=None,
            metadata={"model": "gpt-reasoning", "total_tokens": 17},
        ),
    ]
    manager.prompt_execution_analytics = PromptExecutionAnalytics(
        prompt_id=prompt_id,
        name="CI Failure Triage",
        total_runs=2,
        success_rate=0.5,
        average_duration_ms=555.0,
        average_rating=4.5,
        rating_trend=0.0,
        last_executed_at=datetime(2026, 4, 29, 10, 30, tzinfo=UTC),
        prompt_tokens=20,
        completion_tokens=39,
        total_tokens=59,
        decision_summary="Keep prompt but inspect unstable model responses.",
        next_action_summary="Retry with the fast model for baseline checks.",
        freshness_summary="Validation freshness: recent",
    )
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert f"id: {prompt_id}" in output
    assert "name: CI Failure Triage" in output
    assert "runs: 2" in output
    assert "success: 50.0%" in output
    assert "avg_rating: 4.5" in output
    assert "decision: Keep prompt but inspect unstable model responses." in output
    assert "recent executions:" in output
    assert (
        "[1] 2026-04-29T10:30:00+00:00 | success | 210 ms | rating: 4.5 | "
        "model: gpt-fast | tokens: 42" in output
    )
    assert "request: First request payload" in output
    assert "response: First response payload" in output
    assert (
        "[2] 2026-04-29T09:15:00+00:00 | failed | 900 ms | rating: n/a | "
        "model: gpt-reasoning | tokens: 17" in output
    )
    assert "error: Timeout while calling model" in output
    assert manager.closed is True


def test_prompt_history_command_outputs_json_payload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompt_id = uuid.uuid4()
    monkeypatch.setattr(
        "sys.argv",
        ["prompt-manager", "prompt-history", str(prompt_id), "--limit", "2", "--json"],
    )
    settings = _DummySettings()
    _patch_main(monkeypatch, "load_settings", lambda: settings)
    manager = _DummyManager()
    manager.repository.store.append(
        Prompt(
            id=prompt_id,
            name="CI Failure Triage",
            description="Summarise the first-pass diagnosis for a failing workflow.",
            category="Debugging",
            tags=["ci", "triage"],
            context="Inspect logs, isolate the first failing step, and propose next checks.",
            is_active=True,
            source="catalog",
        )
    )
    manager.prompt_executions = [
        PromptExecution(
            id=uuid.uuid4(),
            prompt_id=prompt_id,
            request_text="First request payload",
            response_text="First response payload",
            status=ExecutionStatus.SUCCESS,
            duration_ms=210,
            executed_at=datetime(2026, 4, 29, 10, 30, tzinfo=UTC),
            rating=4.5,
            metadata={"model": "gpt-fast", "total_tokens": 42},
        )
    ]
    manager.prompt_execution_analytics = PromptExecutionAnalytics(
        prompt_id=prompt_id,
        name="CI Failure Triage",
        total_runs=1,
        success_rate=1.0,
        average_duration_ms=210.0,
        average_rating=4.5,
        rating_trend=0.2,
        last_executed_at=datetime(2026, 4, 29, 10, 30, tzinfo=UTC),
        prompt_tokens=20,
        completion_tokens=22,
        total_tokens=42,
        decision_summary="Keep prompt for baseline incident triage.",
        next_action_summary="Reuse for first-pass diagnostics.",
        freshness_summary="Validation freshness: recent",
    )
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))

    exit_code = main.main()

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["prompt"]["id"] == str(prompt_id)
    assert output["prompt"]["name"] == "CI Failure Triage"
    assert output["analytics"]["total_runs"] == 1
    assert output["analytics"]["decision_summary"] == "Keep prompt for baseline incident triage."
    assert len(output["executions"]) == 1
    assert output["executions"][0]["status"] == "success"
    assert output["executions"][0]["metadata"]["model"] == "gpt-fast"
    assert manager.closed is True


def test_prompt_history_command_filters_by_status_and_window_days(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompt_id = uuid.uuid4()
    now = datetime.now(UTC)
    monkeypatch.setattr(
        "sys.argv",
        [
            "prompt-manager",
            "prompt-history",
            str(prompt_id),
            "--limit",
            "5",
            "--status",
            "failed",
            "--window-days",
            "1",
        ],
    )
    settings = _DummySettings()
    _patch_main(monkeypatch, "load_settings", lambda: settings)
    manager = _DummyManager()
    old_execution_at = now - timedelta(days=2)
    recent_failed_at = now - timedelta(hours=2)
    recent_success_at = now - timedelta(minutes=90)
    manager.repository.store.append(
        Prompt(
            id=prompt_id,
            name="CI Failure Triage",
            description="Summarise the first-pass diagnosis for a failing workflow.",
            category="Debugging",
            tags=["ci", "triage"],
            context="Inspect logs, isolate the first failing step, and propose next checks.",
            is_active=True,
            source="catalog",
        )
    )
    manager.prompt_executions = [
        PromptExecution(
            id=uuid.uuid4(),
            prompt_id=prompt_id,
            request_text="Old failed request",
            response_text=None,
            status=ExecutionStatus.FAILED,
            error_message="Old timeout",
            duration_ms=901,
            executed_at=old_execution_at,
            metadata={"model": "gpt-reasoning", "total_tokens": 18},
        ),
        PromptExecution(
            id=uuid.uuid4(),
            prompt_id=prompt_id,
            request_text="Recent failed request",
            response_text=None,
            status=ExecutionStatus.FAILED,
            error_message="Recent timeout",
            duration_ms=321,
            executed_at=recent_failed_at,
            metadata={"model": "gpt-fast", "total_tokens": 12},
        ),
        PromptExecution(
            id=uuid.uuid4(),
            prompt_id=prompt_id,
            request_text="Recent success request",
            response_text="Recovered response",
            status=ExecutionStatus.SUCCESS,
            duration_ms=210,
            executed_at=recent_success_at,
            metadata={"model": "gpt-fast", "total_tokens": 20},
        ),
    ]
    manager.prompt_execution_analytics = PromptExecutionAnalytics(
        prompt_id=prompt_id,
        name="CI Failure Triage",
        total_runs=3,
        success_rate=1 / 3,
        average_duration_ms=477.3,
        average_rating=None,
        rating_trend=0.0,
        last_executed_at=recent_success_at,
        prompt_tokens=22,
        completion_tokens=28,
        total_tokens=50,
    )
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Recent failed request" in output
    assert "Recent timeout" in output
    assert old_execution_at.isoformat(timespec="seconds") not in output
    assert "Old failed request" not in output
    assert "Recent success request" not in output
    assert manager.closed is True


def test_prompt_show_command_outputs_json_payload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompt_id = uuid.uuid4()
    monkeypatch.setattr(
        "sys.argv",
        ["prompt-manager", "prompt-show", str(prompt_id), "--json"],
    )
    settings = _DummySettings()
    _patch_main(monkeypatch, "load_settings", lambda: settings)
    manager = _DummyManager()
    manager.repository.store.append(
        Prompt(
            id=prompt_id,
            name="CI Failure Triage",
            description="Summarise the first-pass diagnosis for a failing workflow.",
            category="Debugging",
            tags=["ci", "triage"],
            context="Inspect logs, isolate the first failing step, and propose next checks.",
            is_active=True,
            source="catalog",
        )
    )
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))

    exit_code = main.main()

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["id"] == str(prompt_id)
    assert output["name"] == "CI Failure Triage"
    assert output["tags"] == ["ci", "triage"]
    assert output["source"] == "catalog"
    assert output["is_active"] is True
    assert manager.closed is True


def test_setup_logging_basic_config_fallback(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    missing_path = tmp_path / "absent.ini"
    runtime_setup_logging = cast("Any", main)._runtime_setup_logging
    runtime_setup_logging(missing_path)
    with caplog.at_level(logging.INFO):
        logging.getLogger("prompt_manager.testing").info("hello from fallback")

    assert "hello from fallback" in caplog.text


def test_configure_litellm_logging_toggles_loggers() -> None:
    """Toggle LiteLLM loggers on and off."""
    from cli.runtime import configure_litellm_logging

    targets = [
        logging.getLogger("litellm"),
        logging.getLogger("LiteLLM"),
        logging.getLogger("litellm.proxy"),
        logging.getLogger("litellm.proxy.proxy_server"),
    ]
    original_states = [(logger.disabled, logger.level, logger.propagate) for logger in targets]
    try:
        configure_litellm_logging(False)
        for logger in targets:
            assert logger.disabled is True
            assert logger.level == logging.CRITICAL

        configure_litellm_logging(True)
        for logger in targets:
            assert logger.disabled is False
            assert logger.level == logging.NOTSET
    finally:
        for logger, (disabled, level, propagate) in zip(targets, original_states, strict=False):
            logger.disabled = disabled
            logger.setLevel(level)
            logger.propagate = propagate


def test_main_entrypoint_guard_executes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager"])

    config_stub = _mock_module("config")
    config_stub.load_settings = _load_entrypoint_settings
    config_stub.PromptManagerSettings = type("PromptManagerSettings", (), {})
    config_stub.LITELLM_ROUTED_WORKFLOWS = {"prompt_execution": "Prompt execution"}
    config_stub.DEFAULT_EMBEDDING_BACKEND = DEFAULT_EMBEDDING_BACKEND
    config_stub.DEFAULT_EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL
    core_stub = _mock_module("core")
    dummy_manager = _DummyManager()
    core_stub.build_prompt_manager = _build_manager_with(dummy_manager)
    core_stub.export_prompt_catalog = _export_catalog_stub
    core_stub.diff_prompt_catalog = _diff_catalog_stub
    core_stub.import_prompt_catalog = _import_catalog_stub
    core_stub.PromptManagerError = RuntimeError
    core_stub.build_analytics_snapshot = _analytics_snapshot_stub
    core_stub.snapshot_dataset_rows = _snapshot_rows_stub

    monkeypatch.setitem(sys.modules, "config", config_stub)
    monkeypatch.setitem(sys.modules, "core", core_stub)
    gui_stub = types.SimpleNamespace(
        launch_prompt_manager=_launch_prompt_manager_stub,
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
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))

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
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))

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


def test_prompt_add_command_accepts_json_file_alias(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    payload_path = tmp_path / "prompt-payload.json"
    payload_path.write_text(
        json.dumps(
            {
                "name": "JSON File Diagnostics Helper",
                "description": "Guide a calm first-pass diagnosis.",
                "context": "Summarise symptoms, likely causes, and next checks.",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "prompt-manager",
            "prompt-add",
            "--input-file",
            str(payload_path),
            "--dry-run",
        ],
    )
    manager = _DummyManager()
    _patch_main(monkeypatch, "load_settings", lambda: _DummySettings())
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Catalog import preview" in output
    assert "added=1" in output
    assert manager.closed is True


def test_prompt_add_command_rejects_path_and_input_file_mix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "prompt.json"
    catalog_path.write_text(
        json.dumps(
            {
                "name": "Diagnostics Helper",
                "description": "Assist with debugging failing CI jobs.",
            }
        ),
        encoding="utf-8",
    )
    alias_path = tmp_path / "prompt-payload.json"
    alias_path.write_text(
        json.dumps(
            {
                "name": "JSON File Diagnostics Helper",
                "description": "Guide a calm first-pass diagnosis.",
                "context": "Summarise symptoms, likely causes, and next checks.",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "prompt-manager",
            "prompt-add",
            str(catalog_path),
            "--input-file",
            str(alias_path),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        main.main()

    assert excinfo.value.code == 2


def test_prompt_add_command_rejects_json_string_missing_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.dumps(
        {
            "description": "Guide a calm first-pass diagnosis.",
            "context": "Summarise symptoms, likely causes, and next checks.",
        }
    )
    monkeypatch.setattr(
        "sys.argv",
        ["prompt-manager", "prompt-add", "--json", payload, "--dry-run"],
    )

    with pytest.raises(SystemExit) as excinfo:
        main.main()

    assert excinfo.value.code == 2


def test_prompt_add_command_rejects_json_string_missing_description(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.dumps(
        {
            "name": "JSON Diagnostics Helper",
            "context": "Summarise symptoms, likely causes, and next checks.",
        }
    )
    monkeypatch.setattr(
        "sys.argv",
        ["prompt-manager", "prompt-add", "--json", payload, "--dry-run"],
    )

    with pytest.raises(SystemExit) as excinfo:
        main.main()

    assert excinfo.value.code == 2


def test_prompt_add_command_rejects_json_string_with_non_object_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["prompt-manager", "prompt-add", "--json", '["not-an-object"]', "--dry-run"],
    )

    with pytest.raises(SystemExit) as excinfo:
        main.main()

    assert excinfo.value.code == 2


def test_prompt_add_command_accepts_json_string(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = json.dumps(
        {
            "name": "JSON Diagnostics Helper",
            "description": "Guide a calm first-pass diagnosis.",
            "context": "Summarise symptoms, likely causes, and next checks.",
        }
    )
    monkeypatch.setattr(
        "sys.argv",
        ["prompt-manager", "prompt-add", "--json", payload, "--dry-run"],
    )
    manager = _DummyManager()
    _patch_main(monkeypatch, "load_settings", lambda: _DummySettings())
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Catalog import preview" in output
    assert "added=1" in output
    assert manager.closed is True


def test_prompt_add_command_accepts_stdin_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = json.dumps(
        {
            "name": "STDIN Diagnostics Helper",
            "description": "Guide a calm first-pass diagnosis.",
            "context": "Summarise symptoms, likely causes, and next checks.",
        }
    )
    monkeypatch.setattr(
        "sys.argv",
        ["prompt-manager", "prompt-add", "--from-stdin", "--dry-run"],
    )
    monkeypatch.setattr(main.sys, "stdin", SimpleNamespace(read=lambda: payload))
    manager = _DummyManager()
    _patch_main(monkeypatch, "load_settings", lambda: _DummySettings())
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Catalog import preview" in output
    assert "added=1" in output
    assert manager.closed is True


def test_prompt_add_command_rejects_json_and_stdin_mix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.dumps(
        {
            "name": "JSON Diagnostics Helper",
            "description": "Guide a calm first-pass diagnosis.",
            "context": "Summarise symptoms, likely causes, and next checks.",
        }
    )
    monkeypatch.setattr(
        "sys.argv",
        ["prompt-manager", "prompt-add", "--json", payload, "--from-stdin"],
    )
    monkeypatch.setattr(main.sys, "stdin", SimpleNamespace(read=lambda: payload))

    with pytest.raises(SystemExit) as excinfo:
        main.main()

    assert excinfo.value.code == 2


def test_prompt_add_command_rejects_invalid_json_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["prompt-manager", "prompt-add", "--json", "{not-valid-json}", "--dry-run"],
    )

    with pytest.raises(SystemExit) as excinfo:
        main.main()

    assert excinfo.value.code == 2


def test_prompt_add_command_rejects_blank_stdin_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["prompt-manager", "prompt-add", "--from-stdin", "--dry-run"],
    )
    monkeypatch.setattr(main.sys, "stdin", SimpleNamespace(read=lambda: "   \n  "))

    with pytest.raises(SystemExit) as excinfo:
        main.main()

    assert excinfo.value.code == 2


def test_prompt_add_inline_scenario_is_preserved_in_payload_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inline prompt-add scenario should use the prompt model's serialized shape."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "prompt-manager",
            "prompt-add",
            "--name",
            "Inline prompt",
            "--description",
            "Prompt description",
            "--prompt-text",
            "Prompt body",
            "--scenario",
            "Use during an incident handoff.",
        ],
    )

    from cli.parser import parse_args

    args = parse_args()
    payload = json.loads(args.path.read_text())
    args.path.unlink()

    assert payload["ext5"] == {"scenarios": ["Use during an incident handoff."]}


def test_prompt_add_command_accepts_inline_fields(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prompt-manager",
            "prompt-add",
            "--name",
            "Inline Diagnostics Helper",
            "--description",
            "Guide a calm first-pass diagnosis.",
            "--prompt-text",
            "Summarise symptoms, likely causes, and next checks.",
            "--category",
            "Operations",
            "--tags",
            "diagnostics,incident,triage",
            "--dry-run",
        ],
    )
    manager = _DummyManager()
    _patch_main(monkeypatch, "load_settings", lambda: _DummySettings())
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Catalog import preview" in output
    assert "added=1" in output
    assert manager.closed is True


def test_prompt_add_command_rejects_missing_input_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager", "prompt-add"])

    with pytest.raises(SystemExit) as excinfo:
        main.main()

    assert excinfo.value.code == 2


def test_prompt_add_command_rejects_path_and_inline_mix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "prompt.json"
    catalog_path.write_text(
        json.dumps(
            {
                "name": "Diagnostics Helper",
                "description": "Assist with debugging failing CI jobs.",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "prompt-manager",
            "prompt-add",
            str(catalog_path),
            "--name",
            "Inline Diagnostics Helper",
            "--description",
            "Guide a calm first-pass diagnosis.",
            "--prompt-text",
            "Summarise symptoms, likely causes, and next checks.",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        main.main()

    assert excinfo.value.code == 2


def test_prompt_add_command_requires_description_for_inline_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prompt-manager",
            "prompt-add",
            "--name",
            "Inline Diagnostics Helper",
            "--prompt-text",
            "Summarise symptoms, likely causes, and next checks.",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        main.main()

    assert excinfo.value.code == 2


def test_prompt_add_command_dry_run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "prompt.json"
    catalog_path.write_text(
        json.dumps(
            {
                "name": "Diagnostics Helper",
                "description": "Assist with debugging failing CI jobs.",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        ["prompt-manager", "prompt-add", str(catalog_path), "--dry-run"],
    )
    manager = _DummyManager()
    _patch_main(monkeypatch, "load_settings", lambda: _DummySettings())
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Catalog import preview" in output
    assert "added=1" in output
    assert manager.closed is True


def test_prompt_add_command_applies_changes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "prompt.json"
    catalog_path.write_text(
        json.dumps(
            {
                "name": "Diagnostics Helper",
                "description": "Assist with debugging failing CI jobs.",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        ["prompt-manager", "prompt-add", str(catalog_path)],
    )
    manager = _DummyManager()
    _patch_main(monkeypatch, "load_settings", lambda: _DummySettings())
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Catalog import applied" in output
    assert "added=1" in output
    assert manager.closed is True


def test_prompt_add_command_no_overwrite_skips_existing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "prompt.json"
    catalog_path.write_text(
        json.dumps(
            {
                "name": "Diagnostics Helper",
                "description": "Incoming replacement",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        ["prompt-manager", "prompt-add", str(catalog_path), "--no-overwrite"],
    )
    manager = _DummyManager()
    manager.create_prompt(
        Prompt(
            id=uuid.uuid4(),
            name="Diagnostics Helper",
            description="Existing prompt",
            category="General",
            tags=[],
        )
    )
    _patch_main(monkeypatch, "load_settings", lambda: _DummySettings())
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Catalog import applied" in output
    assert "skipped=1" in output
    stored = cast("Prompt", manager.repository.list()[0])
    assert stored.description == "Existing prompt"
    assert manager.closed is True


def test_prompt_add_command_returns_error_when_importer_reports_failures(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "prompt.json"
    catalog_path.write_text(
        json.dumps(
            {
                "name": "Diagnostics Helper",
                "description": "Assist with debugging failing CI jobs.",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        ["prompt-manager", "prompt-add", str(catalog_path)],
    )
    manager = _DummyManager()
    _patch_main(monkeypatch, "load_settings", lambda: _DummySettings())
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))

    def _import_stub(*args: object, **kwargs: object):
        return SimpleNamespace(added=0, updated=0, skipped=0, errors=1)

    _patch_main(monkeypatch, "import_prompt_catalog", _import_stub)

    exit_code = main.main()

    assert exit_code == 6
    output = capsys.readouterr().out
    assert "Catalog import applied" in output
    assert "errors=1" in output
    assert manager.closed is True


def test_reembed_command_succeeds(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["prompt-manager", "reembed"])
    manager = _DummyManager()
    manager.reembed_result = (4, 0)
    _patch_main(monkeypatch, "load_settings", lambda: _DummySettings())
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))

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
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))

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
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))

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
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))

    exit_code = main.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Execution analytics" in output
    assert "Prompt Alpha" in output
    assert "decision: Keep baseline" in output
    assert "next: Prefer baseline before reuse" in output
    assert "freshness: Validation freshness: recent" in output
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
    _patch_main(monkeypatch, "build_prompt_manager", _build_manager_with(manager))

    exit_code = main.main()

    assert exit_code == 0
    assert "No execution history" in capsys.readouterr().out
    assert manager.closed is True
