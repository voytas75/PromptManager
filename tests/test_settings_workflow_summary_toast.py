"""Tests for user-facing settings summary toasts in Prompt Manager."""

from __future__ import annotations

from typing import Any, cast

from gui.runtime_settings_service import RuntimeSettingsResult
from gui.settings_workflow import SettingsWorkflow


class _DummyRuntimeSettingsService:
    def __init__(self, result: RuntimeSettingsResult) -> None:
        self.result = result

    def apply_updates(
        self,
        runtime: dict[str, object | None],
        updates: dict[str, object | None],
    ) -> RuntimeSettingsResult:
        runtime.update(updates)
        return self.result


class _DummyAppearanceController:
    def __init__(self) -> None:
        self.applied_theme: str | None = None

    def apply_theme(self, theme_mode: str) -> None:
        self.applied_theme = theme_mode


class _DummyRunButton:
    def __init__(self) -> None:
        self.enabled = True
        self.tooltip = ""

    def setToolTip(self, text: str) -> None:
        self.tooltip = text

    def setEnabled(self, value: bool) -> None:
        self.enabled = value


class _DummyManager:
    def __init__(self) -> None:
        self.executor = object()
        self.llm_available = True

    def llm_status_message(self, context: str) -> str:
        return f"{context} unavailable"


def test_apply_settings_emits_summary_toast_after_success() -> None:
    runtime_settings: dict[str, object | None] = {"theme_mode": "light"}
    toasts: list[str] = []
    load_calls: list[str] = []
    refresh_calls: list[str] = []
    tooltip_calls: list[str] = []
    appearance = _DummyAppearanceController()
    run_button = _DummyRunButton()
    service = _DummyRuntimeSettingsService(
        RuntimeSettingsResult(
            theme_mode="dark",
            has_executor=True,
            summary_message="Fast model: azure/gpt-4.1-mini | Inference model: azure/gpt-5.4 | Routing: inference for: prompt_generation",
        )
    )

    def _load_prompts(search_text: str = "", *, use_indicator: bool = True) -> None:
        del use_indicator
        load_calls.append(search_text)

    workflow = SettingsWorkflow(
        parent=cast("Any", None),
        manager=cast("Any", _DummyManager()),
        runtime_settings_service=cast("Any", service),
        runtime_settings=runtime_settings,
        quick_action_supplier=lambda: None,
        prompt_generation_refresher=lambda: refresh_calls.append("prompt"),
        appearance_controller=cast("Any", appearance),
        execution_controller_supplier=lambda: None,
        load_prompts=_load_prompts,
        current_search_text=lambda: "hello",
        run_button=cast("Any", run_button),
        template_preview_supplier=lambda: None,
        toast_callback=toasts.append,
        web_search_tooltip_updater=lambda: tooltip_calls.append("tooltip"),
    )

    workflow.apply_settings({"litellm_model": "azure/gpt-4.1-mini"})

    assert toasts == [
        "Fast model: azure/gpt-4.1-mini | Inference model: azure/gpt-5.4 | Routing: inference for: prompt_generation"
    ]
    assert appearance.applied_theme == "dark"
    assert load_calls == ["hello"]
    assert refresh_calls == ["prompt"]
    assert tooltip_calls == ["tooltip"]
    assert run_button.enabled is True
