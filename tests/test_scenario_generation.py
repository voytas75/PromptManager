"""Tests for LiteLLM-backed scenario generation utilities."""

from __future__ import annotations

from typing import Any, Callable, Dict, Sequence

import pytest

from core.scenario_generation import (
    LiteLLMScenarioGenerator,
    ScenarioGenerationError,
    _extract_candidates,
    _normalise_scenarios,
)


def test_normalise_scenarios_trims_and_limits() -> None:
    """Scenarios should be deduplicated, trimmed, and limited."""

    candidates = [
        "  -Investigate outages  ",
        "Investigate outages",
        "* Draft release notes ",
        "",
        " Summarise logs ",
    ]
    assert _normalise_scenarios(candidates, limit=2) == [
        "Investigate outages",
        "Draft release notes",
    ]


def test_extract_candidates_supports_multiple_payload_shapes() -> None:
    """Ensure JSON arrays, dicts, and plain text inputs are parsed."""

    array_json = '["A", "B"]'
    dict_json = '{"first": "One", "second": "Two"}'
    plain_text = "First line\n\nSecond line"

    assert _extract_candidates(array_json) == ["A", "B"]
    assert _extract_candidates(dict_json) == ["One", "Two"]
    assert _extract_candidates(plain_text) == ["First line", "Second line"]


def test_generate_invokes_litellm_with_drop_params(monkeypatch: pytest.MonkeyPatch) -> None:
    """LiteLLMScenarioGenerator should drop configured params and return normalised scenarios."""

    captured: Dict[str, Any] = {}

    def fake_get_completion() -> tuple[Callable[..., Any], type[Exception]]:
        def _completion(**_: Any) -> Dict[str, Any]:
            return {"choices": [{"message": {"content": '["Duplicate", "-Unique"]'}}]}

        return _completion, RuntimeError

    def fake_apply_drop_params(request: Dict[str, Any], drop_params: Sequence[str]) -> tuple[str, ...]:
        captured["request_before_drop"] = dict(request)
        for key in list(request.keys()):
            if key in drop_params:
                request.pop(key, None)
        return tuple(drop_params)

    def fake_call_completion(
        request: Dict[str, Any],
        completion: Callable[..., Any],  # noqa: ARG001
        lite_llm_exception: type[Exception],  # noqa: ARG001
        *,
        drop_candidates: Sequence[str],  # noqa: ARG001
        pre_dropped: Sequence[str],
    ) -> Dict[str, Any]:
        captured["request_after_drop"] = dict(request)
        captured["pre_dropped"] = tuple(pre_dropped)
        return {"choices": [{"message": {"content": '["Duplicate", "* Keep concise "]'}}]}

    monkeypatch.setattr("core.scenario_generation.get_completion", fake_get_completion)
    monkeypatch.setattr("core.scenario_generation.apply_configured_drop_params", fake_apply_drop_params)
    monkeypatch.setattr(
        "core.scenario_generation.call_completion_with_fallback",
        fake_call_completion,
    )

    generator = LiteLLMScenarioGenerator(
        model="fast-model",
        api_key="secret",
        api_base="https://api.example.com",
        api_version="2024-06-01",
        timeout_seconds=5.0,
        drop_params=("api_key",),
        default_max_scenarios=2,
        system_prompt="  Use the template  ",
    )

    scenarios = generator.generate("Use tests to catch regressions.", max_scenarios=10)

    assert scenarios == ["Duplicate", "Keep concise"]
    assert captured["request_before_drop"]["api_key"] == "secret"
    assert "api_key" not in captured["request_after_drop"]
    assert captured["pre_dropped"] == ("api_key",)
    assert captured["request_after_drop"]["messages"][0]["content"] == "Use the template"
    assert captured["request_after_drop"]["messages"][1]["content"].startswith("Return up to 5 scenarios")


def test_generate_requires_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty prompts should raise errors instead of calling LiteLLM."""

    generator = LiteLLMScenarioGenerator(model="fast-model")
    with pytest.raises(ScenarioGenerationError):
        generator.generate("   ")


def test_generate_raises_when_no_scenarios(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing LiteLLM content should raise ScenarioGenerationError."""

    def fake_get_completion() -> tuple[Callable[..., Any], type[Exception]]:
        def _completion(**_: Any) -> Dict[str, Any]:
            return {"choices": [{"message": {"content": "[]"}}]}

        return _completion, RuntimeError

    def fake_call_completion(
        request: Dict[str, Any],  # noqa: ARG001
        completion: Callable[..., Any],  # noqa: ARG001
        lite_llm_exception: type[Exception],  # noqa: ARG001
        *,
        drop_candidates: Sequence[str],  # noqa: ARG001
        pre_dropped: Sequence[str],  # noqa: ARG001
    ) -> Dict[str, Any]:
        return {"choices": [{"message": {"content": "[]"}}]}

    monkeypatch.setattr("core.scenario_generation.get_completion", fake_get_completion)
    monkeypatch.setattr(
        "core.scenario_generation.call_completion_with_fallback",
        fake_call_completion,
    )

    generator = LiteLLMScenarioGenerator(model="fast-model")

    with pytest.raises(ScenarioGenerationError):
        generator.generate("Draft release notes.")
