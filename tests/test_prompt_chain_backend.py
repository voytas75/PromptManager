"""Prompt chain backend summary tests.

Updates:
  v0.1.1 - 2025-12-06 - Assert chain summary prompt overrides propagate to LiteLLM calls.
  v0.1.0 - 2025-12-06 - Cover LiteLLM-driven summaries and deterministic fallback.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from core.execution import CodexExecutionResult
from core.litellm_adapter import LiteLLMNotInstalledError
from core.prompt_manager.chains import PromptChainMixin, PromptChainStepRun
from core.prompt_manager.execution_history import ExecutionOutcome
from models.prompt_chain_model import PromptChainStep


class _ChainSummaryHarness(PromptChainMixin):
    def __init__(self) -> None:
        self._litellm_fast_model = "fast-model"
        self._litellm_inference_model = None
        self._litellm_drop_params = None
        self._prompt_templates: dict[str, str] = {}
        self._executor = SimpleNamespace(
            model="fast-model",
            api_key="test-key",
            api_base=None,
            api_version=None,
            timeout_seconds=5,
            drop_params=None,
        )


def _step_run(order_index: int, response_text: str, *, status: str = "success") -> PromptChainStepRun:
    step = PromptChainStep(
        id=uuid.uuid4(),
        chain_id=uuid.uuid4(),
        prompt_id=uuid.uuid4(),
        order_index=order_index,
        input_template="{{ body }}",
        output_variable=f"var_{order_index}",
    )
    execution_result = CodexExecutionResult(
        prompt_id=step.prompt_id,
        request_text="input",
        response_text=response_text,
        duration_ms=1,
        usage={},
        raw_response={},
    )
    outcome = ExecutionOutcome(result=execution_result, history_entry=None, conversation=[])
    return PromptChainStepRun(step=step, status=status, outcome=outcome if status == "success" else None)


def test_chain_summary_prefers_litellm_last_step(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _ChainSummaryHarness()
    captured: dict[str, object] = {}

    class _FakeLiteError(Exception):
        pass

    def fake_get_completion():  # noqa: ANN202 - helper for monkeypatch
        return (lambda **_kwargs: None, _FakeLiteError)

    def fake_call_completion(request, *_args, **_kwargs):  # noqa: ANN202 - helper stub
        captured["request"] = request
        return {"choices": [{"message": {"content": "LLM summary text"}}]}

    monkeypatch.setattr("core.prompt_manager.chains.get_completion", fake_get_completion)
    monkeypatch.setattr(
        "core.prompt_manager.chains.call_completion_with_fallback",
        fake_call_completion,
    )

    first = _step_run(1, "Initial output.")
    final = _step_run(2, "Final output to summarise.")

    summary = harness._build_chain_summary([first, final])

    assert summary == "LLM summary text"
    assert "request" in captured
    request = captured["request"]
    assert isinstance(request, dict)
    assert request.get("model") == "fast-model"
    user_prompt = request["messages"][1]["content"]  # type: ignore[index]
    assert "Final output to summarise." in user_prompt


def test_chain_summary_respects_prompt_template_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _ChainSummaryHarness()
    harness._prompt_templates = {"chain_summary": "Override summary prompt."}
    captured: dict[str, object] = {}

    def fake_get_completion():  # noqa: ANN202 - helper for monkeypatch
        return (lambda **_kwargs: None, Exception)

    def fake_call_completion(request, *_args, **_kwargs):  # noqa: ANN202
        captured["request"] = request
        return {"choices": [{"message": {"content": "Result"}}]}

    monkeypatch.setattr("core.prompt_manager.chains.get_completion", fake_get_completion)
    monkeypatch.setattr(
        "core.prompt_manager.chains.call_completion_with_fallback",
        fake_call_completion,
    )

    harness._build_chain_summary([_step_run(1, "Content")])
    request = captured["request"]
    messages = request["messages"]  # type: ignore[index]
    assert messages[0]["content"] == "Override summary prompt."


def test_chain_summary_falls_back_without_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = _ChainSummaryHarness()

    def fake_get_completion():  # noqa: ANN202 - helper for monkeypatch
        raise LiteLLMNotInstalledError("liteLLM missing")

    monkeypatch.setattr("core.prompt_manager.chains.get_completion", fake_get_completion)

    summary = harness._build_chain_summary([_step_run(1, "Short final response.")])

    assert summary == "Short final response."


def test_chain_summary_returns_none_without_successful_steps() -> None:
    harness = _ChainSummaryHarness()
    failed = _step_run(1, "", status="failed")

    summary = harness._build_chain_summary([failed])

    assert summary is None
