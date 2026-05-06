"""Prompt chain backend summary tests.

Updates:
  v0.1.1 - 2025-12-06 - Assert chain summary prompt overrides propagate to LiteLLM calls.
  v0.1.0 - 2025-12-06 - Cover LiteLLM-driven summaries and deterministic fallback.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import TYPE_CHECKING

from core.execution import CodexExecutionResult
from core.litellm_adapter import LiteLLMNotInstalledError
from core.prompt_manager.chains import PromptChainMixin, PromptChainRunResult, PromptChainStepRun
from core.prompt_manager.execution_history import ExecutionOutcome
from models.prompt_chain_model import PromptChain, PromptChainStep

if TYPE_CHECKING:
    import pytest


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


def _step_run(
    order_index: int,
    response_text: str,
    *,
    status: str = "success",
) -> PromptChainStepRun:
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
    return PromptChainStepRun(
        step=step,
        status=status,
        outcome=outcome if status == "success" else None,
    )


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
    assert user_prompt == "Final output to summarise."


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


def test_prompt_chain_step_run_exposes_execution_metadata() -> None:
    chain_id = uuid.uuid4()
    step = PromptChainStep(
        id=uuid.uuid4(),
        chain_id=chain_id,
        prompt_id=uuid.uuid4(),
        order_index=1,
        input_template="{{ body }}",
        output_variable="draft",
    )
    result = CodexExecutionResult(
        prompt_id=step.prompt_id,
        request_text="raw request",
        response_text="raw response",
        duration_ms=42,
        usage={},
        raw_response={},
    )
    outcome = ExecutionOutcome(result=result, history_entry=None, conversation=[])

    step_run = PromptChainStepRun(
        step=step,
        status="success",
        outcome=outcome,
        prompt_name="Draft Prompt",
        request_text="enriched request",
        response_text="raw response",
        duration_ms=42,
        web_search_requested=True,
        web_search_applied=True,
        skip_reason=None,
    )

    assert step_run.prompt_name == "Draft Prompt"
    assert step_run.request_text == "enriched request"
    assert step_run.response_text == "raw response"
    assert step_run.duration_ms == 42
    assert step_run.web_search_requested is True
    assert step_run.web_search_applied is True
    assert step_run.skip_reason is None


def test_prompt_chain_step_run_exposes_human_and_machine_step_identity() -> None:
    chain_id = uuid.uuid4()
    step = PromptChainStep(
        id=uuid.uuid4(),
        chain_id=chain_id,
        prompt_id=uuid.uuid4(),
        order_index=2,
        input_template="{{ body }}",
        output_variable="final_draft",
    )

    step_run = PromptChainStepRun(
        step=step,
        status="success",
        outcome=None,
        step_label="final_draft",
        step_output_key="step_2",
    )

    assert step_run.step_label == "final_draft"
    assert step_run.step_output_key == "step_2"


def test_prompt_chain_run_result_exposes_canonical_machine_outputs_and_final_step_fields() -> None:
    chain_id = uuid.uuid4()
    first_step = PromptChainStep(
        id=uuid.uuid4(),
        chain_id=chain_id,
        prompt_id=uuid.uuid4(),
        order_index=1,
        input_template="",
        output_variable="draft",
    )
    second_step = PromptChainStep(
        id=uuid.uuid4(),
        chain_id=chain_id,
        prompt_id=uuid.uuid4(),
        order_index=2,
        input_template="",
        output_variable="final",
    )
    chain = PromptChain(
        id=chain_id,
        name="Contract Chain",
        description="",
        steps=[first_step, second_step],
    )
    step_runs = [
        PromptChainStepRun(
            step=first_step,
            status="success",
            outcome=None,
            step_label="draft",
            step_output_key="step_1",
        ),
        PromptChainStepRun(
            step=second_step,
            status="success",
            outcome=None,
            step_label="final",
            step_output_key="step_2",
        ),
    ]

    run_result = PromptChainRunResult(
        chain=chain,
        chain_input="Seed input",
        step_outputs={"step_1": "Draft", "step_2": "Final"},
        steps=step_runs,
        final_output_text="Final",
        final_summary_text="Summary",
        step_aliases={"draft": "step_1", "final": "step_2"},
        final_step_id=second_step.id,
        final_step_output_key="step_2",
        final_step_label="final",
    )

    assert run_result.step_outputs == {"step_1": "Draft", "step_2": "Final"}
    assert run_result.step_aliases == {"draft": "step_1", "final": "step_2"}
    assert run_result.final_step_id == second_step.id
    assert run_result.final_step_output_key == "step_2"
    assert run_result.final_step_label == "final"


def test_prompt_chain_run_result_exposes_run_status_and_terminal_step_fields() -> None:
    chain_id = uuid.uuid4()
    success_step = PromptChainStep(
        id=uuid.uuid4(),
        chain_id=chain_id,
        prompt_id=uuid.uuid4(),
        order_index=1,
        input_template="",
        output_variable="draft",
    )
    terminal_step = PromptChainStep(
        id=uuid.uuid4(),
        chain_id=chain_id,
        prompt_id=uuid.uuid4(),
        order_index=2,
        input_template="",
        output_variable="final",
    )
    chain = PromptChain(
        id=chain_id,
        name="Run Status Chain",
        description="",
        steps=[success_step, terminal_step],
    )
    step_runs = [
        PromptChainStepRun(
            step=success_step,
            status="success",
            outcome=None,
            step_label="draft",
            step_output_key="step_1",
        ),
        PromptChainStepRun(
            step=terminal_step,
            status="failed",
            outcome=None,
            step_label="final",
            step_output_key="step_2",
            error="boom",
        ),
    ]

    run_result = PromptChainRunResult(
        chain=chain,
        chain_input="Seed input",
        step_outputs={"step_1": "Draft"},
        steps=step_runs,
        final_output_text="Draft",
        final_summary_text=None,
        run_status="partial_success",
        final_step_id=success_step.id,
        final_step_output_key="step_1",
        final_step_label="draft",
        terminal_step_id=terminal_step.id,
        terminal_step_output_key="step_2",
        terminal_step_label="final",
        terminal_step_status="failed",
    )

    assert run_result.run_status == "partial_success"
    assert run_result.final_step_id == success_step.id
    assert run_result.final_step_output_key == "step_1"
    assert run_result.terminal_step_id == terminal_step.id
    assert run_result.terminal_step_output_key == "step_2"
    assert run_result.terminal_step_label == "final"
    assert run_result.terminal_step_status == "failed"


class _ChainHistoryHarness(PromptChainMixin):
    def __init__(self) -> None:
        self._chain_run_history_records: list[dict[str, str | None]] = []

    def _record_chain_run_history(self, result: PromptChainRunResult) -> None:
        super()._record_chain_run_history(result)

    def list_recent_prompt_chain_runs(self, *, limit: int = 20) -> list[dict[str, str | None]]:
        return super().list_recent_prompt_chain_runs(limit=limit)


def test_build_chain_history_record_returns_minimum_bounded_evidence() -> None:
    harness = _ChainHistoryHarness()
    chain_id = uuid.uuid4()
    final_step_id = uuid.uuid4()
    chain = PromptChain(
        id=chain_id,
        name="History Chain",
        description="",
        steps=[],
    )
    result = PromptChainRunResult(
        chain=chain,
        chain_input="Input that should be trimmed into a bounded preview.",
        step_outputs={"step_1": "Final response that should also be trimmed."},
        steps=[],
        final_output_text="Final response that should also be trimmed.",
        final_step_id=final_step_id,
        final_step_output_key="step_1",
        final_step_label="final",
    )

    record = harness._build_chain_history_record(result)

    assert record["chain_id"] == str(chain_id)
    assert record["chain_name"] == "History Chain"
    assert record["status"] == "success"
    assert record["final_step_output_key"] == "step_1"
    assert record["final_step_id"] == str(final_step_id)
    assert record["final_step_label"] == "final"
    assert isinstance(record["run_timestamp"], str)
    assert "Input that should be trimmed" in str(record["input_preview"])
    assert "Final response that should also be trimmed" in str(record["final_output_preview"])
    assert "steps" not in record
    assert "step_outputs" not in record
    assert "request_text" not in record
    assert "response_text" not in record


def test_build_chain_history_record_prefers_backend_run_status() -> None:
    harness = _ChainHistoryHarness()
    chain = PromptChain(
        id=uuid.uuid4(),
        name="Partial Chain",
        description="",
        steps=[],
    )
    result = PromptChainRunResult(
        chain=chain,
        chain_input="input",
        step_outputs={"step_1": "Draft"},
        steps=[],
        final_output_text="Draft",
        run_status="partial_success",
        final_step_output_key="step_1",
    )

    record = harness._build_chain_history_record(result)

    assert record["status"] == "partial_success"


def test_list_recent_prompt_chain_runs_normalizes_zero_limit_to_one() -> None:
    harness = _ChainHistoryHarness()
    for index in range(3):
        chain = PromptChain(
            id=uuid.uuid4(),
            name=f"Chain {index}",
            description="",
            steps=[],
        )
        result = PromptChainRunResult(
            chain=chain,
            chain_input=f"input {index}",
            step_outputs={f"step_{index}": f"output {index}"},
            steps=[],
            final_output_text=f"output {index}",
            run_status="success",
            final_step_output_key=f"step_{index}",
        )
        harness._record_chain_run_history(result)

    recent = harness.list_recent_prompt_chain_runs(limit=0)

    assert len(recent) == 3
    assert recent[0]["chain_name"] == "Chain 2"


def test_record_chain_run_history_stores_newest_first_and_bounded_retention() -> None:
    harness = _ChainHistoryHarness()
    for index in range(25):
        chain = PromptChain(
            id=uuid.uuid4(),
            name=f"Chain {index}",
            description="",
            steps=[],
        )
        result = PromptChainRunResult(
            chain=chain,
            chain_input=f"input {index}",
            step_outputs={f"step_{index}": f"output {index}"},
            steps=[],
            final_output_text=f"output {index}",
            final_step_output_key=f"step_{index}",
        )
        harness._record_chain_run_history(result)

    recent = harness.list_recent_prompt_chain_runs(limit=20)

    assert len(recent) == 20
    assert recent[0]["chain_name"] == "Chain 24"
    assert recent[-1]["chain_name"] == "Chain 5"


def test_prompt_chain_run_result_exposes_step_metadata_in_order() -> None:
    chain_id = uuid.uuid4()
    step = PromptChainStep(
        id=uuid.uuid4(),
        chain_id=chain_id,
        prompt_id=uuid.uuid4(),
        order_index=1,
        input_template="{{ body }}",
        output_variable="draft",
    )
    chain = PromptChain(
        id=chain_id,
        name="Metadata Chain",
        description="",
        steps=[step],
    )
    step_run = PromptChainStepRun(
        step=step,
        status="failed",
        outcome=None,
        error="boom",
        prompt_name="Draft Prompt",
        request_text="attempted request",
        response_text=None,
        duration_ms=None,
        web_search_requested=True,
        web_search_applied=False,
        skip_reason="Web search unavailable",
    )

    run_result = PromptChainRunResult(
        chain=chain,
        chain_input="input",
        step_outputs={},
        steps=[step_run],
        final_output_text=None,
        final_summary_text=None,
    )

    assert run_result.steps[0].prompt_name == "Draft Prompt"
    assert run_result.steps[0].web_search_requested is True
    assert run_result.steps[0].web_search_applied is False
    assert run_result.steps[0].skip_reason == "Web search unavailable"
