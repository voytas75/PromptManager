"""Prompt chain CLI legibility tests.

Updates:
  v0.1.0 - 2026-05-04 - Cover bounded prompt-chain show/run output clarity cues.
"""

from __future__ import annotations

import argparse
import logging
import uuid
from typing import TYPE_CHECKING, Any

from cli.commands import run_prompt_chain_run, run_prompt_chain_show
from core.execution import CodexExecutionResult
from core.prompt_manager.execution_history import ExecutionOutcome
from models.prompt_chain_model import PromptChain, PromptChainStep

if TYPE_CHECKING:
    import pytest


class _ChainCliManagerStub:
    def __init__(self) -> None:
        self.chain = _make_chain()

    def get_prompt_chain(self, chain_id: uuid.UUID) -> PromptChain:
        if chain_id != self.chain.id:
            raise ValueError("Chain not found")
        return self.chain

    def run_prompt_chain(
        self,
        chain_id: uuid.UUID,
        *,
        chain_input: str,
        use_web_search: bool,
    ) -> Any:
        if chain_id != self.chain.id:
            raise ValueError("Chain not found")
        del use_web_search
        final_step = self.chain.steps[-1]
        outcome = ExecutionOutcome(
            result=CodexExecutionResult(
                prompt_id=final_step.prompt_id,
                request_text="Previous output",
                response_text="Final response text that should stay visible.",
                duration_ms=12,
                usage={},
                raw_response={},
            ),
            history_entry=None,
            conversation=[],
        )
        from core import PromptChainRunResult, PromptChainStepRun

        return PromptChainRunResult(
            chain=self.chain,
            chain_input=chain_input,
            outputs={"final": "Final response text that should stay visible."},
            steps=[
                PromptChainStepRun(
                    step=final_step,
                    status="success",
                    outcome=outcome,
                    error=None,
                )
            ],
            summary="Calm final summary",
        )


def _make_chain() -> PromptChain:
    chain_id = uuid.uuid4()
    step = PromptChainStep(
        id=uuid.uuid4(),
        chain_id=chain_id,
        prompt_id=uuid.uuid4(),
        order_index=1,
        input_template="",
        output_variable="final",
    )
    return PromptChain(
        id=chain_id,
        name="Demo Chain",
        description="Chain for CLI tests",
        steps=[step],
        summarize_last_response=True,
    )


def test_prompt_chain_show_lists_step_labels_with_failure_behavior(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Chain show output should read as a calm structure summary."""

    manager = _ChainCliManagerStub()
    logger = logging.getLogger("test_prompt_chain_show")
    args = argparse.Namespace(chain_id=str(manager.chain.id))

    exit_code = run_prompt_chain_show(manager, args, logger)

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Chain: Demo Chain" in output
    assert "Steps:" in output
    assert "1. final" in output
    assert "Prompt:" in output
    assert "Failure:" in output
    assert "Stop chain on failure" in output


def test_prompt_chain_run_surfaces_final_chain_result_label(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Chain run output should make the terminal result explicit."""

    manager = _ChainCliManagerStub()
    logger = logging.getLogger("test_prompt_chain_run")
    args = argparse.Namespace(
        chain_id=str(manager.chain.id),
        chain_input="CLI input text",
        chain_input_file=None,
        no_web_search=False,
    )

    exit_code = run_prompt_chain_run(manager, args, logger)

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Chain 'Demo Chain'" in output
    assert "Input to chain:" in output
    assert "Final chain result:" in output
    assert "Calm final summary" in output
    assert "Chain outputs:" in output
    assert "Step summary:" in output
