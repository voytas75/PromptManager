"""Prompt chain CLI legibility tests.

Updates:
  v0.1.0 - 2026-05-04 - Cover bounded prompt-chain show/run output clarity cues.
"""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, cast

from cli.commands import (
    run_prompt_chain_apply,
    run_prompt_chain_export,
    run_prompt_chain_run,
    run_prompt_chain_show,
    run_prompt_chain_validate,
)
from core.execution import CodexExecutionResult
from core.prompt_manager.execution_history import ExecutionOutcome
from models.prompt_chain_model import PromptChain, PromptChainStep

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class _ChainCliManagerStub:
    def __init__(self) -> None:
        self.chain = _make_chain()
        self.saved_chains: list[PromptChain] = []
        self.run_mode = "success"

    def get_prompt_chain(self, chain_id: uuid.UUID) -> PromptChain:
        if chain_id != self.chain.id:
            raise ValueError("Chain not found")
        return self.chain

    def save_prompt_chain(self, chain: PromptChain) -> PromptChain:
        self.saved_chains.append(chain)
        self.chain = chain
        return chain

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
        from core import PromptChainRunResult, PromptChainStepRun

        if self.run_mode == "partial_success":
            previous_step = PromptChainStep(
                id=self.chain.steps[0].id,
                chain_id=self.chain.id,
                prompt_id=uuid.uuid4(),
                order_index=1,
                input_template="",
                output_variable="draft",
            )
            return PromptChainRunResult(
                chain=self.chain,
                chain_input=chain_input,
                step_outputs={"step_1": "Draft response that remains final."},
                final_output_text="Draft response that remains final.",
                final_summary_text="Draft summary",
                step_aliases={"draft": "step_1"},
                run_status="partial_success",
                final_step_id=previous_step.id,
                final_step_output_key="step_1",
                final_step_label="draft",
                terminal_step_id=final_step.id,
                terminal_step_output_key="step_2",
                terminal_step_label="final",
                terminal_step_status="failed",
                steps=[
                    PromptChainStepRun(
                        step=previous_step,
                        status="success",
                        outcome=None,
                        prompt_name="Draft Prompt",
                        request_text="Initial input",
                        response_text="Draft response that remains final.",
                        duration_ms=9,
                        web_search_requested=False,
                        web_search_applied=False,
                        skip_reason=None,
                        error=None,
                        step_label="draft",
                        step_output_key="step_1",
                    ),
                    PromptChainStepRun(
                        step=final_step,
                        status="failed",
                        outcome=None,
                        prompt_name="Final Prompt",
                        request_text="Draft response that remains final.",
                        response_text=None,
                        duration_ms=None,
                        web_search_requested=False,
                        web_search_applied=False,
                        skip_reason=None,
                        error="terminal failure",
                        step_label="final",
                        step_output_key="step_2",
                    ),
                ],
            )

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

        return PromptChainRunResult(
            chain=self.chain,
            chain_input=chain_input,
            step_outputs={"step_1": "Final response text that should stay visible."},
            final_output_text="Final response text that should stay visible.",
            final_summary_text="Calm final summary",
            step_aliases={"final": "step_1"},
            run_status="success",
            final_step_id=final_step.id,
            final_step_output_key="step_1",
            final_step_label="final",
            terminal_step_id=final_step.id,
            terminal_step_output_key="step_1",
            terminal_step_label="final",
            terminal_step_status="success",
            steps=[
                PromptChainStepRun(
                    step=final_step,
                    status="success",
                    outcome=outcome,
                    prompt_name="Final Prompt",
                    request_text="Previous output\n\n[Search context attached]",
                    response_text="Final response text that should stay visible.",
                    duration_ms=12,
                    web_search_requested=True,
                    web_search_applied=True,
                    skip_reason=None,
                    error=None,
                )
            ],
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
        metadata={
            "legacy_runtime_fields": {
                "input_template": "{{legacy_input}}",
                "output_variable": "final",
                "condition": None,
            },
            "legacy_runtime_fields_status": "inactive",
            "legacy_runtime_fields_note": (
                "Compatibility-only fields preserved for import/export boundaries. "
                "They do not affect the active linear runner."
            ),
        },
    )
    return PromptChain(
        id=chain_id,
        name="Demo Chain",
        description="Chain for CLI tests",
        steps=[step],
        summarize_last_response=True,
        metadata={
            "legacy_chain_fields": {
                "variables_schema": {
                    "status": "inactive",
                    "note": "Compatibility-only field; not used by the active linear runner.",
                }
            }
        },
    )


def test_prompt_chain_show_lists_step_labels_with_failure_behavior(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Chain show output should read as a calm structure summary."""

    manager = _ChainCliManagerStub()
    logger = logging.getLogger("test_prompt_chain_show")
    args = argparse.Namespace(chain_id=str(manager.chain.id), json=False)

    exit_code = run_prompt_chain_show(
        cast("Any", manager),
        args,
        logger,
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Chain: Demo Chain" in output
    assert "Steps:" in output
    assert "1. final" in output
    assert "Prompt:" in output
    assert "Failure:" in output
    assert "Stop chain on failure" in output


def test_prompt_chain_show_supports_json_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Chain show JSON output should be deterministic and machine-readable."""

    manager = _ChainCliManagerStub()
    logger = logging.getLogger("test_prompt_chain_show_json")
    args = argparse.Namespace(chain_id=str(manager.chain.id), json=True)

    exit_code = run_prompt_chain_show(
        cast("Any", manager),
        args,
        logger,
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["id"] == str(manager.chain.id)
    assert payload["name"] == "Demo Chain"
    assert payload["description"] == "Chain for CLI tests"
    assert payload["status"] == "active"
    assert payload["is_active"] is True
    assert payload["summarize_last_response"] is True
    assert payload["legacy_semantics"] == {
        "variables_schema": {
            "status": "inactive",
            "note": "Compatibility-only field; not used by the active linear runner.",
        }
    }
    assert payload["steps"] == [
        {
            "id": str(manager.chain.steps[0].id),
            "order_index": 1,
            "prompt_id": str(manager.chain.steps[0].prompt_id),
            "step_label": "final",
            "stop_on_failure": True,
            "legacy_semantics": {
                "input_template": {
                    "status": "inactive",
                    "note": "Compatibility-only field preserved for import/export boundaries.",
                },
                "condition": {
                    "status": "inactive",
                    "note": "Compatibility-only field preserved for import/export boundaries.",
                },
                "output_variable": {
                    "status": "inactive_alias",
                    "note": (
                        "Legacy alias field preserved for import/export boundaries; "
                        "active runtime uses canonical step_output_key."
                    ),
                },
            },
        }
    ]



def test_prompt_chain_show_text_surfaces_inactive_legacy_semantics(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Human-readable show output should flag inactive compatibility fields."""

    manager = _ChainCliManagerStub()
    logger = logging.getLogger("test_prompt_chain_show_legacy_semantics")
    args = argparse.Namespace(chain_id=str(manager.chain.id), json=False)

    exit_code = run_prompt_chain_show(
        cast("Any", manager),
        args,
        logger,
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Legacy semantics:" in output
    assert "variables_schema: inactive compatibility-only field" in output
    assert (
        "Legacy fields: input_template, output_variable, condition "
        "(inactive semantics)" in output
    )


def test_prompt_chain_export_writes_deterministic_json(tmp_path: Path) -> None:
    """Chain export should persist a deterministic JSON payload for one selected chain."""

    manager = _ChainCliManagerStub()
    logger = logging.getLogger("test_prompt_chain_export")
    export_path = tmp_path / "chain-export.json"
    args = argparse.Namespace(chain_id=str(manager.chain.id), path=export_path)

    exit_code = run_prompt_chain_export(
        cast("Any", manager),
        args,
        logger,
    )

    assert exit_code == 0
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    assert payload == {
        "id": str(manager.chain.id),
        "name": "Demo Chain",
        "description": "Chain for CLI tests",
        "is_active": True,
        "summarize_last_response": True,
        "steps": [
            {
                "id": str(manager.chain.steps[0].id),
                "prompt_id": str(manager.chain.steps[0].prompt_id),
                "order_index": 1,
                "stop_on_failure": True,
            }
        ],
    }


def test_prompt_chain_validate_accepts_valid_definition(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Validation should confirm a bounded valid chain definition without writing."""

    payload = {
        "name": "Validate me",
        "steps": [{"prompt_id": str(uuid.uuid4())}],
    }
    path = tmp_path / "valid-chain.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    logger = logging.getLogger("test_prompt_chain_validate_ok")
    args = argparse.Namespace(path=path, json=False)

    exit_code = run_prompt_chain_validate(None, args, logger)

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Valid prompt chain definition" in output
    assert str(path) in output



def test_prompt_chain_validate_supports_json_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Validation JSON output should be deterministic and bounded."""

    payload = {
        "name": "Validate me",
        "variables_schema": {"customer": {"type": "string"}},
        "steps": [
            {
                "prompt_id": str(uuid.uuid4()),
                "input_template": "{{legacy_input}}",
                "condition": "legacy_condition",
                "output_variable": "legacy_alias",
            }
        ],
    }
    path = tmp_path / "valid-chain.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    logger = logging.getLogger("test_prompt_chain_validate_json")
    args = argparse.Namespace(path=path, json=True)

    exit_code = run_prompt_chain_validate(None, args, logger)

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report == {
        "valid": True,
        "warnings": [
            "Chain uses compatibility-only legacy field: variables_schema",
            "Step 1 uses inactive legacy field: input_template",
            "Step 1 uses inactive legacy alias field: output_variable",
            "Step 1 uses inactive legacy field: condition",
        ],
        "step_count": 1,
        "legacy_fields_detected": {
            "chain": ["variables_schema"],
            "steps": {
                "1": ["input_template", "output_variable", "condition"]
            },
        },
    }


def test_prompt_chain_validate_rejects_empty_step_list(
    tmp_path: Path,
) -> None:
    """Validation should fail when active semantic requirements are not met."""

    payload = {
        "name": "Invalid chain",
        "steps": [],
    }
    path = tmp_path / "invalid-chain.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    logger = logging.getLogger("test_prompt_chain_validate_bad")
    args = argparse.Namespace(path=path, json=False)

    exit_code = run_prompt_chain_validate(None, args, logger)

    assert exit_code == 5



def test_prompt_chain_validate_json_reports_invalid_payload_shape(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Validation JSON output should keep a stable shape for invalid payloads."""

    payload = {
        "name": "Invalid chain",
        "steps": [],
    }
    path = tmp_path / "invalid-chain.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    logger = logging.getLogger("test_prompt_chain_validate_bad_json")
    args = argparse.Namespace(path=path, json=True)

    exit_code = run_prompt_chain_validate(None, args, logger)

    assert exit_code == 5
    report = json.loads(capsys.readouterr().out)
    assert report == {
        "valid": False,
        "warnings": [],
        "step_count": 0,
        "legacy_fields_detected": {"chain": [], "steps": {}},
        "error": "Prompt chain definition must include at least one step.",
    }



def test_prompt_chain_apply_dry_run_does_not_persist_chain(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Dry-run apply should preview without writing to the manager."""

    manager = _ChainCliManagerStub()
    payload = {
        "name": "Preview only",
        "variables_schema": {"customer": {"type": "string"}},
        "steps": [
            {
                "prompt_id": str(uuid.uuid4()),
                "input_template": "{{legacy_input}}",
                "condition": "legacy_condition",
                "output_variable": "legacy_alias",
            }
        ],
    }
    path = tmp_path / "dry-run-chain.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    logger = logging.getLogger("test_prompt_chain_apply_dry_run")
    args = argparse.Namespace(path=path, dry_run=True)

    exit_code = run_prompt_chain_apply(cast("Any", manager), args, logger)

    assert exit_code == 0
    assert manager.saved_chains == []
    preview = json.loads(capsys.readouterr().out)
    assert preview == {
        "dry_run": True,
        "valid": True,
        "mode": "create",
        "name": "Preview only",
        "step_count": 1,
        "warnings": [
            "Chain uses compatibility-only legacy field: variables_schema",
            "Step 1 uses inactive legacy field: input_template",
            "Step 1 uses inactive legacy alias field: output_variable",
            "Step 1 uses inactive legacy field: condition",
        ],
        "legacy_fields_detected": {
            "chain": ["variables_schema"],
            "steps": {
                "1": ["input_template", "output_variable", "condition"]
            },
        },
    }


def test_prompt_chain_run_supports_json_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Chain run JSON output should be deterministic and machine-readable."""

    manager = _ChainCliManagerStub()
    logger = logging.getLogger("test_prompt_chain_run_json")
    args = argparse.Namespace(
        chain_id=str(manager.chain.id),
        chain_input="CLI input text",
        chain_input_file=None,
        no_web_search=False,
        json=True,
        output_file=None,
    )

    exit_code = run_prompt_chain_run(
        cast("Any", manager),
        args,
        logger,
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
            "chain": {
                "id": str(manager.chain.id),
                "name": "Demo Chain",
            },
            "chain_input": "CLI input text",
            "final_output_text": "Final response text that should stay visible.",
            "final_summary_text": "Calm final summary",
            "run_status": "success",
            "final_step_id": str(manager.chain.steps[0].id),
            "final_step_output_key": "step_1",
            "final_step_label": "final",
            "terminal_step_id": str(manager.chain.steps[0].id),
            "terminal_step_output_key": "step_1",
            "terminal_step_label": "final",
            "terminal_step_status": "success",
            "step_aliases": {"final": "step_1"},
            "step_outputs": {"step_1": "Final response text that should stay visible."},
            "steps": [
                {
                    "step_id": str(manager.chain.steps[0].id),
                    "order_index": 1,
                    "prompt_id": str(manager.chain.steps[0].prompt_id),
                    "step_label": "final",
                    "step_output_key": "step_1",
                    "output_variable": "final",
                    "prompt_name": "Final Prompt",
                    "status": "success",
                    "duration_ms": 12,
                    "error": None,
                    "request_text": "Previous output\n\n[Search context attached]",
                    "response_text": "Final response text that should stay visible.",
                    "web_search_requested": True,
                    "web_search_applied": True,
                    "skip_reason": None,
                }
            ],
        }


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
        json=False,
        output_file=None,
        final_output_only=False,
        summary_only=False,
        status_only=False,
    )

    exit_code = run_prompt_chain_run(
        cast("Any", manager),
        args,
        logger,
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Chain 'Demo Chain'" in output
    assert "Input to chain:" in output
    assert "Final output:" in output
    assert "Final response text that should stay visible." in output
    assert "Final summary:" in output
    assert "Calm final summary" in output
    assert "Step outputs:" in output
    assert "Step summary:" in output
    assert "Step 1 (final | output key: step_1): success" in output
    assert "Prompt: Final Prompt" in output
    assert "Duration: 12 ms" in output
    assert "Web search: requested=yes, applied=yes" in output
    assert "Request text:" in output


def test_prompt_chain_run_supports_final_output_only_mode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    manager = _ChainCliManagerStub()
    logger = logging.getLogger("test_prompt_chain_run_final_output_only")
    args = argparse.Namespace(
        chain_id=str(manager.chain.id),
        chain_input="CLI input text",
        chain_input_file=None,
        no_web_search=False,
        json=False,
        output_file=None,
        final_output_only=True,
        summary_only=False,
        status_only=False,
    )

    exit_code = run_prompt_chain_run(cast("Any", manager), args, logger)

    assert exit_code == 0
    output = capsys.readouterr().out
    assert output == "Final response text that should stay visible.\n"


def test_prompt_chain_run_supports_output_file_for_text_mode(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manager = _ChainCliManagerStub()
    logger = logging.getLogger("test_prompt_chain_run_output_file_text")
    output_path = tmp_path / "chain-run.txt"
    args = argparse.Namespace(
        chain_id=str(manager.chain.id),
        chain_input="CLI input text",
        chain_input_file=None,
        no_web_search=False,
        json=False,
        output_file=output_path,
        final_output_only=False,
        summary_only=False,
        status_only=False,
    )

    exit_code = run_prompt_chain_run(cast("Any", manager), args, logger)

    assert exit_code == 0
    saved = output_path.read_text(encoding="utf-8")
    assert "Chain 'Demo Chain'" in saved
    assert "Final output:" in saved
    assert "Calm final summary" in saved
    assert output_path.as_posix() in capsys.readouterr().out


def test_prompt_chain_run_supports_output_file_for_json_mode(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manager = _ChainCliManagerStub()
    logger = logging.getLogger("test_prompt_chain_run_output_file_json")
    output_path = tmp_path / "chain-run.json"
    args = argparse.Namespace(
        chain_id=str(manager.chain.id),
        chain_input="CLI input text",
        chain_input_file=None,
        no_web_search=False,
        json=True,
        output_file=output_path,
    )

    exit_code = run_prompt_chain_run(cast("Any", manager), args, logger)

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["chain"]["id"] == str(manager.chain.id)
    assert payload["final_step_output_key"] == "step_1"
    assert payload["step_aliases"] == {"final": "step_1"}
    assert output_path.as_posix() in capsys.readouterr().out



def test_prompt_chain_run_supports_summary_only_mode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    manager = _ChainCliManagerStub()
    logger = logging.getLogger("test_prompt_chain_run_summary_only")
    args = argparse.Namespace(
        chain_id=str(manager.chain.id),
        chain_input="CLI input text",
        chain_input_file=None,
        no_web_search=False,
        json=False,
        output_file=None,
        final_output_only=False,
        summary_only=True,
        status_only=False,
    )

    exit_code = run_prompt_chain_run(cast("Any", manager), args, logger)

    assert exit_code == 0
    output = capsys.readouterr().out
    assert output == "Calm final summary\n"



def test_prompt_chain_run_supports_status_only_mode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    manager = _ChainCliManagerStub()
    logger = logging.getLogger("test_prompt_chain_run_status_only")
    args = argparse.Namespace(
        chain_id=str(manager.chain.id),
        chain_input="CLI input text",
        chain_input_file=None,
        no_web_search=False,
        json=False,
        output_file=None,
        final_output_only=False,
        summary_only=False,
        status_only=True,
        step_output=None,
        step_alias=None,
        final_step_meta=False,
        compact=False,
    )

    exit_code = run_prompt_chain_run(cast("Any", manager), args, logger)

    assert exit_code == 0
    output = capsys.readouterr().out
    assert output == "success\n"



def test_prompt_chain_run_status_only_uses_backend_run_status(
    capsys: pytest.CaptureFixture[str],
) -> None:
    manager = _ChainCliManagerStub()
    manager.run_mode = "partial_success"
    logger = logging.getLogger("test_prompt_chain_run_status_only_backend_status")
    args = argparse.Namespace(
        chain_id=str(manager.chain.id),
        chain_input="CLI input text",
        chain_input_file=None,
        no_web_search=False,
        json=False,
        output_file=None,
        final_output_only=False,
        summary_only=False,
        status_only=True,
        step_output=None,
        step_alias=None,
        final_step_meta=False,
        compact=False,
    )

    exit_code = run_prompt_chain_run(cast("Any", manager), args, logger)

    assert exit_code == 0
    assert capsys.readouterr().out == "partial_success\n"



def test_prompt_chain_run_supports_step_output_mode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    manager = _ChainCliManagerStub()
    logger = logging.getLogger("test_prompt_chain_run_step_output")
    args = argparse.Namespace(
        chain_id=str(manager.chain.id),
        chain_input="CLI input text",
        chain_input_file=None,
        no_web_search=False,
        json=False,
        output_file=None,
        final_output_only=False,
        summary_only=False,
        status_only=False,
        step_output="step_1",
        step_alias=None,
        final_step_meta=False,
        compact=False,
    )

    exit_code = run_prompt_chain_run(cast("Any", manager), args, logger)

    assert exit_code == 0
    assert capsys.readouterr().out == "Final response text that should stay visible.\n"



def test_prompt_chain_run_supports_step_alias_mode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    manager = _ChainCliManagerStub()
    logger = logging.getLogger("test_prompt_chain_run_step_alias")
    args = argparse.Namespace(
        chain_id=str(manager.chain.id),
        chain_input="CLI input text",
        chain_input_file=None,
        no_web_search=False,
        json=False,
        output_file=None,
        final_output_only=False,
        summary_only=False,
        status_only=False,
        step_output=None,
        step_alias="final",
        final_step_meta=False,
        compact=False,
    )

    exit_code = run_prompt_chain_run(cast("Any", manager), args, logger)

    assert exit_code == 0
    assert capsys.readouterr().out == "Final response text that should stay visible.\n"



def test_prompt_chain_run_supports_final_step_meta_mode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    manager = _ChainCliManagerStub()
    logger = logging.getLogger("test_prompt_chain_run_final_step_meta")
    args = argparse.Namespace(
        chain_id=str(manager.chain.id),
        chain_input="CLI input text",
        chain_input_file=None,
        no_web_search=False,
        json=False,
        output_file=None,
        final_output_only=False,
        summary_only=False,
        status_only=False,
        step_output=None,
        step_alias=None,
        final_step_meta=True,
        compact=False,
    )

    exit_code = run_prompt_chain_run(cast("Any", manager), args, logger)

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "final_step_id": str(manager.chain.steps[0].id),
        "final_step_output_key": "step_1",
        "final_step_label": "final",
        "terminal_step_id": str(manager.chain.steps[0].id),
        "terminal_step_output_key": "step_1",
        "terminal_step_label": "final",
        "terminal_step_status": "success",
        "run_status": "success",
    }



def test_prompt_chain_run_final_step_meta_keeps_final_and_terminal_semantics_separate(
    capsys: pytest.CaptureFixture[str],
) -> None:
    manager = _ChainCliManagerStub()
    manager.run_mode = "partial_success"
    logger = logging.getLogger("test_prompt_chain_run_final_step_meta_partial")
    args = argparse.Namespace(
        chain_id=str(manager.chain.id),
        chain_input="CLI input text",
        chain_input_file=None,
        no_web_search=False,
        json=False,
        output_file=None,
        final_output_only=False,
        summary_only=False,
        status_only=False,
        step_output=None,
        step_alias=None,
        final_step_meta=True,
        compact=False,
    )

    exit_code = run_prompt_chain_run(cast("Any", manager), args, logger)

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "final_step_id": str(manager.chain.steps[0].id),
        "final_step_output_key": "step_1",
        "final_step_label": "draft",
        "terminal_step_id": str(manager.chain.steps[-1].id),
        "terminal_step_output_key": "step_2",
        "terminal_step_label": "final",
        "terminal_step_status": "failed",
        "run_status": "partial_success",
    }



def test_prompt_chain_run_rejects_conflicting_selective_output_modes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    manager = _ChainCliManagerStub()
    logger = logging.getLogger("test_prompt_chain_run_conflicting_output_modes")
    args = argparse.Namespace(
        chain_id=str(manager.chain.id),
        chain_input="CLI input text",
        chain_input_file=None,
        no_web_search=False,
        json=True,
        output_file=None,
        final_output_only=False,
        summary_only=False,
        status_only=False,
        step_output="step_1",
        step_alias=None,
        final_step_meta=False,
        compact=False,
    )

    exit_code = run_prompt_chain_run(cast("Any", manager), args, logger)

    assert exit_code == 5
    assert capsys.readouterr().out == ""



def test_prompt_chain_run_supports_compact_mode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    manager = _ChainCliManagerStub()
    logger = logging.getLogger("test_prompt_chain_run_compact")
    args = argparse.Namespace(
        chain_id=str(manager.chain.id),
        chain_input="CLI input text",
        chain_input_file=None,
        no_web_search=False,
        json=False,
        output_file=None,
        final_output_only=False,
        summary_only=False,
        status_only=False,
        step_output=None,
        step_alias=None,
        final_step_meta=False,
        compact=True,
    )

    exit_code = run_prompt_chain_run(cast("Any", manager), args, logger)

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Chain: Demo Chain" in output
    assert "Status: success" in output
    assert "Final output preview: Final response text that should stay visible." in output
    assert "Summary preview: Calm final summary" in output
