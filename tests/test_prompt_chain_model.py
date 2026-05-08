"""Prompt chain model helper tests.

Updates:
  v0.3.0 - 2025-12-06 - Align payload parsing tests with plain-text chaining.
  v0.2.0 - 2025-12-05 - Validate summarize-last-response flag parsing.
  v0.1.0 - 2025-12-04 - Cover chain_from_payload parsing and validation cases.
"""

from __future__ import annotations

import uuid

import pytest

from models.prompt_chain_model import PromptChainStep, chain_from_payload


def test_chain_from_payload_creates_ordered_steps() -> None:
    """Ensure the helper constructs chains with ordered steps and metadata."""

    prompt_a = uuid.uuid4()
    prompt_b = uuid.uuid4()
    payload = {
        "name": "Demo",
        "description": "Example chain",
        "is_active": True,
        "steps": [
            {
                "prompt_id": str(prompt_b),
                "order_index": 2,
                "stop_on_failure": False,
            },
            {
                "prompt_id": str(prompt_a),
                "order_index": 1,
            },
        ],
    }

    chain = chain_from_payload(payload)

    assert chain.name == "Demo"
    assert len(chain.steps) == 2
    assert chain.steps[0].order_index == 1
    assert chain.steps[0].prompt_id == prompt_a
    assert chain.steps[1].order_index == 2
    assert chain.steps[1].stop_on_failure is False


def test_chain_from_payload_requires_name() -> None:
    """Missing chain name raises ``ValueError`` for clarity."""

    with pytest.raises(ValueError):
        chain_from_payload({"steps": []})


def test_chain_from_payload_allows_summary_toggle() -> None:
    """Summarize flag should be read from JSON payloads when provided."""

    prompt_id = uuid.uuid4()
    payload = {
        "id": str(uuid.uuid4()),
        "name": "Summaries",
        "description": "",
        "summarize_last_response": False,
        "steps": [{"prompt_id": str(prompt_id)}],
    }

    chain = chain_from_payload(payload)

    assert chain.summarize_last_response is False
    assert chain.steps[0].output_variable == "step_1"


def test_prompt_chain_step_runtime_metadata_defaults_are_derived() -> None:
    """Runtime labels should be derived, not treated as stored runtime semantics."""

    chain_id = uuid.uuid4()
    prompt_id = uuid.uuid4()
    step = PromptChainStep(
        id=uuid.uuid4(),
        chain_id=chain_id,
        prompt_id=prompt_id,
        order_index=3,
        stop_on_failure=False,
    )

    assert step.input_template == ""
    assert step.output_variable == "step_3"
    assert step.condition is None


def test_chain_from_payload_ignores_legacy_runtime_step_fields() -> None:
    """Legacy import fields may be accepted without driving active runtime semantics."""

    prompt_id = uuid.uuid4()
    payload = {
        "name": "Legacy import",
        "steps": [
            {
                "prompt_id": str(prompt_id),
                "order_index": 2,
                "input_template": "{{ old }}",
                "output_variable": "custom_name",
                "condition": "value == 1",
            }
        ],
    }

    chain = chain_from_payload(payload)

    step = chain.steps[0]
    assert step.order_index == 2
    assert step.input_template == ""
    assert step.output_variable == "step_2"
    assert step.condition is None


def test_chain_from_payload_preserves_legacy_fields_only_as_inactive_metadata() -> None:
    """Legacy step fields should survive only as compatibility metadata."""

    prompt_id = uuid.uuid4()
    chain = chain_from_payload(
        {
            "name": "Legacy metadata",
            "steps": [
                {
                    "prompt_id": str(prompt_id),
                    "input_template": "{{legacy_input}}",
                    "output_variable": "legacy_alias",
                    "condition": "legacy_condition",
                }
            ],
        }
    )

    step = chain.steps[0]
    assert step.metadata is not None
    assert step.metadata["legacy_runtime_fields"] == {
        "input_template": "{{legacy_input}}",
        "output_variable": "legacy_alias",
        "condition": "legacy_condition",
    }
    assert step.metadata["legacy_runtime_fields_status"] == "inactive"
    assert step.metadata["legacy_runtime_fields_note"] == (
        "Compatibility-only fields preserved for import/export boundaries. "
        "They do not affect the active linear runner."
    )


def test_prompt_chain_storage_fields_do_not_imply_active_runtime_semantics() -> None:
    """Persisted legacy-shaped fields should expose inactive semantics cues."""

    chain = chain_from_payload(
        {
            "name": "Legacy chain",
            "variables_schema": {"customer": {"type": "string"}},
            "steps": [{"prompt_id": str(uuid.uuid4())}],
        }
    )

    assert chain.variables_schema == {"customer": {"type": "string"}}
    assert chain.metadata is not None
    assert chain.metadata["legacy_chain_fields"] == {
        "variables_schema": {
            "status": "inactive",
            "note": "Compatibility-only field; not used by the active linear runner.",
        }
    }


def test_prompt_chain_storage_roundtrip_marks_legacy_shape_as_inactive_metadata() -> None:
    """Row hydration should keep stored legacy-shaped fields while flagging them as inactive."""

    chain_id = uuid.uuid4()
    chain = chain_from_payload(
        {
            "id": str(chain_id),
            "name": "Roundtrip chain",
            "description": "",
            "variables_schema": {"customer": {"type": "string"}},
            "steps": [
                {
                    "id": str(uuid.uuid4()),
                    "prompt_id": str(uuid.uuid4()),
                    "order_index": 2,
                    "input_template": "{{legacy_input}}",
                    "output_variable": "legacy_alias",
                    "condition": "legacy_condition",
                }
            ],
        }
    )

    record = chain.to_record()
    hydrated = type(chain).from_row(record, steps=chain.steps)

    assert hydrated.variables_schema == {"customer": {"type": "string"}}
    assert hydrated.metadata is not None
    assert hydrated.metadata["legacy_chain_fields"] == {
        "variables_schema": {
            "status": "inactive",
            "note": "Compatibility-only field; not used by the active linear runner.",
        }
    }
    assert hydrated.steps[0].metadata is not None
    assert hydrated.steps[0].metadata["legacy_runtime_fields_status"] == "inactive"
    assert hydrated.steps[0].metadata["legacy_runtime_fields_note"] == (
        "Compatibility-only fields preserved for import/export boundaries. "
        "They do not affect the active linear runner."
    )
