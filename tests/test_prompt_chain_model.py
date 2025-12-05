"""Prompt chain model helper tests.

Updates:
  v0.2.0 - 2025-12-05 - Validate summarize-last-response flag parsing.
  v0.1.0 - 2025-12-04 - Cover chain_from_payload parsing and validation cases.
"""

from __future__ import annotations

import uuid

import pytest

from models.prompt_chain_model import chain_from_payload


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
                "input_template": "{{ value }}",
                "output_variable": "second",
                "condition": "{{ first }}",
            },
            {
                "prompt_id": str(prompt_a),
                "order_index": 1,
                "input_template": "{{ source }}",
                "output_variable": "first",
            },
        ],
        "variables_schema": {"type": "object"},
    }

    chain = chain_from_payload(payload)

    assert chain.name == "Demo"
    assert len(chain.steps) == 2
    assert chain.steps[0].order_index == 1
    assert chain.steps[0].prompt_id == prompt_a
    assert chain.steps[1].order_index == 2
    assert chain.steps[1].condition == "{{ first }}"
    assert chain.variables_schema == {"type": "object"}


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
        "steps": [
            {
                "prompt_id": str(prompt_id),
                "order_index": 1,
                "input_template": "{{ body }}",
                "output_variable": "result",
            }
        ],
    }

    chain = chain_from_payload(payload)

    assert chain.summarize_last_response is False
