"""Tests for effective built-in workflow prompt template listing."""

from __future__ import annotations

from core.prompt_template_listing import list_effective_prompt_templates
from prompt_templates import DEFAULT_PROMPT_TEMPLATES, PROMPT_TEMPLATE_KEYS


def test_list_effective_prompt_templates_preserves_canonical_order_and_defaults() -> None:
    """Expose every known template exactly once in the source-defined order."""
    records = list_effective_prompt_templates()

    assert [record.key for record in records] == list(PROMPT_TEMPLATE_KEYS)
    assert [record.source for record in records] == ["default"] * len(PROMPT_TEMPLATE_KEYS)
    assert [record.text for record in records] == [
        DEFAULT_PROMPT_TEMPLATES[key] for key in PROMPT_TEMPLATE_KEYS
    ]
    assert all(record.characters == len(record.text) for record in records)
    assert all(record.lines >= 1 for record in records)


def test_list_effective_prompt_templates_applies_only_known_nonempty_overrides() -> None:
    """Report effective provenance without creating a second override policy."""
    records = list_effective_prompt_templates(
        {
            "name_generation": "Use compact safe names.",
            "scenario_generation": "  ",
            "unknown": "Ignored.",
        }
    )
    by_key = {record.key: record for record in records}

    assert by_key["name_generation"].source == "override"
    assert by_key["name_generation"].text == "Use compact safe names."
    assert by_key["scenario_generation"].source == "default"
    assert by_key["scenario_generation"].text == DEFAULT_PROMPT_TEMPLATES["scenario_generation"]
    assert set(by_key) == set(PROMPT_TEMPLATE_KEYS)
