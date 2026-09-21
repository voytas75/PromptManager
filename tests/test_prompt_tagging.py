"""Tests for deterministic prompt tag catalogue and mutations."""

from __future__ import annotations

import uuid

import pytest

from core.prompt_tagging import (
    apply_prompt_tag,
    build_tag_catalog,
    find_tagged_prompts,
    normalize_tag,
)
from models.prompt_model import Prompt


def _prompt(name: str, tags: list[str], *, active: bool = True) -> Prompt:
    return Prompt(
        id=uuid.uuid4(),
        name=name,
        description="Description",
        category="Test",
        tags=tags,
        is_active=active,
    )


def test_build_tag_catalog_deduplicates_within_prompt_and_sorts_by_count() -> None:
    records = build_tag_catalog(
        [
            _prompt("One", ["Ops", " ops ", "CI"]),
            _prompt("Two", ["ops", "Review"], active=False),
            _prompt("Three", ["CI"]),
        ]
    )

    actual = [(record.tag, record.prompt_count, record.active_prompt_count) for record in records]
    assert actual == [
        ("CI", 2, 2),
        ("Ops", 2, 1),
        ("Review", 1, 0),
    ]


def test_find_and_apply_prompt_tag_are_case_insensitive_and_idempotent() -> None:
    prompt = _prompt("Alpha", ["Ops"])
    other = _prompt("Beta", ["review"])

    assert find_tagged_prompts([other, prompt], " OPS ") == (prompt,)
    assert apply_prompt_tag(prompt, "add", "ops") is False
    assert prompt.tags == ["Ops"]
    assert apply_prompt_tag(prompt, "add", "CI") is True
    assert prompt.tags == ["Ops", "CI"]
    assert apply_prompt_tag(prompt, "remove", " ops ") is True
    assert prompt.tags == ["CI"]
    assert apply_prompt_tag(prompt, "remove", "ops") is False


def test_normalize_tag_rejects_blank_values() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        normalize_tag("  ")
