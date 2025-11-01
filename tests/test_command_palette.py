"""Tests for command palette utilities."""

from __future__ import annotations

import uuid

import pytest

pytest.importorskip("PySide6")

from gui.command_palette import QuickAction, rank_prompts_for_action
from models.prompt_model import Prompt


def _make_prompt(name: str, category: str, tags: list[str]) -> Prompt:
    return Prompt(
        id=uuid.uuid4(),
        name=name,
        description=f"Prompt for {name}",
        category=category,
        tags=tags,
        context="",
    )


def test_rank_prompts_prioritises_category_and_tags() -> None:
    prompts = [
        _make_prompt("General Helper", "General", ["misc"]),
        _make_prompt("Debug Trace Investigator", "Reasoning / Debugging", ["debugging", "incident-response"]),
        _make_prompt("Refactor Navigator", "Refactoring", ["refactor"]),
    ]

    action = QuickAction(
        identifier="fix-errors",
        title="Fix Errors",
        description="",
        category_hint="Reasoning / Debugging",
        tag_hints=("debugging",),
    )

    ranked = rank_prompts_for_action(prompts, action)
    assert ranked
    assert ranked[0].name == "Debug Trace Investigator"


def test_rank_prompts_empty_when_no_match() -> None:
    prompts = [_make_prompt("Feature Booster", "Enhancement", ["enhancement"])]
    action = QuickAction(
        identifier="docs",
        title="Add Comments",
        description="",
        category_hint="Documentation",
        tag_hints=("documentation",),
    )
    assert rank_prompts_for_action(prompts, action) == []
