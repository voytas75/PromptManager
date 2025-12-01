"""Tests for category hint matching helper."""
from __future__ import annotations

from gui.main_window import _match_category_label
from models.category_model import PromptCategory


def _category(slug: str, label: str) -> PromptCategory:
    return PromptCategory(slug=slug, label=label, description=f"{label} prompts")


def test_match_category_label_exact_label() -> None:
    categories = [_category("documentation", "Documentation")]
    assert _match_category_label("Documentation", categories) == "Documentation"


def test_match_category_label_uses_slug_and_partial_matches() -> None:
    categories = [_category("reasoning-debugging", "Reasoning / Debugging")]
    assert _match_category_label("Reasoning Debugging", categories) == "Reasoning / Debugging"
    assert _match_category_label("debugging", categories) == "Reasoning / Debugging"


def test_match_category_label_returns_none_when_unknown() -> None:
    categories = [_category("refactoring", "Refactoring")]
    assert _match_category_label("Observability", categories) is None
