"""Focused tests for bounded prompt-list filtering helpers.

Updates:
  v0.1.0 - 2026-04-12 - Cover favorites-only filtering without widening list behavior.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import cast

from gui.prompt_list_coordinator import PromptListCoordinator
from models.prompt_model import Prompt


class _ManagerStub:
    def list_categories(self) -> list[object]:
        return []


class _FilterPanelStub:
    def __init__(
        self,
        *,
        category_slug: str | None = None,
        tag_value: str | None = None,
        favorites_only: bool = False,
        min_quality: float = 0.0,
    ) -> None:
        self._category_slug = category_slug
        self._tag_value = tag_value
        self._favorites_only = favorites_only
        self._min_quality = min_quality

    def category_slug(self) -> str | None:
        return self._category_slug

    def tag_value(self) -> str | None:
        return self._tag_value

    def favorites_only(self) -> bool:
        return self._favorites_only

    def min_quality(self) -> float:
        return self._min_quality


def _prompt(name: str, *, is_favorite: bool) -> Prompt:
    return Prompt(
        id=uuid.uuid4(),
        name=name,
        description=f"{name} description",
        category="General",
        context=f"{name} body",
        is_favorite=is_favorite,
        created_at=datetime(2026, 4, 12, 12, 0, tzinfo=UTC),
        last_modified=datetime(2026, 4, 12, 12, 0, tzinfo=UTC),
    )


def test_apply_filters_keeps_only_favorite_prompts_when_requested() -> None:
    """Favorites-only filtering should exclude non-favorite prompts from the list."""
    coordinator = PromptListCoordinator(cast("object", _ManagerStub()))
    prompts = [_prompt("Alpha", is_favorite=True), _prompt("Beta", is_favorite=False)]
    panel = _FilterPanelStub(favorites_only=True)

    filtered = coordinator.apply_filters(cast("object", panel), prompts)

    assert [prompt.name for prompt in filtered] == ["Alpha"]


def test_apply_filters_leaves_existing_results_unchanged_when_favorites_disabled() -> None:
    """The added favorite filter should stay inert when the checkbox is off."""
    coordinator = PromptListCoordinator(cast("object", _ManagerStub()))
    prompts = [_prompt("Alpha", is_favorite=True), _prompt("Beta", is_favorite=False)]
    panel = _FilterPanelStub(favorites_only=False)

    filtered = coordinator.apply_filters(cast("object", panel), prompts)

    assert [prompt.name for prompt in filtered] == ["Alpha", "Beta"]
