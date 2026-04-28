"""Focused tests for bounded prompt-list filtering helpers.

Updates:
  v0.2.0 - 2026-04-25 - Distinguish search no-match vs search-error retrieval states.
  v0.1.0 - 2026-04-12 - Cover favorites-only filtering without widening list behavior.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import cast

from core import PromptManagerError
from gui.prompt_list_coordinator import PromptListCoordinator
from models.prompt_model import Prompt


class _ManagerStub:
    def __init__(self) -> None:
        self.repository = _RepositoryStub([])
        self.search_results: list[Prompt] = []
        self.search_error: Exception | None = None

    def list_categories(self) -> list[object]:
        return []

    def search_prompts(self, query_text: str, limit: int = 50) -> list[Prompt]:
        if self.search_error is not None:
            raise self.search_error
        return list(self.search_results[:limit])


class _RepositoryStub:
    def __init__(self, prompts: list[Prompt]) -> None:
        self._prompts = list(prompts)

    def list(self) -> list[Prompt]:
        return list(self._prompts)


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


def test_fetch_prompts_keeps_search_mode_for_no_match_results() -> None:
    """Active search with zero matches should stay distinct from the default full-catalog state."""
    manager = _ManagerStub()
    manager.repository = _RepositoryStub([_prompt("Alpha", is_favorite=True)])
    manager.search_results = []
    coordinator = PromptListCoordinator(cast("object", manager))

    result = coordinator.fetch_prompts("rollback")

    assert [prompt.name for prompt in result.all_prompts] == ["Alpha"]
    assert result.search_results == []
    assert result.preserve_search_order is True
    assert result.search_error is None


def test_fetch_prompts_records_search_errors_without_faking_empty_results() -> None:
    """Search failures should stay distinguishable from no-match search states."""
    manager = _ManagerStub()
    manager.repository = _RepositoryStub([_prompt("Alpha", is_favorite=True)])
    manager.search_error = PromptManagerError("search backend down")
    coordinator = PromptListCoordinator(cast("object", manager))

    result = coordinator.fetch_prompts("rollback")

    assert [prompt.name for prompt in result.all_prompts] == ["Alpha"]
    assert result.search_results is None
    assert result.preserve_search_order is False
    assert result.search_error == "search backend down"


def test_fetch_prompts_exposes_operator_state_for_no_match_search() -> None:
    """No-match search state should stay explicit instead of falling back to catalog wording."""
    manager = _ManagerStub()
    manager.repository = _RepositoryStub([_prompt("Alpha", is_favorite=True)])
    manager.search_results = []
    coordinator = PromptListCoordinator(cast("object", manager))

    result = coordinator.fetch_prompts("rollback")

    assert result.operator_state_label == "No matches for search"


def test_fetch_prompts_exposes_operator_state_for_search_errors() -> None:
    """Search backend failures should surface an explicit search-error trust cue."""
    manager = _ManagerStub()
    manager.repository = _RepositoryStub([_prompt("Alpha", is_favorite=True)])
    manager.search_error = PromptManagerError("search backend down")
    coordinator = PromptListCoordinator(cast("object", manager))

    result = coordinator.fetch_prompts("rollback")

    assert result.operator_state_label == "Search unavailable"


def test_fetch_prompts_exposes_operator_state_for_default_catalog() -> None:
    """Blank search should expose the ordinary catalog posture instead of a search-state cue."""
    manager = _ManagerStub()
    manager.repository = _RepositoryStub([_prompt("Alpha", is_favorite=True)])
    coordinator = PromptListCoordinator(cast("object", manager))

    result = coordinator.fetch_prompts("")

    assert result.operator_state_label == "Browsing all prompts"


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
