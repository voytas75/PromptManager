"""Unit tests for analytics mixin helpers."""

from __future__ import annotations

from typing import Any, cast

from core.prompt_manager.analytics import AnalyticsMixin, CategoryHealth
from models.category_model import PromptCategory


class _RegistryStub:
    def __init__(self, categories: list[PromptCategory]) -> None:
        self._categories = categories

    def all(self, *, include_archived: bool = False) -> list[PromptCategory]:  # noqa: FBT001
        return list(self._categories)


class _RepositoryStub:
    def get_category_prompt_counts(self) -> dict[str, dict[str, int]]:
        return {
            "general": {"total_prompts": 2, "active_prompts": 2},
            "archived": {"total_prompts": 1, "active_prompts": 0},
        }

    def get_category_execution_statistics(self) -> dict[str, dict[str, Any]]:
        return {
            "general": {
                "total_runs": 10,
                "success_runs": 8,
                "last_executed_at": "2025-12-02T12:30:00+00:00",
            },
            "": {"total_runs": 0, "success_runs": 0},
        }


class _AnalyticsHarness(AnalyticsMixin):
    def __init__(self) -> None:
        categories = [
            PromptCategory(slug="general", label="General", description=""),
            PromptCategory(slug="archived", label="Archived", description="", is_active=False),
        ]
        self._repository = cast("Any", _RepositoryStub())
        self._category_registry = cast("Any", _RegistryStub(categories))
        self._embedding_provider = cast("Any", object())


def test_get_category_health_merges_counts_and_stats() -> None:
    harness = _AnalyticsHarness()

    result = harness.get_category_health()

    assert isinstance(result, list)
    general = next(item for item in result if item.slug == "general")
    assert isinstance(general, CategoryHealth)
    assert general.total_prompts == 2
    assert general.active_prompts == 2
    assert general.success_rate == 0.8
    archived = next(item for item in result if item.slug == "archived")
    assert archived.label == "Archived"
    assert archived.success_rate is None
