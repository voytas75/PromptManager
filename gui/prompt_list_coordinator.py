"""Prompt list loading, filtering, and sorting helpers for the main window.

Updates:
  v0.1.0 - 2025-12-01 - Introduce PromptListCoordinator and associated models.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from core import PromptManager, PromptManagerError
from models.category_model import slugify_category

if TYPE_CHECKING:
    from collections.abc import Sequence

    from models.prompt_model import Prompt

    from .widgets import PromptFilterPanel


@dataclass(slots=True)
class PromptLoadResult:
    """Fetched prompt data assembled from repository and optional search."""
    all_prompts: list[Prompt]
    search_results: list[Prompt] | None
    preserve_search_order: bool
    search_error: str | None


class PromptSortOrder(Enum):
    """Supported sorting orders for the prompt list view."""
    NAME_ASC = "name_asc"
    NAME_DESC = "name_desc"
    QUALITY_DESC = "quality_desc"
    MODIFIED_DESC = "modified_desc"
    CREATED_DESC = "created_desc"
    BODY_SIZE_DESC = "body_size_desc"
    RATING_DESC = "rating_desc"
    USAGE_DESC = "usage_desc"


class PromptListCoordinator:
    """Encapsulates prompt loading, filtering, and sorting logic."""
    def __init__(self, manager: PromptManager) -> None:
        """Store PromptManager dependency for data access."""
        self._manager = manager

    def fetch_prompts(self, search_text: str) -> PromptLoadResult:
        """Return prompt data queried from the repository and optional search."""
        stripped = search_text.strip()
        all_prompts = list(self._manager.repository.list())
        search_results: list[Prompt] | None = None
        search_error: str | None = None
        preserve_order = False

        if stripped:
            try:
                search_results = list(self._manager.search_prompts(stripped, limit=50))
            except PromptManagerError as exc:
                search_error = str(exc)
            else:
                preserve_order = True

        return PromptLoadResult(
            all_prompts=all_prompts,
            search_results=search_results,
            preserve_search_order=preserve_order,
            search_error=search_error,
        )

    def populate_filters(
        self,
        panel: PromptFilterPanel | None,
        prompts: Sequence[Prompt],
        *,
        pending_category_slug: str | None,
        pending_tag_value: str | None,
    ) -> tuple[str | None, str | None]:
        """Refresh category and tag filter options and mutate pending values."""
        if panel is None:
            return pending_category_slug, pending_tag_value
        categories = self._manager.list_categories()
        target_category = panel.category_slug() or pending_category_slug
        panel.set_categories(categories, target_category)
        if target_category and panel.category_slug() == target_category:
            pending_category_slug = None

        tags = sorted({tag for prompt in prompts for tag in prompt.tags})
        target_tag = panel.tag_value() or pending_tag_value
        panel.set_tags(tags, target_tag)
        if target_tag and panel.tag_value() == target_tag:
            pending_tag_value = None

        return pending_category_slug, pending_tag_value

    def apply_filters(
        self,
        panel: PromptFilterPanel | None,
        prompts: Sequence[Prompt],
    ) -> list[Prompt]:
        """Apply category, tag, and quality filters to prompts."""
        if panel is None:
            return list(prompts)

        selected_category = panel.category_slug()
        selected_tag = panel.tag_value()
        min_quality = panel.min_quality()

        filtered: list[Prompt] = []
        for prompt in prompts:
            if selected_category:
                prompt_slug = self._prompt_category_slug(prompt)
                if prompt_slug != selected_category:
                    continue
            if selected_tag and selected_tag not in prompt.tags:
                continue
            if min_quality > 0.0:
                quality = prompt.quality_score or 0.0
                if quality < min_quality:
                    continue
            filtered.append(prompt)
        return filtered

    def sort_prompts(
        self,
        prompts: Sequence[Prompt],
        order: PromptSortOrder,
    ) -> list[Prompt]:
        """Return prompts sorted according to the requested sort order."""
        if not prompts:
            return []

        if order is PromptSortOrder.NAME_ASC:
            return sorted(prompts, key=lambda prompt: (prompt.name.casefold(), str(prompt.id)))
        if order is PromptSortOrder.NAME_DESC:
            return sorted(
                prompts,
                key=lambda prompt: (prompt.name.casefold(), str(prompt.id)),
                reverse=True,
            )
        if order is PromptSortOrder.QUALITY_DESC:

            def quality_key(prompt: Prompt) -> tuple[float, str, str]:
                quality = (
                    prompt.quality_score if prompt.quality_score is not None else float("-inf")
                )
                return (-quality, prompt.name.casefold(), str(prompt.id))

            return sorted(prompts, key=quality_key)
        if order is PromptSortOrder.MODIFIED_DESC:

            def modified_key(prompt: Prompt) -> tuple[float, str, str]:
                timestamp = prompt.last_modified.timestamp() if prompt.last_modified else 0.0
                return (-timestamp, prompt.name.casefold(), str(prompt.id))

            return sorted(prompts, key=modified_key)
        if order is PromptSortOrder.CREATED_DESC:

            def created_key(prompt: Prompt) -> tuple[float, str, str]:
                timestamp = prompt.created_at.timestamp() if prompt.created_at else 0.0
                return (-timestamp, prompt.name.casefold(), str(prompt.id))

            return sorted(prompts, key=created_key)
        if order is PromptSortOrder.BODY_SIZE_DESC:

            def body_size_key(prompt: Prompt) -> tuple[int, str, str]:
                length = self._prompt_body_length(prompt)
                return (-length, prompt.name.casefold(), str(prompt.id))

            return sorted(prompts, key=body_size_key)
        if order is PromptSortOrder.RATING_DESC:

            def rating_key(prompt: Prompt) -> tuple[float, int, str, str]:
                average = self._prompt_average_rating(prompt)
                count = prompt.rating_count
                return (-average, -(count or 0), prompt.name.casefold(), str(prompt.id))

            return sorted(prompts, key=rating_key)
        if order is PromptSortOrder.USAGE_DESC:

            def usage_key(prompt: Prompt) -> tuple[int, float, str, str]:
                usage = prompt.usage_count
                modified = prompt.last_modified.timestamp() if prompt.last_modified else 0.0
                return (-(usage or 0), -modified, prompt.name.casefold(), str(prompt.id))

            return sorted(prompts, key=usage_key)
        return list(prompts)

    @staticmethod
    def _prompt_category_slug(prompt: Prompt) -> str | None:
        """Return the prompt's category slug, deriving it when missing."""
        if prompt.category_slug:
            return prompt.category_slug
        return slugify_category(prompt.category)

    @staticmethod
    def _prompt_body_length(prompt: Prompt) -> int:
        """Return the length of the prompt body used for embedding/search."""
        payload = prompt.context or prompt.description or ""
        return len(payload)

    @staticmethod
    def _prompt_average_rating(prompt: Prompt) -> float:
        """Return the average rating for a prompt, defaulting to zero."""
        if prompt.rating_count and prompt.rating_sum is not None:
            try:
                return float(prompt.rating_sum) / float(prompt.rating_count)
            except ZeroDivisionError:  # pragma: no cover - defensive
                return 0.0
        return 0.0


__all__ = ["PromptListCoordinator", "PromptLoadResult", "PromptSortOrder"]
