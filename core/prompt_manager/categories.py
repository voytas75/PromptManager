"""Category management helpers for Prompt Manager.

Updates:
  v0.1.0 - 2025-12-02 - Extract category APIs into mixin for modularisation.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from models.category_model import PromptCategory

from ..exceptions import CategoryNotFoundError, CategoryStorageError
from ..repository import RepositoryError, RepositoryNotFoundError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..category_registry import CategoryRegistry
    from ..repository import PromptRepository

__all__ = ["CategorySupport"]


class CategorySupport:
    """Mixin exposing category CRUD helpers backed by the repository and registry."""

    _category_registry: CategoryRegistry
    _repository: PromptRepository

    def list_categories(self, include_archived: bool = False) -> list[PromptCategory]:
        """Return cached categories."""
        return self._category_registry.all(include_archived)

    def refresh_categories(self) -> list[PromptCategory]:
        """Reload categories from the repository."""
        return self._category_registry.refresh()

    def create_category(
        self,
        *,
        label: str,
        slug: str | None = None,
        description: str | None = None,
        parent_slug: str | None = None,
        color: str | None = None,
        icon: str | None = None,
        min_quality: float | None = None,
        default_tags: Sequence[str] | None = None,
        is_active: bool = True,
    ) -> PromptCategory:
        """Create a new category entry."""
        category = PromptCategory(
            slug=slug or label,
            label=label,
            description=description or label,
            parent_slug=parent_slug,
            color=color,
            icon=icon,
            min_quality=min_quality,
            default_tags=list(default_tags or []),
            is_active=is_active,
        )
        try:
            created = self._repository.create_category(category)
        except RepositoryError as exc:
            raise CategoryStorageError("Unable to create category") from exc
        self._category_registry.refresh()
        return created

    def update_category(
        self,
        slug: str,
        *,
        label: str | None = None,
        description: str | None = None,
        parent_slug: str | None = None,
        color: str | None = None,
        icon: str | None = None,
        min_quality: float | None = None,
        default_tags: Sequence[str] | None = None,
        is_active: bool | None = None,
    ) -> PromptCategory:
        """Update the specified category."""
        current = self._category_registry.require(slug)
        updated = replace(
            current,
            label=label or current.label,
            description=description or current.description,
            parent_slug=parent_slug if parent_slug is not None else current.parent_slug,
            color=color if color is not None else current.color,
            icon=icon if icon is not None else current.icon,
            min_quality=min_quality if min_quality is not None else current.min_quality,
            default_tags=list(default_tags) if default_tags is not None else current.default_tags,
            is_active=is_active if is_active is not None else current.is_active,
            updated_at=datetime.now(UTC),
        )
        try:
            persisted = self._repository.update_category(updated)
        except RepositoryNotFoundError as exc:
            raise CategoryNotFoundError(str(exc)) from exc
        except RepositoryError as exc:
            raise CategoryStorageError("Unable to update category") from exc
        self._category_registry.refresh()
        return persisted

    def set_category_active(self, slug: str, is_active: bool) -> PromptCategory:
        """Toggle category visibility."""
        try:
            category = self._repository.set_category_active(slug, is_active)
        except RepositoryNotFoundError as exc:
            raise CategoryNotFoundError(str(exc)) from exc
        except RepositoryError as exc:
            raise CategoryStorageError("Unable to update category state") from exc
        self._category_registry.refresh()
        return category

    def resolve_category_label(self, slug: str | None, fallback: str | None = None) -> str:
        """Return the human-readable label for a slug."""
        category = self._category_registry.get(slug)
        if category:
            return category.label
        return fallback or ""
