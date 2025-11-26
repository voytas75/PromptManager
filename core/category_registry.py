"""Prompt category registry helpers and defaults.

Updates: v0.1.0 - 2025-11-22 - Introduce PromptCategory registry and defaults.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Mapping, MutableMapping, Optional, Sequence, cast

from models.category_model import PromptCategory, slugify_category

from .exceptions import CategoryError, CategoryNotFoundError, CategoryStorageError
from .repository import PromptRepository, RepositoryError

logger = logging.getLogger(__name__)

DEFAULT_CATEGORY_DEFINITIONS: Sequence[Mapping[str, str]] = (
    {
        "slug": "general",
        "label": "General",
        "description": "Miscellaneous prompts or staging entries awaiting classification.",
        "color": "#6b7280",
        "icon": "mdi-inbox-arrow-down",
    },
    {
        "slug": "code-analysis",
        "label": "Code Analysis",
        "description": "Explaining code behaviour, risks, performance, and intent.",
        "color": "#2563eb",
        "icon": "mdi-file-search-outline",
    },
    {
        "slug": "reasoning-debugging",
        "label": "Reasoning / Debugging",
        "description": "Tracing defects, analysing logs, and suggesting fixes.",
        "color": "#dc2626",
        "icon": "mdi-bug-outline",
    },
    {
        "slug": "refactoring",
        "label": "Refactoring",
        "description": "Improving structure, readability, and maintainability without changing behaviour.",
        "color": "#7c3aed",
        "icon": "mdi-code-json",
        "parent_slug": "code-analysis",
    },
    {
        "slug": "documentation",
        "label": "Documentation",
        "description": "Generating docstrings, READMEs, and developer guides.",
        "color": "#059669",
        "icon": "mdi-file-document-edit-outline",
    },
    {
        "slug": "enhancement",
        "label": "Enhancement",
        "description": "Brainstorming improvements, tests, and feature extensions.",
        "color": "#ea580c",
        "icon": "mdi-lightbulb-on-outline",
    },
    {
        "slug": "reporting",
        "label": "Reporting",
        "description": "Summarising metrics, crafting updates, and turning data into prose.",
        "color": "#0f172a",
        "icon": "mdi-chart-box-outline",
    },
)


def load_category_definitions(
    inline_definitions: Optional[Sequence[Mapping[str, object]]] = None,
    *,
    path: Optional[Path] = None,
) -> List[PromptCategory]:
    """Return PromptCategory definitions from defaults plus overrides."""

    catalog: MutableMapping[str, PromptCategory] = {
        entry["slug"]: PromptCategory.from_mapping(entry)
        for entry in DEFAULT_CATEGORY_DEFINITIONS
    }

    payloads: List[Mapping[str, object]] = []
    if path is not None:
        try:
            text = path.expanduser().read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning("Category catalogue file missing at %s", path)
        except OSError as exc:
            logger.warning("Unable to read categories file %s: %s", path, exc)
        else:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                logger.warning("Invalid category JSON in %s: %s", path, exc)
            else:
                if isinstance(parsed, list):
                    parsed_entries = cast("Sequence[object]", parsed)
                    for raw_entry in parsed_entries:
                        if isinstance(raw_entry, Mapping):
                            typed_entry = cast("Mapping[str, object]", raw_entry)
                            payloads.append(dict(typed_entry))
                else:
                    logger.warning("Expected a list of category mappings in %s", path)

    if inline_definitions:
        payloads.extend(inline_definitions)

    for payload in payloads:
        try:
            category = PromptCategory.from_mapping(payload)
        except (ValueError, TypeError) as exc:
            logger.warning("Skipping invalid category definition: %s", exc)
            continue
        catalog[category.slug] = category
    return list(catalog.values())


class CategoryRegistry:
    """Cached view of stored categories with helper utilities."""

    def __init__(
        self,
        repository: PromptRepository,
        defaults: Optional[Sequence[PromptCategory]] = None,
    ) -> None:
        self._repository = repository
        self._defaults = list(defaults or [])
        self._all: dict[str, PromptCategory] = {}
        self._active: dict[str, PromptCategory] = {}
        if self._defaults:
            try:
                self._repository.sync_category_definitions(self._defaults)
            except RepositoryError as exc:  # pragma: no cover - initialization
                raise CategoryStorageError("Unable to seed default categories") from exc
        self.refresh()

    def refresh(self) -> List[PromptCategory]:
        """Reload categories from the repository."""

        try:
            categories = self._repository.list_categories(include_archived=True)
        except RepositoryError as exc:
            raise CategoryStorageError("Unable to load categories") from exc
        self._all = {category.slug: category for category in categories}
        self._active = {
            category.slug: category for category in categories if category.is_active
        }
        return categories

    def all(self, include_archived: bool = False) -> List[PromptCategory]:
        """Return cached categories."""

        source = self._all if include_archived else self._active
        return list(source.values())

    def get(self, slug: Optional[str]) -> Optional[PromptCategory]:
        """Return category by slug, if available."""

        if not slug:
            return None
        return self._all.get(slugify_category(slug))

    def find_by_label(self, label: Optional[str]) -> Optional[PromptCategory]:
        """Return the first category matching a label (case-insensitive)."""

        if not label:
            return None
        target = label.strip().lower()
        for category in self._all.values():
            if category.label.lower() == target:
                return category
        return None

    def ensure(
        self,
        slug: Optional[str],
        *,
        label: Optional[str],
        description: Optional[str] = None,
    ) -> PromptCategory:
        """Return an existing category or create a new placeholder."""

        resolved_slug = slugify_category(slug or label)
        if not resolved_slug:
            raise CategoryError("Category slug cannot be empty.")
        existing = self.get(resolved_slug)
        if existing:
            return existing
        by_label = self.find_by_label(label)
        if by_label:
            return by_label
        category = PromptCategory(
            slug=resolved_slug,
            label=label or resolved_slug.replace("-", " ").title(),
            description=description or f"{label or resolved_slug} prompts",
        )
        try:
            self._repository.create_category(category)
        except RepositoryError as exc:
            raise CategoryStorageError(f"Unable to create category '{category.label}'") from exc
        self.refresh()
        return category

    def require(self, slug: str) -> PromptCategory:
        """Return a category or raise if not found."""

        category = self.get(slug)
        if category is None:
            raise CategoryNotFoundError(f"Category '{slug}' does not exist.")
        return category


__all__ = ["CategoryRegistry", "load_category_definitions", "DEFAULT_CATEGORY_DEFINITIONS"]
