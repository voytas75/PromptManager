"""Helpers that persist and restore user layout/preferences via QSettings.

Updates:
  v0.1.0 - 2025-11-30 - Extract layout persistence helpers from main window.
"""
from __future__ import annotations

import json
import logging
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from PySide6.QtCore import QSettings

logger = logging.getLogger(__name__)

_EXECUTE_CONTEXT_TASK_KEY = "lastExecuteContextTask"
_EXECUTE_CONTEXT_HISTORY_KEY = "executeContextTaskHistory"
_EXECUTE_CONTEXT_HISTORY_LIMIT = 15
_FILTER_CATEGORY_KEY = "filterCategorySlug"
_FILTER_TAG_KEY = "filterTag"
_FILTER_QUALITY_KEY = "filterQuality"
_FILTER_SORT_KEY = "filterSortOrder"


def _load_last_execute_context_task(settings: QSettings) -> str:
    """Return the persisted execute-context task text."""
    raw_value = settings.value(_EXECUTE_CONTEXT_TASK_KEY, "", str)
    if raw_value is None:
        return ""
    return str(raw_value).strip()


def _store_last_execute_context_task(settings: QSettings, value: str) -> None:
    """Persist the execute-context task text for future sessions."""
    settings.setValue(_EXECUTE_CONTEXT_TASK_KEY, value)
    try:
        settings.sync()
    except Exception:  # pragma: no cover - platform specific
        logger.warning("Unable to sync execute-context history", exc_info=True)


def _load_execute_context_history(
    settings: QSettings,
    *,
    limit: int = _EXECUTE_CONTEXT_HISTORY_LIMIT,
) -> list[str]:
    """Return the stored execute-context descriptions limited to *limit* entries."""
    raw_value = settings.value(_EXECUTE_CONTEXT_HISTORY_KEY, "[]", str)
    if raw_value is None:
        return []
    text = str(raw_value).strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Invalid execute-context history payload; resetting history.")
        return []
    if not isinstance(payload, list):
        return []
    entries: list[str] = []
    for entry in payload:
        candidate = str(entry)
        if not candidate.strip() or candidate in entries:
            continue
        entries.append(candidate)
        if len(entries) >= max(limit, 0):
            break
    return entries


def _store_execute_context_history(settings: QSettings, history: Sequence[str]) -> None:
    """Persist execute-context descriptions as a JSON-encoded list."""
    entries: list[str] = []
    for entry in history:
        candidate = str(entry)
        if not candidate.strip() or candidate in entries:
            continue
        entries.append(candidate)
        if len(entries) >= _EXECUTE_CONTEXT_HISTORY_LIMIT:
            break
    settings.setValue(_EXECUTE_CONTEXT_HISTORY_KEY, json.dumps(entries))
    try:
        settings.sync()
    except Exception:  # pragma: no cover - platform specific
        logger.warning("Unable to sync execute-context history list", exc_info=True)


def _load_filter_preferences(
    settings: QSettings,
) -> tuple[str | None, str | None, float | None, str | None]:
    """Return persisted filter selections (category, tag, quality, sort)."""
    def _clean_text(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    category = _clean_text(settings.value(_FILTER_CATEGORY_KEY, None))
    tag = _clean_text(settings.value(_FILTER_TAG_KEY, None))
    raw_quality = settings.value(_FILTER_QUALITY_KEY, None)
    quality: float | None = None
    if raw_quality not in {None, ""}:
        try:
            quality = float(raw_quality)
        except (TypeError, ValueError):
            logger.warning("Invalid quality filter value '%s'; resetting to default.", raw_quality)
            quality = None
    sort_value = _clean_text(settings.value(_FILTER_SORT_KEY, None))
    return category, tag, quality, sort_value


def _store_filter_preferences(
    settings: QSettings,
    *,
    category_slug: str | None,
    tag: str | None,
    min_quality: float,
) -> None:
    """Persist current filter selections."""
    settings.setValue(_FILTER_CATEGORY_KEY, category_slug or "")
    settings.setValue(_FILTER_TAG_KEY, tag or "")
    settings.setValue(_FILTER_QUALITY_KEY, float(min_quality))
    try:
        settings.sync()
    except Exception:  # pragma: no cover - platform specific
        logger.warning("Unable to sync filter preferences", exc_info=True)


def _store_sort_preference(settings: QSettings, sort_value: Enum | str) -> None:
    """Persist the selected prompt sort order."""
    if isinstance(sort_value, Enum):
        payload = sort_value.value
    else:
        payload = str(sort_value)
    settings.setValue(_FILTER_SORT_KEY, payload)
    try:
        settings.sync()
    except Exception:  # pragma: no cover
        logger.warning("Unable to sync sort preference", exc_info=True)


__all__ = [
    "_EXECUTE_CONTEXT_HISTORY_KEY",
    "_EXECUTE_CONTEXT_HISTORY_LIMIT",
    "_EXECUTE_CONTEXT_TASK_KEY",
    "_FILTER_CATEGORY_KEY",
    "_FILTER_QUALITY_KEY",
    "_FILTER_SORT_KEY",
    "_FILTER_TAG_KEY",
    "_load_execute_context_history",
    "_load_filter_preferences",
    "_load_last_execute_context_task",
    "_store_execute_context_history",
    "_store_filter_preferences",
    "_store_last_execute_context_task",
    "_store_sort_preference",
]
