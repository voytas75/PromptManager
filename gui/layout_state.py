"""Helpers that persist and restore user layout/preferences via QSettings.

Updates:
  v0.1.1 - 2025-12-01 - Add WindowStateManager for geometry/splitter/history helpers.
  v0.1.0 - 2025-11-30 - Extract layout persistence helpers from main window.
"""
from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from PySide6.QtCore import QByteArray

if TYPE_CHECKING:
    from collections.abc import Sequence

    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QSplitter, QWidget

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


@dataclass(slots=True)
class FilterPreferences:
    """Persisted filters restored from user settings."""
    category_slug: str | None
    tag: str | None
    min_quality: float | None
    sort_value: str | None


@dataclass(slots=True)
class ExecuteContextState:
    """Execute-context history pulled from persistent storage."""
    last_task: str
    history: deque[str]


class WindowStateManager:
    """High-level helper that encapsulates window/state persistence logic."""
    def __init__(
        self,
        settings: QSettings,
        *,
        history_limit: int = _EXECUTE_CONTEXT_HISTORY_LIMIT,
    ) -> None:
        """Initialise the manager with the provided QSettings handle."""
        self._settings = settings
        self._history_limit = max(history_limit, 1)

    @property
    def history_limit(self) -> int:
        """Return the execute-context history limit."""
        return self._history_limit

    @property
    def settings(self) -> QSettings:
        """Expose the underlying QSettings for callers needing raw access."""
        return self._settings

    def load_filter_preferences(self) -> FilterPreferences:
        """Return persisted filter settings."""
        category, tag, quality, sort_value = _load_filter_preferences(self._settings)
        return FilterPreferences(
            category_slug=category,
            tag=tag,
            min_quality=quality,
            sort_value=sort_value,
        )

    def persist_filter_preferences(
        self,
        *,
        category_slug: str | None,
        tag: str | None,
        min_quality: float,
    ) -> None:
        """Persist the provided filter selections."""
        _store_filter_preferences(
            self._settings,
            category_slug=category_slug,
            tag=tag,
            min_quality=min_quality,
        )

    def persist_sort_order(self, value: Enum | str) -> None:
        """Persist the selected sort order."""
        _store_sort_preference(self._settings, value)

    def load_execute_context_state(self) -> ExecuteContextState:
        """Load execute-context task state and ensure history respects limits."""
        last_task = _load_last_execute_context_task(self._settings)
        entries = deque(
            _load_execute_context_history(self._settings, limit=self._history_limit),
            maxlen=self._history_limit,
        )
        if last_task.strip() and last_task not in entries:
            entries.appendleft(last_task)
            self.persist_execute_context_history(entries)
        return ExecuteContextState(last_task=last_task, history=entries)

    def persist_last_execute_task(self, task_text: str) -> None:
        """Persist the latest execute-context task text."""
        _store_last_execute_context_task(self._settings, task_text.strip())

    def persist_execute_context_history(self, history: deque[str]) -> None:
        """Persist the execute-context history entries."""
        _store_execute_context_history(self._settings, tuple(history))

    def record_execute_task(self, task_text: str, history: deque[str]) -> None:
        """Insert *task_text* into history while enforcing uniqueness."""
        cleaned = task_text.strip()
        if not cleaned:
            return
        entries: list[str] = [cleaned]
        for existing in history:
            if existing == cleaned:
                continue
            entries.append(existing)
            if len(entries) >= (history.maxlen or self._history_limit):
                break
        history.clear()
        history.extend(entries)
        self.persist_execute_context_history(history)

    def restore_window_geometry(
        self,
        window: QWidget,
        *,
        fallback_size: tuple[int, int] = (1024, 640),
    ) -> None:
        """Restore window geometry or resize to fallback dimensions."""
        stored = self._settings.value("windowGeometry")
        if isinstance(stored, QByteArray):
            if not window.restoreGeometry(stored):
                window.resize(*fallback_size)
            return
        if isinstance(stored, (bytes, bytearray)):
            if not window.restoreGeometry(QByteArray(stored)):
                window.resize(*fallback_size)
            return
        if isinstance(stored, str):
            try:
                width_str, height_str, *_ = stored.split(",")
                width = int(width_str)
                height = int(height_str)
            except (ValueError, TypeError):
                window.resize(*fallback_size)
            else:
                if width > 0 and height > 0:
                    window.resize(width, height)
                else:
                    window.resize(*fallback_size)
            return
        window.resize(*fallback_size)

    def save_window_geometry(self, window: QWidget) -> None:
        """Persist the provided window's geometry."""
        geometry = window.saveGeometry()
        self._settings.setValue("windowGeometry", geometry)

    def restore_splitter_sizes(
        self,
        entries: Sequence[tuple[str, QSplitter | None]],
    ) -> None:
        """Restore splitter sizes for the provided keyed entries."""
        for key, splitter in entries:
            if splitter is None:
                continue
            stored = self._settings.value(key)
            if stored is None:
                continue
            if isinstance(stored, str):
                parts = [segment for segment in stored.split(",") if segment]
            else:
                parts = list(stored) if isinstance(stored, (list, tuple)) else []
            try:
                sizes = [int(part) for part in parts]
            except (TypeError, ValueError):
                continue
            if len(sizes) != splitter.count() or not sizes or sum(sizes) <= 0:
                continue
            splitter.setSizes(sizes)

    def save_splitter_sizes(
        self,
        entries: Sequence[tuple[str, QSplitter | None]],
    ) -> None:
        """Persist splitter sizes for the given keyed entries."""
        for key, splitter in entries:
            if splitter is None:
                continue
            self._settings.setValue(key, splitter.sizes())
        try:
            self._settings.sync()
        except Exception:  # pragma: no cover - platform dependent
            logger.warning("Unable to sync splitter state", exc_info=True)


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
    "ExecuteContextState",
    "FilterPreferences",
    "WindowStateManager",
]
