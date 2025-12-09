"""Tests for settings/persistence helpers.

Updates:
  v0.3.2 - 2025-12-08 - Cast fake settings objects to QSettings for Pyright compliance.
  v0.3.1 - 2025-11-30 - Point helpers to layout_state module exports.
  v0.3.0 - 2025-11-26 - Cover filter preference helpers for category/tag/sort persistence.
  v0.2.0 - 2025-11-26 - Cover history load/store helpers and trimming logic.
  v0.1.0 - 2025-11-22 - Cover helper functions that load/store the last task text.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import pytest

pytest.importorskip("PySide6")

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PySide6.QtCore import QSettings
else:  # pragma: no cover - fallback type for runtime
    QSettings = object

from gui.layout_state import (
    _EXECUTE_CONTEXT_HISTORY_KEY,
    _EXECUTE_CONTEXT_TASK_KEY,
    _FILTER_CATEGORY_KEY,
    _FILTER_QUALITY_KEY,
    _FILTER_SORT_KEY,
    _FILTER_TAG_KEY,
    _load_execute_context_history,
    _load_filter_preferences,
    _load_last_execute_context_task,
    _store_execute_context_history,
    _store_filter_preferences,
    _store_last_execute_context_task,
    _store_sort_preference,
)
from gui.main_window import PromptSortOrder


class _FakeSettings:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}
        self.synced = False

    def value(self, key: str, default: object = None, _type: object | None = None) -> object:
        return self.values.get(key, default)

    def setValue(self, key: str, value: object) -> None:  # noqa: N802 - Qt-style API
        self.values[key] = value

    def sync(self) -> None:
        self.synced = True


def _as_qsettings(settings: _FakeSettings) -> QSettings:
    return cast("QSettings", settings)


def test_load_last_execute_context_task_returns_trimmed_value() -> None:
    """Trim whitespace when loading the last recorded execute-context task."""
    settings = _FakeSettings()
    settings.values[_EXECUTE_CONTEXT_TASK_KEY] = "  Summarise logs  "

    assert _load_last_execute_context_task(_as_qsettings(settings)) == "Summarise logs"


def test_store_last_execute_context_task_persists_text_and_syncs() -> None:
    """Persist last task text and sync the settings store."""
    settings = _FakeSettings()

    _store_last_execute_context_task(_as_qsettings(settings), "Investigate timeouts")

    assert settings.values[_EXECUTE_CONTEXT_TASK_KEY] == "Investigate timeouts"
    assert settings.synced is True


def test_load_execute_context_history_preserves_whitespace() -> None:
    """Keep intentional whitespace and drop illegal entries when loading history."""
    settings = _FakeSettings()
    settings.values[_EXECUTE_CONTEXT_HISTORY_KEY] = json.dumps(
        ["  Summarise logs  ", "Summarise logs", "Investigate outages", " "]
    )

    entries = _load_execute_context_history(_as_qsettings(settings), limit=3)

    assert entries == ["  Summarise logs  ", "Summarise logs", "Investigate outages"]


def test_load_execute_context_history_handles_invalid_payload() -> None:
    """Return an empty list when execute-context history JSON is invalid."""
    settings = _FakeSettings()
    settings.values[_EXECUTE_CONTEXT_HISTORY_KEY] = "{"  # invalid JSON

    assert _load_execute_context_history(_as_qsettings(settings)) == []


def test_store_execute_context_history_preserves_whitespace_and_limits() -> None:
    """Persist a deduplicated execute-context history payload and sync settings."""
    settings = _FakeSettings()

    _store_execute_context_history(
        _as_qsettings(settings),
        ["  Summarise logs  ", "Investigate outages", "Summarise logs", ""],
    )

    stored_raw = settings.values[_EXECUTE_CONTEXT_HISTORY_KEY]
    stored = json.loads(str(stored_raw))
    assert stored == ["  Summarise logs  ", "Investigate outages", "Summarise logs"]
    assert settings.synced is True


def test_load_filter_preferences_returns_trimmed_state() -> None:
    """Load filters and convert numbers from strings when the payload is valid."""
    settings = _FakeSettings()
    settings.values[_FILTER_CATEGORY_KEY] = "  incident_response  "
    settings.values[_FILTER_TAG_KEY] = "  outages  "
    settings.values[_FILTER_QUALITY_KEY] = "4.5"
    settings.values[_FILTER_SORT_KEY] = "usage_desc"

    category, tag, quality, sort_value = _load_filter_preferences(_as_qsettings(settings))

    assert category == "incident_response"
    assert tag == "outages"
    assert quality == 4.5
    assert sort_value == "usage_desc"


def test_load_filter_preferences_handles_invalid_quality() -> None:
    """Ignore invalid quality inputs and return defaults for the rest."""
    settings = _FakeSettings()
    settings.values[_FILTER_QUALITY_KEY] = "fast"

    _, _, quality, sort_value = _load_filter_preferences(_as_qsettings(settings))

    assert quality is None
    assert sort_value is None


def test_store_filter_preferences_persists_values_and_syncs() -> None:
    """Persist filter settings and ensure the store is synced."""
    settings = _FakeSettings()

    _store_filter_preferences(
        _as_qsettings(settings),
        category_slug="incident_response",
        tag="outages",
        min_quality=3.2,
    )

    assert settings.values[_FILTER_CATEGORY_KEY] == "incident_response"
    assert settings.values[_FILTER_TAG_KEY] == "outages"
    assert settings.values[_FILTER_QUALITY_KEY] == 3.2
    assert settings.synced is True


def test_store_sort_preference_persists_value() -> None:
    """Persist the sort order using the enum value representation."""
    settings = _FakeSettings()

    _store_sort_preference(_as_qsettings(settings), PromptSortOrder.USAGE_DESC)

    assert settings.values[_FILTER_SORT_KEY] == PromptSortOrder.USAGE_DESC.value
    assert settings.synced is True
