"""Tests for execute-as-context task persistence helpers.

Updates: v0.2.0 - 2025-11-26 - Cover history load/store helpers and trimming logic.
Updates: v0.1.0 - 2025-11-22 - Cover helper functions that load/store the last task text.
"""

from __future__ import annotations

import json

from gui.main_window import (
    _EXECUTE_CONTEXT_HISTORY_KEY,
    _EXECUTE_CONTEXT_TASK_KEY,
    _load_execute_context_history,
    _load_last_execute_context_task,
    _store_execute_context_history,
    _store_last_execute_context_task,
)


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


def test_load_last_execute_context_task_returns_trimmed_value() -> None:
    settings = _FakeSettings()
    settings.values[_EXECUTE_CONTEXT_TASK_KEY] = "  Summarise logs  "

    assert _load_last_execute_context_task(settings) == "Summarise logs"


def test_store_last_execute_context_task_persists_text_and_syncs() -> None:
    settings = _FakeSettings()

    _store_last_execute_context_task(settings, "Investigate timeouts")

    assert settings.values[_EXECUTE_CONTEXT_TASK_KEY] == "Investigate timeouts"
    assert settings.synced is True


def test_load_execute_context_history_preserves_whitespace() -> None:
    settings = _FakeSettings()
    settings.values[_EXECUTE_CONTEXT_HISTORY_KEY] = json.dumps(
        ["  Summarise logs  ", "Summarise logs", "Investigate outages", " "]
    )

    entries = _load_execute_context_history(settings, limit=3)

    assert entries == ["  Summarise logs  ", "Summarise logs", "Investigate outages"]


def test_load_execute_context_history_handles_invalid_payload() -> None:
    settings = _FakeSettings()
    settings.values[_EXECUTE_CONTEXT_HISTORY_KEY] = "{"  # invalid JSON

    assert _load_execute_context_history(settings) == []


def test_store_execute_context_history_preserves_whitespace_and_limits() -> None:
    settings = _FakeSettings()

    _store_execute_context_history(
        settings,
        ["  Summarise logs  ", "Investigate outages", "Summarise logs", ""]
    )

    stored = json.loads(settings.values[_EXECUTE_CONTEXT_HISTORY_KEY])
    assert stored == ["  Summarise logs  ", "Investigate outages", "Summarise logs"]
    assert settings.synced is True
