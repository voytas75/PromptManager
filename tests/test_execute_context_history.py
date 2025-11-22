"""Tests for execute-as-context task persistence helpers.

Updates: v0.1.0 - 2025-11-22 - Cover helper functions that load/store the last task text.
"""

from __future__ import annotations

from gui.main_window import (
    _EXECUTE_CONTEXT_TASK_KEY,
    _load_last_execute_context_task,
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
