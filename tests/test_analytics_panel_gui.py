"""GUI-focused tests for the analytics dashboard panel.

Updates:
  v0.1.1 - 2025-12-08 - Cast manager stubs and ensure QApplication fixtures satisfy Pyright.
  v0.1.0 - 2025-12-05 - Cover usage table prompt activation behaviour.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from core import PromptManager
from core.analytics_dashboard import AnalyticsSnapshot, UsageFrequencyEntry
from gui.analytics_panel import AnalyticsDashboardPanel

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from uuid import UUID


class _StubRepository:
    def list(self):  # noqa: D401 - lightweight stub
        return []

    def get_prompt_execution_statistics(self, *_args, **_kwargs):
        return {}

    def get_model_usage_breakdown(self, *_args, **_kwargs):
        return []

    def get_benchmark_execution_stats(self, *_args, **_kwargs):
        return []


class _StubManager:
    def __init__(self) -> None:
        self.repository = _StubRepository()

    def get_execution_analytics(self, *_args, **_kwargs):
        return None

    def diagnose_embeddings(self):
        return None


def _as_prompt_manager(manager: _StubManager) -> PromptManager:
    return cast(PromptManager, manager)


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    """Provide a shared Qt application instance for analytics panel tests."""

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return cast(QApplication, app)


def test_usage_prompt_activation_opens_editor(qt_app: QApplication) -> None:
    """Double-clicking a usage row should trigger the prompt edit callback."""

    fired: list[UUID] = []
    panel = AnalyticsDashboardPanel(
        _as_prompt_manager(_StubManager()),
        prompt_edit_callback=lambda prompt_id: fired.append(prompt_id),
    )
    prompt_id = uuid.uuid4()
    snapshot = AnalyticsSnapshot(
        execution=None,
        usage_frequency=[
            UsageFrequencyEntry(
                prompt_id=prompt_id,
                name="Example",
                usage_count=4,
                success_rate=0.5,
                last_executed_at=datetime.now(UTC),
            )
        ],
        model_costs=[],
        benchmark_stats=[],
        intent_success=[],
        embedding=None,
    )
    try:
        panel._snapshot = snapshot  # type: ignore[attr-defined]
        panel._populate_table()  # type: ignore[attr-defined]
        panel._handle_table_cell_activated(0, 0)  # type: ignore[attr-defined]
        assert fired == [prompt_id]
    finally:
        panel.deleteLater()
