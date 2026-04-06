"""Focused tests for bounded retrieval previews in the main prompt list.

Updates:
  v0.1.0 - 2026-04-06 - Cover visible, hidden, and truncated retrieval-preview paths.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import cast

import pytest

from gui.prompt_list_delegate import PromptListDelegate
from gui.prompt_list_model import PromptListModel
from models.prompt_model import Prompt

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QStyleOptionViewItem
except ImportError:  # pragma: no cover - optional dependency in test environments
    pytest.skip("PySide6 is not available", allow_module_level=True)


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    """Provide a shared Qt application instance for prompt-list model tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return cast("QApplication", app)


def _build_prompt(
    *,
    description: str = "",
    scenarios: list[str] | None = None,
    source: str = "local",
) -> Prompt:
    """Create a prompt tailored to retrieval-preview tests."""
    return Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000201"),
        name="Incident triage",
        description=description,
        category="Ops",
        context="Review the latest incident timeline and summarise the next operator actions.",
        scenarios=scenarios or [],
        source=source,
        created_at=datetime(2026, 4, 6, 9, 0, tzinfo=UTC),
        last_modified=datetime(2026, 4, 6, 10, 0, tzinfo=UTC),
    )


def test_prompt_list_model_prefers_description_preview_when_available(qt_app: QApplication) -> None:
    """Description should provide the preview before scenario or source fallbacks."""
    prompt = _build_prompt(
        description="Summarise incident updates for the next on-call handoff.",
        scenarios=["Use after timeline review."],
        source="ops notebook",
    )
    model = PromptListModel([prompt])

    index = model.index(0, 0)

    assert index.data(Qt.ItemDataRole.DisplayRole) == "Incident triage (Ops)"
    assert index.data(PromptListModel.PreviewRole) == (
        "Summarise incident updates for the next on-call handoff."
    )


def test_prompt_list_model_hides_preview_for_low_signal_prompt_data(qt_app: QApplication) -> None:
    """No preview should render when only empty fields or generic source markers exist."""
    prompt = _build_prompt(description=" ", scenarios=[], source="quick_capture")
    model = PromptListModel([prompt])

    index = model.index(0, 0)

    assert index.data(PromptListModel.PreviewRole) is None


def test_prompt_list_model_flattens_and_truncates_scenario_preview(qt_app: QApplication) -> None:
    """Scenario fallback should stay on one line and use deterministic truncation."""
    prompt = _build_prompt(
        description="",
        scenarios=[
            "Use after collecting timeline notes,\n"
            "then compare responder actions against the handoff checklist before posting."
        ],
        source="local",
    )
    model = PromptListModel([prompt])

    index = model.index(0, 0)
    preview = index.data(PromptListModel.PreviewRole)

    assert isinstance(preview, str)
    assert "\n" not in preview
    assert preview.endswith("...")
    assert len(preview) <= PromptListModel.PreviewMaxLength


def test_prompt_list_delegate_returns_taller_rows_when_preview_exists(
    qt_app: QApplication,
) -> None:
    """Rows with a preview should reserve enough height for the second line."""
    with_preview = PromptListModel(
        [_build_prompt(description="", scenarios=[], source="support queue")]
    )
    without_preview = PromptListModel([_build_prompt(description="", scenarios=[], source="local")])
    delegate = PromptListDelegate()
    option = QStyleOptionViewItem()

    with_height = delegate.sizeHint(option, with_preview.index(0, 0)).height()
    without_height = delegate.sizeHint(option, without_preview.index(0, 0)).height()

    assert with_height > without_height
