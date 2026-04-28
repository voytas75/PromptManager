"""Focused tests for bounded retrieval previews in the main prompt list.

Updates:
  v0.1.5 - 2026-04-12 - Cover active-search scenario-priority while keeping
             source and description precedence unchanged.
  v0.1.4 - 2026-04-12 - Cover prompt-body lead fallback while keeping stronger
             preview priorities unchanged.
  v0.1.3 - 2026-04-12 - Cover active-search source-priority while keeping
             ordinary preview fallback unchanged.
  v0.1.2 - 2026-04-12 - Cover active-search match spans and bounded delegate emphasis runs.
  v0.1.1 - 2026-04-11 - Keep preview text at the row base font size for readability.
  v0.1.0 - 2026-04-06 - Cover visible, hidden, and truncated retrieval-preview paths.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, cast

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
    context: str = "Review the latest incident timeline and summarise the next operator actions.",
) -> Prompt:
    """Create a prompt tailored to retrieval-preview tests."""
    return Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000201"),
        name="Incident triage",
        description=description,
        category="Ops",
        context=context,
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


def test_prompt_list_model_prefers_matching_source_preview_for_active_search(
    qt_app: QApplication,
) -> None:
    """Active plain-text search can promote a credible matching source cue."""
    prompt = _build_prompt(
        description="Summarise incident updates for the next on-call handoff.",
        scenarios=["Use after notebook review before the next handoff."],
        source="PagerDuty ops notebook",
    )
    model = PromptListModel([prompt])
    model.set_active_search_text("notebook")

    index = model.index(0, 0)

    assert index.data(PromptListModel.PreviewRole) == "Source: PagerDuty ops notebook"
    assert index.data(PromptListModel.MatchReasonRole) == "Matched in source"


def test_prompt_list_model_prefers_matching_scenario_over_non_matching_description(
    qt_app: QApplication,
) -> None:
    """Active search can promote the first matching credible scenario over a generic description."""
    prompt = _build_prompt(
        description="Reusable incident handoff prompt for routine operator transitions.",
        scenarios=[
            "Use after rollback review for release readiness decisions.",
            "Use after on-call handoff cleanup.",
        ],
        source="ops notebook",
    )
    model = PromptListModel([prompt])
    model.set_active_search_text("rollback review")

    index = model.index(0, 0)
    preview = cast("str", index.data(PromptListModel.PreviewRole))
    preview_spans = cast(
        "tuple[tuple[int, int], ...]",
        index.data(PromptListModel.PreviewMatchRole),
    )

    assert preview == "Use after rollback review for release readiness decisions."
    assert [preview[start : start + length].lower() for start, length in preview_spans] == [
        "rollback",
        "review",
    ]


def test_prompt_list_model_keeps_matching_description_over_matching_scenario_for_active_search(
    qt_app: QApplication,
) -> None:
    """A matching description should stay visible instead of switching to a matching scenario."""
    prompt = _build_prompt(
        description="Rollback review checklist for release readiness and operator handoff.",
        scenarios=["Use after rollback review for release readiness decisions."],
        source="ops notebook",
    )
    model = PromptListModel([prompt])
    model.set_active_search_text("rollback review")

    index = model.index(0, 0)

    assert index.data(PromptListModel.PreviewRole) == (
        "Rollback review checklist for release readiness and operator handoff."
    )


def test_prompt_list_model_keeps_no_search_preview_priority_unchanged(
    qt_app: QApplication,
) -> None:
    """Without active search, description-first preview selection stays unchanged."""
    prompt = _build_prompt(
        description="Summarise incident updates for the next on-call handoff.",
        scenarios=["Use after timeline review."],
        source="PagerDuty ops notebook",
    )
    model = PromptListModel([prompt])
    model.set_active_search_text("")

    index = model.index(0, 0)

    assert index.data(PromptListModel.PreviewRole) == (
        "Summarise incident updates for the next on-call handoff."
    )


def test_prompt_list_model_ignores_generic_or_weak_source_for_active_search_priority(
    qt_app: QApplication,
) -> None:
    """Generic or weak source values should not override the ordinary preview path."""
    low_signal_prompt = _build_prompt(
        description="Summarise incident updates for the next on-call handoff.",
        scenarios=[],
        source="local",
    )
    weak_prompt = _build_prompt(
        description="Summarise incident updates for the next on-call handoff.",
        scenarios=[],
        source="qa",
    )

    low_signal_model = PromptListModel([low_signal_prompt])
    low_signal_model.set_active_search_text("local")
    weak_model = PromptListModel([weak_prompt])
    weak_model.set_active_search_text("qa")

    assert low_signal_model.index(0, 0).data(PromptListModel.PreviewRole) == (
        "Summarise incident updates for the next on-call handoff."
    )
    assert weak_model.index(0, 0).data(PromptListModel.PreviewRole) == (
        "Summarise incident updates for the next on-call handoff."
    )


def test_prompt_list_model_falls_back_to_body_lead_when_metadata_is_weak(
    qt_app: QApplication,
) -> None:
    """Prompt body lead should provide one compact preview only after stronger cues fail."""
    prompt = _build_prompt(
        description=" ",
        scenarios=[],
        source="quick_capture",
        context="Summarize deployment risks for this release and call out rollback concerns.",
    )
    model = PromptListModel([prompt])

    index = model.index(0, 0)

    assert index.data(PromptListModel.PreviewRole) == (
        "Summarize deployment risks for this release and call out rollback concerns."
    )


def test_prompt_list_model_hides_preview_for_low_signal_prompt_data(qt_app: QApplication) -> None:
    """No preview should render when only empty fields or generic source markers exist."""
    prompt = _build_prompt(
        description=" ",
        scenarios=[],
        source="quick_capture",
        context="Draft body",
    )
    model = PromptListModel([prompt])

    index = model.index(0, 0)

    assert index.data(PromptListModel.PreviewRole) is None


def test_prompt_list_model_flattens_and_truncates_scenario_preview(qt_app: QApplication) -> None:
    """Scenario fallback should stay on one line and outrank the later body fallback."""
    prompt = _build_prompt(
        description="",
        scenarios=[
            "Use after collecting timeline notes,\n"
            "then compare responder actions against the handoff checklist before posting."
        ],
        source="local",
        context="Summarize deployment risks for this release and call out rollback concerns.",
    )
    model = PromptListModel([prompt])

    index = model.index(0, 0)
    preview = index.data(PromptListModel.PreviewRole)

    assert isinstance(preview, str)
    assert "\n" not in preview
    assert preview.endswith("...")
    assert len(preview) <= PromptListModel.PreviewMaxLength
    assert preview.startswith("Use after collecting timeline notes,")


def test_prompt_list_model_exposes_title_and_preview_match_spans_for_active_search(
    qt_app: QApplication,
) -> None:
    """Active plain-text search should expose bounded match spans for visible row text."""
    prompt = _build_prompt(
        description="Summarise incident updates for the next on-call handoff.",
        scenarios=[],
        source="ops notebook",
    )
    model = PromptListModel([prompt])
    model.set_active_search_text("incident updates")

    index = model.index(0, 0)
    title = cast("str", index.data(Qt.ItemDataRole.DisplayRole))
    preview = cast("str", index.data(PromptListModel.PreviewRole))
    title_spans = cast("tuple[tuple[int, int], ...]", index.data(PromptListModel.TitleMatchRole))
    preview_spans = cast(
        "tuple[tuple[int, int], ...]",
        index.data(PromptListModel.PreviewMatchRole),
    )

    assert [title[start : start + length].lower() for start, length in title_spans] == ["incident"]
    assert [preview[start : start + length].lower() for start, length in preview_spans] == [
        "incident",
        "updates",
    ]


def test_prompt_list_model_emits_preview_role_when_search_changes_preview_choice(
    qt_app: QApplication,
) -> None:
    """Changing the active search should notify views when the preview text itself changes."""
    prompt = _build_prompt(
        description="Summarise incident updates for the next on-call handoff.",
        scenarios=[],
        source="PagerDuty ops notebook",
    )
    model = PromptListModel([prompt])
    emitted_roles: list[list[int]] = []

    model.dataChanged.connect(
        lambda _top_left, _bottom_right, roles: emitted_roles.append(list(roles))
    )

    model.set_active_search_text("notebook")

    assert emitted_roles == [
        [
            PromptListModel.PreviewRole,
            PromptListModel.TitleMatchRole,
            PromptListModel.PreviewMatchRole,
        ]
    ]


def test_prompt_list_model_keeps_match_roles_empty_without_active_search(
    qt_app: QApplication,
) -> None:
    """No active search should leave match roles empty while row text stays unchanged."""
    prompt = _build_prompt(description="Summarise incident updates for the next on-call handoff.")
    model = PromptListModel([prompt])

    index = model.index(0, 0)

    assert index.data(Qt.ItemDataRole.DisplayRole) == "Incident triage (Ops)"
    assert not index.data(PromptListModel.TitleMatchRole)
    assert not index.data(PromptListModel.PreviewMatchRole)


def test_prompt_list_delegate_returns_taller_rows_when_preview_exists(
    qt_app: QApplication,
) -> None:
    """Rows with a preview should reserve enough height for the second line."""
    with_preview = PromptListModel(
        [_build_prompt(description="", scenarios=[], source="support queue", context=" ")]
    )
    without_preview = PromptListModel(
        [_build_prompt(description="", scenarios=[], source="local", context=" ")]
    )
    delegate = PromptListDelegate()
    option = QStyleOptionViewItem()

    with_height = delegate.sizeHint(option, with_preview.index(0, 0)).height()
    without_height = delegate.sizeHint(option, without_preview.index(0, 0)).height()

    assert with_height > without_height


def test_prompt_list_delegate_keeps_preview_font_at_base_size(qt_app: QApplication) -> None:
    """Preview text should not shrink below the row base font size."""
    delegate = PromptListDelegate()
    option = QStyleOptionViewItem()
    option_any = cast("Any", option)
    title_font = cast("Any", option_any).font
    preview_font = delegate._preview_font(title_font)  # noqa: SLF001

    assert preview_font.pointSizeF() == title_font.pointSizeF()


def test_prompt_list_delegate_builds_emphasis_runs_from_match_spans(qt_app: QApplication) -> None:
    """Delegate should split text into plain and emphasized fragments for drawing."""
    delegate = PromptListDelegate()

    runs = delegate._build_text_runs(  # noqa: SLF001
        "Incident triage (Ops)",
        ((0, 8),),
    )

    assert runs == (("Incident", True), (" triage (Ops)", False))
