"""Tests for the bounded recent prompt reopen workflow.

Updates:
  v0.1.0 - 2026-04-04 - Cover deterministic recent ordering and selection handoff.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast

from PySide6.QtWidgets import QApplication, QDialog, QLabel, QMessageBox

from core import RepositoryError
from gui.dialogs.recent_prompts import RecentPromptsDialog, recent_prompts
from gui.main_window import MainWindow
from gui.main_window_handlers import PromptActionsHandler
from gui.widgets.prompt_filter_panel import PromptFilterPanel
from gui.widgets.prompt_toolbar import PromptToolbar
from models.prompt_model import Prompt

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Sequence

    from PySide6.QtWidgets import QWidget

    from core import PromptManager
    from gui.dialogs.recent_prompts import RecentPromptsDialogFactory
    from gui.main_window_handlers import (
        CatalogWorkflowController,
        DialogLauncher,
        PromptActionsController,
        PromptEditorFlow,
        PromptSearchController,
        SettingsWorkflow,
        ShareWorkflowCoordinator,
    )
    from gui.widgets import PromptDetailWidget
else:  # pragma: no cover - runtime placeholders
    PromptManager = object  # type: ignore[assignment]
    RecentPromptsDialogFactory = object  # type: ignore[assignment]
    PromptSearchController = object  # type: ignore[assignment]
    PromptActionsController = object  # type: ignore[assignment]
    PromptEditorFlow = object  # type: ignore[assignment]
    CatalogWorkflowController = object  # type: ignore[assignment]
    SettingsWorkflow = object  # type: ignore[assignment]
    DialogLauncher = object  # type: ignore[assignment]
    ShareWorkflowCoordinator = object  # type: ignore[assignment]
    PromptDetailWidget = object  # type: ignore[assignment]
    QWidget = object  # type: ignore[assignment]


import pytest


@pytest.fixture
def qt_app() -> QApplication:
    """Provide a shared Qt application instance for recent prompt dialog tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return cast("QApplication", app)


def _build_prompt(
    *,
    prompt_id: uuid.UUID,
    name: str,
    modified_at: datetime,
) -> Prompt:
    """Create a minimal prompt object for recent prompt tests."""
    return Prompt(
        id=prompt_id,
        name=name,
        description=f"{name} description",
        category="General",
        context=f"{name} body",
        created_at=modified_at - timedelta(days=1),
        last_modified=modified_at,
    )


@dataclass
class _RecentPromptsDialogStub:
    selected_prompt_id: uuid.UUID | None
    dialog_code: int = QDialog.DialogCode.Accepted

    def exec(self) -> int:
        """Return the preconfigured dialog result."""
        return self.dialog_code


@dataclass
class _RecentPromptsDialogFactoryStub:
    dialog: _RecentPromptsDialogStub
    built_prompts: list[list[Prompt]]

    def build(self, _parent: object, prompts: Sequence[Prompt]) -> _RecentPromptsDialogStub:
        """Capture ordered prompts before returning the shared dialog."""
        self.built_prompts.append(list(prompts))
        return self.dialog


def test_recent_prompts_dialog_summary_mentions_reopen_for_further_work(
    qt_app: QApplication,
) -> None:
    """Recent dialog summary should make the reopen-and-continue posture explicit."""
    prompt = _build_prompt(
        prompt_id=uuid.UUID("00000000-0000-0000-0000-0000000000ac"),
        name="Reopen Intent Prompt",
        modified_at=datetime(2026, 4, 28, 10, 0, tzinfo=UTC),
    )

    dialog = RecentPromptsDialog([prompt])
    summary = dialog.findChildren(QLabel)[0]

    assert summary.text() == (
        "Reopen one of the prompts you touched most recently to continue refining it."
    )


def test_open_recent_prompts_orders_by_last_modified_and_selects_prompt() -> None:
    """Recent prompt flow should sort deterministically and reuse the selection path."""
    newest = _build_prompt(
        prompt_id=uuid.UUID("00000000-0000-0000-0000-000000000003"),
        name="Newest",
        modified_at=datetime(2026, 4, 4, 12, 0, tzinfo=UTC),
    )
    alpha = _build_prompt(
        prompt_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
        name="Alpha",
        modified_at=datetime(2026, 4, 3, 9, 0, tzinfo=UTC),
    )
    beta = _build_prompt(
        prompt_id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
        name="Beta",
        modified_at=datetime(2026, 4, 3, 9, 0, tzinfo=UTC),
    )
    prompts = [beta, newest, alpha]
    selected_prompt_ids: list[uuid.UUID] = []
    status_messages: list[tuple[str, int]] = []
    factory = _RecentPromptsDialogFactoryStub(
        dialog=_RecentPromptsDialogStub(selected_prompt_id=alpha.id),
        built_prompts=[],
    )
    handler = PromptActionsHandler(
        parent=cast("QWidget", object()),
        manager=cast("PromptManager", object()),
        model_prompts_supplier=lambda: prompts,
        recent_catalog_prompts_supplier=lambda: prompts,
        current_prompt_supplier=lambda: None,
        detail_widget=cast("PromptDetailWidget", object()),
        prompt_search_controller=cast("PromptSearchController", object()),
        prompt_actions_controller_supplier=lambda: cast("PromptActionsController | None", None),
        prompt_editor_flow_supplier=lambda: cast("PromptEditorFlow | None", None),
        catalog_controller_supplier=lambda: cast("CatalogWorkflowController | None", None),
        settings_workflow_supplier=lambda: cast("SettingsWorkflow | None", None),
        dialog_launcher_supplier=lambda: cast("DialogLauncher | None", None),
        share_workflow_supplier=lambda: cast("ShareWorkflowCoordinator | None", None),
        recent_prompts_dialog_factory=cast("RecentPromptsDialogFactory", factory),
        select_prompt=selected_prompt_ids.append,
        reveal_recent_prompt=selected_prompt_ids.append,
        load_prompts=lambda _text: None,
        current_search_text=lambda: "",
        status_callback=lambda message, duration: status_messages.append((message, duration)),
        exit_callback=lambda: None,
    )

    ordered = recent_prompts(prompts)
    handler.open_recent_prompts()

    assert [prompt.id for prompt in ordered] == [newest.id, alpha.id, beta.id]
    assert factory.built_prompts and [prompt.id for prompt in factory.built_prompts[0]] == [
        newest.id,
        alpha.id,
        beta.id,
    ]
    assert selected_prompt_ids == [alpha.id]
    assert status_messages == []


def test_recent_can_reopen_a_prompt_hidden_by_search_results() -> None:
    """Recent uses the catalog, not the current result subset, and reveals its selection."""
    recent = _build_prompt(
        prompt_id=uuid.UUID("00000000-0000-0000-0000-000000000011"),
        name="Recent outside search",
        modified_at=datetime(2026, 4, 4, 12, 0, tzinfo=UTC),
    )
    visible = _build_prompt(
        prompt_id=uuid.UUID("00000000-0000-0000-0000-000000000012"),
        name="Visible search result",
        modified_at=datetime(2026, 4, 3, 12, 0, tzinfo=UTC),
    )
    factory = _RecentPromptsDialogFactoryStub(
        dialog=_RecentPromptsDialogStub(selected_prompt_id=recent.id), built_prompts=[]
    )
    revealed: list[uuid.UUID] = []
    handler = PromptActionsHandler(
        parent=cast("QWidget", object()),
        manager=cast("PromptManager", object()),
        model_prompts_supplier=lambda: [visible],
        recent_catalog_prompts_supplier=lambda: [visible, recent],
        current_prompt_supplier=lambda: visible,
        detail_widget=cast("PromptDetailWidget", object()),
        prompt_search_controller=cast("PromptSearchController", object()),
        prompt_actions_controller_supplier=lambda: None,
        prompt_editor_flow_supplier=lambda: None,
        catalog_controller_supplier=lambda: None,
        settings_workflow_supplier=lambda: None,
        dialog_launcher_supplier=lambda: None,
        share_workflow_supplier=lambda: None,
        recent_prompts_dialog_factory=cast("RecentPromptsDialogFactory", factory),
        select_prompt=revealed.append,
        reveal_recent_prompt=revealed.append,
        load_prompts=lambda _text: None,
        current_search_text=lambda: "visible",
        status_callback=lambda _message, _duration: None,
        exit_callback=lambda: None,
    )

    handler.open_recent_prompts()

    assert [prompt.id for prompt in factory.built_prompts[0]] == [recent.id, visible.id]
    assert revealed == [recent.id]


def test_recent_catalog_read_failure_preserves_current_view() -> None:
    """A failed catalog read must report an error without opening or changing Recent."""
    from types import SimpleNamespace
    from unittest.mock import patch

    def failed_read() -> list[Prompt]:
        raise RepositoryError("catalog unavailable")

    errors: list[tuple[str, str]] = []

    def record_error(_parent: QWidget, title: str, message: str) -> None:
        errors.append((title, message))

    window = SimpleNamespace(
        _recent_catalog_prompts_supplier=failed_read,
        _parent=cast("QWidget", object()),
    )

    with patch.object(QMessageBox, "critical", side_effect=record_error):
        PromptActionsHandler.open_recent_prompts(
            cast("PromptActionsHandler", cast("object", window))
        )

    assert errors == [("Unable to load prompts", "catalog unavailable")]


def test_reveal_hidden_recent_resets_narrowing_before_selection(qt_app: QApplication) -> None:
    """A hidden recent entry needs truthful list/search/filter cues before detail selection."""
    from types import SimpleNamespace

    from models.category_model import PromptCategory

    wanted_id = uuid.uuid4()
    other_id = uuid.uuid4()
    toolbar = PromptToolbar()
    toolbar.set_search_text("other")
    panel = PromptFilterPanel(sort_options=[("Name", "name_asc")])
    panel.set_categories([PromptCategory(slug="ops", label="Ops", description="")], "ops")
    panel.set_tags(["ops"], "ops")
    panel.set_favorites_only(True)
    panel.set_min_quality(7.0)
    panel.set_active_search_text("other")
    panel.set_sort_enabled(False)
    events: list[object] = []

    def record_load() -> list[Prompt]:
        events.append("load")
        return [
            _build_prompt(
                prompt_id=wanted_id,
                name="Hidden",
                modified_at=datetime(2026, 4, 4, 12, 0, tzinfo=UTC),
            )
        ]

    def record_selection(prompt_id: uuid.UUID) -> None:
        events.append(("select", prompt_id))

    def record_display(_prompts: Sequence[Prompt], prompt_id: uuid.UUID) -> None:
        assert toolbar.search_text() == ""
        assert panel.category_slug() is None
        assert panel.tag_value() is None
        assert not panel.favorites_only()
        assert panel.min_quality() == 0.0
        assert panel.is_sort_enabled()
        events.append(("display", prompt_id))

    presenter = SimpleNamespace(
        load_catalog_for_recent=record_load,
        display_catalog_for_recent=record_display,
    )
    window = SimpleNamespace(
        _model=SimpleNamespace(prompts=lambda: [SimpleNamespace(id=other_id)]),
        _prompt_presenter=presenter,
        _toolbar=toolbar,
        _filter_panel=panel,
        _prompt_search_controller=SimpleNamespace(
            reset_search_state=lambda: events.append("reset")
        ),
        _layout_controller=SimpleNamespace(
            persist_filter_preferences=lambda: events.append("persist")
        ),
        _select_prompt=record_selection,
    )

    MainWindow._reveal_recent_prompt(  # pyright: ignore[reportPrivateUsage]
        cast("MainWindow", cast("object", window)), wanted_id
    )

    assert toolbar.search_text() == ""
    assert panel.category_slug() is None
    assert panel.tag_value() is None
    assert not panel.favorites_only()
    assert panel.min_quality() == 0.0
    assert panel.is_sort_enabled()
    summary = panel.findChild(QLabel, "activeNarrowingSummaryLabel")
    assert summary is not None and summary.text() == "Showing all prompts"
    assert events == ["load", "reset", ("display", wanted_id), "persist"]


def test_failed_hidden_recent_reload_keeps_old_search_and_filters(qt_app: QApplication) -> None:
    """No selection or misleading all-prompts cues when the underlying reload fails."""
    from types import SimpleNamespace

    wanted_id = uuid.uuid4()
    other_id = uuid.uuid4()
    toolbar = PromptToolbar()
    toolbar.set_search_text("other")
    panel = PromptFilterPanel(sort_options=[("Name", "name_asc")])
    panel.set_favorites_only(True)
    panel.set_active_search_text("other")
    panel.set_sort_enabled(False)
    events: list[object] = []

    def failed_reload() -> list[Prompt] | None:
        events.append("load")
        return None

    def record_selection(prompt_id: uuid.UUID) -> None:
        events.append(("select", prompt_id))

    window = SimpleNamespace(
        _model=SimpleNamespace(prompts=lambda: [SimpleNamespace(id=other_id)]),
        _prompt_presenter=SimpleNamespace(load_catalog_for_recent=failed_reload),
        _toolbar=toolbar,
        _filter_panel=panel,
        _layout_controller=SimpleNamespace(
            persist_filter_preferences=lambda: events.append("persist")
        ),
        _select_prompt=record_selection,
    )

    MainWindow._reveal_recent_prompt(  # pyright: ignore[reportPrivateUsage]
        cast("MainWindow", cast("object", window)), wanted_id
    )

    assert toolbar.search_text() == "other"
    assert panel.favorites_only()
    assert not panel.is_sort_enabled()
    summary = panel.findChild(QLabel, "activeNarrowingSummaryLabel")
    assert summary is not None and summary.text() == (
        "Showing prompts narrowed by search: other • favorites only"
    )
    assert events == ["load"]


def test_reveal_visible_recent_prompt_keeps_search_and_filters(qt_app: QApplication) -> None:
    """Reopening a result already on screen should not reset the operator's view."""
    from types import SimpleNamespace

    wanted_id = uuid.uuid4()
    toolbar = PromptToolbar()
    toolbar.set_search_text("match")
    events: list[object] = []

    def record_selection(prompt_id: uuid.UUID) -> None:
        events.append(("select", prompt_id))

    window = SimpleNamespace(
        _model=SimpleNamespace(prompts=lambda: [SimpleNamespace(id=wanted_id)]),
        _toolbar=toolbar,
        _select_prompt=record_selection,
    )

    MainWindow._reveal_recent_prompt(  # pyright: ignore[reportPrivateUsage]
        cast("MainWindow", cast("object", window)), wanted_id
    )

    assert toolbar.search_text() == "match"
    assert events == [("select", wanted_id)]


def _first_row_labels(dialog: RecentPromptsDialog) -> list[str]:
    """Return visible QLabel texts for the first recent-prompt row."""
    labels = dialog.findChildren(QLabel)
    return [label.text() for label in labels[1:3]]


def test_recent_prompts_dialog_shows_visible_row_metadata(qt_app: QApplication) -> None:
    """Recent dialog rows should show compact visible metadata without tooltip-only reliance."""
    prompt = _build_prompt(
        prompt_id=uuid.UUID("00000000-0000-0000-0000-0000000000aa"),
        name="Visible Metadata Prompt",
        modified_at=datetime(2026, 4, 28, 8, 14, tzinfo=UTC),
    )
    prompt.category = "Research"

    dialog = RecentPromptsDialog([prompt])

    assert _first_row_labels(dialog) == [
        "Visible Metadata Prompt",
        "Modified 2026-04-28 08:14 UTC • Category: Research",
    ]


def test_recent_prompts_dialog_uses_uncategorised_fallback_in_visible_metadata(
    qt_app: QApplication,
) -> None:
    """Recent dialog metadata should reuse the uncategorised fallback in visible rows."""
    prompt = _build_prompt(
        prompt_id=uuid.UUID("00000000-0000-0000-0000-0000000000ab"),
        name="No Category Prompt",
        modified_at=datetime(2026, 4, 28, 9, 0, tzinfo=UTC),
    )
    prompt.category = "   "

    dialog = RecentPromptsDialog([prompt])

    assert _first_row_labels(dialog) == [
        "No Category Prompt",
        "Modified 2026-04-28 09:00 UTC • Category: Uncategorised",
    ]
