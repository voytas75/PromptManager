"""Focused tests for bounded prompt reuse actions.

Updates:
  v0.1.2 - 2026-04-06 - Lock the prompt actions menu copy label to the shared Copy Prompt wording.
  v0.1.1 - 2026-04-06 - Lock prompt copying to the stored body with deterministic toast feedback.
  v0.1.0 - 2026-04-04 - Cover clipboard copy and non-executing workspace handoff paths.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import pytest

from gui.prompt_actions_controller import PromptActionsController
from gui.usage_logger import IntentUsageLogger
from gui.workspace_view_controller import WorkspaceViewController
from models.prompt_model import Prompt

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from collections.abc import Callable

    from gui.layout_state import WindowStateManager
    from gui.prompt_list_model import PromptListModel

try:
    from PySide6.QtWidgets import (
        QApplication,
        QLabel,
        QListView,
        QPlainTextEdit,
        QTabWidget,
        QWidget,
    )
except ImportError:  # pragma: no cover - optional dependency in test environments
    pytest.skip("PySide6 is not available", allow_module_level=True)


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    """Provide a shared Qt application instance for controller tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return cast("QApplication", app)


@dataclass
class _ExecuteState:
    last_task: str = ""
    history: tuple[str, ...] = ()


@dataclass
class _LayoutStateStub:
    execute_state: _ExecuteState = field(default_factory=_ExecuteState)

    def load_execute_context_state(self) -> _ExecuteState:
        """Return a stable execute-context state for controller bootstrap."""
        return self.execute_state

    def persist_last_execute_task(self, _task: str) -> None:
        """Accept persisted task writes without side effects."""
        return None

    def record_execute_task(self, _task: str, _history: object) -> None:
        """Accept execute history writes without side effects."""
        return None


class _DummyClipboard:
    def __init__(self) -> None:
        self.text: str | None = None

    def setText(self, text: str) -> None:  # noqa: N802 - Qt style API
        self.text = text


class _MenuAction:
    def __init__(self, text: str) -> None:
        self.text = text
        self.enabled = True

    def setEnabled(self, enabled: bool) -> None:
        """Store enabled state like a Qt action."""
        self.enabled = enabled


class _MenuStub:
    instances: list[_MenuStub] = []

    def __init__(self, _parent: QWidget) -> None:
        self.actions: list[_MenuAction] = []
        _MenuStub.instances.append(self)

    def addAction(self, text: str) -> _MenuAction:  # noqa: N802 - Qt style API
        action = _MenuAction(text)
        self.actions.append(action)
        return action

    def exec(self, _point: object) -> None:  # noqa: A003 - Qt style API
        return None


def _build_prompt(*, context: str | None, description: str) -> Prompt:
    """Create a minimal prompt for reuse-action tests."""
    return Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000125"),
        name="Reusable prompt",
        description=description,
        category="General",
        context=context,
        created_at=datetime(2026, 4, 4, 9, 0, tzinfo=UTC),
        last_modified=datetime(2026, 4, 4, 10, 30, tzinfo=UTC),
    )


def _build_controller(
    *,
    query_input: QPlainTextEdit,
    workspace_view: WorkspaceViewController | None,
    status_messages: list[tuple[str, int]],
    toast_messages: list[tuple[str, int]],
    execution_supplier: Callable[[], None],
) -> PromptActionsController:
    """Create a prompt-actions controller with bounded test doubles."""
    return PromptActionsController(
        parent=QWidget(),
        model=cast("PromptListModel", object()),
        list_view=QListView(),
        query_input=query_input,
        layout_state=cast("WindowStateManager", _LayoutStateStub()),
        workspace_view=workspace_view,
        execution_controller_supplier=execution_supplier,
        current_prompt_supplier=lambda: None,
        edit_callback=lambda: None,
        duplicate_callback=lambda _prompt: None,
        fork_callback=lambda _prompt: None,
        similar_callback=None,
        status_callback=lambda message, duration: status_messages.append((message, duration)),
        error_callback=lambda _title, _message: None,
        toast_callback=lambda message, duration: toast_messages.append((message, duration)),
        usage_logger=IntentUsageLogger(enabled=False),
    )


def test_open_prompt_in_workspace_seeds_text_without_running(qt_app: QApplication) -> None:
    """Workspace handoff should populate the editor and avoid execution."""
    query_input = QPlainTextEdit()
    workspace_view = WorkspaceViewController(
        query_input,
        QTabWidget(),
        QLabel(),
        status_callback=lambda *_: None,
        execution_controller_supplier=lambda: None,
        quick_action_controller_supplier=lambda: None,
    )
    status_messages: list[tuple[str, int]] = []
    toast_messages: list[tuple[str, int]] = []
    execution_calls = 0

    def _execution_supplier() -> None:
        nonlocal execution_calls
        execution_calls += 1
        return None

    controller = _build_controller(
        query_input=query_input,
        workspace_view=workspace_view,
        status_messages=status_messages,
        toast_messages=toast_messages,
        execution_supplier=_execution_supplier,
    )
    prompt = _build_prompt(context="Prompt body to reuse", description="Fallback description")

    controller.open_prompt_in_workspace(prompt)
    qt_app.processEvents()

    assert query_input.toPlainText() == "Prompt body to reuse"
    assert execution_calls == 0
    assert status_messages == []
    assert toast_messages == [("Opened 'Reusable prompt' in the workspace.", 2500)]


def test_copy_prompt_to_clipboard_copies_the_prompt_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clipboard copy should place only the stored prompt body on the clipboard."""
    clipboard = _DummyClipboard()
    monkeypatch.setattr("PySide6.QtGui.QGuiApplication.clipboard", lambda: clipboard)
    status_messages: list[tuple[str, int]] = []
    toast_messages: list[tuple[str, int]] = []
    controller = _build_controller(
        query_input=QPlainTextEdit(),
        workspace_view=None,
        status_messages=status_messages,
        toast_messages=toast_messages,
        execution_supplier=lambda: None,
    )
    prompt = _build_prompt(
        context="Prompt body to reuse immediately",
        description="Description fallback payload",
    )

    controller.copy_prompt_to_clipboard(prompt)

    assert clipboard.text == "Prompt body to reuse immediately"
    assert status_messages == []
    assert toast_messages == [("Copied 'Reusable prompt' to the clipboard.", 2500)]


def test_show_context_menu_uses_shared_copy_prompt_label(
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Context menu should label body-only prompt copying as Copy Prompt."""
    _MenuStub.instances.clear()
    monkeypatch.setattr("gui.prompt_actions_controller.QMenu", _MenuStub)
    query_input = QPlainTextEdit()
    list_view = QListView()
    status_messages: list[tuple[str, int]] = []
    toast_messages: list[tuple[str, int]] = []
    prompt = _build_prompt(context="Prompt body to reuse", description="Fallback description")

    controller = PromptActionsController(
        parent=QWidget(),
        model=cast("PromptListModel", object()),
        list_view=list_view,
        query_input=query_input,
        layout_state=cast("WindowStateManager", _LayoutStateStub()),
        workspace_view=None,
        execution_controller_supplier=lambda: None,
        current_prompt_supplier=lambda: prompt,
        edit_callback=lambda: None,
        duplicate_callback=lambda _prompt: None,
        fork_callback=lambda _prompt: None,
        similar_callback=None,
        status_callback=lambda message, duration: status_messages.append((message, duration)),
        error_callback=lambda _title, _message: None,
        toast_callback=lambda message, duration: toast_messages.append((message, duration)),
        usage_logger=IntentUsageLogger(enabled=False),
    )

    controller.show_context_menu(list_view.viewport().rect().center())
    qt_app.processEvents()

    assert len(_MenuStub.instances) == 1
    menu_texts = [action.text for action in _MenuStub.instances[0].actions]
    assert "Copy Prompt" in menu_texts
    assert "Copy Prompt Text" not in menu_texts
