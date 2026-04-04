"""Tests covering :mod:`gui.prompt_editor_flow` helpers.

Updates:
  v0.1.2 - 2026-04-04 - Add quick-capture happy-path coverage for draft creation handoff.
  v0.1.1 - 2025-12-08 - Cast QWidget parent and use DialogCode.Accepted for Pyright.
  v0.1.0 - 2025-12-02 - Ensure delete flow skips confirmation via keyword argument.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from PySide6.QtWidgets import QDialog

from gui.dialogs.quick_capture import QuickCaptureDraft
from gui.prompt_editor_flow import PromptEditorFlow
from models.prompt_model import Prompt

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Callable

    from PySide6.QtWidgets import QWidget

    from core import PromptManager
    from gui.prompt_editor_flow import PromptDialogFactory, QuickCaptureDialogFactory
else:  # pragma: no cover - runtime placeholders to avoid heavy imports
    PromptManager = object  # type: ignore[assignment]
    PromptDialogFactory = object  # type: ignore[assignment]
    QuickCaptureDialogFactory = object  # type: ignore[assignment]
    QWidget = object  # type: ignore[assignment]
    Callable = object  # type: ignore[assignment]


class _SignalStub:
    def connect(self, _callback: object) -> None:  # noqa: D401 - testing helper
        """Record callback without invoking it."""


@dataclass
class _DialogStub:
    delete_requested: bool
    result_prompt: Prompt | None = None
    dialog_code: int = QDialog.DialogCode.Accepted
    applied: _SignalStub = field(default_factory=_SignalStub)

    def exec(self) -> int:
        """Return the pre-configured dialog result."""
        return self.dialog_code

    def prefill_from_prompt(self, _prompt: Prompt) -> None:
        """Mirror the PromptDialog API used by duplicate flows."""

    def setWindowTitle(self, _title: str) -> None:
        """Mirror the PromptDialog API used by fork flows."""


@dataclass
class _DialogFactoryStub:
    dialog: _DialogStub
    built_prompts: list[Prompt | None]

    def build(self, *args: Any, prompt: Prompt | None = None, **__: Any) -> _DialogStub:
        """Return the shared dialog and record the requested prompt."""
        captured_prompt = prompt
        if captured_prompt is None and len(args) >= 2:
            maybe_prompt = args[1]
            if isinstance(maybe_prompt, Prompt):
                captured_prompt = maybe_prompt
        self.built_prompts.append(captured_prompt)
        return self.dialog


@dataclass
class _QuickCaptureDialogStub:
    result_draft: QuickCaptureDraft | None = None
    dialog_code: int = QDialog.DialogCode.Accepted

    def exec(self) -> int:
        """Return the pre-configured dialog result."""
        return self.dialog_code


@dataclass
class _QuickCaptureDialogFactoryStub:
    dialog: _QuickCaptureDialogStub

    def build(self, *_: Any, **__: Any) -> _QuickCaptureDialogStub:
        """Return the shared quick-capture dialog stub."""
        return self.dialog


class _ManagerStub:
    def __init__(self) -> None:
        """Track created prompts for happy-path assertions."""
        self.created_prompts: list[Prompt] = []

    def create_prompt(self, prompt: Prompt) -> Prompt:
        """Record and return the created prompt."""
        self.created_prompts.append(prompt)
        return prompt


def _build_flow(
    *,
    manager: PromptManager,
    dialog_factory: PromptDialogFactory,
    quick_capture_dialog_factory: QuickCaptureDialogFactory,
    delete_prompt: Callable[..., None],
    load_prompts: Callable[[str], None],
    select_prompt: Callable[[uuid.UUID], None],
    status_callback: Callable[[str, int], None],
) -> PromptEditorFlow:
    """Create a PromptEditorFlow with lightweight doubles."""
    return PromptEditorFlow(
        parent=cast("QWidget", object()),
        manager=manager,
        dialog_factory=dialog_factory,
        quick_capture_dialog_factory=quick_capture_dialog_factory,
        load_prompts=load_prompts,
        current_search_text=lambda: "",
        select_prompt=select_prompt,
        delete_prompt=delete_prompt,
        status_callback=status_callback,
        error_callback=lambda *_: None,
    )


def test_edit_prompt_delete_requests_skip_confirmation_keyword() -> None:
    """Delete action in editor flow must force confirmation bypass via keyword."""
    captured: dict[str, Any] = {}

    def _delete_prompt(prompt: Prompt, *, skip_confirmation: bool = False) -> None:
        captured["prompt"] = prompt
        captured["skip_confirmation"] = skip_confirmation

    dialog_factory = cast(
        "PromptDialogFactory",
        _DialogFactoryStub(_DialogStub(delete_requested=True), []),
    )
    quick_capture_factory = cast(
        "QuickCaptureDialogFactory",
        _QuickCaptureDialogFactoryStub(
            _QuickCaptureDialogStub(dialog_code=QDialog.DialogCode.Rejected)
        ),
    )
    flow = _build_flow(
        manager=cast("PromptManager", object()),
        dialog_factory=dialog_factory,
        quick_capture_dialog_factory=quick_capture_factory,
        delete_prompt=_delete_prompt,
        load_prompts=lambda _: None,
        select_prompt=lambda _: None,
        status_callback=lambda *_: None,
    )
    prompt = Prompt(
        id=uuid.uuid4(),
        name="Test",
        description="Desc",
        category="General",
    )

    flow.edit_prompt(prompt)

    assert captured == {"prompt": prompt, "skip_confirmation": True}


def test_quick_capture_creates_draft_selects_prompt_and_opens_editor() -> None:
    """Quick capture should persist a draft and hand it off to the editor flow."""
    manager = _ManagerStub()
    load_calls: list[str] = []
    selected_ids: list[uuid.UUID] = []
    status_messages: list[tuple[str, int]] = []
    editor_builds: list[Prompt | None] = []

    dialog_factory = cast(
        "PromptDialogFactory",
        _DialogFactoryStub(
            _DialogStub(
                delete_requested=False,
                dialog_code=QDialog.DialogCode.Rejected,
            ),
            editor_builds,
        ),
    )
    quick_capture_factory = cast(
        "QuickCaptureDialogFactory",
        _QuickCaptureDialogFactoryStub(
            _QuickCaptureDialogStub(
                result_draft=QuickCaptureDraft(
                    body="Summarize this incident report in three bullets.\nHighlight the risk.",
                    source_label="chat thread",
                    tags_text="incident, summary",
                )
            )
        ),
    )
    flow = _build_flow(
        manager=cast("PromptManager", manager),
        dialog_factory=dialog_factory,
        quick_capture_dialog_factory=quick_capture_factory,
        delete_prompt=lambda *_args, **_kwargs: None,
        load_prompts=load_calls.append,
        select_prompt=selected_ids.append,
        status_callback=lambda message, duration: status_messages.append((message, duration)),
    )

    flow.quick_capture_prompt()

    assert len(manager.created_prompts) == 1
    created = manager.created_prompts[0]
    assert created.name == "Summarize this incident report in three bullets."
    assert created.description == "Quick capture draft."
    assert created.category == "General"
    assert created.tags == ["incident", "summary"]
    assert created.context == (
        "Summarize this incident report in three bullets.\nHighlight the risk."
    )
    assert created.source == "chat thread"
    assert created.ext2 == {
        "capture_state": "draft",
        "capture_method": "quick_capture",
    }
    assert load_calls == [""]
    assert selected_ids == [created.id]
    assert status_messages == [("Draft prompt created.", 4000)]
    assert editor_builds == [created]
