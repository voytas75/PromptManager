"""Tests covering :mod:`gui.prompt_editor_flow` helpers.

Updates:
  v0.1.0 - 2025-12-02 - Ensure delete flow skips confirmation via keyword argument.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from PySide6.QtWidgets import QDialog

from gui.prompt_editor_flow import PromptEditorFlow
from models.prompt_model import Prompt

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from core import PromptManager
    from gui.prompt_editor_flow import PromptDialogFactory
else:  # pragma: no cover - runtime placeholders to avoid heavy imports
    PromptManager = object  # type: ignore[assignment]
    PromptDialogFactory = object  # type: ignore[assignment]


class _SignalStub:
    def connect(self, _callback) -> None:  # noqa: D401 - testing helper
        """Record callback without invoking it."""


@dataclass
class _DialogStub:
    delete_requested: bool
    result_prompt: Prompt | None = None
    applied: _SignalStub = _SignalStub()

    def exec(self) -> int:
        return QDialog.Accepted


@dataclass
class _DialogFactoryStub:
    dialog: _DialogStub

    def build(self, *_: Any, **__: Any) -> _DialogStub:
        return self.dialog


def test_edit_prompt_delete_requests_skip_confirmation_keyword() -> None:
    """Delete action in editor flow must force confirmation bypass via keyword."""

    captured: dict[str, Any] = {}

    def _delete_prompt(prompt: Prompt, *, skip_confirmation: bool = False) -> None:
        captured["prompt"] = prompt
        captured["skip_confirmation"] = skip_confirmation

    dialog = _DialogStub(delete_requested=True)
    dialog_factory = cast("PromptDialogFactory", _DialogFactoryStub(dialog))

    flow = PromptEditorFlow(
        parent=object(),
        manager=cast("PromptManager", object()),
        dialog_factory=dialog_factory,
        load_prompts=lambda _: None,
        current_search_text=lambda: "",
        select_prompt=lambda _: None,
        delete_prompt=_delete_prompt,
        status_callback=lambda *_: None,
        error_callback=lambda *_: None,
    )

    prompt = Prompt(
        id=uuid.uuid4(),
        name="Test",
        description="Desc",
        category="General",
    )

    flow.edit_prompt(prompt)

    assert captured == {"prompt": prompt, "skip_confirmation": True}
