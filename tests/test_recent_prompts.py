"""Tests for the bounded recent prompt reopen workflow.

Updates:
  v0.1.0 - 2026-04-04 - Cover deterministic recent ordering and selection handoff.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast

from PySide6.QtWidgets import QDialog

from gui.dialogs.recent_prompts import recent_prompts
from gui.main_window_handlers import PromptActionsHandler
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
