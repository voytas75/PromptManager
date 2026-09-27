"""Provider-free GUI execution guards for inactive catalog prompts."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from core import PromptManager, PromptRepository
from gui.controllers.execution_controller import ExecutionController
from models.prompt_model import Prompt

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_gui_context_refuses_inactive_catalog_prompt_before_web_or_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = PromptRepository(str(tmp_path / "catalog.db"))
    prompt = Prompt(
        id=uuid.uuid4(),
        name="Archived prompt",
        description="test",
        category="test",
        context="old body",
    )
    repository.add(prompt)
    manager = PromptManager(
        chroma_path=str(tmp_path / "chroma"),
        db_path=str(tmp_path / "catalog.db"),
        repository=repository,
        enable_background_sync=False,
    )
    try:
        repository.set_prompt_active(prompt.id, active=False, expect_active=True)
        controller = object.__new__(ExecutionController)
        controller._manager = manager  # type: ignore[reportPrivateUsage]
        failures: list[str] = []

        def record_failure(_title: str, text: str) -> None:
            failures.append(text)

        def noop_status(_message: str, _duration: int) -> None:
            return None

        monkeypatch.setattr(controller, "_error", record_failure, raising=False)
        monkeypatch.setattr(controller, "_status", noop_status, raising=False)

        def fail_web(*_args: object, **_kwargs: object) -> str:
            raise AssertionError("web enrichment reached for inactive prompt")

        monkeypatch.setattr(controller, "_maybe_enrich_request", fail_web, raising=False)
        manager._executor = object()  # type: ignore[reportPrivateUsage,reportAttributeAccessIssue]
        monkeypatch.setattr(type(manager), "llm_available", property(lambda _self: True))
        controller.execute_prompt_as_context(prompt, task_text="request", context_text="context")
        assert failures and "inactive" in failures[0]
    finally:
        manager.close()
