"""Prompt chain manager dialog tests.

Updates:
  v0.1.0 - 2025-12-04 - Ensure dialog populates details and executes chains.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from core import PromptChainRunResult, PromptChainStepRun
from gui.dialogs.prompt_chains import PromptChainManagerDialog
from models.prompt_chain_model import PromptChain, PromptChainStep


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    """Provide a shared Qt application instance for dialog tests."""

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _ManagerStub:
    def __init__(self) -> None:
        self._chain = _make_chain()
        self.saved_chain: PromptChain | None = None
        self.runs: list[dict[str, Any]] = []

    @property
    def repository(self):  # pragma: no cover - only used by DialogLauncher in real app
        return type("Repo", (), {"list": lambda *_: []})()

    def list_prompt_chains(self, include_inactive: bool = False):  # noqa: ARG002
        return [self._chain]

    def save_prompt_chain(self, chain: PromptChain) -> PromptChain:
        self.saved_chain = chain
        return chain

    def run_prompt_chain(self, chain_id: uuid.UUID, *, variables: dict[str, Any] | None = None):
        self.runs.append({"chain_id": chain_id, "variables": variables or {}})
        return PromptChainRunResult(
            chain=self._chain,
            variables=variables or {},
            outputs={"summary": "ok"},
            steps=[
                PromptChainStepRun(
                    step=self._chain.steps[0],
                    status="success",
                    outcome=None,
                    error=None,
                )
            ],
        )


def _make_chain() -> PromptChain:
    chain_id = uuid.uuid4()
    prompt_id = uuid.uuid4()
    step = PromptChainStep(
        id=uuid.uuid4(),
        chain_id=chain_id,
        prompt_id=prompt_id,
        order_index=1,
        input_template="{{ body }}",
        output_variable="result",
    )
    return PromptChain(
        id=chain_id,
        name="Demo Chain",
        description="Example",
        steps=[step],
    )


def test_prompt_chain_dialog_populates_details(qt_app: QApplication) -> None:
    """Dialog should load chains and render their metadata immediately."""

    manager = _ManagerStub()
    dialog = PromptChainManagerDialog(manager)
    try:
        assert dialog._chain_list.count() == 1  # noqa: SLF001 - accessed for test visibility
        assert dialog._detail_title.text() == "Demo Chain"
        assert "Steps" not in dialog._description_view.placeholderText()
        assert "Example" in dialog._description_view.toPlainText()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_runs_chain(qt_app: QApplication) -> None:
    """Running a chain should invoke the manager with parsed variables."""

    manager = _ManagerStub()
    dialog = PromptChainManagerDialog(manager)
    try:
        dialog._variables_input.setPlainText('{"foo": "bar"}')  # noqa: SLF001
        dialog._run_selected_chain()  # noqa: SLF001
        assert manager.runs
        assert manager.runs[0]["variables"] == {"foo": "bar"}
        assert "Outputs" in dialog._result_view.toPlainText()  # noqa: SLF001
    finally:
        dialog.close()
        dialog.deleteLater()
