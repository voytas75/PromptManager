"""Prompt chain manager dialog tests.

Updates:
  v0.2.0 - 2025-12-04 - Cover GUI CRUD helpers via editor and manager dialog actions.
  v0.1.0 - 2025-12-04 - Ensure dialog populates details and executes chains.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from core import PromptChainRunResult, PromptChainStepRun
from gui.dialogs.prompt_chain_editor import PromptChainEditorDialog
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
        self._chains = [_make_chain()]
        self.saved_chain: PromptChain | None = None
        self.runs: list[dict[str, Any]] = []
        self.deleted_chain_ids: list[uuid.UUID] = []

    @property
    def repository(self):  # pragma: no cover - only used by DialogLauncher in real app
        return type("Repo", (), {"list": lambda *_: []})()

    def list_prompt_chains(self, include_inactive: bool = False):  # noqa: ARG002
        return list(self._chains)

    def save_prompt_chain(self, chain: PromptChain) -> PromptChain:
        self.saved_chain = chain
        for index, existing in enumerate(self._chains):
            if existing.id == chain.id:
                self._chains[index] = chain
                break
        else:
            self._chains.append(chain)
        return chain

    def run_prompt_chain(self, chain_id: uuid.UUID, *, variables: dict[str, Any] | None = None):
        self.runs.append({"chain_id": chain_id, "variables": variables or {}})
        chain = next((entry for entry in self._chains if entry.id == chain_id), self._chains[0])
        return PromptChainRunResult(
            chain=chain,
            variables=variables or {},
            outputs={"summary": "ok"},
            steps=[
                PromptChainStepRun(
                    step=chain.steps[0],
                    status="success",
                    outcome=None,
                    error=None,
                )
            ],
        )

    def delete_prompt_chain(self, chain_id: uuid.UUID) -> None:
        self.deleted_chain_ids.append(chain_id)
        self._chains = [chain for chain in self._chains if chain.id != chain_id]


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


def test_prompt_chain_editor_dialog_creates_chain(qt_app: QApplication) -> None:
    """Editor dialog should build a PromptChain when inputs are valid."""

    editor = PromptChainEditorDialog(None, manager=None)
    editor._name_input.setText("New Chain")  # noqa: SLF001
    editor._description_input.setPlainText("Describe")  # noqa: SLF001
    step = PromptChainStep(
        id=uuid.uuid4(),
        chain_id=editor._chain_id,  # noqa: SLF001
        prompt_id=uuid.uuid4(),
        order_index=1,
        input_template="{{ body }}",
        output_variable="result",
    )
    editor._steps = [step]  # noqa: SLF001
    editor._handle_accept()  # noqa: SLF001
    chain = editor.result_chain()
    assert chain is not None
    assert chain.name == "New Chain"
    assert len(chain.steps) == 1


def test_prompt_chain_manager_creates_chain_via_editor(
    qt_app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Manager dialog should persist the chain returned by the editor."""

    manager = _ManagerStub()
    dialog = PromptChainManagerDialog(manager)

    class _EditorStub:
        def __init__(self, *_: object, **__: object) -> None:
            self._chain = _make_chain()

        def exec(self) -> int:
            return QDialog.Accepted

        def result_chain(self) -> PromptChain:
            return self._chain

    monkeypatch.setattr(
        "gui.dialogs.prompt_chains.PromptChainEditorDialog",
        _EditorStub,
    )
    dialog._create_chain()  # noqa: SLF001
    assert manager.saved_chain is not None
    dialog.close()
    dialog.deleteLater()


def test_prompt_chain_manager_deletes_chain(
    qt_app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Delete action should invoke PromptManager.delete_prompt_chain."""

    manager = _ManagerStub()
    dialog = PromptChainManagerDialog(manager)
    monkeypatch.setattr(
        "gui.dialogs.prompt_chains.QMessageBox.question",
        lambda *_, **__: QMessageBox.Yes,
    )
    dialog._delete_chain()  # noqa: SLF001
    assert manager.deleted_chain_ids
    dialog.close()
    dialog.deleteLater()
