"""Prompt chain manager dialog tests.

Updates:
  v0.2.6 - 2025-12-05 - Cover summarize toggle persistence and UI output section.
  v0.2.5 - 2025-12-05 - Ensure step IO renders outside code fences for Markdown formatting.
  v0.2.4 - 2025-12-05 - Verify Markdown toggle preserves rendered text.
  v0.2.3 - 2025-12-05 - Cover streaming preview rendering and settings detection.
  v0.2.2 - 2025-12-05 - Validate expanded chain result text sections.
  v0.2.1 - 2025-12-05 - Adjust assertions for new description label.
  v0.2.0 - 2025-12-04 - Cover GUI CRUD helpers via editor and manager dialog actions.
  v0.1.0 - 2025-12-04 - Ensure dialog populates details and executes chains.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox, QTextEdit

from core import PromptChainRunResult, PromptChainStepRun
from core.execution import CodexExecutionResult
from core.prompt_manager.execution_history import ExecutionOutcome
from gui.dialogs.prompt_chain_editor import PromptChainEditorDialog
from gui.dialogs.prompt_chains import PromptChainManagerDialog
from models.prompt_chain_model import PromptChain, PromptChainStep

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from collections.abc import Callable, Mapping


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    """Provide a shared Qt application instance for dialog tests."""

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _ManagerStub:
    def __init__(
        self,
        *,
        stream_enabled: bool = False,
        stream_chunks: tuple[str, ...] = (),
        step_request_text: str = "{{ body }}",
        step_response_text: str = "Demo response",
        step_reasoning_text: str | None = None,
    ) -> None:
        self._chains = [_make_chain()]
        self.saved_chain: PromptChain | None = None
        self.runs: list[dict[str, Any]] = []
        self.deleted_chain_ids: list[uuid.UUID] = []
        self._executor = SimpleNamespace(stream=stream_enabled) if stream_enabled else None
        self._litellm_stream = stream_enabled
        self._stream_chunks = stream_chunks
        self.received_stream_callback: Callable[[PromptChainStep, str], None] | None = None
        self._step_request_text = step_request_text
        self._step_response_text = step_response_text
        self._step_reasoning_text = step_reasoning_text

    @property
    def repository(self):  # pragma: no cover - only used by DialogLauncher in real app
        return type("Repo", (), {"list": lambda *_: []})()

    def list_prompt_chains(self, include_inactive: bool = False):  # noqa: ARG002
        return list(self._chains)

    def list_prompts(self, limit: int | None = None):  # noqa: ARG002
        return [SimpleNamespace(id=uuid.uuid4(), name="Example Prompt")]

    def save_prompt_chain(self, chain: PromptChain) -> PromptChain:
        self.saved_chain = chain
        for index, existing in enumerate(self._chains):
            if existing.id == chain.id:
                self._chains[index] = chain
                break
        else:
            self._chains.append(chain)
        return chain

    def run_prompt_chain(
        self,
        chain_id: uuid.UUID,
        *,
        variables: dict[str, Any] | None = None,
        stream_callback: Callable[[PromptChainStep, str, bool], None] | None = None,
    ):
        self.runs.append({"chain_id": chain_id, "variables": variables or {}})
        self.received_stream_callback = stream_callback
        chain = next((entry for entry in self._chains if entry.id == chain_id), self._chains[0])
        if stream_callback is not None:
            for chunk in self._stream_chunks:
                stream_callback(chain.steps[0], chunk, False)
            stream_callback(chain.steps[0], "final text", True)
        step = chain.steps[0]
        raw_response: Mapping[str, Any] = {}
        if self._step_reasoning_text:
            raw_response = {
                "output": [
                    {
                        "content": [
                            {"type": "reasoning", "text": self._step_reasoning_text},
                        ]
                    }
                ]
            }
        execution_result = CodexExecutionResult(
            prompt_id=step.prompt_id,
            request_text=self._step_request_text,
            response_text=self._step_response_text,
            duration_ms=123,
            usage={},
            raw_response=raw_response,
        )
        outcome = ExecutionOutcome(result=execution_result, history_entry=None, conversation=[])
        return PromptChainRunResult(
            chain=chain,
            variables=variables or {},
            outputs={"summary": "ok"},
            steps=[
                PromptChainStepRun(
                    step=step,
                    status="success",
                    outcome=outcome,
                    error=None,
                )
            ],
            summary="Demo summary",
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
        assert "Example" in dialog._description_label.text()
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
        text = dialog._result_view.toPlainText()  # noqa: SLF001
        assert "Input to chain" in text
        assert "Chain outputs" in text
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_renders_chain_summary(qt_app: QApplication) -> None:
    """Execution results should include the final summary section when present."""

    manager = _ManagerStub()
    dialog = PromptChainManagerDialog(manager)
    try:
        dialog._run_selected_chain()  # noqa: SLF001
        text = dialog._result_view.toPlainText()  # noqa: SLF001
        assert "Chain summary" in text
        assert "Demo summary" in text
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_markdown_toggle_preserves_text(qt_app: QApplication) -> None:
    """Disabling Markdown rendering must not clear previously rendered results."""

    manager = _ManagerStub()
    dialog = PromptChainManagerDialog(manager)
    try:
        dialog._run_selected_chain()  # noqa: SLF001 - exercising private helper for test
        initial_plain = dialog._result_view.toPlainText().strip()  # noqa: SLF001
        assert initial_plain
        assert dialog._result_richtext.strip()  # noqa: SLF001
        dialog._result_plaintext = ""  # noqa: SLF001 - simulate missing plain text snapshot
        dialog._result_format_checkbox.setChecked(True)  # noqa: SLF001
        dialog._result_format_checkbox.setChecked(False)  # noqa: SLF001
        toggled_text = dialog._result_view.toPlainText().strip()  # noqa: SLF001
        assert toggled_text
        assert "Input to chain" in toggled_text
        assert "Chain Outputs" in toggled_text
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_wrap_toggle_changes_line_mode(qt_app: QApplication) -> None:
    """Wrap checkbox should control the result view line wrap mode."""

    manager = _ManagerStub()
    dialog = PromptChainManagerDialog(manager)
    try:
        assert dialog._wrap_checkbox.isChecked() is True  # noqa: SLF001
        assert dialog._result_view.lineWrapMode() == QTextEdit.WidgetWidth  # noqa: SLF001
        dialog._wrap_checkbox.setChecked(False)  # noqa: SLF001
        assert dialog._result_view.lineWrapMode() == QTextEdit.NoWrap  # noqa: SLF001
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_markdown_omits_code_fences(qt_app: QApplication) -> None:
    """Markdown output should avoid code fences so wrapping works everywhere."""

    manager = _ManagerStub()
    dialog = PromptChainManagerDialog(manager)
    try:
        dialog._run_selected_chain()  # noqa: SLF001
        assert "```" not in dialog._result_richtext  # noqa: SLF001
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_results_use_colored_sections(qt_app: QApplication) -> None:
    """Rich text output should include styled blocks for key sections."""

    manager = _ManagerStub()
    dialog = PromptChainManagerDialog(manager)
    try:
        dialog._run_selected_chain()  # noqa: SLF001
        rich = dialog._result_richtext  # noqa: SLF001
        assert "chain-block--input" in rich
        assert "chain-block--summary" in rich
        assert "#e8f5e9" in rich  # light green blocks
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_renders_reasoning_summary(qt_app: QApplication) -> None:
    """Reasoning snippets should appear with dedicated styling when available."""

    manager = _ManagerStub(step_reasoning_text="Deliberate reasoning path.")
    dialog = PromptChainManagerDialog(manager)
    try:
        dialog._run_selected_chain()  # noqa: SLF001
        plain = dialog._result_view.toPlainText()  # noqa: SLF001
        assert "Reasoning summary" in plain
        assert "Deliberate reasoning path." in plain
        assert "#e3f2fd" in dialog._result_richtext  # noqa: SLF001
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_step_markdown_renders_without_code_fences(qt_app: QApplication) -> None:
    """Step inputs/outputs should render markdown content directly."""

    manager = _ManagerStub(
        step_request_text="### Step input heading",
        step_response_text="### Step output\n\n- Item one",
    )
    dialog = PromptChainManagerDialog(manager)
    try:
        dialog._run_selected_chain()  # noqa: SLF001
        rich = dialog._result_richtext  # noqa: SLF001
        assert "chain-step-output" in rich
        assert "### Step output" in rich
        assert "```" not in rich
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
    assert chain.summarize_last_response is True


def test_prompt_chain_editor_respects_summary_flag(qt_app: QApplication) -> None:
    """Editing an existing chain should reflect and persist the summary preference."""

    chain = _make_chain()
    chain.summarize_last_response = False
    editor = PromptChainEditorDialog(None, manager=None, chain=chain)
    assert editor._summarize_checkbox.isChecked() is False  # noqa: SLF001
    editor._handle_accept()  # noqa: SLF001
    updated = editor.result_chain()
    assert updated is not None
    assert updated.summarize_last_response is False


def test_prompt_chain_editor_prompt_tooltip_and_double_click(
    qt_app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt = SimpleNamespace(
        id=uuid.uuid4(),
        name="Demo Prompt",
        context="Body text",
        description="",
    )
    editor = PromptChainEditorDialog(None, manager=None, prompts=[prompt])
    step = PromptChainStep(
        id=uuid.uuid4(),
        chain_id=editor._chain_id,  # noqa: SLF001
        prompt_id=prompt.id,
        order_index=1,
        input_template="{{ body }}",
        output_variable="result",
    )
    editor._steps = [step]  # noqa: SLF001
    editor._refresh_steps()  # noqa: SLF001
    item = editor._steps_table.item(0, 1)  # noqa: SLF001
    assert item is not None
    assert "Demo Prompt" in (item.toolTip() or "")
    captured: dict[str, str] = {}

    def _capture(parent: object, title: str, text: str) -> None:  # noqa: ANN001
        captured["title"] = title
        captured["text"] = text

    monkeypatch.setattr("gui.dialogs.prompt_chain_editor.QMessageBox.information", _capture)
    editor._handle_step_double_click(item)  # noqa: SLF001
    assert captured["title"] == "Demo Prompt"
    assert "Body text" in captured["text"]


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
def test_prompt_chain_dialog_stream_preview_renders(qt_app: QApplication) -> None:
    """Streaming preview should show incremental text in the result area."""

    manager = _ManagerStub()
    dialog = PromptChainManagerDialog(manager)
    step = manager._chains[0].steps[0]  # noqa: SLF001
    try:
        dialog._begin_stream_preview(manager._chains[0], {"foo": "bar"})  # noqa: SLF001
        dialog._register_stream_chunk(step, "partial response", False)  # noqa: SLF001
        text = dialog._result_view.toPlainText()  # noqa: SLF001
        assert "Input to chain" in text
        assert '"foo": "bar"' in text
        assert "partial response" in text
    finally:
        dialog._end_stream_preview()  # noqa: SLF001
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_stream_detection_uses_executor(qt_app: QApplication) -> None:
    """Streaming flag should be inferred from the manager executor."""

    manager = _ManagerStub(stream_enabled=True)
    dialog = PromptChainManagerDialog(manager)
    try:
        assert dialog._is_streaming_enabled() is True  # noqa: SLF001
    finally:
        dialog.close()
        dialog.deleteLater()
