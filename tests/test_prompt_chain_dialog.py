"""Prompt chain manager dialog tests.

Updates:
  v0.4.3 - 2025-12-12 - Assert chain web search tooltip reflects the active provider and Random rotation.
  v0.4.2 - 2025-12-08 - Accept case-insensitive chain headings for Markdown toggles.
  v0.4.1 - 2025-12-08 - Cast manager stubs, use Qt enums, and swap SimpleNamespace prompts.
  v0.4.0 - 2025-12-06 - Adapt to the plain-text chain manager/editor UX.
  v0.3.0 - 2025-12-06 - Gate reasoning summary rendering and extend coverage.
  v0.2.9 - 2025-12-05 - Cover schema toggle visibility and chain list activation.
  v0.2.8 - 2025-12-05 - Cover prompt name rendering and editor activation from the Chain tab.
  v0.2.7 - 2025-12-05 - Cover default-on web search toggle behaviour for chains.
  v0.2.6 - 2025-12-05 - Cover summarize toggle persistence and UI output section.
  v0.2.5 - 2025-12-05 - Ensure step IO renders outside code fences for Markdown formatting.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, cast

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox, QTextEdit

from core import PromptChainRunResult, PromptChainStepRun, PromptManager, PromptManagerError
from core.execution import CodexExecutionResult
from core.prompt_manager.execution_history import ExecutionOutcome
from core.web_search import RandomWebSearchProvider, WebSearchService
from gui.dialogs.prompt_chain_editor import PromptChainEditorDialog
from gui.dialogs.prompt_chains import PromptChainManagerDialog, PromptChainManagerPanel
from models.prompt_chain_model import PromptChain, PromptChainStep
from models.prompt_model import Prompt

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from collections.abc import Callable, Iterator, Mapping
else:  # pragma: no cover - runtime placeholders
    from typing import Any as _Any

    Callable = _Any
    Iterator = _Any
    Mapping = _Any


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    """Provide a shared Qt application instance for dialog tests."""

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return cast("QApplication", app)


@pytest.fixture(autouse=True)
def reset_chain_panel_settings() -> Iterator[None]:
    """Reset QSettings used by the chain panel between tests."""

    settings = QSettings("PromptManager", "PromptChainManagerPanel")
    settings.clear()
    yield
    settings.clear()


class _ExecutorStub:
    def __init__(self, stream_enabled: bool) -> None:
        self.stream = stream_enabled


def _make_prompt_record(prompt_id: uuid.UUID) -> Prompt:
    return Prompt(
        id=prompt_id,
        name="Example Prompt",
        description="Example prompt",
        category="tests",
    )


def _as_prompt_manager(manager: _ManagerStub) -> PromptManager:
    return cast("PromptManager", manager)


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
        self._executor = _ExecutorStub(stream_enabled) if stream_enabled else None
        self._litellm_stream = stream_enabled
        self._stream_chunks = stream_chunks
        self.received_stream_callback: Callable[[PromptChainStep, str, bool], None] | None = None
        self._step_request_text = step_request_text
        self._step_response_text = step_response_text
        self._step_reasoning_text = step_reasoning_text
        self.last_use_web_search: bool | None = None
        prompt_id = self._chains[0].steps[0].prompt_id
        self._prompt_record = _make_prompt_record(prompt_id)

    @property
    def repository(self):  # pragma: no cover - only used by DialogLauncher in real app
        return type("Repo", (), {"list": lambda *_: []})()

    def list_prompt_chains(self, include_inactive: bool = False):  # noqa: ARG002
        return list(self._chains)

    def list_prompts(self, limit: int | None = None):  # noqa: ARG002
        return [self._prompt_record]

    def get_prompt(self, prompt_id: uuid.UUID):  # noqa: D401
        if prompt_id == self._prompt_record.id:
            return self._prompt_record
        raise PromptManagerError("Prompt not found")

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
        chain_input: str,
        stream_callback: Callable[[PromptChainStep, str, bool], None] | None = None,
        use_web_search: bool | None = None,
        web_search_limit: int = 10,
    ):
        self.runs.append(
            {
                "chain_id": chain_id,
                "chain_input": chain_input,
                "use_web_search": use_web_search,
                "web_search_limit": web_search_limit,
            }
        )
        self.received_stream_callback = stream_callback
        self.last_use_web_search = use_web_search
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
            chain_input=chain_input,
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


class _FakeProvider:
    def __init__(self, slug: str, display_name: str | None = None) -> None:
        self.slug = slug
        self.display_name = display_name or slug

    async def search(self, query: str, *, limit: int = 5, **kwargs: Any) -> object:  # noqa: ARG002
        return {}


def _build_dialog(
    manager: _ManagerStub | None = None,
    **kwargs: Any,
) -> tuple[PromptChainManagerDialog, PromptChainManagerPanel, _ManagerStub]:
    stub = manager or _ManagerStub()
    dialog = PromptChainManagerDialog(_as_prompt_manager(stub), **kwargs)
    panel: PromptChainManagerPanel = dialog._panel
    return dialog, panel, stub


def _make_chain() -> PromptChain:
    chain_id = uuid.uuid4()
    prompt_id = uuid.uuid4()
    step = PromptChainStep(
        id=uuid.uuid4(),
        chain_id=chain_id,
        prompt_id=prompt_id,
        order_index=1,
        input_template="",
        output_variable="step_1",
    )
    return PromptChain(
        id=chain_id,
        name="Demo Chain",
        description="Example",
        steps=[step],
    )


def _make_outcome(
    *,
    request_text: str,
    response_text: str,
    reasoning_text: str | None = None,
) -> ExecutionOutcome:
    """Build an execution outcome with optional reasoning payload."""

    raw_response: dict[str, Any] = {}
    if reasoning_text:
        raw_response = {
            "output": [
                {
                    "content": [
                        {"type": "reasoning", "text": reasoning_text},
                    ]
                }
            ]
        }
    execution_result = CodexExecutionResult(
        prompt_id=uuid.uuid4(),
        request_text=request_text,
        response_text=response_text,
        duration_ms=1,
        usage={},
        raw_response=raw_response,
    )
    return ExecutionOutcome(result=execution_result, history_entry=None, conversation=[])


def test_prompt_chain_dialog_populates_details(qt_app: QApplication) -> None:
    """Dialog should load chains and render their metadata immediately."""

    manager = _ManagerStub()
    dialog, panel, manager = _build_dialog(manager)
    try:
        assert panel._chain_list.count() == 1  # noqa: SLF001
        assert panel._detail_title.text() == "Demo Chain"
        assert "Example" in panel._description_label.text()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_sets_provider_tooltip(qt_app: QApplication) -> None:
    """Tooltip should describe the configured provider."""

    manager = _ManagerStub()
    manager.web_search_service = WebSearchService(_FakeProvider("exa", "Exa Search"))
    manager.web_search = manager.web_search_service
    dialog, panel, manager = _build_dialog(manager)
    try:
        assert panel._web_search_checkbox is not None  # noqa: SLF001
        assert panel._web_search_checkbox.toolTip() == (
            "Include live web search findings via Exa Search before each chain step executes."
        )  # noqa: SLF001
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_sets_random_provider_tooltip(qt_app: QApplication) -> None:
    """Tooltip should list available providers when Random is configured."""

    manager = _ManagerStub()
    random_provider = RandomWebSearchProvider(
        (_FakeProvider("exa", "Exa"), _FakeProvider("tavily", "Tavily"))
    )
    manager.web_search_service = WebSearchService(random_provider)
    manager.web_search = manager.web_search_service
    dialog, panel, manager = _build_dialog(manager)
    try:
        assert panel._web_search_checkbox is not None  # noqa: SLF001
        assert panel._web_search_checkbox.toolTip() == (
            "Include live web search findings via the Random provider, rotating between "
            "Exa and Tavily before each chain step executes."
        )  # noqa: SLF001
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_runs_chain(qt_app: QApplication) -> None:
    """Running a chain should send the plain-text input to the manager."""

    manager = _ManagerStub()
    dialog, panel, manager = _build_dialog(manager)
    try:
        panel._chain_input_edit.setPlainText("Chain input text")  # noqa: SLF001
        panel._run_selected_chain()  # noqa: SLF001
        assert manager.runs
        assert manager.runs[0]["chain_input"].strip() == "Chain input text"
        assert manager.runs[0]["use_web_search"] is True
        text = panel._result_view.toPlainText()  # noqa: SLF001
        assert "Input to chain" in text
        assert "Chain outputs" in text
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_renders_chain_summary(qt_app: QApplication) -> None:
    """Execution results should include the final summary section when present."""

    manager = _ManagerStub()
    dialog, panel, manager = _build_dialog(manager)
    try:
        panel._chain_input_edit.setPlainText("Summary input")  # noqa: SLF001
        panel._run_selected_chain()  # noqa: SLF001
        text = panel._result_view.toPlainText()  # noqa: SLF001
        assert "Chain summary" in text
        assert "Demo summary" in text
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_respects_web_search_toggle(qt_app: QApplication) -> None:
    """Web search checkbox should control manager invocation flag."""

    manager = _ManagerStub()
    dialog, panel, manager = _build_dialog(manager)
    try:
        panel._chain_input_edit.setPlainText("Toggle input")  # noqa: SLF001
        assert panel._web_search_checkbox is not None
        checkbox = panel._web_search_checkbox
        checkbox.setChecked(False)  # noqa: SLF001
        panel._run_selected_chain()  # noqa: SLF001
        assert manager.last_use_web_search is False
        checkbox.setChecked(True)  # noqa: SLF001
        panel._run_selected_chain()  # noqa: SLF001
        assert manager.last_use_web_search is True
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_displays_prompt_names(qt_app: QApplication) -> None:
    """Steps table should render prompt names instead of UUIDs."""

    manager = _ManagerStub()
    dialog, panel, manager = _build_dialog(manager)
    try:
        item = panel._steps_table.item(0, 1)  # noqa: SLF001
        assert item is not None
        assert item.text() == "Example Prompt"
        assert item.toolTip() == str(manager._prompt_record.id)
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_step_activation_opens_prompt_editor(qt_app: QApplication) -> None:
    """Double-clicking a step should invoke the prompt edit callback."""

    manager = _ManagerStub()
    invoked: list[uuid.UUID] = []
    dialog, panel, manager = _build_dialog(
        manager,
        prompt_edit_callback=lambda prompt_id: invoked.append(prompt_id),
    )
    try:
        panel._handle_step_table_activated(0, 0)  # noqa: SLF001
        assert invoked == [manager._prompt_record.id]
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_markdown_toggle_preserves_text(qt_app: QApplication) -> None:
    """Disabling Markdown rendering must not clear previously rendered results."""

    manager = _ManagerStub()
    dialog, panel, manager = _build_dialog(manager)
    try:
        panel._chain_input_edit.setPlainText("Markdown input")  # noqa: SLF001
        panel._run_selected_chain()  # noqa: SLF001
        initial_plain = panel._result_view.toPlainText().strip()  # noqa: SLF001
        assert initial_plain
        assert panel._result_richtext.strip()  # noqa: SLF001
        panel._result_plaintext = ""  # noqa: SLF001
        panel._result_format_checkbox.setChecked(True)  # noqa: SLF001
        panel._result_format_checkbox.setChecked(False)  # noqa: SLF001
        toggled_text = panel._result_view.toPlainText().strip()  # noqa: SLF001
        assert toggled_text
        assert "Input to chain" in toggled_text
        assert "chain outputs" in toggled_text.lower()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_wrap_toggle_changes_line_mode(qt_app: QApplication) -> None:
    """Wrap checkbox should control the result view line wrap mode."""

    manager = _ManagerStub()
    dialog, panel, manager = _build_dialog(manager)
    try:
        assert panel._wrap_checkbox.isChecked() is True  # noqa: SLF001
        assert panel._result_view.lineWrapMode() == QTextEdit.LineWrapMode.WidgetWidth
        panel._wrap_checkbox.setChecked(False)  # noqa: SLF001
        assert panel._result_view.lineWrapMode() == QTextEdit.LineWrapMode.NoWrap
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_markdown_omits_code_fences(qt_app: QApplication) -> None:
    """Markdown output should avoid code fences so wrapping works everywhere."""

    manager = _ManagerStub()
    dialog, panel, manager = _build_dialog(manager)
    try:
        panel._chain_input_edit.setPlainText("Markdown fences")  # noqa: SLF001
        panel._run_selected_chain()  # noqa: SLF001
        assert "```" not in panel._result_richtext  # noqa: SLF001
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_results_use_colored_sections(qt_app: QApplication) -> None:
    """Rich text output should include styled blocks for key sections."""

    manager = _ManagerStub()
    dialog, panel, manager = _build_dialog(manager)
    try:
        panel._chain_input_edit.setPlainText("Colored sections")  # noqa: SLF001
        panel._run_selected_chain()  # noqa: SLF001
        rich = panel._result_richtext  # noqa: SLF001
        assert "chain-block--input" in rich
        assert "chain-block--summary" in rich
        assert "#66bb6a" in rich  # light green text
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_renders_reasoning_summary(qt_app: QApplication) -> None:
    """Reasoning snippets should appear with dedicated styling when available."""

    manager = _ManagerStub(step_reasoning_text="Deliberate reasoning path.")
    dialog, panel, manager = _build_dialog(manager)
    try:
        panel._chain_input_edit.setPlainText("Reasoning input")  # noqa: SLF001
        panel._run_selected_chain()  # noqa: SLF001
        plain = panel._result_view.toPlainText()  # noqa: SLF001
        assert "Reasoning summary" in plain
        assert "Deliberate reasoning path." in plain
        assert "#1e88e5" in panel._result_richtext  # noqa: SLF001
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_omits_reasoning_when_summary_disabled(
    qt_app: QApplication,
) -> None:
    """Reasoning and chain summary must disappear when the preference is off."""

    manager = _ManagerStub(step_reasoning_text="Should not appear.")
    manager._chains[0].summarize_last_response = False
    dialog, panel, manager = _build_dialog(manager)
    try:
        panel._chain_input_edit.setPlainText("No reasoning")  # noqa: SLF001
        panel._run_selected_chain()  # noqa: SLF001
        plain = panel._result_view.toPlainText()  # noqa: SLF001
        assert "Reasoning summary" not in plain
        assert "Chain summary" not in plain
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_only_last_step_has_reasoning_summary(
    qt_app: QApplication,
) -> None:
    """Only the final successful step should surface the reasoning summary."""

    manager = _ManagerStub()
    dialog, panel, manager = _build_dialog(manager)
    try:
        base_chain = _make_chain()
        step_one = base_chain.steps[0]
        step_two = PromptChainStep(
            id=uuid.uuid4(),
            chain_id=base_chain.id,
            prompt_id=uuid.uuid4(),
            order_index=2,
            input_template="",
            output_variable="final",
        )
        base_chain.steps = [step_one, step_two]
        result = PromptChainRunResult(
            chain=base_chain,
            chain_input="Initial text",
            outputs={"final": "done"},
            steps=[
                PromptChainStepRun(
                    step=step_one,
                    status="success",
                    outcome=_make_outcome(
                        request_text="req1",
                        response_text="res1",
                        reasoning_text="First reason",
                    ),
                ),
                PromptChainStepRun(
                    step=step_two,
                    status="success",
                    outcome=_make_outcome(
                        request_text="req2",
                        response_text="res2",
                        reasoning_text="Final reason",
                    ),
                ),
            ],
            summary="Demo summary",
        )
        panel._display_run_result(result)  # noqa: SLF001
        plain = panel._result_plaintext  # noqa: SLF001
        assert plain.count("Reasoning summary") == 1
        assert "Final reason" in plain
        assert "First reason" not in plain
        rich = panel._result_richtext  # noqa: SLF001
        assert "Final reason" in rich
        assert "First reason" not in rich
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_step_markdown_renders_without_code_fences(qt_app: QApplication) -> None:
    """Step inputs/outputs should render markdown content directly."""

    manager = _ManagerStub(
        step_request_text="### Step input heading",
        step_response_text="### Step output\n\n- Item one",
    )
    dialog, panel, manager = _build_dialog(manager)
    try:
        panel._chain_input_edit.setPlainText("Markdown step input")  # noqa: SLF001
        panel._run_selected_chain()  # noqa: SLF001
        rich = panel._result_richtext  # noqa: SLF001
        assert "chain-block--outputs" in rich
        assert "```" not in rich
        plain = panel._result_plaintext  # noqa: SLF001
        assert "Chain summary" in plain
        assert "### Step output" not in plain
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
        input_template="",
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
    prompt = Prompt(
        id=uuid.uuid4(),
        name="Demo Prompt",
        description="Body text",
        category="tests",
        context="Body text",
    )
    editor = PromptChainEditorDialog(None, manager=None, prompts=[prompt])
    step = PromptChainStep(
        id=uuid.uuid4(),
        chain_id=editor._chain_id,  # noqa: SLF001
        prompt_id=prompt.id,
        order_index=1,
        input_template="",
        output_variable="result",
    )
    editor._steps = [step]  # noqa: SLF001
    editor._refresh_steps()  # noqa: SLF001
    item = editor._steps_table.item(0, 1)  # noqa: SLF001
    assert item is not None
    assert "Demo Prompt" in (item.toolTip() or "")
    captured: dict[str, str] = {}

    def _capture(parent: object, title: str, text: str) -> QMessageBox.StandardButton:  # noqa: ANN001
        captured["title"] = title
        captured["text"] = text
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr("gui.dialogs.prompt_chain_editor.QMessageBox.information", _capture)
    editor._handle_step_double_click(item)  # noqa: SLF001
    assert captured["title"] == "Demo Prompt"
    assert "Body text" in captured["text"]


def test_prompt_chain_list_activation_opens_editor(
    qt_app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Double-clicking a chain should invoke the editor workflow."""

    manager = _ManagerStub()
    dialog, panel, manager = _build_dialog(manager)

    class _EditorStub:
        def __init__(self, *_: object, **kwargs: object) -> None:
            called["chain"] = kwargs.get("chain")

        def exec(self) -> int:
            return int(QDialog.DialogCode.Accepted)

        def result_chain(self) -> PromptChain:
            chain = called["chain"]
            assert isinstance(chain, PromptChain)
            return chain

    called: dict[str, object] = {}
    monkeypatch.setattr(
        "gui.dialogs.prompt_chains.PromptChainEditorDialog",
        _EditorStub,
    )
    try:
        item = panel._chain_list.item(0)  # noqa: SLF001
        panel._handle_chain_activation(item)  # noqa: SLF001
        assert called["chain"] is not None
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_manager_creates_chain_via_editor(
    qt_app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Manager dialog should persist the chain returned by the editor."""

    manager = _ManagerStub()
    dialog, panel, manager = _build_dialog(manager)

    class _EditorStub:
        def __init__(self, *_: object, **__: object) -> None:
            self._chain = _make_chain()

        def exec(self) -> int:
            return int(QDialog.DialogCode.Accepted)

        def result_chain(self) -> PromptChain:
            return self._chain

    monkeypatch.setattr(
        "gui.dialogs.prompt_chains.PromptChainEditorDialog",
        _EditorStub,
    )
    panel._create_chain()  # noqa: SLF001
    assert manager.saved_chain is not None
    dialog.close()
    dialog.deleteLater()


def test_prompt_chain_manager_deletes_chain(
    qt_app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Delete action should invoke PromptManager.delete_prompt_chain."""

    manager = _ManagerStub()
    dialog, panel, manager = _build_dialog(manager)
    monkeypatch.setattr(
        "gui.dialogs.prompt_chains.QMessageBox.question",
        lambda *_, **__: QMessageBox.StandardButton.Yes,
    )
    panel._delete_chain()  # noqa: SLF001
    assert manager.deleted_chain_ids
    dialog.close()
    dialog.deleteLater()


def test_prompt_chain_dialog_stream_preview_renders(qt_app: QApplication) -> None:
    """Streaming preview should show incremental text in the result area."""

    manager = _ManagerStub()
    dialog, panel, manager = _build_dialog(manager)
    step = manager._chains[0].steps[0]  # noqa: SLF001
    try:
        panel._begin_stream_preview(manager._chains[0], "Streaming input")  # noqa: SLF001
        panel._register_stream_chunk(step, "partial response", False)  # noqa: SLF001
        text = panel._result_view.toPlainText()  # noqa: SLF001
        assert "Input to chain" in text
        assert "Streaming input" in text
        assert "partial response" in text
    finally:
        panel._end_stream_preview()  # noqa: SLF001
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_stream_detection_uses_executor(qt_app: QApplication) -> None:
    """Streaming flag should be inferred from the manager executor."""

    manager = _ManagerStub(stream_enabled=True)
    dialog, panel, manager = _build_dialog(manager)
    try:
        assert panel._is_streaming_enabled() is True  # noqa: SLF001
    finally:
        dialog.close()
        dialog.deleteLater()
