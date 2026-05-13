"""Prompt chain manager dialog tests.

Updates:
  v0.4.4 - 2025-12-12 - Tighten web search stubs to satisfy Pyright strict mode.
  v0.4.3 - 2025-12-12 - Assert chain web search tooltip matches provider/Random rotation.
  v0.4.2 - 2025-12-08 - Accept case-insensitive chain headings for Markdown toggles.
  v0.4.1 - 2025-12-08 - Cast manager stubs, use Qt enums, and swap SimpleNamespace prompts.
  v0.4.0 - 2025-12-06 - Adapt to the plain-text chain manager/editor UX.
  v0.3.0 - 2025-12-06 - Gate reasoning summary rendering and extend coverage.
  v0.2.9 - 2025-12-05 - Cover schema toggle visibility and chain list activation.
  v0.2.8 - 2025-12-05 - Cover prompt name rendering and editor activation from the Chain tab.
  v0.2.7 - 2025-12-05 - Cover default-on web search toggle behaviour for chains.
  v0.2.6 - 2025-12-05 - Cover summarize toggle persistence and Markdown step IO rendering.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, cast

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QLabel,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QTableWidget,
    QTextEdit,
    QWidget,
)

from core import PromptChainRunResult, PromptChainStepRun, PromptManager, PromptManagerError
from core.execution import CodexExecutionResult
from core.prompt_manager.execution_history import ExecutionOutcome
from core.web_search import RandomWebSearchProvider, WebSearchResult, WebSearchService
from gui.dialogs.prompt_chain_editor import PromptChainEditorDialog, PromptChainStepDialog
from gui.dialogs.prompt_chains import PromptChainManagerDialog, PromptChainManagerPanel
from models.prompt_chain_model import PromptChain, PromptChainStep
from models.prompt_model import Prompt

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from collections.abc import Callable, Iterator, Mapping
    from pathlib import Path
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


def _repo_list(*_: object) -> list[object]:
    return []


def _record_prompt_edit(invoked: list[uuid.UUID]) -> Callable[[uuid.UUID], None]:
    def _record(prompt_id: uuid.UUID) -> None:
        invoked.append(prompt_id)

    return _record


def _select_save_path(output_path: Path) -> Callable[..., tuple[str, str]]:
    def _stub(*args: object, **kwargs: object) -> tuple[str, str]:
        del args, kwargs
        return (str(output_path), "Text files (*.txt)")

    return _stub


def _confirm_yes(*args: object, **kwargs: object) -> QMessageBox.StandardButton:
    del args, kwargs
    return QMessageBox.StandardButton.Yes


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
        self._recent_runs: list[dict[str, str | None]] = []
        self.web_search_service: WebSearchService = WebSearchService()
        self.web_search: WebSearchService = self.web_search_service
        prompt_id = self._chains[0].steps[0].prompt_id
        self._prompt_record = _make_prompt_record(prompt_id)

    @property
    def repository(self):  # pragma: no cover - only used by DialogLauncher in real app
        return type("Repo", (), {"list": staticmethod(_repo_list)})()

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

    def list_recent_prompt_chain_runs(self, *, limit: int = 20) -> list[dict[str, str | None]]:
        safe_limit = max(1, int(limit or 20))
        return list(self._recent_runs[:safe_limit])

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
        result = PromptChainRunResult(
            chain=chain,
            chain_input=chain_input,
            step_outputs={"summary": "ok"},
            final_output_text=self._step_response_text,
            final_summary_text="Demo summary",
            steps=[
                PromptChainStepRun(
                    step=step,
                    status="success",
                    outcome=outcome,
                    prompt_name="Example Prompt",
                    request_text=self._step_request_text,
                    response_text=self._step_response_text,
                    duration_ms=123,
                    web_search_requested=bool(use_web_search),
                    web_search_applied=bool(use_web_search),
                    skip_reason=None,
                    error=None,
                    step_label="final",
                    step_output_key="step_1",
                )
            ],
            run_status="success",
            final_step_id=step.id,
            final_step_output_key="step_1",
            final_step_label="final",
        )
        self._recent_runs.insert(
            0,
            {
                "chain_id": str(chain.id),
                "chain_name": chain.name,
                "run_timestamp": "2026-05-08T12:00:00+00:00",
                "status": "success",
                "input_preview": chain_input,
                "final_output_preview": self._step_response_text,
                "final_step_output_key": "step_1",
                "final_step_id": str(step.id),
                "final_step_label": "final",
            },
        )
        del self._recent_runs[20:]
        return result

    def delete_prompt_chain(self, chain_id: uuid.UUID) -> None:
        self.deleted_chain_ids.append(chain_id)
        self._chains = [chain for chain in self._chains if chain.id != chain_id]


class _FakeProvider:
    def __init__(self, slug: str, display_name: str | None = None) -> None:
        self.slug = slug
        self.display_name = display_name or slug

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> WebSearchResult:
        return WebSearchResult(provider=self.slug, query=query, documents=[])


def _find_child(root: QWidget, widget_type: type[QWidget], name: str) -> QWidget:
    widget = root.findChild(widget_type, name)
    assert widget is not None, f"Expected {widget_type.__name__} named {name!r}"
    return widget


def _required_list_widget(root: QWidget, name: str) -> QListWidget:
    return cast("QListWidget", _find_child(root, QListWidget, name))


def _required_label(root: QWidget, name: str) -> QLabel:
    return cast("QLabel", _find_child(root, QLabel, name))


def _required_plain_text_edit(root: QWidget, name: str) -> QPlainTextEdit:
    return cast("QPlainTextEdit", _find_child(root, QPlainTextEdit, name))


def _required_text_edit(root: QWidget, name: str) -> QTextEdit:
    return cast("QTextEdit", _find_child(root, QTextEdit, name))


def _required_check_box(root: QWidget, name: str) -> QCheckBox:
    return cast("QCheckBox", _find_child(root, QCheckBox, name))


def _required_table_widget(root: QWidget, name: str) -> QTableWidget:
    return cast("QTableWidget", _find_child(root, QTableWidget, name))


def _panel_widget(dialog: PromptChainManagerDialog) -> PromptChainManagerPanel:
    return cast("PromptChainManagerPanel", _find_child(dialog, PromptChainManagerPanel, ""))


def _chain_input_edit(panel: PromptChainManagerPanel) -> QPlainTextEdit:
    return _required_plain_text_edit(panel, "promptChainInputEdit")


def _result_view(panel: PromptChainManagerPanel) -> QTextEdit:
    return _required_text_edit(panel, "promptChainResultView")


def _markdown_checkbox(panel: PromptChainManagerPanel) -> QCheckBox:
    return _required_check_box(panel, "promptChainMarkdownCheckbox")


def _wrap_checkbox(panel: PromptChainManagerPanel) -> QCheckBox:
    return _required_check_box(panel, "promptChainWrapCheckbox")


def _chain_list_widget(panel: PromptChainManagerPanel) -> QListWidget:
    return _required_list_widget(panel, "promptChainList")


def _manager_prompt_id(manager: _ManagerStub) -> uuid.UUID:
    return cast("uuid.UUID", cast("Any", manager)._prompt_record.id)


def _manager_chain(manager: _ManagerStub, index: int = 0) -> PromptChain:
    return cast("PromptChain", cast("Any", manager)._chains[index])


def _manager_step_response_text(manager: _ManagerStub) -> str:
    return cast("str", cast("Any", manager)._step_response_text)


def _panel_plaintext(panel: PromptChainManagerPanel) -> str:
    return cast("str", cast("Any", panel)._result_plaintext)


def _panel_richtext(panel: PromptChainManagerPanel) -> str:
    return cast("str", cast("Any", panel)._result_richtext)


def _editor_name_input(editor: PromptChainEditorDialog) -> QWidget:
    return _find_child(editor, QWidget, "promptChainEditorNameInput")


def _editor_description_input(editor: PromptChainEditorDialog) -> QPlainTextEdit:
    return _required_plain_text_edit(editor, "promptChainEditorDescriptionInput")


def _editor_summarize_checkbox(editor: PromptChainEditorDialog) -> QCheckBox:
    return _required_check_box(editor, "promptChainEditorSummarizeCheckbox")


def _editor_steps_table(editor: PromptChainEditorDialog) -> QTableWidget:
    return _required_table_widget(editor, "promptChainEditorStepsTable")


def _editor_warning_label(editor: PromptChainEditorDialog) -> QLabel:
    return _required_label(editor, "promptChainEditorWarningLabel")


def _editor_move_up_button(editor: PromptChainEditorDialog) -> QWidget:
    return _find_child(editor, QWidget, "promptChainEditorMoveStepUpButton")


def _editor_move_down_button(editor: PromptChainEditorDialog) -> QWidget:
    return _find_child(editor, QWidget, "promptChainEditorMoveStepDownButton")


def _build_dialog(
    manager: _ManagerStub | None = None,
    **kwargs: Any,
) -> tuple[PromptChainManagerDialog, PromptChainManagerPanel, _ManagerStub]:
    stub = manager or _ManagerStub()
    dialog = PromptChainManagerDialog(_as_prompt_manager(stub), **kwargs)
    panel = _panel_widget(dialog)
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
    chain_list = _required_list_widget(panel, "promptChainList")
    detail_title = _required_label(panel, "promptChainDetailTitle")
    description_label = _required_label(panel, "promptChainDescription")
    try:
        assert chain_list.count() == 1
        assert detail_title.text() == "Demo Chain"
        assert "Example" in description_label.text()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_sets_provider_tooltip(qt_app: QApplication) -> None:
    """Tooltip should describe the configured provider."""

    manager = _ManagerStub()
    manager.web_search_service = WebSearchService(_FakeProvider("exa", "Exa Search"))
    manager.web_search = manager.web_search_service
    dialog, panel, manager = _build_dialog(manager)
    web_search_checkbox = _required_check_box(panel, "promptChainWebSearchCheckbox")
    try:
        assert web_search_checkbox.toolTip() == (
            "Include live web search findings via Exa Search before each chain step executes."
        )
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_sets_random_provider_tooltip(qt_app: QApplication) -> None:
    """Random provider tooltip should mention each configured backend exactly once."""

    manager = _ManagerStub()
    random_provider = RandomWebSearchProvider(
        providers=[_FakeProvider("exa", "Exa Search"), _FakeProvider("brave", "Brave Search")]
    )
    manager.web_search_service = WebSearchService(random_provider)
    manager.web_search = manager.web_search_service
    dialog, panel, manager = _build_dialog(manager)
    web_search_checkbox = _required_check_box(panel, "promptChainWebSearchCheckbox")
    try:
        assert web_search_checkbox.toolTip() == (
            "Include live web search findings via the Random provider, rotating between "
            "Exa Search and Brave Search before each chain step executes."
        )
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_runs_selected_chain(qt_app: QApplication) -> None:
    """Run action should forward the plain-text input into the PromptManager chain runner."""

    manager = _ManagerStub()
    dialog, panel, manager = _build_dialog(manager)
    chain_input_edit = _chain_input_edit(panel)
    result_view = _result_view(panel)
    try:
        chain_input_edit.setPlainText("Chain input text")
        panel.run_selected_chain()
        assert manager.runs[0]["chain_input"] == "Chain input text"
        assert manager.runs[0]["use_web_search"] is True
        text = result_view.toPlainText()
        assert "Input to chain" in text

        assert "Step outputs" in text
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_renders_chain_summary(qt_app: QApplication) -> None:
    """Execution results should include the supporting summary section when present."""

    manager = _ManagerStub()
    dialog, panel, manager = _build_dialog(manager)
    chain_input_edit = _chain_input_edit(panel)
    result_view = _result_view(panel)
    try:
        chain_input_edit.setPlainText("Summary input")
        panel.run_selected_chain()
        text = result_view.toPlainText()
        assert "Supporting summary" in text
        assert "Demo summary" in text
        assert "Final chain result" not in text
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_marks_final_summary_as_supporting_context(
    qt_app: QApplication,
) -> None:
    """When both result blocks exist, the summary should read as supporting context."""

    manager = _ManagerStub()
    dialog, panel, manager = _build_dialog(manager)
    chain_input_edit = _chain_input_edit(panel)
    result_view = _result_view(panel)
    try:
        chain_input_edit.setPlainText("Summary context input")
        panel.run_selected_chain()
        text = result_view.toPlainText()
        assert "Final output" in text
        assert "Supporting summary" in text
        assert "Demo summary" in text
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_respects_web_search_toggle(qt_app: QApplication) -> None:
    """Web search checkbox should control manager invocation flag."""

    manager = _ManagerStub()
    dialog, panel, manager = _build_dialog(manager)
    chain_input_edit = _chain_input_edit(panel)
    web_search_checkbox = _required_check_box(panel, "promptChainWebSearchCheckbox")
    try:
        chain_input_edit.setPlainText("Toggle input")
        web_search_checkbox.setChecked(False)
        panel.run_selected_chain()
        assert manager.last_use_web_search is False
        web_search_checkbox.setChecked(True)
        panel.run_selected_chain()
        assert manager.last_use_web_search is True
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_displays_prompt_names(qt_app: QApplication) -> None:
    """Steps table should render prompt names instead of UUIDs."""

    manager = _ManagerStub()
    dialog, panel, manager = _build_dialog(manager)
    steps_table = _required_table_widget(panel, "promptChainStepsTable")
    try:
        item = steps_table.item(0, 1)
        assert item is not None
        assert item.text() == "Example Prompt"
        assert item.toolTip() == str(_manager_prompt_id(manager))
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_step_activation_opens_prompt_editor(qt_app: QApplication) -> None:
    """Double-clicking a step should invoke the prompt edit callback."""

    manager = _ManagerStub()
    invoked: list[uuid.UUID] = []
    dialog, panel, manager = _build_dialog(
        manager,
        prompt_edit_callback=_record_prompt_edit(invoked),
    )
    try:
        panel.activate_step_at(0, 0)
        assert invoked == [_manager_prompt_id(manager)]
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_markdown_toggle_preserves_rendered_output(
    qt_app: QApplication,
) -> None:
    """Disabling Markdown rendering must not clear previously rendered results."""

    manager = _ManagerStub()
    dialog, panel, manager = _build_dialog(manager)
    chain_input_edit = _chain_input_edit(panel)
    result_view = _result_view(panel)
    markdown_checkbox = _markdown_checkbox(panel)
    try:
        chain_input_edit.setPlainText("Markdown input")
        panel.run_selected_chain()
        initial_plain = result_view.toPlainText().strip()
        assert initial_plain
        assert _panel_richtext(panel).strip()
        cast("Any", panel)._result_plaintext = ""
        markdown_checkbox.setChecked(True)
        markdown_checkbox.setChecked(False)
        toggled_text = result_view.toPlainText().strip()
        assert toggled_text
        assert "Input to chain" in toggled_text
        assert "step outputs" in toggled_text.lower()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_wrap_toggle_changes_line_mode(qt_app: QApplication) -> None:
    """Wrap checkbox should control the result view line wrap mode."""

    manager = _ManagerStub()
    dialog, panel, manager = _build_dialog(manager)
    result_view = _result_view(panel)
    wrap_checkbox = _wrap_checkbox(panel)
    try:
        assert wrap_checkbox.isChecked() is True
        assert result_view.lineWrapMode() == QTextEdit.LineWrapMode.WidgetWidth
        wrap_checkbox.setChecked(False)
        assert result_view.lineWrapMode() == QTextEdit.LineWrapMode.NoWrap
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_markdown_omits_code_fences(qt_app: QApplication) -> None:
    """Markdown output should avoid code fences so wrapping works everywhere."""

    manager = _ManagerStub()
    dialog, panel, manager = _build_dialog(manager)
    chain_input_edit = _chain_input_edit(panel)
    try:
        chain_input_edit.setPlainText("Markdown fences")
        panel.run_selected_chain()
        assert "```" not in _panel_richtext(panel)
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_results_use_colored_sections(qt_app: QApplication) -> None:
    """Rich text output should include styled blocks for key sections."""

    manager = _ManagerStub()
    dialog, panel, manager = _build_dialog(manager)
    chain_input_edit = _chain_input_edit(panel)
    try:
        chain_input_edit.setPlainText("Colored sections")
        panel.run_selected_chain()
        rich = _panel_richtext(panel)
        assert "chain-block--input" in rich
        assert "chain-block--summary" in rich
        assert "#66bb6a" in rich  # light green text
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_clear_results_resets_to_neutral_state(qt_app: QApplication) -> None:
    """Clearing results should remove stale run cues and leave an empty results pane."""

    manager = _ManagerStub()
    dialog, panel, manager = _build_dialog(manager)
    chain_input_edit = _chain_input_edit(panel)
    result_view = _result_view(panel)
    try:
        chain_input_edit.setPlainText("First run")
        panel.run_selected_chain()
        assert "Input to chain" in result_view.toPlainText()

        panel.clear_results()

        assert result_view.toPlainText() == ""
        assert _panel_plaintext(panel) == ""
        assert _panel_richtext(panel) == ""
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_can_copy_final_output_to_clipboard(qt_app: QApplication) -> None:
    manager = _ManagerStub()
    dialog, panel, manager = _build_dialog(manager)
    chain_input_edit = _chain_input_edit(panel)
    try:
        chain_input_edit.setPlainText("Clipboard output")
        panel.run_selected_chain()
        panel.copy_final_output()
        clipboard = QApplication.clipboard()
        assert clipboard is not None
        assert clipboard.text() == _manager_step_response_text(manager)
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_can_copy_final_summary_to_clipboard(qt_app: QApplication) -> None:
    manager = _ManagerStub()
    dialog, panel, manager = _build_dialog(manager)
    chain_input_edit = _chain_input_edit(panel)
    try:
        chain_input_edit.setPlainText("Clipboard summary")
        panel.run_selected_chain()
        panel.copy_final_summary()
        clipboard = QApplication.clipboard()
        assert clipboard is not None
        assert clipboard.text() == "Demo summary"
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_can_save_displayed_result_to_file(
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manager = _ManagerStub()
    dialog, panel, manager = _build_dialog(manager)
    chain_input_edit = _chain_input_edit(panel)
    output_path = tmp_path / "prompt-chain-result.txt"
    try:
        chain_input_edit.setPlainText("Save result")
        panel.run_selected_chain()
        monkeypatch.setattr(
            "gui.dialogs.prompt_chains.QFileDialog.getSaveFileName",
            _select_save_path(output_path),
        )

        panel.save_result_to_file()

        saved = output_path.read_text(encoding="utf-8")
        assert "Input to chain" in saved
        assert "Final output" in saved
        assert "Demo summary" in saved
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_second_run_replaces_previous_result_cues(qt_app: QApplication) -> None:
    """Running a chain again should replace prior result text instead of appending stale cues."""

    manager = _ManagerStub(step_response_text="First response")
    dialog, panel, manager = _build_dialog(manager)
    try:
        _chain_input_edit(panel).setPlainText("First input")
        panel.run_selected_chain()
        first_plain = _result_view(panel).toPlainText()
        assert "First input" in first_plain
        assert "Supporting summary" in first_plain

        cast("Any", manager)._step_response_text = "Second response"
        _chain_input_edit(panel).setPlainText("Second input")
        panel.run_selected_chain()

        second_plain = _result_view(panel).toPlainText()
        assert "Second input" in second_plain
        assert second_plain.count("Input to chain") == 1
        assert second_plain.count("Supporting summary") == 1
        assert "First input" not in second_plain
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_renders_reasoning_summary(qt_app: QApplication) -> None:
    """Reasoning snippets should appear with dedicated styling when available."""

    manager = _ManagerStub(step_reasoning_text="Deliberate reasoning path.")
    dialog, panel, manager = _build_dialog(manager)
    try:
        _chain_input_edit(panel).setPlainText("Reasoning input")
        panel.run_selected_chain()
        plain = _result_view(panel).toPlainText()
        assert "Reasoning summary" in plain
        assert "Deliberate reasoning path." in plain
        assert "#1e88e5" in _panel_richtext(panel)
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_omits_reasoning_when_summary_disabled(
    qt_app: QApplication,
) -> None:
    """Reasoning and chain summary must disappear when the preference is off."""

    manager = _ManagerStub(step_reasoning_text="Should not appear.")
    _manager_chain(manager).summarize_last_response = False
    dialog, panel, manager = _build_dialog(manager)
    try:
        _chain_input_edit(panel).setPlainText("No reasoning")
        panel.run_selected_chain()
        plain = _result_view(panel).toPlainText()
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
            step_outputs={"final": "done"},
            final_output_text="Second final output",
            final_summary_text="Summary text",
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
        )
        panel.display_run_result(result)
        plain = _panel_plaintext(panel)
        assert plain.count("Reasoning summary") == 1
        assert "Final reason" in plain
        assert "First reason" not in plain
        rich = _panel_richtext(panel)
        assert "Final reason" in rich
        assert "First reason" not in rich
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_shows_later_step_handoff_input(qt_app: QApplication) -> None:
    """Later successful steps should explicitly show the input received from the previous step."""

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
            step_outputs={"final": "Second response"},
            final_output_text="Second response",
            final_summary_text="Demo summary",
            steps=[
                PromptChainStepRun(
                    step=step_one,
                    status="success",
                    outcome=_make_outcome(
                        request_text="Initial text",
                        response_text="First response",
                    ),
                ),
                PromptChainStepRun(
                    step=step_two,
                    status="success",
                    outcome=_make_outcome(
                        request_text="First response",
                        response_text="Second response",
                    ),
                ),
            ],
        )
        panel.display_run_result(result)
        plain = _panel_plaintext(panel)
        assert plain.count("Input to step:") == 1
        assert plain.count("Input from previous step output:") == 1
        assert "First response" in plain
        rich = _panel_richtext(panel)
        assert "Input from previous step output" in rich
        assert "First response" in rich
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
        _chain_input_edit(panel).setPlainText("Markdown step input")
        panel.run_selected_chain()
        rich = _panel_richtext(panel)
        assert "chain-block--outputs" in rich
        assert "```" not in rich
        plain = _panel_plaintext(panel)
        assert "Supporting summary" in plain
        assert "### Step output" in plain
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_shows_step_label_and_machine_output_key(
    qt_app: QApplication,
) -> None:
    """Rendered step blocks should expose human label separately from machine output key."""

    manager = _ManagerStub()
    dialog, panel, manager = _build_dialog(manager)
    try:
        _chain_input_edit(panel).setPlainText("Label demo")
        panel.run_selected_chain()

        plain = _panel_plaintext(panel)
        rich = _panel_richtext(panel)

        assert "Step 1: final" in plain
        assert "Output key: step_1" in plain
        assert "Step 1 – final" in rich
        assert "Output key:</strong> <code>step_1</code>" in rich
    finally:
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_editor_dialog_creates_chain(qt_app: QApplication) -> None:
    """Editor dialog should build a PromptChain when inputs are valid."""

    editor = PromptChainEditorDialog(None, manager=None)
    cast("Any", _editor_name_input(editor)).setText("New Chain")
    _editor_description_input(editor).setPlainText("Describe")
    step = PromptChainStep(
        id=uuid.uuid4(),
        chain_id=editor.chain_id(),
        prompt_id=uuid.uuid4(),
        order_index=1,
        input_template="",
        output_variable="result",
    )
    editor.set_steps([step])
    editor.accept_chain()
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
    assert _editor_summarize_checkbox(editor).isChecked() is False
    editor.accept_chain()
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
        chain_id=editor.chain_id(),
        prompt_id=prompt.id,
        order_index=1,
        input_template="",
        output_variable="result",
    )
    editor.set_steps([step])
    item = _editor_steps_table(editor).item(0, 1)
    assert item is not None
    assert "Demo Prompt" in (item.toolTip() or "")
    captured: dict[str, str] = {}

    def _capture(parent: object, title: str, text: str) -> QMessageBox.StandardButton:  # noqa: ANN001
        captured["title"] = title
        captured["text"] = text
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr("gui.dialogs.prompt_chain_editor.QMessageBox.information", _capture)
    editor.open_step_preview(item)
    assert captured["title"] == "Demo Prompt"
    assert "Body text" in captured["text"]


def test_prompt_chain_step_dialog_shows_selected_prompt_preview(
    qt_app: QApplication,
) -> None:
    prompt = Prompt(
        id=uuid.uuid4(),
        name="Preview Prompt",
        description="Short fallback",
        category="tests",
        context="Full preview body for the selected prompt.",
    )
    dialog = PromptChainStepDialog(None, chain_id=uuid.uuid4(), prompts=[prompt])

    assert "Preview Prompt" in dialog.prompt_preview_text()
    assert "Full preview body for the selected prompt." in dialog.prompt_preview_text()


def test_prompt_chain_step_dialog_updates_prompt_preview_on_selection_change(
    qt_app: QApplication,
) -> None:
    first_prompt = Prompt(
        id=uuid.uuid4(),
        name="First Prompt",
        description="First fallback",
        category="tests",
        context="First preview body.",
    )
    second_prompt = Prompt(
        id=uuid.uuid4(),
        name="Second Prompt",
        description="Second fallback",
        category="tests",
        context="Second preview body.",
    )
    dialog = PromptChainStepDialog(
        None,
        chain_id=uuid.uuid4(),
        prompts=[first_prompt, second_prompt],
    )

    dialog.prompt_combo().setCurrentIndex(1)

    preview_text = dialog.prompt_preview_text()
    assert "Second Prompt" in preview_text
    assert "Second preview body." in preview_text


def test_prompt_chain_editor_reorder_controls_move_selected_step(
    qt_app: QApplication,
) -> None:
    editor = PromptChainEditorDialog(None, manager=None)
    first_step = PromptChainStep(
        id=uuid.uuid4(),
        chain_id=editor.chain_id(),
        prompt_id=uuid.uuid4(),
        order_index=1,
        input_template="",
        output_variable="first",
    )
    second_step = PromptChainStep(
        id=uuid.uuid4(),
        chain_id=editor.chain_id(),
        prompt_id=uuid.uuid4(),
        order_index=2,
        input_template="",
        output_variable="second",
    )
    editor.set_steps([first_step, second_step])

    _editor_steps_table(editor).selectRow(1)
    editor.move_selected_step_up()

    assert [step.output_variable for step in editor.steps()] == ["second", "first"]

    editor.move_selected_step_down()

    assert [step.output_variable for step in editor.steps()] == ["first", "second"]


def test_prompt_chain_editor_reorder_controls_update_button_state(
    qt_app: QApplication,
) -> None:
    editor = PromptChainEditorDialog(None, manager=None)
    editor.set_steps(
        [
            PromptChainStep(
                id=uuid.uuid4(),
                chain_id=editor.chain_id(),
                prompt_id=uuid.uuid4(),
                order_index=1,
                input_template="",
                output_variable="first",
            ),
            PromptChainStep(
                id=uuid.uuid4(),
                chain_id=editor.chain_id(),
                prompt_id=uuid.uuid4(),
                order_index=2,
                input_template="",
                output_variable="second",
            ),
        ]
    )

    _editor_steps_table(editor).selectRow(0)
    editor.update_step_action_state()
    assert cast("Any", _editor_move_up_button(editor)).isEnabled() is False
    assert cast("Any", _editor_move_down_button(editor)).isEnabled() is True

    _editor_steps_table(editor).selectRow(1)
    editor.update_step_action_state()
    assert cast("Any", _editor_move_up_button(editor)).isEnabled() is True
    assert cast("Any", _editor_move_down_button(editor)).isEnabled() is False


def test_prompt_chain_editor_reindexes_steps_after_reorder_on_save(
    qt_app: QApplication,
) -> None:
    editor = PromptChainEditorDialog(None, manager=None)
    cast("Any", _editor_name_input(editor)).setText("Ordered Chain")
    editor.set_steps(
        [
            PromptChainStep(
                id=uuid.uuid4(),
                chain_id=editor.chain_id(),
                prompt_id=uuid.uuid4(),
                order_index=2,
                input_template="",
                output_variable="second",
            ),
            PromptChainStep(
                id=uuid.uuid4(),
                chain_id=editor.chain_id(),
                prompt_id=uuid.uuid4(),
                order_index=1,
                input_template="",
                output_variable="first",
            ),
        ]
    )

    editor.accept_chain()

    chain = editor.result_chain()
    assert chain is not None
    assert [step.output_variable for step in chain.steps] == ["first", "second"]
    assert [step.order_index for step in chain.steps] == [1, 2]


def test_prompt_chain_editor_warns_when_same_prompt_is_reused(
    qt_app: QApplication,
) -> None:
    prompt_id = uuid.uuid4()
    editor = PromptChainEditorDialog(None, manager=None)
    editor.set_steps(
        [
            PromptChainStep(
                id=uuid.uuid4(),
                chain_id=editor.chain_id(),
                prompt_id=prompt_id,
                order_index=1,
                input_template="",
                output_variable="first",
            ),
            PromptChainStep(
                id=uuid.uuid4(),
                chain_id=editor.chain_id(),
                prompt_id=prompt_id,
                order_index=2,
                input_template="",
                output_variable="second",
            ),
        ]
    )

    warning_text = _editor_warning_label(editor).text()
    assert "Warning:" in warning_text
    assert "same prompt" in warning_text.lower()


def test_prompt_chain_editor_hides_duplicate_prompt_warning_when_not_needed(
    qt_app: QApplication,
) -> None:
    editor = PromptChainEditorDialog(None, manager=None)
    editor.set_steps(
        [
            PromptChainStep(
                id=uuid.uuid4(),
                chain_id=editor.chain_id(),
                prompt_id=uuid.uuid4(),
                order_index=1,
                input_template="",
                output_variable="first",
            ),
            PromptChainStep(
                id=uuid.uuid4(),
                chain_id=editor.chain_id(),
                prompt_id=uuid.uuid4(),
                order_index=2,
                input_template="",
                output_variable="second",
            ),
        ]
    )

    assert _editor_warning_label(editor).text() == ""


def test_prompt_chain_editor_duplicate_step_copies_selected_prompt_with_new_identity(
    qt_app: QApplication,
) -> None:
    prompt = Prompt(
        id=uuid.uuid4(),
        name="Reusable Prompt",
        description="Body",
        category="tests",
    )
    editor = PromptChainEditorDialog(None, manager=None, prompts=[prompt])
    original_step = PromptChainStep(
        id=uuid.uuid4(),
        chain_id=editor.chain_id(),
        prompt_id=prompt.id,
        order_index=1,
        input_template="{{ legacy }}",
        output_variable="first",
        condition="legacy condition",
        stop_on_failure=False,
    )
    editor.set_steps([original_step])
    _editor_steps_table(editor).selectRow(0)

    editor.duplicate_selected_step()

    assert len(editor.steps()) == 2
    duplicated_step = editor.steps()[1]
    assert duplicated_step.id != original_step.id
    assert duplicated_step.prompt_id == original_step.prompt_id
    assert duplicated_step.order_index == 2
    assert duplicated_step.output_variable == "step_2"


def test_prompt_chain_editor_pre_save_warning_summary_surfaces_bounded_issues(
    qt_app: QApplication,
) -> None:
    prompt_id = uuid.uuid4()
    editor = PromptChainEditorDialog(None, manager=None)
    cast("Any", _editor_name_input(editor)).setText("Warning Chain")
    editor.set_steps(
        [
            PromptChainStep(
                id=uuid.uuid4(),
                chain_id=editor.chain_id(),
                prompt_id=prompt_id,
                order_index=1,
                input_template="{{ legacy }}",
                output_variable="first",
                condition="legacy condition",
            ),
            PromptChainStep(
                id=uuid.uuid4(),
                chain_id=editor.chain_id(),
                prompt_id=prompt_id,
                order_index=2,
                input_template="",
                output_variable="second",
            ),
        ]
    )
    editor.accept_chain()

    warning_text = _editor_warning_label(editor).text()
    assert "same prompt" in warning_text.lower()
    assert "legacy/inactive semantics" in warning_text.lower()
    assert editor.result_chain() is not None


def test_prompt_chain_editor_legacy_warning_mentions_inactive_import_fields(
    qt_app: QApplication,
) -> None:
    editor = PromptChainEditorDialog(None, manager=None)
    editor.set_steps(
        [
            PromptChainStep(
                id=uuid.uuid4(),
                chain_id=editor.chain_id(),
                prompt_id=uuid.uuid4(),
                order_index=1,
                input_template="{{ imported }}",
                output_variable="first",
                condition="legacy condition",
            )
        ]
    )

    warning_text = _editor_warning_label(editor).text()
    assert "legacy/inactive semantics" in warning_text.lower()
    assert "input_template" in warning_text
    assert "condition" in warning_text


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
        item = _chain_list_widget(panel).item(0)
        panel.activate_chain_item(item)
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
    panel.create_chain()
    assert manager.saved_chain is not None
    dialog.close()
    dialog.deleteLater()


def test_prompt_chain_manager_duplicates_chain_via_editor(
    qt_app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Duplicate action should open the editor with a copied chain and persist the returned copy."""

    manager = _ManagerStub()
    dialog, panel, manager = _build_dialog(manager)
    source_chain = manager.list_prompt_chains()[0]
    panel.select_chain(source_chain.id)
    captured: dict[str, PromptChain] = {}

    class _EditorStub:
        def __init__(self, *_: object, **kwargs: object) -> None:
            chain = kwargs.get("chain")
            assert isinstance(chain, PromptChain)
            captured["editor_chain"] = chain
            self._chain = chain

        def exec(self) -> int:
            return int(QDialog.DialogCode.Accepted)

        def result_chain(self) -> PromptChain:
            return self._chain

    monkeypatch.setattr(
        "gui.dialogs.prompt_chains.PromptChainEditorDialog",
        _EditorStub,
    )

    panel.duplicate_chain()

    duplicated = manager.saved_chain
    assert duplicated is not None
    assert duplicated.id != source_chain.id
    assert duplicated.name == f"{source_chain.name} (Copy)"
    assert len(duplicated.steps) == len(source_chain.steps)
    assert duplicated.steps[0].id != source_chain.steps[0].id
    assert duplicated.steps[0].chain_id == duplicated.id
    assert captured["editor_chain"].id == duplicated.id
    dialog.close()
    dialog.deleteLater()


def test_prompt_chain_manager_deletes_chain(
    qt_app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Delete action should invoke PromptManager.delete_prompt_chain."""

    manager = _ManagerStub()
    expected_chain_id = manager.list_prompt_chains()[0].id
    dialog, panel, manager = _build_dialog(manager)
    monkeypatch.setattr(
        "gui.dialogs.prompt_chains.QMessageBox.question",
        _confirm_yes,
    )
    panel.delete_chain()
    assert manager.deleted_chain_ids == [expected_chain_id]
    dialog.close()
    dialog.deleteLater()


def test_prompt_chain_panel_refreshes_backend_recent_run_history(
    qt_app: QApplication,
) -> None:
    manager = _ManagerStub()
    dialog, panel, manager = _build_dialog(manager)
    result = manager.run_prompt_chain(
        manager.list_prompt_chains()[0].id,
        chain_input="history input",
        use_web_search=True,
    )

    panel.record_run_history(result)

    assert len(panel.run_history()) == 1
    entry = panel.run_history()[0]
    assert entry["chain_name"] == manager.list_prompt_chains()[0].name
    assert entry["chain_id"] == str(manager.list_prompt_chains()[0].id)
    assert entry["input_preview"] == "history input"
    assert entry["status"] == "success"
    assert entry["run_timestamp"] == "2026-05-08T12:00:00+00:00"
    assert entry["final_step_label"] == "final"
    assert "Recent runs for 'Demo Chain'" in panel.history_label_text()
    assert "output:" in panel.history_label_text()
    assert "in this session" not in panel.history_label_text()
    dialog.close()
    dialog.deleteLater()


def test_prompt_chain_panel_uses_backend_bounded_recent_run_history(
    qt_app: QApplication,
) -> None:
    manager = _ManagerStub()
    dialog, panel, manager = _build_dialog(manager)

    for index in range(6):
        result = manager.run_prompt_chain(
            manager.list_prompt_chains()[0].id,
            chain_input=f"input {index}",
            use_web_search=True,
        )
        panel.record_run_history(result)

    assert len(panel.run_history()) == 6
    assert panel.run_history()[0]["input_preview"] == "input 5"
    assert panel.run_history()[-1]["input_preview"] == "input 0"
    assert "input 5" in panel.history_label_text()
    assert "input 0" in panel.history_label_text()
    dialog.close()
    dialog.deleteLater()


def test_prompt_chain_panel_filters_backend_recent_history_to_selected_chain(
    qt_app: QApplication,
) -> None:
    manager = _ManagerStub()
    second_chain = PromptChain(
        id=uuid.uuid4(),
        name="Second Chain",
        description="Another chain",
        steps=[
            PromptChainStep(
                id=uuid.uuid4(),
                chain_id=uuid.uuid4(),
                prompt_id=uuid.uuid4(),
                order_index=1,
                input_template="",
                output_variable="step_1",
            )
        ],
    )
    second_chain.steps[0].chain_id = second_chain.id
    cast("Any", manager)._chains.append(second_chain)
    dialog, panel, manager = _build_dialog(manager)

    first_result = manager.run_prompt_chain(
        manager.list_prompt_chains()[0].id,
        chain_input="first chain input",
        use_web_search=True,
    )
    panel.record_run_history(first_result)

    second_result = manager.run_prompt_chain(
        second_chain.id,
        chain_input="second chain input",
        use_web_search=True,
    )
    panel.record_run_history(second_result)

    panel.set_selected_chain_id(str(second_chain.id))
    panel.refresh_run_history_label()
    history_text = panel.history_label_text()
    assert "Recent runs for 'Second Chain'" in history_text
    assert "second chain input" in history_text
    assert "first chain input" not in history_text
    dialog.close()
    dialog.deleteLater()


def test_prompt_chain_dialog_stream_preview_renders(qt_app: QApplication) -> None:
    """Streaming preview should show incremental text in the result area."""

    manager = _ManagerStub()
    dialog, panel, manager = _build_dialog(manager)
    step = manager.list_prompt_chains()[0].steps[0]
    try:
        panel.begin_stream_preview(manager.list_prompt_chains()[0], "Streaming input")
        panel.register_stream_chunk(step, "partial response", False)
        text = panel.result_view_plaintext()
        assert "Input to chain" in text
        assert "Streaming input" in text
        assert "partial response" in text
    finally:
        panel.end_stream_preview()
        dialog.close()
        dialog.deleteLater()


def test_prompt_chain_dialog_stream_detection_uses_executor(qt_app: QApplication) -> None:
    """Streaming flag should be inferred from the manager executor."""

    manager = _ManagerStub(stream_enabled=True)
    dialog, panel, manager = _build_dialog(manager)
    try:
        assert panel.is_streaming_enabled() is True
    finally:
        dialog.close()
        dialog.deleteLater()
