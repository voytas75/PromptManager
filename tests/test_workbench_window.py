"""Focused tests for bounded WorkbenchWindow lifecycle and run fallback seams.

Updates:
  v0.1.2 - 2026-05-08 - Cover run-to-refine status cues and feedback focus messaging.
  v0.1.1 - 2026-05-08 - Use typed harness helpers so Pyright accepts bounded GUI seam tests.
  v0.1.0 - 2026-05-08 - Add RED tests for begin_session, template mode,
  run fallbacks, and offline gating.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, cast

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QTextEdit,
)

from core.execution import CodexExecutionResult
from gui.workbench import WorkbenchMode, WorkbenchWindow
from gui.workbench.session import WorkbenchExecutionRecord, WorkbenchSession
from models.prompt_model import Prompt

if TYPE_CHECKING:
    from core.prompt_manager import PromptManager


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    """Provide a shared Qt application instance for Workbench tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return cast("QApplication", app)


class _ExecutorStub:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.stream = False

    def execute(
        self,
        prompt: Prompt,
        request_text: str,
        *,
        conversation: Any = None,
        stream: bool | None = None,
        on_stream: Any = None,
    ) -> CodexExecutionResult:
        self.calls.append(request_text)
        return CodexExecutionResult(
            prompt_id=prompt.id,
            request_text=request_text,
            response_text=f"response for: {request_text}",
            duration_ms=12,
            usage={},
            raw_response={},
        )


class _PromptManagerStub:
    def __init__(
        self, *, llm_available: bool = True, executor: _ExecutorStub | None = None
    ) -> None:
        self.executor = executor
        self.history_tracker = None
        self.llm_available = llm_available
        self.created_prompts: list[Prompt] = []

    def llm_status_message(self, action: str) -> str:
        return f"{action} unavailable in tests"

    def create_prompt(self, prompt: Prompt) -> Prompt:
        self.created_prompts.append(prompt)
        return prompt


class _WorkbenchHarness:
    def __init__(self, window: WorkbenchWindow, executor: _ExecutorStub | None = None) -> None:
        self.window = window
        self.executor = executor
        self._window_any = cast("Any", window)

    @property
    def session(self) -> WorkbenchSession:
        return cast("WorkbenchSession", self._window_any._session)

    @property
    def editor(self) -> QPlainTextEdit:
        return cast("QPlainTextEdit", self._window_any._editor)

    @property
    def output_view(self) -> QTextEdit:
        return cast("QTextEdit", self._window_any._output_view)

    @property
    def history_list(self) -> QListWidget:
        return cast("QListWidget", self._window_any._history_list)

    @property
    def feedback_input(self) -> QLineEdit:
        return cast("QLineEdit", self._window_any._feedback_input)

    @property
    def test_input(self) -> QPlainTextEdit:
        return cast("QPlainTextEdit", self._window_any._test_input)

    def handle_preview_run(self, rendered_text: str, variables: dict[str, str]) -> None:
        self._window_any._handle_preview_run(rendered_text, variables)

    def record_rating(self, rating: float) -> None:
        self._window_any._record_rating(rating)

    def run_brainstorm(self) -> None:
        self._window_any._run_brainstorm()

    def run_peek(self) -> None:
        self._window_any._run_peek()

    @property
    def summary_label_text(self) -> str:
        summary_label = cast("QLabel", self._window_any._summary_label)
        return summary_label.text()


def _make_prompt_manager_stub(
    *, llm_available: bool = True, executor: _ExecutorStub | None = None
) -> PromptManager:
    manager = _PromptManagerStub(llm_available=llm_available, executor=executor)
    return cast("PromptManager", cast("Any", manager))


def _make_window(
    *,
    mode: str = WorkbenchMode.BLANK,
    template_prompt: Prompt | None = None,
    llm_available: bool = True,
    executor: _ExecutorStub | None = None,
) -> _WorkbenchHarness:
    window = WorkbenchWindow(
        _make_prompt_manager_stub(llm_available=llm_available, executor=executor),
        mode=mode,
        template_prompt=template_prompt,
    )
    return _WorkbenchHarness(window, executor=executor)


def _make_prompt(
    *, name: str = "Template Prompt", description: str = "Summarise incidents"
) -> Prompt:
    return Prompt(
        id=uuid.uuid4(),
        name=name,
        description=description,
        category="Operations",
        category_slug="operations",
        context="### System Role\nYou are helpful.\n\n### Goal\nSummarise the incident.",
    )


def _capture_information_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, str]]:
    shown: list[tuple[str, str]] = []

    def _record_information(
        parent: Any,
        title: str,
        message: str,
        /,
    ) -> None:
        del parent
        shown.append((title, message))

    monkeypatch.setattr(QMessageBox, "information", _record_information)
    return shown


def test_workbench_window_begin_session_clears_runtime_state_for_blank_mode(
    qt_app: QApplication,
) -> None:
    """Starting a fresh blank session should clear old runtime state.

    The editor, output, and transient form fields should all reset too.
    """
    harness = _make_window(executor=_ExecutorStub())
    harness.session.prompt_name = "Old draft"
    harness.session.goal_statement = "Old goal"
    harness.session.record_execution(
        WorkbenchExecutionRecord(request_text="old", response_text="old response")
    )
    harness.editor.setPlainText("Old text")
    harness.output_view.setPlainText("Old output")
    harness.history_list.addItem("old history")
    harness.feedback_input.setText("old feedback")
    harness.test_input.setPlainText("old request")
    harness.window.statusBar().showMessage("old status")
    harness.session.template_text = "### Context\nOld context"
    harness.editor.setPlainText(harness.session.template_text)
    harness.handle_preview_run(harness.session.template_text, {})
    assert harness.summary_label_text.endswith("Refinement focus: Context")

    harness.window.begin_session(WorkbenchMode.BLANK)

    assert harness.session.prompt_name == ""
    assert harness.session.goal_statement == ""
    assert harness.session.execution_history == []
    assert harness.editor.toPlainText() == ""
    assert harness.output_view.toPlainText() == ""
    assert harness.history_list.count() == 0
    assert harness.feedback_input.text() == ""
    assert harness.test_input.toPlainText() == ""
    assert harness.window.statusBar().currentMessage() == ""
    assert harness.summary_label_text == "No goal defined yet."


def test_workbench_window_template_mode_prefills_session_from_prompt(
    qt_app: QApplication,
) -> None:
    """Template mode should preload the selected prompt into the new workbench session."""
    prompt = _make_prompt(name="Incident Template", description="Summarise outage")

    harness = _make_window(
        executor=_ExecutorStub(),
        mode=WorkbenchMode.TEMPLATE,
        template_prompt=prompt,
    )

    assert harness.session.prompt_name == "Incident Template"
    assert harness.session.goal_statement == "Summarise outage"
    assert harness.session.template_text == prompt.context
    assert harness.editor.toPlainText() == prompt.context


def test_workbench_window_run_uses_first_variable_when_test_input_missing(
    qt_app: QApplication,
) -> None:
    """Without explicit test input, the first populated variable should drive the request text."""
    executor = _ExecutorStub()
    harness = _make_window(executor=executor)
    harness.session.goal_statement = "Fallback goal"
    harness.session.template_text = "### Context\nOriginal context\n\n### Goal\nUse variables"
    harness.editor.setPlainText(harness.session.template_text)
    harness.session.link_variable("topic", sample_value="database outage")

    harness.handle_preview_run(harness.session.template_text, harness.session.variable_payload())

    assert executor.calls == ["database outage"]
    record = harness.session.execution_history[-1]
    assert record.request_text == "database outage"
    assert record.suggested_focus == "context"
    assert harness.window.statusBar().currentMessage() == (
        "Refine Context next — test input was empty, so the preview used "
        "the 'topic' variable value."
    )
    selections = harness.editor.extraSelections()
    assert len(selections) == 1
    selection = cast("Any", selections[0])
    assert selection.cursor.selectedText() == "### Context"


def test_workbench_window_run_uses_goal_when_no_input_or_variables_exist(
    qt_app: QApplication,
) -> None:
    """The goal statement should become the request when no test input or variable value exists."""
    executor = _ExecutorStub()
    harness = _make_window(executor=executor)
    harness.session.goal_statement = "Summarise the deployment impact"
    harness.session.template_text = (
        "### Context\nDeployment notes\n\n"
        "### Goal\nSummarise the deployment impact"
    )
    harness.editor.setPlainText(harness.session.template_text)

    harness.handle_preview_run(harness.session.template_text, {})

    assert executor.calls == ["Summarise the deployment impact"]
    record = harness.session.execution_history[-1]
    assert record.request_text == "Summarise the deployment impact"
    assert record.suggested_focus == "context"
    assert harness.window.statusBar().currentMessage() == (
        "Refine Context next — test input was empty, so the preview used the prompt goal."
    )


def test_workbench_window_run_uses_generic_fallback_when_session_has_no_guidance(
    qt_app: QApplication,
) -> None:
    """A generic fallback request should be used when the session has no input guidance at all."""
    executor = _ExecutorStub()
    harness = _make_window(executor=executor)
    harness.session.template_text = "### Constraints\n- Keep it short"
    harness.editor.setPlainText(harness.session.template_text)

    harness.handle_preview_run(harness.session.template_text, {})

    assert executor.calls == ["Run a preview based on the current prompt."]
    record = harness.session.execution_history[-1]
    assert record.request_text == "Run a preview based on the current prompt."
    assert record.suggested_focus == "constraints"
    assert harness.window.statusBar().currentMessage() == (
        "Refine Constraints next — no test input or prompt goal was available, "
        "so the preview used a generic request."
    )


def test_workbench_window_feedback_save_mentions_refinement_focus(qt_app: QApplication) -> None:
    """Saving feedback should preserve the current refinement target in the status cue."""
    executor = _ExecutorStub()
    harness = _make_window(executor=executor)
    harness.session.template_text = "### Constraints\n- Keep it short"
    harness.editor.setPlainText(harness.session.template_text)
    harness.handle_preview_run(harness.session.template_text, {})
    harness.feedback_input.setText("Needs sharper output format")

    harness.record_rating(0.0)

    record = harness.session.execution_history[-1]
    assert record.rating == 0.0
    assert record.feedback == "Needs sharper output format"
    assert harness.window.statusBar().currentMessage() == (
        "Feedback saved for last run — current refinement focus: Constraints."
    )


def test_workbench_window_brainstorm_sets_persistent_refinement_summary(
    qt_app: QApplication,
) -> None:
    """Brainstorm should keep the last refinement focus visible outside the transient status bar."""
    executor = _ExecutorStub()
    harness = _make_window(executor=executor)
    harness.session.template_text = "### Context\nOriginal context\n\n### Goal\nUse variables"
    harness.editor.setPlainText(harness.session.template_text)
    harness.session.link_variable("topic", sample_value="database outage")
    harness.handle_preview_run(harness.session.template_text, harness.session.variable_payload())

    harness.run_brainstorm()

    assert executor.calls[-1].startswith(
        "Provide three alternative phrasings that could strengthen this prompt."
    )
    assert harness.window.statusBar().currentMessage() == "Brainstorm suggestions ready."
    assert harness.summary_label_text.endswith("Refinement focus: Context")


def test_workbench_window_ai_peek_preserves_existing_refinement_focus(qt_app: QApplication) -> None:
    """AI Peek should not clear the persistent refinement cue established by the last run."""
    executor = _ExecutorStub()
    harness = _make_window(executor=executor)
    harness.session.template_text = "### Constraints\n- Keep it short"
    harness.editor.setPlainText(harness.session.template_text)
    harness.handle_preview_run(harness.session.template_text, {})

    harness.run_peek()

    assert executor.calls[-1].startswith(
        "Summarise this prompt in two sentences and point out obvious gaps."
    )
    assert harness.summary_label_text.endswith("Refinement focus: Constraints")


def test_workbench_window_run_is_blocked_when_llm_unavailable(
    qt_app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline LLM state should block execution before the executor path is touched."""
    executor = _ExecutorStub()
    harness = _make_window(llm_available=False, executor=executor)
    shown = _capture_information_messages(monkeypatch)

    harness.handle_preview_run("### Goal\nOffline", {})

    assert executor.calls == []
    assert shown == [
        (
            "Prompt execution unavailable",
            "Prompt execution unavailable in tests",
        )
    ]
    assert harness.session.execution_history == []
