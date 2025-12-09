"""Qt widgets for the Enhanced Prompt Workbench experience.

Updates:
  v0.1.25 - 2025-12-09 - Disable template runs when LiteLLM is offline.
  v0.1.24 - 2025-12-08 - Align PySide6 enums and text selection handling with Pyright.
  v0.1.23 - 2025-12-07 - Support embedded tab sessions and add begin_session helper.
  v0.1.22 - 2025-12-04 - Modularize dialogs, wizard, editor, and utilities into separate modules.
  v0.1.21 - 2025-11-30 - Skip busy indicator whenever LiteLLM streaming is enabled.
  v0.1.20 - 2025-11-30 - Respect LiteLLM streaming flag when running prompts.
  v0.1.19 - 2025-11-29 - Fix toast calls to pass the parent widget first.
  v0.1.18 - 2025-11-29 - Persist Workbench window geometry between sessions.
  v0.1.17 - 2025-11-29 - Stack the prompt editor above Run Output/History in the center column.
  v0.1.16 - 2025-11-29 - Relocate output/history tabs into the center column and
    collapse the bottom panel.
  v0.1.15 - 2025-11-29 - Move output/history tabs below the editor and persist
    output splitter widths.
Earlier versions: v0.1.0-v0.1.14 - Introduced the guided Workbench window plus
  iterative palette refinements.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, TypeVar, cast

from PySide6.QtCore import QByteArray, QEventLoop, QSettings, Qt
from PySide6.QtGui import QCloseEvent, QGuiApplication, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core import PromptManager, PromptManagerError
from core.execution import CodexExecutionResult, CodexExecutor, ExecutionError
from core.history_tracker import HistoryTracker, HistoryTrackerError

from ..processing_indicator import ProcessingIndicator
from ..template_preview import TemplatePreviewWidget
from ..toast import show_toast
from .dialogs import VariableCaptureDialog, WorkbenchExportDialog, WorkbenchMode
from .editor import WorkbenchPromptEditor
from .session import WorkbenchExecutionRecord, WorkbenchSession
from .utils import BLOCK_SNIPPETS, StreamRelay, normalise_variable_token, variable_at_cursor
from .wizard import GuidedPromptWizard

_T = TypeVar("_T")

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Callable, Mapping

    from models.prompt_model import Prompt
else:  # pragma: no cover - runtime placeholders for type-only imports
    from typing import Any as _Any

    Callable = _Any
    Mapping = _Any
    Prompt = _Any

logger = logging.getLogger("prompt_manager.gui.workbench")


class WorkbenchWindow(QMainWindow):
    """Modal workspace that guides users through crafting prompts iteratively."""

    def __init__(
        self,
        prompt_manager: PromptManager,
        *,
        mode: str = WorkbenchMode.GUIDED,
        template_prompt: Prompt | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Initialise the window, settings, and supporting controllers."""
        super().__init__(parent)
        self._manager = prompt_manager
        self._session = WorkbenchSession()
        self._settings = QSettings("PromptManager", "WorkbenchWindow")
        self._executor: CodexExecutor | None = prompt_manager.executor
        self._history_tracker: HistoryTracker | None = prompt_manager.history_tracker
        self._suppress_editor_signal = False
        self._active_refinement_target: str | None = None
        self._main_splitter: QSplitter | None = None
        self._middle_splitter: QSplitter | None = None
        self._stream_relay = StreamRelay(self)
        self._stream_relay.chunk.connect(self._handle_stream_chunk)
        self._streaming_active = False
        self._streaming_buffer: list[str] = []
        self._build_ui()
        self._load_initial_state(mode, template_prompt)

    def _build_ui(self) -> None:
        self.setWindowTitle("Enhanced Prompt Workbench")
        if self.isWindow():
            self.setWindowModality(Qt.WindowModality.ApplicationModal)
            self.resize(1280, 860)

        toolbar = QToolBar("Workbench Controls", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        wizard_action = toolbar.addAction("🧙 Wizard", self._launch_wizard)
        wizard_action.setToolTip("Restart the guided wizard")
        link_action = toolbar.addAction("🔗 Link Variable", self._link_variable)
        link_action.setToolTip("Capture metadata and sample values for the selected variable.")
        self._brainstorm_action = toolbar.addAction("🤔 Brainstorm", self._run_brainstorm)
        self._brainstorm_action.setToolTip("Ask CodexExecutor to propose alternative phrasing.")
        self._peek_action = toolbar.addAction("👀 AI Peek", self._run_peek)
        self._peek_action.setToolTip("Request a quick summary of the prompt's behaviour.")
        validate_action = toolbar.addAction("✅ Validate", self._validate_template)
        validate_action.setToolTip("Re-run template preview validation and report status.")
        run_action = toolbar.addAction("▶️ Run Once", self._trigger_run)
        run_action.setToolTip(
            "Render the prompt with sample data and execute it via CodexExecutor."
        )
        export_action = toolbar.addAction("💾 Export", self._export_prompt)
        export_action.setToolTip("Persist this prompt into the repository.")
        executor_ready = self._executor is not None
        self._brainstorm_action.setEnabled(executor_ready)
        self._peek_action.setEnabled(executor_ready)
        run_action.setEnabled(executor_ready)

        container = QWidget(self)
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(8, 8, 8, 8)
        container_layout.setSpacing(8)
        self.setCentralWidget(container)

        self._summary_label = QLabel("", container)
        self._summary_label.setWordWrap(True)
        self._summary_label.setStyleSheet("font-weight: 500;")
        container_layout.addWidget(self._summary_label)

        self._main_splitter = QSplitter(Qt.Orientation.Horizontal, container)
        self._main_splitter.setChildrenCollapsible(False)
        container_layout.addWidget(self._main_splitter, 1)

        palette_frame = QFrame(self._main_splitter)
        palette_layout = QVBoxLayout(palette_frame)
        palette_layout.setContentsMargins(8, 8, 8, 8)
        palette_layout.addWidget(QLabel("Guidance blocks", palette_frame))
        self._palette_list = QListWidget(palette_frame)
        for label, snippet in BLOCK_SNIPPETS.items():
            item = QListWidgetItem(label)
            item.setToolTip(snippet.splitlines()[0])
            self._palette_list.addItem(item)
        self._palette_list.itemDoubleClicked.connect(self._insert_block)  # type: ignore[arg-type]
        palette_layout.addWidget(self._palette_list, 1)

        middle_panel = QWidget(self._main_splitter)
        middle_layout = QVBoxLayout(middle_panel)
        middle_layout.setContentsMargins(8, 8, 8, 8)
        middle_layout.setSpacing(8)
        self._middle_splitter = QSplitter(Qt.Orientation.Vertical, middle_panel)
        self._middle_splitter.setChildrenCollapsible(False)
        middle_layout.addWidget(self._middle_splitter, 1)

        editor_container = QWidget(self._middle_splitter)
        editor_container_layout = QVBoxLayout(editor_container)
        editor_container_layout.setContentsMargins(0, 0, 0, 0)
        editor_container_layout.setSpacing(6)
        editor_container_layout.addWidget(QLabel("Prompt Draft", editor_container))
        self._editor = WorkbenchPromptEditor(editor_container)
        self._editor.setPlaceholderText("Use the wizard or block palette to start…")
        self._editor.textChanged.connect(self._on_editor_changed)  # type: ignore[arg-type]
        self._editor.variableActivated.connect(self._open_variable_editor)
        editor_container_layout.addWidget(self._editor, 1)

        output_panel = QFrame(self._middle_splitter)
        output_layout = QVBoxLayout(output_panel)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(6)
        output_title = QLabel("Run Output & History", output_panel)
        output_title.setStyleSheet("font-weight: 500;")
        output_layout.addWidget(output_title)
        self._output_tabs = QTabWidget(output_panel)
        self._output_view = QTextEdit(output_panel)
        self._output_view.setReadOnly(True)
        self._output_tabs.addTab(self._output_view, "Run Output")
        self._history_list = QListWidget(output_panel)
        self._output_tabs.addTab(self._history_list, "History")
        output_layout.addWidget(self._output_tabs, 1)

        feedback_row = QHBoxLayout()
        feedback_row.setSpacing(6)
        feedback_row.addWidget(QLabel("Was the last run helpful?", output_panel))
        self._thumbs_up = QToolButton(output_panel)
        self._thumbs_up.setText("👍")
        self._thumbs_up.clicked.connect(lambda: self._record_rating(1.0))
        feedback_row.addWidget(self._thumbs_up)
        self._thumbs_down = QToolButton(output_panel)
        self._thumbs_down.setText("👎")
        self._thumbs_down.clicked.connect(lambda: self._record_rating(0.0))
        feedback_row.addWidget(self._thumbs_down)
        feedback_row.addWidget(QLabel("Feedback", output_panel))
        self._feedback_input = QLineEdit(output_panel)
        feedback_row.addWidget(self._feedback_input, 1)
        apply_feedback = QPushButton("Save", output_panel)
        apply_feedback.clicked.connect(self._apply_feedback)
        feedback_row.addWidget(apply_feedback)
        output_layout.addLayout(feedback_row)

        right_panel = QWidget(self._main_splitter)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(8)
        self._preview = TemplatePreviewWidget(right_panel)
        self._preview.set_run_enabled(self._executor is not None and getattr(self._manager, "llm_available", False))
        self._preview.run_requested.connect(self._handle_preview_run)  # type: ignore[arg-type]
        right_layout.addWidget(self._preview, 1)
        right_layout.addWidget(QLabel("Test input", right_panel))
        self._test_input = QPlainTextEdit(right_panel)
        self._test_input.setPlaceholderText("Provide user input to test this prompt…")
        self._test_input.setFixedHeight(120)
        right_layout.addWidget(self._test_input)

        self._status = QStatusBar(self)
        self.setStatusBar(self._status)
        self._restore_layout_state()

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        """Persist splitters and clean up when closing the window."""
        self._persist_layout_state()
        super().closeEvent(event)

    def begin_session(self, mode: str, template_prompt: Prompt | None = None) -> None:
        """Start a fresh Workbench session, clearing editor, output, and history."""
        self._session = WorkbenchSession()
        self._active_refinement_target = None
        self._streaming_active = False
        self._streaming_buffer = []
        self._summary_label.clear()
        self._editor.clear()
        self._output_view.clear()
        self._history_list.clear()
        self._feedback_input.clear()
        self._test_input.clear()
        self._status.clearMessage()
        self._load_initial_state(mode, template_prompt)

    def _restore_layout_state(self) -> None:
        if not self.isWindow():
            return
        geometry = self._settings.value("windowGeometry")
        if isinstance(geometry, QByteArray):
            self.restoreGeometry(geometry)
        main_state = self._settings.value("mainSplitterState")
        if isinstance(main_state, QByteArray) and self._main_splitter is not None:
            self._main_splitter.restoreState(main_state)
        middle_state = self._settings.value("middleSplitterState")
        if isinstance(middle_state, QByteArray) and self._middle_splitter is not None:
            self._middle_splitter.restoreState(middle_state)

    def _persist_layout_state(self) -> None:
        if not self.isWindow():
            return
        self._settings.setValue("windowGeometry", self.saveGeometry())
        if self._main_splitter is not None:
            self._settings.setValue("mainSplitterState", self._main_splitter.saveState())
        if self._middle_splitter is not None:
            self._settings.setValue("middleSplitterState", self._middle_splitter.saveState())

    def _load_initial_state(self, mode: str, template_prompt: Prompt | None) -> None:
        if mode == WorkbenchMode.TEMPLATE and template_prompt is not None:
            self._session.prompt_name = template_prompt.name
            self._session.goal_statement = template_prompt.description
            self._session.system_role = template_prompt.context or ""
            self._session.context = template_prompt.context or ""
            self._session.template_text = template_prompt.context or ""
            self._apply_editor_text(template_prompt.context or "", from_wizard=True)
        elif mode == WorkbenchMode.BLANK:
            self._apply_editor_text("", from_wizard=True)
        else:
            self._launch_wizard(initial=True)
        self._refresh_summary()
        self._sync_preview()

    def _refresh_summary(self) -> None:
        summary = self._session.goal_statement or "No goal defined yet."
        if self._session.audience:
            summary += f" (Audience: {self._session.audience})"
        self._summary_label.setText(summary)

    def _apply_editor_text(self, text: str, *, from_wizard: bool) -> None:
        self._suppress_editor_signal = True
        self._editor.setPlainText(text)
        self._suppress_editor_signal = False
        self._session.set_template_text(text, source="wizard" if from_wizard else "editor")
        self._sync_preview()

    def _sync_preview(self) -> None:
        prompt_id = self._session.prompt_name or "workbench"
        self._preview.set_template(self._session.template_text, prompt_id)
        self._preview.apply_variable_values(self._session.variable_payload())
        self._preview.refresh_preview()

    def _on_editor_changed(self) -> None:
        if self._suppress_editor_signal:
            return
        self._session.set_template_text(self._editor.toPlainText(), source="editor")
        self._sync_preview()

    def _insert_block(self, item: QListWidgetItem) -> None:
        snippet = BLOCK_SNIPPETS.get(item.text())
        if not snippet:
            return
        cursor = self._editor.textCursor()
        cursor.beginEditBlock()
        cursor.insertText(snippet + "\n\n")
        cursor.endEditBlock()
        self._on_editor_changed()

    def _launch_wizard(self, *, initial: bool = False) -> None:
        wizard = GuidedPromptWizard(self._session, self)
        wizard.updated.connect(self._handle_wizard_update)
        if wizard.exec() == QDialog.DialogCode.Accepted and not initial:
            show_toast(self, "Wizard applied to prompt.")

    def _handle_wizard_update(self, payload: Mapping[str, Any]) -> None:
        constraints = payload.get("constraints") or []
        variables = payload.get("variables") or {}
        self._session.update_from_wizard(
            prompt_name=str(payload.get("prompt_name") or ""),
            goal=str(payload.get("goal") or ""),
            system_role=str(payload.get("system_role") or ""),
            context=str(payload.get("context") or ""),
            audience=str(payload.get("audience") or ""),
            constraints=constraints,
            variables=variables,
        )
        self._apply_editor_text(self._session.template_text, from_wizard=True)
        self._refresh_summary()

    def _open_variable_editor(self, name: str) -> None:
        dialog = VariableCaptureDialog(name, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        variable = dialog.result_variable()
        if variable is None:
            return
        self._session.link_variable(
            variable.name,
            sample_value=variable.sample_value,
            description=variable.description,
        )
        self._preview.apply_variable_values({variable.name: variable.sample_value or ""})
        show_toast(self, f"Variable '{variable.name}' updated.")

    def _link_variable(self) -> None:
        cursor = self._editor.textCursor()
        if cursor.hasSelection():
            candidate = normalise_variable_token(cursor.selectedText())
        else:
            candidate = variable_at_cursor(cursor)
        dialog = VariableCaptureDialog(candidate, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        variable = dialog.result_variable()
        if variable is None:
            return
        stored = self._session.link_variable(
            variable.name,
            sample_value=variable.sample_value,
            description=variable.description,
        )
        self._preview.apply_variable_values({stored.name: stored.sample_value or ""})
        show_toast(self, f"Variable '{stored.name}' saved.")

    def _validate_template(self) -> None:
        self._preview.refresh_preview()
        self._status.showMessage("Template validation refreshed.", 4000)

    def _trigger_run(self) -> None:
        if not self._preview.request_run():
            self._status.showMessage("Preview not ready for execution.", 5000)

    def _is_streaming_enabled(self) -> bool:
        executor = self._executor
        if executor is None:
            return False
        return bool(getattr(executor, "stream", False))

    def _begin_streaming_run(self, status: str) -> None:
        self._streaming_active = True
        self._streaming_buffer = []
        self._output_tabs.setCurrentWidget(self._output_view)
        self._output_view.clear()
        self._status.showMessage(status, 0)

    def _end_streaming_run(self, *, success: bool) -> None:
        if not self._streaming_active:
            return
        self._streaming_active = False
        message = "Streaming complete." if success else "Streaming interrupted."
        self._status.showMessage(message, 4000)

    def _handle_stream_chunk(self, chunk: str) -> None:
        if not self._streaming_active or not chunk:
            return
        self._streaming_buffer.append(chunk)
        cursor = self._output_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(chunk)
        self._output_view.setTextCursor(cursor)
        self._output_view.ensureCursorVisible()
        QGuiApplication.processEvents()

    def _run_in_background(self, func: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
        event = threading.Event()
        result: dict[str, _T] = {}
        error: list[BaseException] = []

        def _worker() -> None:
            try:
                result["value"] = func(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001
                error.append(exc)
            finally:
                event.set()

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        try:
            while not event.is_set():
                QGuiApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
                event.wait(0.01)
        finally:
            thread.join()
        if error:
            raise error[0]
        if "value" not in result:
            raise RuntimeError("Prompt execution did not return a result")
        return result["value"]

    def _handle_preview_run(self, rendered_text: str, variables: Mapping[str, str]) -> None:
        if self._executor is None:
            self._status.showMessage("CodexExecutor is not configured.", 6000)
            return
        raw_request = self._test_input.toPlainText().strip()
        fallback_message = None
        if raw_request:
            request_text = raw_request
        else:
            fallback_value: tuple[str, str] | None = None
            for name, value in variables.items():
                trimmed = value.strip()
                if trimmed:
                    fallback_value = (name, trimmed)
                    break
            if fallback_value is not None:
                request_text = fallback_value[1]
                fallback_message = (
                    f"No test input supplied; using the '{fallback_value[0]}' variable value."
                )
            else:
                request_text = (
                    self._session.goal_statement.strip()
                    or "Run a preview based on the current prompt."
                )
                fallback_message = "No test input supplied; using the prompt goal instead."
        prompt = self._session.build_prompt()
        prompt.context = rendered_text
        streaming_enabled = self._is_streaming_enabled()
        stream_success = False
        indicator: ProcessingIndicator | None = None
        if streaming_enabled:
            self._begin_streaming_run("Streaming preview output…")
        else:
            indicator = ProcessingIndicator(self, "Running prompt…")
        try:
            if streaming_enabled:
                result = self._run_in_background(
                    self._executor.execute,
                    prompt,
                    request_text,
                    conversation=None,
                    stream=True,
                    on_stream=self._stream_relay.chunk.emit,
                )
                stream_success = True
            else:
                assert indicator is not None
                result = indicator.run(
                    self._executor.execute,
                    prompt,
                    request_text,
                    conversation=None,
                    stream=False,
                )
        except (ExecutionError, RuntimeError) as exc:
            logger.exception("Run once failed")
            self._status.showMessage(str(exc), 8000)
            record = WorkbenchExecutionRecord(
                request_text=request_text,
                response_text=str(exc),
                success=False,
                variables=dict(variables),
            )
            self._session.record_execution(record)
            self._update_history()
            return
        finally:
            if streaming_enabled:
                self._end_streaming_run(success=stream_success)
        self._render_execution(result, request_text, variables)
        if fallback_message:
            self._status.showMessage(fallback_message, 5000)

    def _render_execution(
        self,
        result: CodexExecutionResult,
        request_text: str,
        variables: Mapping[str, str],
    ) -> None:
        self._output_tabs.setCurrentWidget(self._output_view)
        self._output_view.setPlainText(result.response_text)
        record = WorkbenchExecutionRecord(
            request_text=request_text,
            response_text=result.response_text,
            duration_ms=result.duration_ms,
            success=True,
            variables=dict(variables),
        )
        record.suggested_focus = self._session.suggest_refinement_target(result.response_text)
        self._session.record_execution(record)
        self._update_history()
        self._apply_highlight(record.suggested_focus)
        self._status.showMessage("Execution complete.", 5000)

    def _apply_highlight(self, target: str | None) -> None:
        if target is None:
            self._editor.setExtraSelections([])
            return
        header = {
            "context": "### Context",
            "system": "### System Role",
            "constraints": "### Constraints",
            "output": "### Output Format",
        }.get(target)
        if not header:
            return
        document = self._editor.document()
        cursor = document.find(header)
        if cursor.isNull():
            return
        selection = QTextEdit.ExtraSelection()
        fmt = QTextCharFormat()
        fmt.setBackground(Qt.GlobalColor.yellow)
        selection_any = cast(Any, selection)
        selection_any.cursor = cursor
        selection_any.cursor.movePosition(
            QTextCursor.MoveOperation.EndOfBlock,
            QTextCursor.MoveMode.KeepAnchor,
        )
        selection_any.format = fmt
        self._editor.setExtraSelections([selection])

    def _run_brainstorm(self) -> None:
        self._invoke_helper(
            "brainstorm", "Provide three alternative phrasings that could strengthen this prompt."
        )

    def _run_peek(self) -> None:
        self._invoke_helper(
            "peek", "Summarise this prompt in two sentences and point out obvious gaps."
        )

    def _invoke_helper(self, label: str, instruction: str) -> None:
        if self._executor is None:
            self._status.showMessage("CodexExecutor unavailable.", 6000)
            return
        prompt = self._session.build_prompt()
        request_text = f"{instruction}\n---\n{self._session.template_text.strip()}"
        streaming_enabled = self._is_streaming_enabled()
        stream_success = False
        indicator: ProcessingIndicator | None = None
        if streaming_enabled:
            self._begin_streaming_run(f"Streaming {label} suggestions…")
        else:
            indicator = ProcessingIndicator(self, f"Running {label}…")
        try:
            if streaming_enabled:
                result = self._run_in_background(
                    self._executor.execute,
                    prompt,
                    request_text,
                    conversation=None,
                    stream=True,
                    on_stream=self._stream_relay.chunk.emit,
                )
                stream_success = True
            else:
                assert indicator is not None
                result = indicator.run(
                    self._executor.execute,
                    prompt,
                    request_text,
                    conversation=None,
                    stream=False,
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s action failed", label)
            self._status.showMessage(str(exc), 8000)
            return
        finally:
            if streaming_enabled:
                self._end_streaming_run(success=stream_success)
        self._output_view.setPlainText(result.response_text)
        self._status.showMessage(f"{label.title()} suggestions ready.", 6000)

    def _update_history(self) -> None:
        self._history_list.clear()
        for record in self._session.execution_history[-20:]:
            status = "✅" if record.success else "⚠️"
            summary = record.response_text.splitlines()[0] if record.response_text else "(empty)"
            item = QListWidgetItem(f"{status} {summary}")
            self._history_list.addItem(item)

    def _record_rating(self, rating: float) -> None:
        if not self._session.execution_history:
            return
        record = self._session.execution_history[-1]
        record.rating = rating
        record.feedback = self._feedback_input.text().strip() or None
        self._status.showMessage("Feedback saved for last run.", 4000)

    def _apply_feedback(self) -> None:
        if not self._session.execution_history:
            return
        record = self._session.execution_history[-1]
        baseline = record.rating if record.rating is not None else 1.0
        self._record_rating(baseline)

    def _export_prompt(self) -> None:
        dialog = WorkbenchExportDialog(self._session, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        kwargs = dialog.prompt_kwargs()
        if kwargs is None:
            return
        prompt = self._session.build_prompt(
            category=kwargs["category"],
            language=kwargs["language"],
            tags=kwargs["tags"],
            author=kwargs["author"],
        )
        prompt.name = kwargs["name"] or prompt.name
        try:
            created = self._manager.create_prompt(prompt)
        except PromptManagerError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self._persist_history(created)
        show_toast(self, f"Prompt '{created.name}' saved.")
        self._session.clear_history()
        self._update_history()

    def _persist_history(self, prompt: Prompt) -> None:
        tracker = self._history_tracker
        if tracker is None:
            return
        for record in self._session.execution_history:
            try:
                if record.success:
                    tracker.record_success(
                        prompt.id,
                        record.request_text,
                        record.response_text,
                        duration_ms=record.duration_ms,
                        metadata={"variables": dict(record.variables), "source": "workbench"},
                        rating=record.rating,
                    )
                else:
                    tracker.record_failure(
                        prompt.id,
                        record.request_text,
                        record.response_text,
                        duration_ms=record.duration_ms,
                        metadata={"variables": dict(record.variables), "source": "workbench"},
                    )
            except HistoryTrackerError as exc:
                logger.warning("Unable to persist history entry: %s", exc)
