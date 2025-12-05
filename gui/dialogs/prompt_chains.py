"""Prompt chain management surfaces for the GUI.

Updates:
  v0.5.4 - 2025-12-05 - Expand execution results with labeled inputs/outputs per step.
  v0.5.3 - 2025-12-05 - Split detail column vertically, persist run variables, and replace description editor with static text.
  v0.5.2 - 2025-12-05 - Persist chain splitters immediately so tabbed windows remember widths.
  v0.5.1 - 2025-12-05 - Add Markdown toggle for execution results with rich-text rendering.
  v0.5.0 - 2025-12-05 - Split execution results into a dedicated column with persisted splitter state.
  v0.4.1 - 2025-12-05 - Document dialog passthrough helper for lint compliance.
  v0.4.0 - 2025-12-05 - Extract reusable panel for embedding plus dialog wrapper.
  v0.3.0 - 2025-12-04 - Prompt selector for chain steps plus inline CRUD actions.
  v0.2.0 - 2025-12-04 - Add create, edit, and delete actions via prompt chain editor dialog.
  v0.1.0 - 2025-12-04 - Introduce prompt chain manager dialog with run/import actions.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QByteArray, QSettings, Qt, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core import (
    PromptChainError,
    PromptChainExecutionError,
    PromptChainRunResult,
    PromptManager,
    PromptManagerError,
)
from models.prompt_chain_model import PromptChain, PromptChainStep, chain_from_payload

from ..processing_indicator import ProcessingIndicator
from ..toast import show_toast
from .prompt_chain_editor import PromptChainEditorDialog

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from uuid import UUID

    from models.prompt_model import Prompt


class PromptChainManagerPanel(QWidget):
    """Widget that lists, imports, and runs stored prompt chains."""

    def __init__(self, manager: PromptManager, parent: QWidget | None = None) -> None:
        """Create the panel and load the initial chain list."""
        super().__init__(parent)
        self._manager = manager
        self._chains: list[PromptChain] = []
        self._selected_chain_id: str | None = None
        self._settings = QSettings("PromptManager", "PromptChainManagerPanel")
        self._result_plaintext: str = ""
        self._result_markdown: str = ""
        self._suppress_variable_signal = False
        self._current_variables_chain_id: str | None = None

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Manage prompt chains defined in the shared repository. "
            "Select a chain to review its steps, optionally import JSON definitions, "
            "and provide variables before executing the workflow.",
            self,
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._outer_splitter = QSplitter(Qt.Horizontal, self)
        self._outer_splitter.setObjectName("promptChainOuterSplitter")
        layout.addWidget(self._outer_splitter, 1)

        management_container = QFrame(self._outer_splitter)
        management_layout = QVBoxLayout(management_container)
        management_layout.setContentsMargins(0, 0, 0, 0)
        management_layout.setSpacing(0)

        self._management_splitter = QSplitter(Qt.Horizontal, management_container)
        self._management_splitter.setObjectName("promptChainSplitter")
        management_layout.addWidget(self._management_splitter, 1)

        # Left column (chain list + actions)
        list_container = QFrame(self._management_splitter)
        list_layout = QVBoxLayout(list_container)
        list_layout.setContentsMargins(8, 8, 8, 8)
        list_layout.setSpacing(8)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.addWidget(QLabel("Chains", list_container))
        header_row.addStretch(1)
        self._refresh_button = QPushButton("Refresh", list_container)
        self._refresh_button.clicked.connect(self._load_chains)  # type: ignore[arg-type]
        header_row.addWidget(self._refresh_button)
        list_layout.addLayout(header_row)

        self._chain_list = QListWidget(list_container)
        self._chain_list.currentRowChanged.connect(self._handle_selection_changed)  # type: ignore[arg-type]
        list_layout.addWidget(self._chain_list, 1)

        import_run_row = QHBoxLayout()
        import_run_row.setContentsMargins(0, 0, 0, 0)
        self._import_button = QPushButton("Import JSON…", list_container)
        self._import_button.clicked.connect(self._import_chain_from_file)  # type: ignore[arg-type]
        import_run_row.addWidget(self._import_button)
        self._new_button = QPushButton("New", list_container)
        self._new_button.clicked.connect(self._create_chain)  # type: ignore[arg-type]
        import_run_row.addWidget(self._new_button)
        self._edit_button = QPushButton("Edit", list_container)
        self._edit_button.clicked.connect(self._edit_chain)  # type: ignore[arg-type]
        import_run_row.addWidget(self._edit_button)
        self._delete_button = QPushButton("Delete", list_container)
        self._delete_button.clicked.connect(self._delete_chain)  # type: ignore[arg-type]
        import_run_row.addWidget(self._delete_button)
        import_run_row.addStretch(1)
        list_layout.addLayout(import_run_row)

        # Right column (details + run form)
        detail_container = QFrame(self._management_splitter)
        detail_layout = QVBoxLayout(detail_container)
        detail_layout.setContentsMargins(8, 8, 8, 8)
        detail_layout.setSpacing(10)

        self._detail_splitter = QSplitter(Qt.Vertical, detail_container)
        self._detail_splitter.setObjectName("promptChainDetailSplitter")
        detail_layout.addWidget(self._detail_splitter, 1)

        info_container = QFrame(self._detail_splitter)
        info_layout = QVBoxLayout(info_container)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(6)

        self._detail_title = QLabel("Select a prompt chain to view details.", info_container)
        self._detail_title.setObjectName("promptChainDetailTitle")
        self._detail_title.setStyleSheet("font-size:16px;font-weight:600;")
        info_layout.addWidget(self._detail_title)

        self._detail_status = QLabel("", info_container)
        info_layout.addWidget(self._detail_status)

        self._description_label = QLabel("(No description provided.)", info_container)
        self._description_label.setWordWrap(True)
        self._description_label.setObjectName("promptChainDescription")
        self._description_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        info_layout.addWidget(self._description_label)

        self._schema_view = QPlainTextEdit(info_container)
        self._schema_view.setReadOnly(True)
        self._schema_view.setPlaceholderText("Variables schema (JSON schema, optional)")
        info_layout.addWidget(self._schema_view, 1)

        steps_container = QFrame(self._detail_splitter)
        steps_layout = QVBoxLayout(steps_container)
        steps_layout.setContentsMargins(0, 0, 0, 0)
        steps_layout.setSpacing(6)

        steps_label = QLabel("Steps", steps_container)
        steps_layout.addWidget(steps_label)

        self._steps_table = QTableWidget(0, 5, steps_container)
        self._steps_table.setHorizontalHeaderLabels(
            ["Order", "Prompt", "Input", "Output variable", "Condition"]
        )
        self._steps_table.horizontalHeader().setStretchLastSection(True)
        self._steps_table.verticalHeader().setVisible(False)
        self._steps_table.setEditTriggers(QTableWidget.NoEditTriggers)
        steps_layout.addWidget(self._steps_table, 1)

        run_container = QFrame(self._detail_splitter)
        run_layout = QVBoxLayout(run_container)
        run_layout.setContentsMargins(0, 0, 0, 0)
        run_layout.setSpacing(6)

        run_header = QLabel("Run Chain", run_container)
        run_header.setStyleSheet("font-weight:600;")
        run_layout.addWidget(run_header)

        self._variables_input = QPlainTextEdit(run_container)
        self._variables_input.setPlaceholderText("Enter JSON object for chain variables")
        self._variables_input.textChanged.connect(self._handle_variables_changed)  # type: ignore[arg-type]
        run_layout.addWidget(self._variables_input, 1)

        actions_row = QHBoxLayout()
        self._run_button = QPushButton("Run Chain", run_container)
        self._run_button.clicked.connect(self._run_selected_chain)  # type: ignore[arg-type]
        actions_row.addWidget(self._run_button)
        self._clear_vars_button = QPushButton("Clear Variables", run_container)
        self._clear_vars_button.clicked.connect(self._variables_input.clear)  # type: ignore[arg-type]
        actions_row.addWidget(self._clear_vars_button)
        actions_row.addStretch(1)
        run_layout.addLayout(actions_row)

        self._detail_splitter.addWidget(info_container)
        self._detail_splitter.addWidget(steps_container)
        self._detail_splitter.addWidget(run_container)
        self._detail_splitter.setStretchFactor(0, 1)
        self._detail_splitter.setStretchFactor(1, 2)
        self._detail_splitter.setStretchFactor(2, 1)

        results_container = QFrame(self._outer_splitter)
        results_layout = QVBoxLayout(results_container)
        results_layout.setContentsMargins(8, 8, 8, 8)
        results_layout.setSpacing(8)

        results_header = QHBoxLayout()
        results_header.setContentsMargins(0, 0, 0, 0)
        results_title = QLabel("Execution results", results_container)
        results_title.setStyleSheet("font-size:15px;font-weight:600;")
        results_header.addWidget(results_title)
        self._clear_results_button = QPushButton("Clear", results_container)
        self._clear_results_button.clicked.connect(self._handle_clear_results)  # type: ignore[arg-type]
        results_header.addWidget(self._clear_results_button)
        results_header.addStretch(1)
        results_layout.addLayout(results_header)

        self._result_format_checkbox = QCheckBox("Markdown", results_container)
        self._result_format_checkbox.setChecked(True)
        self._result_format_checkbox.toggled.connect(self._handle_result_format_changed)  # type: ignore[arg-type]
        results_header.addWidget(self._result_format_checkbox)

        self._result_view = QTextEdit(results_container)
        self._result_view.setReadOnly(True)
        self._result_view.setPlaceholderText("Execution results, outputs, and per-step summary.")
        self._result_view.setAcceptRichText(True)
        results_layout.addWidget(self._result_view, 1)
        self._stream_preview_active = False
        self._stream_buffers: dict[str, str] = {}
        self._stream_labels: dict[str, str] = {}
        self._stream_order: list[str] = []
        self._stream_chain_title = ""

        self._management_splitter.addWidget(list_container)
        self._management_splitter.addWidget(detail_container)
        self._management_splitter.setStretchFactor(0, 1)
        self._management_splitter.setStretchFactor(1, 2)

        self._outer_splitter.addWidget(management_container)
        self._outer_splitter.addWidget(results_container)
        self._outer_splitter.setStretchFactor(0, 3)
        self._outer_splitter.setStretchFactor(1, 2)
        self._outer_splitter.splitterMoved.connect(self._handle_splitter_moved)  # type: ignore[arg-type]
        self._management_splitter.splitterMoved.connect(self._handle_splitter_moved)  # type: ignore[arg-type]
        self._detail_splitter.splitterMoved.connect(self._handle_splitter_moved)  # type: ignore[arg-type]

        self._restore_splitter_state()
        self._load_chains()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Reload the prompt chain list from the repository."""
        self._load_chains()

    # ------------------------------------------------------------------
    # Chain loading and selection
    # ------------------------------------------------------------------
    def _load_chains(self) -> None:
        """Refresh the chain list from the repository."""
        previous_id = self._selected_chain_id
        try:
            chains = self._manager.list_prompt_chains(include_inactive=True)
        except PromptChainError as exc:
            QMessageBox.critical(self, "Unable to load chains", str(exc))
            return
        self._chains = chains
        self._populate_chain_list(previous_id)
        show_toast(self, f"Loaded {len(chains)} prompt chains.")

    def _populate_chain_list(self, preferred_id: str | None = None) -> None:
        """Populate the list widget with available chains."""
        self._chain_list.blockSignals(True)
        selected_chain: PromptChain | None = None
        try:
            self._chain_list.clear()
            for chain in self._chains:
                status = "Active" if chain.is_active else "Inactive"
                item = QListWidgetItem(f"{chain.name} ({status})", self._chain_list)
                item.setData(Qt.UserRole, str(chain.id))
            if self._chains:
                target_id = preferred_id or str(self._chains[0].id)
                for row in range(self._chain_list.count()):
                    item = self._chain_list.item(row)
                    if item.data(Qt.UserRole) == target_id:
                        self._chain_list.setCurrentRow(row)
                        selected_chain = self._chains[row]
                        self._selected_chain_id = str(selected_chain.id)
                        break
                else:
                    self._chain_list.setCurrentRow(0)
                    selected_chain = self._chains[0]
                    self._selected_chain_id = str(selected_chain.id)
            else:
                self._selected_chain_id = None
        finally:
            self._chain_list.blockSignals(False)
        self._render_chain_details(selected_chain)

    def _current_chain(self) -> PromptChain | None:
        if not self._selected_chain_id:
            return None
        return next(
            (chain for chain in self._chains if str(chain.id) == self._selected_chain_id),
            None,
        )

    def _handle_selection_changed(self, row: int) -> None:
        """Render details for the currently selected chain."""
        if row < 0:
            self._selected_chain_id = None
            self._render_chain_details(None)
            return
        item = self._chain_list.item(row)
        if item is None:
            self._selected_chain_id = None
            self._render_chain_details(None)
            return
        chain_id = item.data(Qt.UserRole)
        self._selected_chain_id = chain_id
        chain = next((entry for entry in self._chains if str(entry.id) == chain_id), None)
        self._render_chain_details(chain)

    def _render_chain_details(self, chain: PromptChain | None) -> None:
        """Populate detail widgets with information about *chain*."""
        if chain is None:
            self._detail_title.setText("Select a prompt chain to view details.")
            self._detail_status.setText("")
            self._description_label.setText("(No description provided.)")
            self._schema_view.clear()
            self._steps_table.setRowCount(0)
            self._result_view.clear()
            self._run_button.setEnabled(False)
            self._current_variables_chain_id = None
            self._set_variables_text("")
            return

        self._detail_title.setText(chain.name)
        status = "Active" if chain.is_active else "Inactive"
        updated_at_text = chain.updated_at.strftime("%Y-%m-%d %H:%M")  # noqa: DTZ005
        self._detail_status.setText(
            f"Status: {status} • Steps: {len(chain.steps)} • Updated {updated_at_text}"
        )
        self._description_label.setText(chain.description or "(No description provided.)")
        schema_text = (
            json.dumps(chain.variables_schema, indent=2, sort_keys=True)
            if chain.variables_schema
            else "(No variables schema defined.)"
        )
        self._schema_view.setPlainText(schema_text)
        self._populate_steps_table(chain)
        self._result_view.clear()
        self._run_button.setEnabled(chain.is_active and bool(chain.steps))
        self._current_variables_chain_id = str(chain.id)
        self._set_variables_text(self._load_variables_text(self._current_variables_chain_id))

    def _populate_steps_table(self, chain: PromptChain) -> None:
        """Fill the steps table using ``chain.steps``."""
        self._steps_table.setRowCount(len(chain.steps))
        for row, step in enumerate(chain.steps):
            self._steps_table.setItem(row, 0, QTableWidgetItem(str(step.order_index)))
            self._steps_table.setItem(row, 1, QTableWidgetItem(str(step.prompt_id)))
            self._steps_table.setItem(row, 2, QTableWidgetItem(step.input_template or ""))
            self._steps_table.setItem(row, 3, QTableWidgetItem(step.output_variable))
            condition = step.condition or "Always"
            if not step.stop_on_failure:
                condition = f"{condition} (continues on failure)"
            self._steps_table.setItem(row, 4, QTableWidgetItem(condition))
        self._steps_table.resizeColumnsToContents()

    # ------------------------------------------------------------------
    # Chain import + execution
    # ------------------------------------------------------------------
    def _import_chain_from_file(self) -> None:
        """Import a chain definition from a JSON file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Prompt Chain",
            str(Path.home()),
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            raw_text = Path(path).expanduser().read_text(encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Unable to read file", str(exc))
            return
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            QMessageBox.critical(self, "Invalid JSON", str(exc))
            return
        if not isinstance(payload, Mapping):
            QMessageBox.warning(self, "Invalid payload", "Chain definition must be a JSON object.")
            return
        try:
            chain = chain_from_payload(payload)
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid chain", str(exc))
            return
        try:
            saved = self._manager.save_prompt_chain(chain)
        except PromptChainError as exc:
            QMessageBox.critical(self, "Unable to save chain", str(exc))
            return
        self._load_chains()
        self._select_chain(saved.id)
        show_toast(self, f"Prompt chain '{saved.name}' saved.")

    def _create_chain(self) -> None:
        editor = PromptChainEditorDialog(
            self,
            manager=self._manager,
            prompts=self._available_prompts(),
        )
        if editor.exec() != QDialog.Accepted:
            return
        chain = editor.result_chain()
        if chain is None:
            return
        self._persist_chain(chain)

    def _edit_chain(self) -> None:
        chain = self._current_chain()
        if chain is None:
            QMessageBox.information(self, "Select chain", "Choose a chain to edit first.")
            return
        editor = PromptChainEditorDialog(
            self,
            manager=self._manager,
            prompts=self._available_prompts(),
            chain=chain,
        )
        if editor.exec() != QDialog.Accepted:
            return
        updated = editor.result_chain()
        if updated is None:
            return
        self._persist_chain(updated)

    def _persist_chain(self, chain: PromptChain) -> None:
        try:
            saved = self._manager.save_prompt_chain(chain)
        except PromptChainError as exc:
            QMessageBox.critical(self, "Unable to save chain", str(exc))
            return
        self._load_chains()
        self._select_chain(saved.id)
        show_toast(self, f"Prompt chain '{saved.name}' saved.")

    def _delete_chain(self) -> None:
        chain = self._current_chain()
        if chain is None:
            return
        confirmation = QMessageBox.question(
            self,
            "Delete prompt chain",
            f"Delete '{chain.name}' and all of its steps?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirmation != QMessageBox.Yes:
            return
        try:
            self._manager.delete_prompt_chain(chain.id)
        except PromptChainError as exc:
            QMessageBox.critical(self, "Unable to delete chain", str(exc))
            return
        self._load_chains()
        show_toast(self, f"Prompt chain '{chain.name}' deleted.")

    def _available_prompts(self) -> list[Prompt]:
        if self._manager is None:
            return []
        try:
            list_prompts = getattr(self._manager, "list_prompts", None)
            if callable(list_prompts):
                return list_prompts(limit=500)
            repository = getattr(self._manager, "repository", None)
            if repository is not None:
                return repository.list(limit=500)
        except PromptManagerError as exc:
            logger.warning("Unable to load prompts for chain editor", exc_info=exc)
        except Exception:  # pragma: no cover - defensive guard
            logger.warning("Unexpected prompt lookup failure", exc_info=True)
        return []

    def _select_chain(self, chain_id: UUID) -> None:
        """Select the list entry matching ``chain_id``."""
        chain_id_text = str(chain_id)
        for row in range(self._chain_list.count()):
            item = self._chain_list.item(row)
            if item.data(Qt.UserRole) == chain_id_text:
                self._chain_list.setCurrentRow(row)
                return

    def _run_selected_chain(self) -> None:
        """Execute the currently selected chain with provided variables."""
        chain = next(
            (entry for entry in self._chains if str(entry.id) == self._selected_chain_id),
            None,
        )
        if chain is None:
            QMessageBox.information(self, "Select chain", "Choose a chain to run first.")
            return
        variables_text = self._variables_input.toPlainText().strip()
        if variables_text:
            try:
                variables_obj = json.loads(variables_text)
            except json.JSONDecodeError as exc:
                QMessageBox.critical(self, "Invalid variables", f"JSON decode error: {exc}")
                return
            if not isinstance(variables_obj, Mapping):
                QMessageBox.warning(self, "Invalid format", "Variables must be a JSON object.")
                return
            variables: Mapping[str, Any] | None = variables_obj
        else:
            variables = {}

        indicator = ProcessingIndicator(self, f"Running '{chain.name}'…", title="Prompt Chain")
        previous_status = self._detail_status.text()
        self._detail_status.setText(f"Running '{chain.name}'…")
        streaming_enabled = self._is_streaming_enabled()
        if streaming_enabled:
            self._begin_stream_preview(chain.name)
        stream_callback = self._handle_step_stream if streaming_enabled else None
        try:
            result = indicator.run(
                self._manager.run_prompt_chain,
                chain.id,
                variables=variables,
                stream_callback=stream_callback,
            )
        except PromptChainExecutionError as exc:
            QMessageBox.critical(self, "Chain failed", str(exc))
            self._detail_status.setText(previous_status)
            if streaming_enabled:
                self._end_stream_preview()
            return
        except PromptChainError as exc:
            QMessageBox.critical(self, "Unable to run chain", str(exc))
            self._detail_status.setText(previous_status)
            if streaming_enabled:
                self._end_stream_preview()
            return
        else:
            timestamp = datetime.now().strftime("%H:%M:%S")  # noqa: DTZ005
            self._detail_status.setText(f"Last run succeeded at {timestamp}")
        if streaming_enabled:
            self._end_stream_preview()
        self._display_run_result(result)
        show_toast(self, f"Chain '{chain.name}' completed.")

    def _display_run_result(self, result: PromptChainRunResult) -> None:
        """Render execution outputs and per-step summary for *result*."""
        def _dump_json(payload: Mapping[str, Any]) -> str:
            try:
                return json.dumps(payload, indent=2, sort_keys=True, default=str)
            except TypeError:
                return json.dumps(
                    {str(key): str(value) for key, value in payload.items()},
                    indent=2,
                    sort_keys=True,
                )

        def _indent_lines(text: str, prefix: str = "  ") -> list[str]:
            if not text.strip():
                return [f"{prefix}(empty)"]
            return [f"{prefix}{line}" for line in text.splitlines()]

        plain_sections: list[str] = []
        markdown_lines: list[str] = []

        # Chain inputs
        plain_sections.append("Input to chain")
        if result.variables:
            chain_input = _dump_json(result.variables)
            plain_sections.extend(_indent_lines(chain_input))
            markdown_lines.extend(["## Input to chain", "```json", chain_input, "```"])
        else:
            plain_sections.append("  (no chain variables provided)")
            markdown_lines.extend(["## Input to chain", "- (no chain variables provided)"])
        plain_sections.append("")

        # Chain outputs
        plain_sections.append("Chain Outputs")
        if result.outputs:
            outputs_text = _dump_json(result.outputs)
            plain_sections.extend(_indent_lines(outputs_text))
            markdown_lines.extend(["", "## Chain outputs", "```json", outputs_text, "```"])
        else:
            plain_sections.append("  (no outputs)")
            markdown_lines.extend(["", "## Chain outputs", "- (no outputs)"])
        plain_sections.append("")

        # Steps
        if result.steps:
            markdown_lines.append("")
            markdown_lines.append("## Step details")
        for step_run in result.steps:
            step = step_run.step
            step_label = step.output_variable or str(step.prompt_id)
            plain_sections.append(f"Step {step.order_index}: {step_label}")
            plain_sections.append(f"  Status: {step_run.status}")
            markdown_lines.append(f"### Step {step.order_index} – {step_label}")
            markdown_lines.append(f"- **Status:** `{step_run.status}`")
            if step_run.error:
                plain_sections.append(f"  Error: {step_run.error}")
                markdown_lines.append(f"- **Error:** {step_run.error}")
            elif step_run.status == "skipped":
                plain_sections.append("  Skipped because condition evaluated to false.")
                markdown_lines.append("- _Skipped because condition evaluated to false._")
            outcome = step_run.outcome
            if outcome:
                request_text = outcome.result.request_text.strip()
                response_text = outcome.result.response_text.strip()
                plain_sections.append("  Input to step:")
                plain_sections.extend(_indent_lines(request_text or "(empty input)"))
                plain_sections.append("  Output of step:")
                plain_sections.extend(_indent_lines(response_text or "(empty response)"))
                markdown_lines.append("**Input to step:**")
                markdown_lines.extend(["```", request_text or "(empty input)", "```"])
                markdown_lines.append("**Output of step:**")
                markdown_lines.extend(["```", response_text or "(empty response)", "```"])
            plain_sections.append("")
            markdown_lines.append("")

        plain_text = "\n".join(plain_sections).strip()
        markdown_text = "\n".join(markdown_lines).strip()
        self._result_plaintext = plain_text
        self._result_markdown = markdown_text
        self._apply_result_view()

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        """Persist splitter state before closing."""
        self._persist_splitter_state()
        super().closeEvent(event)

    def _restore_splitter_state(self) -> None:
        """Restore splitter positions from persisted settings."""
        outer_state = self._settings.value("outerSplitterState")
        if isinstance(outer_state, QByteArray):
            self._outer_splitter.restoreState(outer_state)
        inner_state = self._settings.value("managementSplitterState")
        if isinstance(inner_state, QByteArray):
            self._management_splitter.restoreState(inner_state)
        detail_state = self._settings.value("detailSplitterState")
        if isinstance(detail_state, QByteArray):
            self._detail_splitter.restoreState(detail_state)

    def _persist_splitter_state(self) -> None:
        """Save splitter positions for the next session."""
        self._settings.setValue("outerSplitterState", self._outer_splitter.saveState())
        self._settings.setValue("managementSplitterState", self._management_splitter.saveState())
        self._settings.setValue("detailSplitterState", self._detail_splitter.saveState())

    def _handle_clear_results(self) -> None:
        """Reset the execution results pane."""
        self._end_stream_preview()
        self._result_plaintext = ""
        self._result_markdown = ""
        self._apply_result_view()

    def _handle_result_format_changed(self, _: bool) -> None:
        """Switch between Markdown and plain text views."""
        self._apply_result_view()

    def _apply_result_view(self) -> None:
        """Render stored results using the current format preference."""
        if self._result_format_checkbox.isChecked():
            if self._result_markdown:
                self._result_view.setMarkdown(self._result_markdown)
            else:
                self._result_view.clear()
        else:
            if self._result_plaintext:
                self._result_view.setPlainText(self._result_plaintext)
            else:
                self._result_view.clear()

    def _is_streaming_enabled(self) -> bool:
        """Return True when LiteLLM streaming is enabled in runtime settings."""
        executor = getattr(self._manager, "_executor", None)
        if executor is not None and bool(getattr(executor, "stream", False)):
            return True
        return bool(getattr(self._manager, "_litellm_stream", False))

    def _begin_stream_preview(self, chain_name: str) -> None:
        """Initialise buffers used to show streaming output per step."""
        self._stream_preview_active = True
        self._stream_chain_title = chain_name
        self._stream_buffers.clear()
        self._stream_labels.clear()
        self._stream_order.clear()
        self._result_view.setPlainText(f"Streaming '{chain_name}'…")

    def _handle_step_stream(self, step: PromptChainStep, chunk: str) -> None:
        """Schedule GUI updates for streamed chain step output."""
        if not chunk:
            return
        QTimer.singleShot(0, lambda: self._register_stream_chunk(step, chunk))

    def _register_stream_chunk(self, step: PromptChainStep, chunk: str) -> None:
        """Append *chunk* to the preview buffer for *step*."""
        if not self._stream_preview_active:
            return
        step_id = str(step.id)
        if step_id not in self._stream_buffers:
            label = f"Step {step.order_index} – {step.output_variable or step.prompt_id}"
            self._stream_order.append(step_id)
            self._stream_labels[step_id] = label
            self._stream_buffers[step_id] = ""
        self._stream_buffers[step_id] += chunk
        self._refresh_stream_preview()

    def _refresh_stream_preview(self) -> None:
        """Render the live streaming preview text."""
        if not self._stream_preview_active:
            return
        header = self._stream_chain_title or "prompt chain"
        lines = [f"Streaming '{header}'…", ""]
        for step_id in self._stream_order:
            label = self._stream_labels.get(step_id, step_id)
            lines.append(label)
            text = self._stream_buffers.get(step_id, "")
            if text.strip():
                lines.append(text)
            else:
                lines.append("(awaiting tokens…)")
            lines.append("")
        self._result_view.setPlainText("\n".join(lines).rstrip())

    def _end_stream_preview(self) -> None:
        """Clear streaming preview state."""
        self._stream_preview_active = False
        self._stream_chain_title = ""
        self._stream_buffers.clear()
        self._stream_labels.clear()
        self._stream_order.clear()

    def _handle_splitter_moved(self, _: int, __: int) -> None:
        """Persist splitter state whenever the user resizes panes."""
        self._persist_splitter_state()

    def _handle_variables_changed(self) -> None:
        """Persist chain variable text whenever it changes."""
        if self._suppress_variable_signal:
            return
        chain_id = self._current_variables_chain_id
        if not chain_id:
            return
        self._settings.setValue(
            self._variables_settings_key(chain_id),
            self._variables_input.toPlainText(),
        )

    def _set_variables_text(self, text: str) -> None:
        """Apply text to the variables editor without triggering persistence."""
        self._suppress_variable_signal = True
        self._variables_input.setPlainText(text)
        self._suppress_variable_signal = False

    def _load_variables_text(self, chain_id: str | None) -> str:
        """Load saved variable JSON for the given chain."""
        if not chain_id:
            return ""
        stored = self._settings.value(self._variables_settings_key(chain_id))
        if isinstance(stored, str):
            return stored
        if stored is None:
            return ""
        return str(stored)

    def _variables_settings_key(self, chain_id: str) -> str:
        """Return the QSettings key for persisted variable payloads."""
        return f"chainVariables/{chain_id}"


class PromptChainManagerDialog(QDialog):
    """Dialog wrapper that hosts :class:`PromptChainManagerPanel`."""

    def __init__(self, manager: PromptManager, parent: QWidget | None = None) -> None:
        """Create the dialog and embed the shared panel widget."""
        super().__init__(parent)
        self.setWindowTitle("Prompt Chains")
        self.resize(960, 640)
        layout = QVBoxLayout(self)
        self._panel = PromptChainManagerPanel(manager, parent=self)
        layout.addWidget(self._panel)
        buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
        buttons.accepted.connect(self.accept)  # type: ignore[arg-type]
        buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        layout.addWidget(buttons)

    def refresh(self) -> None:
        """Reload prompt chains via the embedded panel."""
        self._panel.refresh()

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - passthrough helper
        """Proxy attribute lookups to the embedded panel when possible."""
        panel = self.__dict__.get("_panel")
        if panel is not None and hasattr(panel, name):
            return getattr(panel, name)
        raise AttributeError(name)


__all__ = ["PromptChainManagerPanel", "PromptChainManagerDialog"]
