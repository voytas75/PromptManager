"""Prompt chain management dialog for the GUI.

Updates:
  v0.2.0 - 2025-12-04 - Add create, edit, and delete actions via prompt chain editor dialog.
  v0.1.0 - 2025-12-04 - Introduce prompt chain manager dialog with run/import actions.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
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
    QVBoxLayout,
    QWidget,
)

from core import PromptChainError, PromptChainExecutionError, PromptChainRunResult, PromptManager
from models.prompt_chain_model import PromptChain, chain_from_payload

from ..processing_indicator import ProcessingIndicator
from ..toast import show_toast
from .prompt_chain_editor import PromptChainEditorDialog

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from uuid import UUID


class PromptChainManagerDialog(QDialog):
    """Dialog that lists, imports, and runs stored prompt chains."""

    def __init__(self, manager: PromptManager, parent: QWidget | None = None) -> None:
        """Create the dialog and load the initial chain list."""
        super().__init__(parent)
        self._manager = manager
        self._chains: list[PromptChain] = []
        self._selected_chain_id: str | None = None
        self.setWindowTitle("Prompt Chains")
        self.resize(960, 640)

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Manage prompt chains defined in the shared repository. "
            "Select a chain to review its steps, optionally import JSON definitions, "
            "and provide variables before executing the workflow.",
            self,
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setObjectName("promptChainSplitter")
        layout.addWidget(splitter, 1)

        # Left column (chain list + actions)
        list_container = QFrame(splitter)
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

        splitter.addWidget(list_container)

        # Right column (details + run form)
        detail_container = QFrame(splitter)
        detail_layout = QVBoxLayout(detail_container)
        detail_layout.setContentsMargins(8, 8, 8, 8)
        detail_layout.setSpacing(10)

        self._detail_title = QLabel("Select a prompt chain to view details.", detail_container)
        self._detail_title.setObjectName("promptChainDetailTitle")
        self._detail_title.setStyleSheet("font-size:16px;font-weight:600;")
        detail_layout.addWidget(self._detail_title)

        self._detail_status = QLabel("", detail_container)
        detail_layout.addWidget(self._detail_status)

        self._description_view = QPlainTextEdit(detail_container)
        self._description_view.setReadOnly(True)
        self._description_view.setPlaceholderText("Chain description")
        detail_layout.addWidget(self._description_view)

        self._schema_view = QPlainTextEdit(detail_container)
        self._schema_view.setReadOnly(True)
        self._schema_view.setPlaceholderText("Variables schema (JSON schema, optional)")
        detail_layout.addWidget(self._schema_view)

        steps_label = QLabel("Steps", detail_container)
        detail_layout.addWidget(steps_label)

        self._steps_table = QTableWidget(0, 5, detail_container)
        self._steps_table.setHorizontalHeaderLabels(
            ["Order", "Prompt", "Input", "Output variable", "Condition"]
        )
        self._steps_table.horizontalHeader().setStretchLastSection(True)
        self._steps_table.verticalHeader().setVisible(False)
        self._steps_table.setEditTriggers(QTableWidget.NoEditTriggers)
        detail_layout.addWidget(self._steps_table, 1)

        run_header = QLabel("Run Chain", detail_container)
        run_header.setStyleSheet("font-weight:600;")
        detail_layout.addWidget(run_header)

        self._variables_input = QPlainTextEdit(detail_container)
        self._variables_input.setPlaceholderText("Enter JSON object for chain variables")
        detail_layout.addWidget(self._variables_input)

        actions_row = QHBoxLayout()
        self._run_button = QPushButton("Run Chain", detail_container)
        self._run_button.clicked.connect(self._run_selected_chain)  # type: ignore[arg-type]
        actions_row.addWidget(self._run_button)
        self._clear_vars_button = QPushButton("Clear Variables", detail_container)
        self._clear_vars_button.clicked.connect(self._variables_input.clear)  # type: ignore[arg-type]
        actions_row.addWidget(self._clear_vars_button)
        actions_row.addStretch(1)
        detail_layout.addLayout(actions_row)

        self._result_view = QPlainTextEdit(detail_container)
        self._result_view.setReadOnly(True)
        self._result_view.setPlaceholderText("Execution results, outputs, and per-step summary.")
        detail_layout.addWidget(self._result_view)

        self._buttons = QDialogButtonBox(QDialogButtonBox.Close, detail_container)
        self._buttons.accepted.connect(self.accept)  # type: ignore[arg-type]
        self._buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        detail_layout.addWidget(self._buttons)

        splitter.addWidget(detail_container)
        splitter.setStretchFactor(1, 2)

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
            self._description_view.clear()
            self._schema_view.clear()
            self._steps_table.setRowCount(0)
            self._result_view.clear()
            self._run_button.setEnabled(False)
            return

        self._detail_title.setText(chain.name)
        status = "Active" if chain.is_active else "Inactive"
        updated_at_text = chain.updated_at.strftime("%Y-%m-%d %H:%M")  # noqa: DTZ005
        self._detail_status.setText(
            f"Status: {status} • Steps: {len(chain.steps)} • Updated {updated_at_text}"
        )
        self._description_view.setPlainText(chain.description or "(No description provided.)")
        schema_text = (
            json.dumps(chain.variables_schema, indent=2, sort_keys=True)
            if chain.variables_schema
            else "(No variables schema defined.)"
        )
        self._schema_view.setPlainText(schema_text)
        self._populate_steps_table(chain)
        self._result_view.clear()
        self._run_button.setEnabled(chain.is_active and bool(chain.steps))

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
        editor = PromptChainEditorDialog(self, manager=self._manager)
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
        editor = PromptChainEditorDialog(self, manager=self._manager, chain=chain)
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
        try:
            result = indicator.run(
                self._manager.run_prompt_chain,
                chain.id,
                variables=variables,
            )
        except PromptChainExecutionError as exc:
            QMessageBox.critical(self, "Chain failed", str(exc))
            return
        except PromptChainError as exc:
            QMessageBox.critical(self, "Unable to run chain", str(exc))
            return
        self._display_run_result(result)
        show_toast(self, f"Chain '{chain.name}' completed.")

    def _display_run_result(self, result: PromptChainRunResult) -> None:
        """Render execution outputs and per-step summary for *result*."""
        outputs_section = ["Outputs:"]
        if not result.outputs:
            outputs_section.append("  (no outputs)")
        else:
            for key, value in result.outputs.items():
                outputs_section.append(f"  {key}: {value}")

        steps_section = ["\nSteps:"]
        for step_run in result.steps:
            prefix = {
                "success": "[OK]",
                "skipped": "[SKIP]",
            }.get(step_run.status, "[ERR]")
            summary = f"{prefix} {step_run.step.order_index}. {step_run.step.output_variable}"
            has_success = step_run.status == "success"
            has_response = bool(step_run.outcome and step_run.outcome.result.response_text)
            if has_success and has_response:
                preview = step_run.outcome.result.response_text.strip()
                if len(preview) > 120:
                    preview = preview[:117].rstrip() + "…"
                summary += f" -> {preview or '(empty response)'}"
            elif step_run.status == "skipped":
                summary += " (condition false)"
            elif step_run.error:
                summary += f": {step_run.error}"
            steps_section.append(summary)

        self._result_view.setPlainText("\n".join(outputs_section + steps_section))


__all__ = ["PromptChainManagerDialog"]
