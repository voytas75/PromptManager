"""Prompt chain management surfaces for the GUI.

Updates:
  v0.6.0 - 2025-12-06 - Redesign chain manager for plain-text inputs and linear chaining.
  v0.5.18 - 2025-12-06 - Improve default contrast and satisfy lint/type checks for toggles.
  v0.5.17 - 2025-12-06 - Ignore reasoning payload tokens that look like opaque IDs.
  v0.5.16 - 2025-12-06 - Limit reasoning summaries to opted-in chains and final steps; fix colors.
  v0.5.15 - 2025-12-05 - Double-clicking chains opens the editor; steps still show prompt names.
  v0.5.14 - 2025-12-05 - Show prompt names in step tables and enable editing via activation.
  v0.5.13 - 2025-12-05 - Add default-on web search toggle for chain executions.
  v0.5.12 - 2025-12-05 - Color-code results sections and surface reasoning summaries.
  v0.5.11 - 2025-12-05 - Ensure wrapped results by removing code fences from chain IO sections.
  v0.5.10 - 2025-12-05 - Add line wrap toggle for execution results pane (on by default).
  v0.5.9 - 2025-12-05 - Display chain summary preference and render condensed outputs.
  v0.5.8 - 2025-12-05 - Render step inputs/outputs without code fences for Markdown previews.
  v0.5.7 - 2025-12-05 - Preserve chain results when toggling Markdown formatting.
  v0.5.6 - 2025-12-05 - Add auto-scroll toggle for streaming/results pane.
  v0.5.5 - 2025-12-05 - Enable LiteLLM streaming preview without blocking busy indicator.
  v0.5.4 - 2025-12-05 - Expand execution results with labeled inputs/outputs per step.
  v0.5.3 - 2025-12-05 - Split detail column vertically, persist variables, swap description editor.
  v0.5.2 - 2025-12-05 - Persist chain splitters immediately so tabbed windows remember widths.
  v0.5.1 - 2025-12-05 - Add Markdown toggle for execution results with rich-text rendering.
  v0.5.0 - 2025-12-05 - Split execution results into dedicated column with persisted splitter state.
  v0.4.1-v0.1.0 - 2025-12-04 - Earlier releases introduced the dialog and editor flows,
    including CRUD/import helpers.
"""

from __future__ import annotations

import html
import json
import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from PySide6.QtCore import QByteArray, QSettings, Qt, QTimer
from PySide6.QtGui import QPalette, QTextDocument  # type: ignore[attr-defined]
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
    PromptChainStepRun,
    PromptManager,
    PromptManagerError,
)
from models.prompt_chain_model import PromptChain, PromptChainStep, chain_from_payload

from ..processing_indicator import ProcessingIndicator
from ..toast import show_toast
from .prompt_chain_editor import PromptChainEditorDialog

logger = logging.getLogger(__name__)

_CHAIN_INPUT_COLOR = "#66bb6a"
_CHAIN_OUTPUT_COLOR = "#66bb6a"
_CHAIN_SUMMARY_COLOR = "#2e7d32"
_STEP_MUTED_COLOR = "#737373"
_REASONING_COLOR = "#1e88e5"

_RESULT_STYLE_TEMPLATE = """
<style>
.chain-results {
  font-family: "Inter", "Segoe UI", sans-serif;
  font-size: 12px;
  line-height: 1.45;
  color: {text_color};
}
.chain-block {
  border-radius: 6px;
  padding: 8px 10px;
  margin-bottom: 10px;
  border: 1px solid #dfe5dc;
}
.chain-block-title {
  font-weight: 600;
  margin-bottom: 4px;
}
.chain-block-body {
  white-space: pre-wrap;
}
.chain-block-body--mono {
  font-family: "JetBrains Mono", "SFMono-Regular", monospace;
}
.chain-block-body--muted {
  color: #A8A8A8;
}
.chain-block pre {
  margin: 0;
  white-space: pre-wrap;
  font-family: "JetBrains Mono", "SFMono-Regular", monospace;
  font-size: 12px;
  border: none;
  background: transparent;
  padding: 0;
}
.chain-steps {
  margin-top: 12px;
}
.chain-step {
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 10px;
  margin-bottom: 12px;
  background-color: transparent;
}
.chain-step-title {
  font-weight: 600;
  margin-bottom: 4px;
}
.chain-step-status {
  font-size: 12px;
  margin-bottom: 6px;
}
.chain-step-error {
  color: #c62828;
  font-weight: 600;
  margin-bottom: 6px;
}
.chain-step-detail {
  margin: 6px 0;
}
.chain-step-detail pre {
  margin-top: 4px;
  padding: 6px;
  border-radius: 4px;
  border: 1px solid #ececec;
  background-color: transparent;
  color: inherit;
}
</style>
"""

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from uuid import UUID

    from PySide6.QtGui import QCloseEvent

    from core.prompt_manager.execution_history import ExecutionOutcome
    from models.prompt_model import Prompt


class PromptChainManagerPanel(QWidget):
    """Widget that lists, imports, and runs stored prompt chains."""

    def __init__(
        self,
        manager: PromptManager,
        parent: QWidget | None = None,
        *,
        prompt_edit_callback: Callable[[UUID], None] | None = None,
    ) -> None:
        """Create the panel and load the initial chain list."""
        super().__init__(parent)
        self._manager = manager
        self._chains: list[PromptChain] = []
        self._selected_chain_id: str | None = None
        self._settings = QSettings("PromptManager", "PromptChainManagerPanel")
        self._result_plaintext: str = ""
        self._result_richtext: str = ""
        self._suppress_input_signal = False
        self._current_input_chain_id: str | None = None
        self._web_search_checkbox: QCheckBox | None = None
        self._prompt_edit_callback = prompt_edit_callback
        self._prompt_name_cache: dict[UUID, str] = {}
        self._step_prompt_ids: dict[int, UUID] = {}

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Manage prompt chains defined in the shared repository. "
            "Select a chain to review its steps, optionally import definitions, "
            "and type the plain-text input that feeds the first step before executing.",
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
        self._chain_list.itemActivated.connect(self._handle_chain_activation)  # type: ignore[arg-type]
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

        guidance = QLabel(
            "Steps automatically receive the previous response. "
            "Provide a single plain-text input when running the chain.",
            info_container,
        )
        guidance.setWordWrap(True)
        guidance.setObjectName("promptChainGuidance")
        info_layout.addWidget(guidance)

        steps_container = QFrame(self._detail_splitter)
        steps_layout = QVBoxLayout(steps_container)
        steps_layout.setContentsMargins(0, 0, 0, 0)
        steps_layout.setSpacing(6)

        steps_label = QLabel("Steps", steps_container)
        steps_layout.addWidget(steps_label)

        self._steps_table = QTableWidget(0, 3, steps_container)
        self._steps_table.setHorizontalHeaderLabels(["Order", "Prompt", "Failure handling"])
        self._steps_table.horizontalHeader().setStretchLastSection(True)
        self._steps_table.verticalHeader().setVisible(False)
        self._steps_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._steps_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._steps_table.setSelectionMode(QTableWidget.SingleSelection)
        self._steps_table.cellActivated.connect(self._handle_step_table_activated)  # type: ignore[arg-type]
        steps_layout.addWidget(self._steps_table, 1)

        run_container = QFrame(self._detail_splitter)
        run_layout = QVBoxLayout(run_container)
        run_layout.setContentsMargins(0, 0, 0, 0)
        run_layout.setSpacing(6)

        run_header = QLabel("Run Chain", run_container)
        run_header.setStyleSheet("font-weight:600;")
        run_layout.addWidget(run_header)

        self._chain_input_edit = QPlainTextEdit(run_container)
        self._chain_input_edit.setPlaceholderText(
            "Type the text you want to feed into the first step…"
        )
        self._chain_input_edit.textChanged.connect(self._handle_chain_input_changed)  # type: ignore[arg-type]
        run_layout.addWidget(self._chain_input_edit, 1)

        actions_row = QHBoxLayout()
        self._run_button = QPushButton("Run Chain", run_container)
        self._run_button.clicked.connect(self._run_selected_chain)  # type: ignore[arg-type]
        actions_row.addWidget(self._run_button)
        self._clear_input_button = QPushButton("Clear Input", run_container)
        self._clear_input_button.clicked.connect(self._chain_input_edit.clear)  # type: ignore[arg-type]
        actions_row.addWidget(self._clear_input_button)
        self._web_search_checkbox = QCheckBox("Use web search", run_container)
        self._web_search_checkbox.setChecked(self._load_web_search_preference())
        self._web_search_checkbox.setToolTip(
            "Include live web search findings before each chain step executes."
        )
        self._web_search_checkbox.stateChanged.connect(  # type: ignore[arg-type]
            self._handle_web_search_toggle
        )
        actions_row.addWidget(self._web_search_checkbox)
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
        self._auto_scroll_checkbox = QCheckBox("Auto-scroll", results_container)
        self._auto_scroll_checkbox.setChecked(True)
        results_header.addWidget(self._auto_scroll_checkbox)
        self._wrap_checkbox = QCheckBox("Wrap lines", results_container)
        self._wrap_checkbox.setChecked(True)
        self._wrap_checkbox.toggled.connect(self._handle_wrap_changed)  # type: ignore[arg-type]
        results_header.addWidget(self._wrap_checkbox)

        self._result_view = QTextEdit(results_container)
        self._result_view.setReadOnly(True)
        self._result_view.setPlaceholderText("Execution results, outputs, and per-step summary.")
        self._result_view.setAcceptRichText(True)
        self._handle_wrap_changed(True)
        results_layout.addWidget(self._result_view, 1)
        self._stream_preview_active = False
        self._stream_buffers: dict[str, str] = {}
        self._stream_labels: dict[str, str] = {}
        self._stream_order: list[str] = []
        self._stream_chain_title = ""
        self._stream_chain_input_text = ""
        self._active_stream_thread: threading.Thread | None = None

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
        self._prompt_name_cache.clear()
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
                    if item is None:
                        continue
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
            self._steps_table.setRowCount(0)
            self._step_prompt_ids.clear()
            self._result_view.clear()
            self._run_button.setEnabled(False)
            if self._web_search_checkbox is not None:
                self._web_search_checkbox.setEnabled(False)
            self._current_input_chain_id = None
            self._set_chain_input_text("")
            return

        self._detail_title.setText(chain.name)
        status = "Active" if chain.is_active else "Inactive"
        updated_at_text = chain.updated_at.strftime("%Y-%m-%d %H:%M")  # noqa: DTZ005
        summary_pref = "On" if chain.summarize_last_response else "Off"
        detail_parts = [
            f"Status: {status}",
            f"Steps: {len(chain.steps)}",
            f"Updated {updated_at_text}",
            f"Summary: {summary_pref}",
        ]
        self._detail_status.setText(" • ".join(detail_parts))
        self._description_label.setText(chain.description or "(No description provided.)")
        self._populate_steps_table(chain)
        self._result_view.clear()
        self._run_button.setEnabled(chain.is_active and bool(chain.steps))
        if self._web_search_checkbox is not None:
            self._web_search_checkbox.setEnabled(bool(chain.steps))
        self._current_input_chain_id = str(chain.id)
        self._set_chain_input_text(self._load_chain_input_text(self._current_input_chain_id))

    def _populate_steps_table(self, chain: PromptChain) -> None:
        """Fill the steps table using ``chain.steps``."""
        self._steps_table.setRowCount(len(chain.steps))
        self._step_prompt_ids.clear()
        for row, step in enumerate(chain.steps):
            self._steps_table.setItem(row, 0, QTableWidgetItem(str(step.order_index)))
            prompt_label = self._resolve_prompt_label(step.prompt_id)
            prompt_item = QTableWidgetItem(prompt_label)
            prompt_item.setToolTip(str(step.prompt_id))  # pyright: ignore[reportAttributeAccessIssue]
            self._steps_table.setItem(row, 1, prompt_item)
            behaviour = "Stop chain on failure" if step.stop_on_failure else "Continue on failure"
            self._steps_table.setItem(row, 2, QTableWidgetItem(behaviour))
            self._step_prompt_ids[row] = step.prompt_id
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
                return cast("list[Prompt]", list_prompts(limit=500))
            repository = getattr(self._manager, "repository", None)
            if repository is not None:
                return cast("list[Prompt]", repository.list(limit=500))
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
            if item is None:
                continue
            if item.data(Qt.UserRole) == chain_id_text:
                self._chain_list.setCurrentRow(row)
                return

    def _run_selected_chain(self) -> None:
        """Execute the currently selected chain with the provided plain-text input."""
        chain = next(
            (entry for entry in self._chains if str(entry.id) == self._selected_chain_id),
            None,
        )
        if chain is None:
            QMessageBox.information(self, "Select chain", "Choose a chain to run first.")
            return
        chain_input_text = self._chain_input_edit.toPlainText()
        if not chain_input_text.strip():
            QMessageBox.warning(self, "Chain input", "Enter text to feed into the first step.")
            return

        previous_status = self._detail_status.text()
        use_web_search = (
            bool(self._web_search_checkbox.isChecked()) if self._web_search_checkbox else True
        )
        self._detail_status.setText(f"Running '{chain.name}'…")
        streaming_enabled = self._is_streaming_enabled()
        stream_callback = self._handle_step_stream if streaming_enabled else None
        if streaming_enabled:
            self._begin_stream_preview(chain, chain_input_text)
            self._run_chain_async(
                chain,
                chain_input_text,
                previous_status,
                stream_callback,
                use_web_search,
            )
            return

        indicator = ProcessingIndicator(self, f"Running '{chain.name}'…", title="Prompt Chain")
        try:
            result = indicator.run(
                self._manager.run_prompt_chain,
                chain.id,
                chain_input=chain_input_text,
                stream_callback=stream_callback,
                use_web_search=use_web_search,
            )
        except PromptChainExecutionError as exc:
            QMessageBox.critical(self, "Chain failed", str(exc))
            self._detail_status.setText(previous_status)
            return
        except PromptChainError as exc:
            QMessageBox.critical(self, "Unable to run chain", str(exc))
            self._detail_status.setText(previous_status)
            return
        else:
            timestamp = datetime.now().strftime("%H:%M:%S")  # noqa: DTZ005
            self._detail_status.setText(f"Last run succeeded at {timestamp}")
        self._display_run_result(result)
        show_toast(self, f"Chain '{chain.name}' completed.")

    def _display_run_result(self, result: PromptChainRunResult) -> None:
        """Render execution outputs and per-step summary for *result*."""

        def _indent_lines(text: str, prefix: str = "  ") -> list[str]:
            if not text.strip():
                return [f"{prefix}(empty)"]
            return [f"{prefix}{line}" for line in text.splitlines()]

        plain_sections: list[str] = []
        html_sections: list[str] = [self._result_style_css(), '<div class="chain-results">']
        step_html_sections: list[str] = []
        summarize_enabled = bool(getattr(result.chain, "summarize_last_response", True))
        summary_text: str | None = None
        if summarize_enabled and result.summary:
            summary_text = result.summary.strip() or None
        last_success_index: int | None = None
        if summarize_enabled:
            for index in range(len(result.steps) - 1, -1, -1):
                candidate = result.steps[index]
                if candidate.outcome is not None and candidate.status == "success":
                    last_success_index = index
                    break

        # Chain inputs
        plain_sections.append("Input to chain")
        chain_input_text = result.chain_input or ""
        html_sections.append(
            self._format_colored_block_html(
                "Input to chain",
                chain_input_text or "- (no input provided)",
                color=_CHAIN_INPUT_COLOR,
                class_name="chain-block--input",
            )
        )
        if chain_input_text:
            plain_sections.extend(_indent_lines(chain_input_text))
        else:
            plain_sections.append("  (no input provided)")
        plain_sections.append("")

        # Chain outputs
        plain_sections.append("Chain Outputs")
        outputs_text = ""
        if result.outputs:
            outputs_text = self._format_json(result.outputs)
            plain_sections.extend(_indent_lines(outputs_text))
        else:
            plain_sections.append("  (no outputs)")
        plain_sections.append("")
        html_sections.append(
            self._format_colored_block_html(
                "Chain outputs",
                outputs_text or "- (no outputs)",
                color=_CHAIN_OUTPUT_COLOR,
                class_name="chain-block--outputs",
            )
        )

        # Steps
        if result.steps:
            html_sections.append('<div class="chain-steps">')
        total_steps = len(result.steps)
        for index, step_run in enumerate(result.steps):
            step = step_run.step
            step_label = step.output_variable or str(step.prompt_id)
            plain_sections.append(f"Step {step.order_index}: {step_label}")
            plain_sections.append(f"  Status: {step_run.status}")
            if step_run.error:
                plain_sections.append(f"  Error: {step_run.error}")
            elif step_run.status == "skipped":
                plain_sections.append("  Skipped because condition evaluated to false.")
            outcome = step_run.outcome
            request_text = ""
            response_text = ""
            reasoning_text = None
            should_render_reasoning = (
                summarize_enabled
                and last_success_index is not None
                and index == last_success_index
                and outcome is not None
            )
            include_step_input = index == 0
            is_last_step = index == total_steps - 1
            show_response_block = True
            if outcome:
                request_text = outcome.result.request_text.strip()
                response_text = outcome.result.response_text.strip()
                if include_step_input:
                    plain_sections.append("  Input to step:")
                    plain_sections.extend(_indent_lines(request_text or "(empty input)"))
                show_response_block = not (is_last_step and should_render_reasoning)
                if show_response_block:
                    plain_sections.append("  Output of step:")
                    plain_sections.extend(_indent_lines(response_text or "(empty response)"))
                if should_render_reasoning:
                    reasoning_text = self._extract_reasoning_summary(outcome)
                    if reasoning_text:
                        plain_sections.append("  Reasoning summary:")
                        plain_sections.extend(_indent_lines(reasoning_text))
            step_html_sections.append(
                self._build_step_html_block(
                    step_run,
                    request_text,
                    response_text,
                    reasoning_text,
                    include_step_input,
                    show_response_block,
                )
            )
            plain_sections.append("")
        if result.steps:
            html_sections.extend(step_html_sections)
            html_sections.append("</div>")
        if summary_text:
            plain_sections.append("Chain summary")
            plain_sections.extend(_indent_lines(summary_text))
            plain_sections.append("")
            html_sections.append(
                self._format_colored_block_html(
                    "Chain summary",
                    summary_text,
                    color=_CHAIN_SUMMARY_COLOR,
                    class_name="chain-block--summary",
                    monospace=False,
                )
            )
        html_sections.append("</div>")
        plain_text = "\n".join(plain_sections).strip()
        html_text = "\n".join(html_sections).strip()
        self._result_plaintext = plain_text
        self._result_richtext = html_text
        self._apply_result_view()

    @staticmethod
    def _format_colored_block_html(
        title: str,
        body_text: str,
        *,
        color: str | None,
        class_name: str,
        monospace: bool = True,
        body_class: str | None = None,
        body_color: str | None = None,
    ) -> str:
        """Return an HTML block with consistent styling and text color."""
        safe_title = html.escape(title)
        safe_body = html.escape(body_text or "(empty)")
        base_classes = ["chain-block-body"]
        body_style_attr = f' style="color:{body_color};"' if body_color else ""
        if monospace:
            base_classes.append("chain-block-body--mono")
        if body_class:
            base_classes.append(body_class)
        body_class_attr = " ".join(base_classes)
        if monospace:
            body_html = f'<pre class="{body_class_attr}"{body_style_attr}>{safe_body}</pre>'
        else:
            body_html = f'<div class="{body_class_attr}"{body_style_attr}>{safe_body}</div>'
        style_attr = f' style="color:{color};"' if color else ""
        return (
            f'<div class="chain-block {class_name}"{style_attr}>'
            f"<div class='chain-block-title'>{safe_title}</div>"
            f"{body_html}"
            "</div>"
        )

    def _result_style_css(self) -> str:
        palette = self._result_view.palette()
        color_role_enum = getattr(QPalette, "ColorRole", None)
        base_role = getattr(color_role_enum, "Base", QPalette.Base)
        text_role = getattr(color_role_enum, "Text", QPalette.Text)
        base_color = palette.color(base_role)
        text_color = palette.color(text_role)
        chosen = self._pick_accessible_text_color(base_color, text_color)
        return _RESULT_STYLE_TEMPLATE.replace("{text_color}", chosen)

    @staticmethod
    def _pick_accessible_text_color(base_color: Any, text_color: Any) -> str:
        def _luminance(color: Any) -> float:
            red = float(getattr(color, "redF", lambda: 0.0)())
            green = float(getattr(color, "greenF", lambda: 0.0)())
            blue = float(getattr(color, "blueF", lambda: 0.0)())
            return 0.2126 * red + 0.7152 * green + 0.0722 * blue

        base_lum = _luminance(base_color)
        if base_lum < 0.4:
            return "#f5f7ff"
        hex_value = PromptChainManagerPanel._color_to_hex(text_color)
        return hex_value or "#1c2837"

    @staticmethod
    def _color_to_hex(color: Any) -> str:
        red = int(max(0, min(255, getattr(color, "red", lambda: 0)())))
        green = int(max(0, min(255, getattr(color, "green", lambda: 0)())))
        blue = int(max(0, min(255, getattr(color, "blue", lambda: 0)())))
        return f"#{red:02x}{green:02x}{blue:02x}"

    def _build_step_html_block(
        self,
        step_run: PromptChainStepRun,
        request_text: str,
        response_text: str,
        reasoning_text: str | None,
        show_request_block: bool,
        show_response_block: bool,
    ) -> str:
        """Return styled HTML for a single step entry."""
        step = step_run.step
        label = step.output_variable or str(step.prompt_id)
        parts = ['<div class="chain-step">']
        parts.append(
            f"<div class='chain-step-title'>Step {step.order_index} – {html.escape(label)}</div>"
        )
        parts.append(
            "<div class='chain-step-status'>"
            f"Status: <code>{html.escape(step_run.status)}</code>"
            "</div>"
        )
        if step_run.error:
            parts.append(f"<div class='chain-step-error'>{html.escape(step_run.error)}</div>")
        elif step_run.status == "skipped":
            parts.append(
                "<div class='chain-step-detail'>Skipped because condition evaluated to false.</div>"
            )
        if request_text and show_request_block:
            parts.append(
                self._format_colored_block_html(
                    "Input to step",
                    request_text or "(empty input)",
                    color=None,
                    class_name="chain-step-detail chain-step-input",
                    body_class="chain-block-body--muted",
                    body_color=_STEP_MUTED_COLOR,
                )
            )
        if show_response_block:
            parts.append(
                self._format_colored_block_html(
                    "Output of step",
                    response_text or "(empty response)",
                    color=None,
                    class_name="chain-step-detail chain-step-output",
                    body_class="chain-block-body--muted",
                    body_color=_STEP_MUTED_COLOR,
                )
            )
        if reasoning_text:
            parts.append(
                self._format_colored_block_html(
                    "Reasoning summary",
                    reasoning_text,
                    color=_REASONING_COLOR,
                    class_name="chain-step-detail chain-step-reasoning",
                    monospace=False,
                )
            )
        parts.append("</div>")
        return "".join(parts)

    @staticmethod
    def _extract_reasoning_summary(outcome: ExecutionOutcome | None) -> str | None:
        """Attempt to pull reasoning content from the raw response payload."""
        if outcome is None:
            return None
        raw = getattr(outcome.result, "raw_response", None)
        if not isinstance(raw, Mapping):
            return None
        return _search_reasoning_payload(raw)

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
        self._result_richtext = ""
        self._apply_result_view()

    def _handle_result_format_changed(self, _: bool) -> None:
        """Switch between Markdown and plain text views."""
        self._apply_result_view()

    def _handle_wrap_changed(self, checked: bool) -> None:
        """Toggle line wrapping for the execution results view."""
        line_wrap_enum = getattr(QTextEdit, "LineWrapMode", None)
        if line_wrap_enum is None:
            return
        mode = line_wrap_enum.WidgetWidth if checked else line_wrap_enum.NoWrap
        self._result_view.setLineWrapMode(mode)

    def _apply_result_view(self) -> None:
        """Render stored results using the current format preference."""
        prefers_markdown = self._result_format_checkbox.isChecked()
        rich_text = self._result_richtext
        plain_text = self._result_plaintext
        has_rich = bool(rich_text.strip())
        has_plain_text = bool(plain_text.strip())

        if prefers_markdown:
            if has_rich:
                self._result_view.setHtml(rich_text)
            elif has_plain_text:
                self._result_view.setPlainText(plain_text)
            else:
                self._result_view.clear()
        else:
            if has_plain_text:
                self._result_view.setPlainText(plain_text)
            elif has_rich:
                doc = QTextDocument()
                doc.setHtml(rich_text)
                self._result_view.setPlainText(doc.toPlainText())
            else:
                self._result_view.clear()
        self._maybe_scroll_results()

    def _is_streaming_enabled(self) -> bool:
        """Return True when LiteLLM streaming is enabled in runtime settings."""
        executor = getattr(self._manager, "_executor", None)
        if executor is None:
            return False
        return bool(getattr(executor, "stream", False))

    def _begin_stream_preview(self, chain: PromptChain, chain_input: str) -> None:
        """Initialise buffers used to show streaming output per step."""
        self._stream_preview_active = True
        self._stream_chain_title = chain.name
        self._stream_buffers.clear()
        self._stream_labels.clear()
        self._stream_order.clear()
        self._stream_chain_input_text = chain_input
        for step in chain.steps:
            self._declare_stream_step(step)
        self._refresh_stream_preview()

    def _handle_step_stream(self, step: PromptChainStep, chunk: str, is_final: bool) -> None:
        """Schedule GUI updates for streamed chain step output."""
        if not chunk:
            return
        QTimer.singleShot(0, self, lambda: self._register_stream_chunk(step, chunk, is_final))

    def _register_stream_chunk(self, step: PromptChainStep, chunk: str, is_final: bool) -> None:
        """Append *chunk* to the preview buffer for *step*."""
        if not self._stream_preview_active:
            return
        step_id = self._declare_stream_step(step)
        existing = self._stream_buffers.get(step_id) or ""
        text = chunk
        if existing:
            existing += text
        else:
            existing = text
        self._stream_buffers[step_id] = existing
        if is_final:
            label = self._stream_labels.get(step_id, "")
            if label and "(completed)" not in label:
                self._stream_labels[step_id] = f"{label} (completed)"
        self._refresh_stream_preview()

    def _declare_stream_step(self, step: PromptChainStep) -> str:
        """Ensure streaming buffers exist for *step* and return its identifier."""
        step_id = str(step.id)
        if step_id not in self._stream_buffers:
            label = f"Step {step.order_index} – {step.output_variable or step.prompt_id}"
            self._stream_order.append(step_id)
            self._stream_labels[step_id] = label
            self._stream_buffers[step_id] = ""
        return step_id

    def _refresh_stream_preview(self) -> None:
        """Render the live streaming preview text."""
        if not self._stream_preview_active:
            return
        header = self._stream_chain_title or "prompt chain"
        lines = [f"Streaming '{header}'…", ""]
        if self._stream_chain_input_text:
            lines.append("Input to chain")
            lines.extend(self._stream_chain_input_text.splitlines())
            lines.append("")
        if self._stream_order:
            lines.append("Step details")
            lines.append("")
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
        self._maybe_scroll_results()

    def _end_stream_preview(self) -> None:
        """Clear streaming preview state."""
        self._stream_preview_active = False
        self._stream_chain_title = ""
        self._stream_chain_input_text = ""
        self._stream_buffers.clear()
        self._stream_labels.clear()
        self._stream_order.clear()

    def _handle_splitter_moved(self, _: int, __: int) -> None:
        """Persist splitter state whenever the user resizes panes."""
        self._persist_splitter_state()

    def _handle_chain_input_changed(self) -> None:
        """Persist chain input text whenever it changes."""
        if self._suppress_input_signal:
            return
        chain_id = self._current_input_chain_id
        if not chain_id:
            return
        self._settings.setValue(
            self._chain_input_settings_key(chain_id),
            self._chain_input_edit.toPlainText(),
        )

    def _set_chain_input_text(self, text: str) -> None:
        """Apply text to the chain input editor without triggering persistence."""
        self._suppress_input_signal = True
        self._chain_input_edit.setPlainText(text)
        self._suppress_input_signal = False

    def _load_web_search_preference(self) -> bool:
        stored = self._settings.value("chainWebSearchEnabled")
        if stored is None:
            return True
        if isinstance(stored, bool):
            return stored
        if isinstance(stored, str):
            lowered = stored.strip().lower()
            if lowered in {"true", "1", "yes"}:
                return True
            if lowered in {"false", "0", "no"}:
                return False
        return True

    def _handle_web_search_toggle(self, state: int) -> None:
        self._settings.setValue("chainWebSearchEnabled", bool(state))

    def _handle_step_table_activated(self, row: int, _: int) -> None:
        prompt_id = self._step_prompt_ids.get(row)
        if prompt_id is None:
            return
        if self._prompt_edit_callback is None:
            QMessageBox.information(
                self,
                "Prompt editor unavailable",
                "Prompt editing is unavailable in this context.",
            )
            return
        self._prompt_edit_callback(prompt_id)

    def _handle_chain_activation(self, item: QListWidgetItem | None) -> None:
        if item is None:
            return
        self._chain_list.setCurrentItem(item)
        self._edit_chain()

    def _resolve_prompt_label(self, prompt_id: UUID) -> str:
        cached = self._prompt_name_cache.get(prompt_id)
        if cached:
            return cached
        try:
            prompt = self._manager.get_prompt(prompt_id)
        except PromptManagerError:
            label = str(prompt_id)
        else:
            name = (prompt.name or "").strip()
            label = name or str(prompt_id)
        self._prompt_name_cache[prompt_id] = label
        return label

    def _load_chain_input_text(self, chain_id: str | None) -> str:
        """Load saved chain input for the given chain."""
        if not chain_id:
            return ""
        stored = self._settings.value(self._chain_input_settings_key(chain_id))
        if stored is None:
            stored = self._settings.value(f"chainVariables/{chain_id}")
        if isinstance(stored, str):
            return stored
        if stored is None:
            return ""
        return str(stored)

    def _chain_input_settings_key(self, chain_id: str) -> str:
        """Return the QSettings key for persisted chain inputs."""
        return f"chainInput/{chain_id}"

    def _run_chain_async(
        self,
        chain: PromptChain,
        chain_input: str,
        previous_status: str,
        stream_callback: Callable[[PromptChainStep, str, bool], None] | None,
        use_web_search: bool,
    ) -> None:
        """Execute the chain on a worker thread while showing live stream updates."""
        payload: dict[str, object] = {}

        def _worker() -> None:
            try:
                payload["result"] = self._manager.run_prompt_chain(
                    chain.id,
                    chain_input=chain_input,
                    stream_callback=stream_callback,
                    use_web_search=use_web_search,
                )
            except Exception as exc:  # noqa: BLE001 - propagate via UI thread handler
                payload["error"] = exc
            finally:
                QTimer.singleShot(
                    0,
                    self,
                    lambda: self._handle_stream_completion(chain, previous_status, payload),
                )

        thread = threading.Thread(target=_worker, daemon=True)
        self._active_stream_thread = thread
        thread.start()

    def _handle_stream_completion(
        self,
        chain: PromptChain,
        previous_status: str,
        payload: dict[str, object],
    ) -> None:
        """Handle completion (success or failure) of the streaming worker."""
        self._active_stream_thread = None
        self._end_stream_preview()
        error = payload.get("error")
        if error is not None:
            self._detail_status.setText(previous_status)
            if isinstance(error, PromptChainExecutionError):
                QMessageBox.critical(self, "Chain failed", str(error))
            elif isinstance(error, PromptChainError):
                QMessageBox.critical(self, "Unable to run chain", str(error))
            else:  # pragma: no cover - defensive guard
                QMessageBox.critical(self, "Chain failed", str(error))
            return
        result = payload.get("result")
        if not isinstance(result, PromptChainRunResult):
            self._detail_status.setText(previous_status)
            return
        timestamp = datetime.now().strftime("%H:%M:%S")  # noqa: DTZ005
        self._detail_status.setText(f"Last run succeeded at {timestamp}")
        self._display_run_result(result)
        show_toast(self, f"Chain '{chain.name}' completed.")

    def _format_json(self, payload: Mapping[str, Any]) -> str:
        """Render mapping payloads as pretty JSON text."""
        try:
            return json.dumps(payload, indent=2, sort_keys=True, default=str)
        except TypeError:
            return json.dumps(
                {str(key): str(value) for key, value in payload.items()},
                indent=2,
                sort_keys=True,
            )

    def _maybe_scroll_results(self) -> None:
        """Scroll the results view to bottom when auto-scroll is enabled."""
        if not self._auto_scroll_checkbox.isChecked():
            return
        bar = self._result_view.verticalScrollBar()
        if bar is not None:
            bar.setValue(bar.maximum())


class PromptChainManagerDialog(QDialog):
    """Dialog wrapper that hosts :class:`PromptChainManagerPanel`."""

    def __init__(
        self,
        manager: PromptManager,
        parent: QWidget | None = None,
        *,
        prompt_edit_callback: Callable[[UUID], None] | None = None,
    ) -> None:
        """Create the dialog and embed the shared panel widget."""
        super().__init__(parent)
        self.setWindowTitle("Prompt Chains")
        self.resize(960, 640)
        layout = QVBoxLayout(self)
        self._panel = PromptChainManagerPanel(
            manager,
            parent=self,
            prompt_edit_callback=prompt_edit_callback,
        )
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


_REASONING_KEYS = (
    "reasoning",
    "reasoning_summary",
    "reasoning_content",
    "chain_of_thought",
    "thoughts",
)


def _search_reasoning_payload(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if any(char.isspace() for char in stripped):
            return stripped
        return None
    if isinstance(value, Mapping):
        for key in _REASONING_KEYS:
            if key in value:
                found = _search_reasoning_payload(value[key])
                if found:
                    return found
        type_value = value.get("type")
        if isinstance(type_value, str) and type_value.lower() in {
            "reasoning",
            "thought",
            "thinking",
        }:
            found = _search_reasoning_payload(value.get("text") or value.get("content"))
            if found:
                return found
        for child in value.values():
            found = _search_reasoning_payload(child)
            if found:
                return found
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            found = _search_reasoning_payload(item)
            if found:
                return found
    return None
