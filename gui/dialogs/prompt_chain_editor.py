"""Prompt chain editor dialogs for creating and updating workflows.

Updates:
  v0.4.0 - 2025-12-06 - Simplify chain editor for plain-text linear chaining.
  v0.3.1 - 2025-12-05 - Hide the variables schema editor behind a toggle button.
  v0.3.0 - 2025-12-05 - Surface summarize toggle for prompt chain results.
  v0.2.0 - 2025-12-04 - Add prompt picker combo box backed by catalog lookups.
  v0.1.0 - 2025-12-04 - Introduce editor dialogs for chain and step CRUD.
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from models.prompt_chain_model import PromptChain, PromptChainStep

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from core import PromptManager
    from models.prompt_model import Prompt


class PromptChainEditorDialog(QDialog):
    """Dialog that edits prompt chain metadata and steps."""

    def __init__(
        self,
        parent: QWidget | None,
        *,
        manager: PromptManager | None,
        prompts: list[Prompt] | None = None,
        chain: PromptChain | None = None,
    ) -> None:
        """Initialise fields for a new or existing chain."""
        super().__init__(parent)
        self._manager = manager
        self._source_chain = chain
        self._chain_id = chain.id if chain else uuid.uuid4()
        self._result_chain: PromptChain | None = None
        self._prompts: list[Prompt] = list(prompts or [])
        self._prompt_lookup: dict[str, Prompt] = {
            str(prompt.id): prompt for prompt in self._prompts
        }
        self.setWindowTitle("Edit Prompt Chain" if chain else "New Prompt Chain")
        self.resize(800, 640)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self._name_input = QLineEdit(self)
        self._name_input.setObjectName("promptChainEditorNameInput")
        self._name_input.setPlaceholderText("e.g. Content Review Flow")
        self._add_form_row(form, "Name", self._name_input, "Friendly chain label shown in lists.")

        self._description_input = QPlainTextEdit(self)
        self._description_input.setObjectName("promptChainEditorDescriptionInput")
        self._description_input.setPlaceholderText("Describe what the chain does…")
        self._add_form_row(
            form,
            "Description",
            self._description_input,
            "Optional summary to help collaborators understand the workflow.",
        )

        info_label = QLabel(
            (
                "Chains no longer define variables or templates—"
                "each step receives the previous output."
            ),
            self,
        )
        info_label.setWordWrap(True)
        form.addRow("", info_label)

        self._active_checkbox = QCheckBox("Chain is active", self)
        self._active_checkbox.setChecked(True)
        form.addRow("Active", self._active_checkbox)

        self._summarize_checkbox = QCheckBox("Summarize final step output", self)
        self._summarize_checkbox.setObjectName("promptChainEditorSummarizeCheckbox")
        self._summarize_checkbox.setChecked(True)
        self._add_form_row(
            form,
            "Summarize",
            self._summarize_checkbox,
            "If enabled, the chain produces a condensed summary from the last step response.",
        )
        layout.addLayout(form)

        self._steps_table = QTableWidget(0, 3, self)
        self._steps_table.setObjectName("promptChainEditorStepsTable")
        self._steps_table.setHorizontalHeaderLabels(["Order", "Prompt", "Failure handling"])
        self._steps_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._steps_table.itemDoubleClicked.connect(self._handle_step_double_click)  # type: ignore[arg-type]
        self._steps_table.itemSelectionChanged.connect(self._update_step_action_state)  # type: ignore[arg-type]
        layout.addWidget(self._steps_table, 1)

        self._warning_label = QLabel("", self)
        self._warning_label.setObjectName("promptChainEditorWarningLabel")
        self._warning_label.setWordWrap(True)
        self._warning_label.setStyleSheet("color: #b26a00;")
        layout.addWidget(self._warning_label)

        step_actions = QHBoxLayout()
        self._add_step_button = QPushButton("Add Step", self)
        self._add_step_button.clicked.connect(self._add_step)  # type: ignore[arg-type]
        step_actions.addWidget(self._add_step_button)

        self._edit_step_button = QPushButton("Edit Step", self)
        self._edit_step_button.clicked.connect(self._edit_selected_step)  # type: ignore[arg-type]
        step_actions.addWidget(self._edit_step_button)

        self._remove_step_button = QPushButton("Remove Step", self)
        self._remove_step_button.clicked.connect(self._remove_selected_step)  # type: ignore[arg-type]
        step_actions.addWidget(self._remove_step_button)

        self._duplicate_step_button = QPushButton("Duplicate Step", self)
        self._duplicate_step_button.clicked.connect(self._duplicate_selected_step)  # type: ignore[arg-type]
        step_actions.addWidget(self._duplicate_step_button)

        self._move_step_up_button = QPushButton("Move Up", self)
        self._move_step_up_button.setObjectName("promptChainEditorMoveStepUpButton")
        self._move_step_up_button.clicked.connect(self._move_selected_step_up)  # type: ignore[arg-type]
        step_actions.addWidget(self._move_step_up_button)

        self._move_step_down_button = QPushButton("Move Down", self)
        self._move_step_down_button.setObjectName("promptChainEditorMoveStepDownButton")
        self._move_step_down_button.clicked.connect(self._move_selected_step_down)  # type: ignore[arg-type]
        step_actions.addWidget(self._move_step_down_button)
        step_actions.addStretch(1)
        layout.addLayout(step_actions)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel, self
        )
        self._buttons.accepted.connect(self._handle_accept)  # type: ignore[arg-type]
        self._buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        layout.addWidget(self._buttons)

        self._steps: list[PromptChainStep] = []
        if chain is not None:
            self._load_chain(chain)
        self._update_step_action_state()

    def result_chain(self) -> PromptChain | None:
        """Return the saved chain instance when available."""
        return self._result_chain

    def chain_id(self) -> uuid.UUID:
        """Return the active chain identifier for tests and callers."""
        return self._chain_id

    def set_steps(self, steps: list[PromptChainStep]) -> None:
        """Replace the current steps and refresh the rendered table."""
        self._steps = list(steps)
        self._refresh_steps()

    def steps(self) -> list[PromptChainStep]:
        """Return a copy of the current step list."""
        return list(self._steps)

    def accept_chain(self) -> None:
        """Public wrapper for validating and accepting the dialog."""
        self._handle_accept()

    def refresh_steps(self) -> None:
        """Public wrapper for refreshing the rendered steps table."""
        self._refresh_steps()

    def move_selected_step_up(self) -> None:
        """Public wrapper for moving the selected step upward."""
        self._move_selected_step_up()

    def move_selected_step_down(self) -> None:
        """Public wrapper for moving the selected step downward."""
        self._move_selected_step_down()

    def update_step_action_state(self) -> None:
        """Public wrapper for recalculating action button enabled state."""
        self._update_step_action_state()

    def open_step_preview(self, item: QTableWidgetItem) -> None:
        """Public wrapper for the prompt preview flow triggered from the steps table."""
        self._handle_step_double_click(item)

    def duplicate_selected_step(self) -> None:
        """Public wrapper for duplicating the selected step."""
        self._duplicate_selected_step()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _load_chain(self, chain: PromptChain) -> None:
        self._name_input.setText(chain.name)
        self._description_input.setPlainText(chain.description)
        self._active_checkbox.setChecked(chain.is_active)
        self._summarize_checkbox.setChecked(chain.summarize_last_response)
        self._steps = [replace(step) for step in chain.steps]
        self._refresh_steps()

    def _refresh_steps(self) -> None:
        self._steps.sort(key=lambda step: step.order_index)
        current_step_id: str | None = None
        current_row = self._steps_table.currentRow()
        if 0 <= current_row < len(self._steps):
            current_step_id = str(self._steps[current_row].id)
        self._steps_table.setRowCount(len(self._steps))
        selected_row = -1
        for row, step in enumerate(self._steps):
            self._steps_table.setItem(row, 0, QTableWidgetItem(str(step.order_index)))
            prompt = self._prompt_lookup.get(str(step.prompt_id))
            prompt_label = prompt.name if prompt and prompt.name else None
            prompt_item = QTableWidgetItem(prompt_label or str(step.prompt_id))
            prompt_item.setData(Qt.ItemDataRole.UserRole, str(step.prompt_id))
            if prompt is not None:
                prompt_item.setToolTip(self._build_prompt_tooltip(prompt))
            self._steps_table.setItem(row, 1, prompt_item)
            behaviour = "Stop chain on failure" if step.stop_on_failure else "Continue on failure"
            self._steps_table.setItem(row, 2, QTableWidgetItem(behaviour))
            if current_step_id is not None and str(step.id) == current_step_id:
                selected_row = row
        if selected_row >= 0:
            self._steps_table.selectRow(selected_row)
        elif self._steps:
            self._steps_table.clearSelection()
        self._update_step_action_state()
        self._update_warning_state()

    def _add_step(self) -> None:
        dialog = PromptChainStepDialog(
            self,
            chain_id=self._chain_id,
            prompts=self._prompts,
        )
        dialog.set_order_index(len(self._steps) + 1)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        step = dialog.result_step()
        if step is None:
            return
        self._steps.append(step)
        self._refresh_steps()

    def _edit_selected_step(self) -> None:
        row = self._steps_table.currentRow()
        if row < 0 or row >= len(self._steps):
            return
        dialog = PromptChainStepDialog(
            self,
            chain_id=self._chain_id,
            step=self._steps[row],
            prompts=self._prompts,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        step = dialog.result_step()
        if step is None:
            return
        self._steps[row] = step
        self._refresh_steps()

    def _remove_selected_step(self) -> None:
        row = self._steps_table.currentRow()
        if row < 0 or row >= len(self._steps):
            return
        del self._steps[row]
        self._refresh_steps()

    def _duplicate_selected_step(self) -> None:
        row = self._steps_table.currentRow()
        if row < 0 or row >= len(self._steps):
            return
        source_step = self._steps[row]
        duplicated_step = replace(
            source_step,
            id=uuid.uuid4(),
            order_index=row + 2,
            output_variable=f"step_{row + 2}",
        )
        self._steps.insert(row + 1, duplicated_step)
        self._steps = self._reindexed_steps()
        self._refresh_steps()
        self._steps_table.selectRow(row + 1)

    def _move_selected_step_up(self) -> None:
        row = self._steps_table.currentRow()
        if row <= 0 or row >= len(self._steps):
            return
        self._steps[row - 1], self._steps[row] = self._steps[row], self._steps[row - 1]
        self._steps[row - 1] = replace(self._steps[row - 1], order_index=row)
        self._steps[row] = replace(self._steps[row], order_index=row + 1)
        self._refresh_steps()
        self._steps_table.selectRow(row - 1)

    def _move_selected_step_down(self) -> None:
        row = self._steps_table.currentRow()
        if row < 0 or row >= len(self._steps) - 1:
            return
        self._steps[row], self._steps[row + 1] = self._steps[row + 1], self._steps[row]
        self._steps[row] = replace(self._steps[row], order_index=row + 1)
        self._steps[row + 1] = replace(self._steps[row + 1], order_index=row + 2)
        self._refresh_steps()
        self._steps_table.selectRow(row + 1)

    def _update_step_action_state(self) -> None:
        row = self._steps_table.currentRow()
        has_selection = 0 <= row < len(self._steps)
        self._edit_step_button.setEnabled(has_selection)
        self._remove_step_button.setEnabled(has_selection)
        self._duplicate_step_button.setEnabled(has_selection)
        self._move_step_up_button.setEnabled(has_selection and row > 0)
        self._move_step_down_button.setEnabled(has_selection and row < len(self._steps) - 1)

    def _update_warning_state(self) -> None:
        warnings: list[str] = []
        duplicate_prompt_ids = {
            str(step.prompt_id)
            for step in self._steps
            if sum(1 for candidate in self._steps if candidate.prompt_id == step.prompt_id) > 1
        }
        if duplicate_prompt_ids:
            duplicate_names: list[str] = []
            for prompt_id in sorted(duplicate_prompt_ids):
                prompt = self._prompt_lookup.get(prompt_id)
                duplicate_names.append(prompt.name if prompt and prompt.name else prompt_id)
            joined_names = ", ".join(duplicate_names)
            warnings.append(
                "Same prompt reused across multiple steps"
                f" ({joined_names}) — verify that repeated execution is intentional."
            )

        legacy_fields: set[str] = set()
        for step in self._steps:
            if step.input_template.strip():
                legacy_fields.add("input_template")
            if step.condition and str(step.condition).strip():
                legacy_fields.add("condition")
        if legacy_fields:
            joined_fields = ", ".join(sorted(legacy_fields))
            warnings.append(
                "Legacy/inactive semantics detected"
                f" ({joined_fields}) — preserved for import compatibility only."
            )

        self._warning_label.setText("Warning: " + " | ".join(warnings) if warnings else "")

    def _handle_accept(self) -> None:
        name = self._name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Validation", "Enter a name for the prompt chain.")
            return
        if not self._steps:
            QMessageBox.warning(self, "Validation", "Add at least one step before saving.")
            return
        description = self._description_input.toPlainText().strip()
        chain = PromptChain(
            id=self._chain_id,
            name=name,
            description=description,
            is_active=self._active_checkbox.isChecked(),
            variables_schema=None,
            metadata=None,
            summarize_last_response=self._summarize_checkbox.isChecked(),
        ).with_steps(self._reindexed_steps())
        self._result_chain = chain
        self.accept()

    def _reindexed_steps(self) -> list[PromptChainStep]:
        for idx, step in enumerate(sorted(self._steps, key=lambda s: s.order_index), start=1):
            self._steps[idx - 1] = replace(step, order_index=idx)
        return self._steps

    def _add_form_row(
        self,
        form: QFormLayout,
        label_text: str,
        widget: QWidget,
        tooltip: str,
    ) -> QLabel:
        label = QLabel(label_text, self)
        label.setToolTip(tooltip)
        label.setBuddy(widget)
        form.addRow(label, widget)
        return label

    def _handle_step_double_click(self, item: QTableWidgetItem) -> None:
        if item.column() != 1:
            return
        prompt_id = item.data(Qt.ItemDataRole.UserRole)
        prompt = self._prompt_lookup.get(str(prompt_id))
        if prompt is None:
            QMessageBox.information(self, "Prompt", "Prompt not found in catalog.")
            return
        body = (prompt.context or prompt.description or "(no body)").strip()
        preview = body if len(body) <= 600 else body[:597].rstrip() + "…"
        QMessageBox.information(
            self,
            prompt.name,
            preview or "Prompt has no saved body.",
        )

    def _build_prompt_tooltip(self, prompt: Prompt) -> str:
        body = (prompt.context or prompt.description or "").strip()
        if len(body) > 160:
            body = body[:157].rstrip() + "…"
        return f"{prompt.name}\n{body}"


class PromptChainStepDialog(QDialog):
    """Dialog for creating or editing a single prompt chain step."""

    def __init__(
        self,
        parent: QWidget | None,
        *,
        chain_id: uuid.UUID,
        step: PromptChainStep | None = None,
        prompts: list[Prompt] | None = None,
    ) -> None:
        """Prepare step editor inputs with optional preloaded data."""
        super().__init__(parent)
        self._chain_id = chain_id
        self._step = step
        self._result_step: PromptChainStep | None = None
        self._prompt_options: list[Prompt] = list(prompts or [])
        self.setWindowTitle("Edit Chain Step" if step else "Add Chain Step")
        self.resize(520, 360)

        layout = QFormLayout(self)
        self._order_input = QSpinBox(self)
        self._order_input.setMinimum(1)
        self._add_step_row(layout, "Order", self._order_input, "Determines the execution sequence.")

        self._prompt_combo = QComboBox(self)
        self._prompt_combo.setObjectName("promptChainStepPromptCombo")
        self._prompt_combo.setEditable(True)
        self._prompt_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._prompt_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._prompt_combo.currentIndexChanged.connect(self._update_prompt_preview)  # type: ignore[arg-type]
        self._add_step_row(
            layout,
            "Prompt",
            self._prompt_combo,
            "Select the catalog prompt to run for this step.",
        )

        self._prompt_preview = QPlainTextEdit(self)
        self._prompt_preview.setObjectName("promptChainStepPromptPreview")
        self._prompt_preview.setReadOnly(True)
        self._prompt_preview.setMaximumBlockCount(0)
        self._prompt_preview.setPlaceholderText("Prompt preview will appear here.")
        self._prompt_preview.setMinimumHeight(120)
        self._add_step_row(
            layout,
            "Prompt preview",
            self._prompt_preview,
            "Read-only preview of the selected prompt body.",
        )

        self._stop_checkbox = QCheckBox("Stop chain when this step fails", self)
        self._stop_checkbox.setChecked(True)
        self._add_step_row(
            layout,
            "Stop on failure",
            self._stop_checkbox,
            "If unchecked the chain continues even when this step errors.",
        )

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttons.accepted.connect(self._handle_accept)  # type: ignore[arg-type]
        buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        layout.addWidget(buttons)

        self._populate_prompt_combo()
        if step is not None:
            self._load_step(step)
        self._update_prompt_preview()

    def set_order_index(self, value: int) -> None:
        """Update the default order index displayed to the user."""
        self._order_input.setValue(max(1, value))

    def result_step(self) -> PromptChainStep | None:
        """Return the saved step when available."""
        return self._result_step

    def prompt_combo(self) -> QComboBox:
        """Expose the prompt selector for tests and callers."""
        return self._prompt_combo

    def prompt_preview_text(self) -> str:
        """Return the current prompt preview body."""
        return self._prompt_preview.toPlainText()

    def _populate_prompt_combo(self) -> None:
        if not self._prompt_options:
            self._prompt_combo.addItem("Enter prompt ID…")
            return
        for prompt in self._prompt_options:
            label = f"{prompt.name} ({prompt.id})"
            self._prompt_combo.addItem(label, str(prompt.id))

    def _load_step(self, step: PromptChainStep) -> None:
        self._order_input.setValue(step.order_index)
        self._select_prompt(step.prompt_id)
        self._stop_checkbox.setChecked(step.stop_on_failure)

    def _select_prompt(self, prompt_id: uuid.UUID) -> None:
        target = str(prompt_id)
        for index in range(self._prompt_combo.count()):
            if str(self._prompt_combo.itemData(index)) == target:
                self._prompt_combo.setCurrentIndex(index)
                return
        fallback_label = f"Unknown ({target})"
        self._prompt_combo.addItem(fallback_label, target)
        self._prompt_combo.setCurrentIndex(self._prompt_combo.count() - 1)

    def _update_prompt_preview(self) -> None:
        prompt_id_text = self._resolve_prompt_uuid_text(silent=True)
        if prompt_id_text is None:
            self._prompt_preview.setPlainText("Select a prompt to preview its body.")
            return
        prompt = next(
            (entry for entry in self._prompt_options if str(entry.id) == prompt_id_text),
            None,
        )
        if prompt is None:
            self._prompt_preview.setPlainText(
                f"Prompt ID: {prompt_id_text}\n\nPrompt body not available in the current catalog."
            )
            return
        body = (prompt.context or prompt.description or "").strip()
        if len(body) > 1200:
            body = body[:1197].rstrip() + "…"
        preview_parts = [f"Prompt: {prompt.name}"]
        if prompt.category:
            preview_parts.append(f"Category: {prompt.category}")
        preview_parts.append("")
        preview_parts.append(body or "Prompt has no saved body.")
        self._prompt_preview.setPlainText("\n".join(preview_parts))

    def _handle_accept(self) -> None:
        prompt_uuid_text = self._resolve_prompt_uuid_text()
        if prompt_uuid_text is None:
            return
        try:
            prompt_id = uuid.UUID(prompt_uuid_text)
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid prompt id", str(exc))
            return
        order_index = self._order_input.value()
        step_id = self._step.id if self._step else uuid.uuid4()
        self._result_step = PromptChainStep(
            id=step_id,
            chain_id=self._chain_id,
            prompt_id=prompt_id,
            order_index=self._order_input.value(),
            input_template="",
            output_variable=f"step_{order_index}",
            condition=None,
            stop_on_failure=self._stop_checkbox.isChecked(),
            metadata=self._step.metadata if self._step else None,
        )
        self.accept()

    def _resolve_prompt_uuid_text(self, *, silent: bool = False) -> str | None:
        data = self._prompt_combo.currentData(Qt.ItemDataRole.UserRole)
        if data:
            return str(data)
        text = self._prompt_combo.currentText().strip()
        if not text:
            if not silent:
                QMessageBox.warning(self, "Validation", "Select a prompt for this step.")
            return None
        return text

    def _add_step_row(
        self,
        layout: QFormLayout,
        label_text: str,
        widget: QWidget,
        tooltip: str,
    ) -> None:
        label = QLabel(label_text, self)
        label.setToolTip(tooltip)
        label.setBuddy(widget)
        layout.addRow(label, widget)


__all__ = ["PromptChainEditorDialog", "PromptChainStepDialog"]
