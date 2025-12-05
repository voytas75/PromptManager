"""Prompt chain editor dialogs for creating and updating workflows.

Updates:
  v0.3.0 - 2025-12-05 - Surface summarize toggle for prompt chain results.
  v0.2.0 - 2025-12-04 - Add prompt picker combo box backed by catalog lookups.
  v0.1.0 - 2025-12-04 - Introduce editor dialogs for chain and step CRUD.
"""

from __future__ import annotations

import json
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
            str(prompt.id): prompt
            for prompt in self._prompts
        }
        self.setWindowTitle("Edit Prompt Chain" if chain else "New Prompt Chain")
        self.resize(800, 640)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self._name_input = QLineEdit(self)
        self._name_input.setPlaceholderText("e.g. Content Review Flow")
        self._add_form_row(form, "Name", self._name_input, "Friendly chain label shown in lists.")

        self._description_input = QPlainTextEdit(self)
        self._description_input.setPlaceholderText("Describe what the chain does…")
        self._add_form_row(
            form,
            "Description",
            self._description_input,
            "Optional summary to help collaborators understand the workflow.",
        )

        self._schema_input = QPlainTextEdit(self)
        self._schema_input.setPlaceholderText("JSON Schema (Draft 2020-12) describing variables…")
        self._add_form_row(
            form,
            "Variables schema",
            self._schema_input,
            "Validates the variables provided before execution.",
        )

        self._active_checkbox = QCheckBox("Chain is active", self)
        self._active_checkbox.setChecked(True)
        form.addRow("Active", self._active_checkbox)

        self._summarize_checkbox = QCheckBox("Summarize final step output", self)
        self._summarize_checkbox.setChecked(True)
        self._add_form_row(
            form,
            "Summarize",
            self._summarize_checkbox,
            "If enabled, the chain produces a condensed summary from the last step response.",
        )
        layout.addLayout(form)

        self._steps_table = QTableWidget(0, 5, self)
        self._steps_table.setHorizontalHeaderLabels(
            ["Order", "Prompt", "Input", "Output", "Condition"]
        )
        self._steps_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._steps_table.itemDoubleClicked.connect(self._handle_step_double_click)  # type: ignore[arg-type]
        layout.addWidget(self._steps_table, 1)

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
        step_actions.addStretch(1)
        layout.addLayout(step_actions)

        self._buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, self)
        self._buttons.accepted.connect(self._handle_accept)  # type: ignore[arg-type]
        self._buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        layout.addWidget(self._buttons)

        self._steps: list[PromptChainStep] = []
        if chain is not None:
            self._load_chain(chain)

    def result_chain(self) -> PromptChain | None:
        """Return the saved chain instance when available."""
        return self._result_chain

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _load_chain(self, chain: PromptChain) -> None:
        self._name_input.setText(chain.name)
        self._description_input.setPlainText(chain.description)
        if chain.variables_schema:
            self._schema_input.setPlainText(json.dumps(chain.variables_schema, indent=2))
        self._active_checkbox.setChecked(chain.is_active)
        self._summarize_checkbox.setChecked(chain.summarize_last_response)
        self._steps = [replace(step) for step in chain.steps]
        self._refresh_steps()

    def _refresh_steps(self) -> None:
        self._steps.sort(key=lambda step: step.order_index)
        self._steps_table.setRowCount(len(self._steps))
        for row, step in enumerate(self._steps):
            self._steps_table.setItem(row, 0, QTableWidgetItem(str(step.order_index)))
            prompt_item = QTableWidgetItem(str(step.prompt_id))
            prompt = self._prompt_lookup.get(str(step.prompt_id))
            if prompt is not None:
                prompt_item.setToolTip(self._build_prompt_tooltip(prompt))
            self._steps_table.setItem(row, 1, prompt_item)
            self._steps_table.setItem(row, 2, QTableWidgetItem(step.input_template))
            self._steps_table.setItem(row, 3, QTableWidgetItem(step.output_variable))
            condition = step.condition or "Always"
            if not step.stop_on_failure:
                condition = f"{condition} (continues on failure)"
            self._steps_table.setItem(row, 4, QTableWidgetItem(condition))

    def _add_step(self) -> None:
        dialog = PromptChainStepDialog(
            self,
            chain_id=self._chain_id,
            prompts=self._prompts,
        )
        dialog.set_order_index(len(self._steps) + 1)
        if dialog.exec() != QDialog.Accepted:
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
        if dialog.exec() != QDialog.Accepted:
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

    def _handle_accept(self) -> None:
        name = self._name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Validation", "Enter a name for the prompt chain.")
            return
        if not self._steps:
            QMessageBox.warning(self, "Validation", "Add at least one step before saving.")
            return
        schema_text = self._schema_input.toPlainText().strip()
        schema = None
        if schema_text:
            try:
                schema = json.loads(schema_text)
            except json.JSONDecodeError as exc:
                QMessageBox.critical(self, "Invalid schema", str(exc))
                return
        description = self._description_input.toPlainText().strip()
        chain = PromptChain(
            id=self._chain_id,
            name=name,
            description=description,
            is_active=self._active_checkbox.isChecked(),
            variables_schema=schema if schema else None,
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
        prompt = self._prompt_lookup.get(str(item.text()).strip())
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
        self._prompt_combo.setEditable(True)
        self._prompt_combo.setInsertPolicy(QComboBox.NoInsert)
        self._prompt_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._add_step_row(
            layout,
            "Prompt",
            self._prompt_combo,
            "Select the catalog prompt to run for this step.",
        )

        self._input_template = QPlainTextEdit(self)
        self._input_template.setPlaceholderText("e.g. Questions: {{ extracted_points }}")
        self._add_step_row(
            layout,
            "Input template",
            self._input_template,
            "Jinja template rendered with chain variables for the prompt body.",
        )
        self._input_hint = QLabel(
            "Jinja-style template that receives chain variables (e.g. {{ source_text }})",
            self,
        )
        self._input_hint.setWordWrap(True)
        self._input_hint.setObjectName("inputTemplateHint")
        layout.addRow("", self._input_hint)

        self._output_variable = QLineEdit(self)
        self._output_variable.setPlaceholderText("e.g. review_summary")
        self._add_step_row(
            layout,
            "Output variable",
            self._output_variable,
            "Name stored in context for later steps (e.g. summary_text).",
        )

        self._condition_input = QLineEdit(self)
        self._condition_input.setPlaceholderText("Optional condition, e.g. {{ score > 0 }}")
        self._add_step_row(
            layout,
            "Condition",
            self._condition_input,
            "If this renders to false/empty the step is skipped.",
        )

        self._stop_checkbox = QCheckBox("Stop chain when this step fails", self)
        self._stop_checkbox.setChecked(True)
        self._add_step_row(
            layout,
            "Stop on failure",
            self._stop_checkbox,
            "If unchecked the chain continues even when this step errors.",
        )

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self._handle_accept)  # type: ignore[arg-type]
        buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        layout.addWidget(buttons)

        self._populate_prompt_combo()
        if step is not None:
            self._load_step(step)

    def set_order_index(self, value: int) -> None:
        """Update the default order index displayed to the user."""
        self._order_input.setValue(max(1, value))

    def result_step(self) -> PromptChainStep | None:
        """Return the saved step when available."""
        return self._result_step

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
        self._input_template.setPlainText(step.input_template)
        self._output_variable.setText(step.output_variable)
        self._condition_input.setText(step.condition or "")
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

    def _handle_accept(self) -> None:
        prompt_uuid_text = self._resolve_prompt_uuid_text()
        if prompt_uuid_text is None:
            return
        try:
            prompt_id = uuid.UUID(prompt_uuid_text)
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid prompt id", str(exc))
            return
        input_template = self._input_template.toPlainText().strip()
        if not input_template:
            QMessageBox.warning(self, "Validation", "Enter an input template for the step.")
            return
        order_index = self._order_input.value()
        output_variable = self._output_variable.text().strip() or f"step_{order_index}"
        condition = self._condition_input.text().strip() or None
        step_id = self._step.id if self._step else uuid.uuid4()
        self._result_step = PromptChainStep(
            id=step_id,
            chain_id=self._chain_id,
            prompt_id=prompt_id,
            order_index=self._order_input.value(),
            input_template=input_template,
            output_variable=output_variable,
            condition=condition,
            stop_on_failure=self._stop_checkbox.isChecked(),
            metadata=self._step.metadata if self._step else None,
        )
        self.accept()

    def _resolve_prompt_uuid_text(self) -> str | None:
        data = self._prompt_combo.currentData(Qt.UserRole)
        if data:
            return str(data)
        text = self._prompt_combo.currentText().strip()
        if not text:
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
