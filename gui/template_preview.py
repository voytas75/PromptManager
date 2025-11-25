"""Workspace template preview widget with live variable validation.

Updates: v0.1.0 - 2025-11-25 - Add dynamic Jinja2 preview with custom filters and schema validation.
"""

from __future__ import annotations

import json
from typing import Dict, List, Mapping, Optional, Sequence, Set

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.templating import SchemaValidationMode, SchemaValidator, TemplateRenderer


class TemplatePreviewWidget(QWidget):
    """Provide a JSON-driven variables editor with live Jinja2 previews."""

    _SUCCESS_COLOR = "#047857"
    _ERROR_COLOR = "#b91c1c"
    _WARNING_COLOR = "#b45309"
    _INFO_COLOR = "#6b7280"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._renderer = TemplateRenderer()
        self._validator = SchemaValidator()
        self._template_text: str = ""
        self._variable_names: List[str] = []
        self._build_ui()
        self._update_preview()

    def set_template(self, template_text: str) -> None:
        """Load a new template and refresh the preview state."""

        self._template_text = template_text or ""
        self._variable_names = self._renderer.extract_variables(self._template_text)
        status = (
            "No template selected."
            if not self._template_text.strip()
            else f"Detected {len(self._variable_names)} variable(s)."
        )
        self._template_hint.setText(status)
        self.setDisabled(not bool(self._template_text.strip()))
        if not self._template_text.strip():
            self._rendered_view.setPlainText("Select a prompt to enable the preview.")
        self._update_preview()

    def clear_template(self) -> None:
        """Reset the widget to an empty template state."""

        self.set_template("")

    def variables_payload(self) -> Mapping[str, object]:
        """Return the current variables dictionary when JSON parsing succeeds."""

        variables, error = self._parse_variables()
        if error:
            return {}
        return variables

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        frame = QFrame(self)
        frame.setObjectName("templatePreviewFrame")
        frame.setFrameShape(QFrame.StyledPanel)
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(8, 8, 8, 8)
        frame_layout.setSpacing(8)

        header = QLabel("Template Preview", frame)
        header.setObjectName("templatePreviewTitle")
        header.setStyleSheet("font-weight: 600;")
        frame_layout.addWidget(header)

        self._template_hint = QLabel("No template selected.", frame)
        self._template_hint.setWordWrap(True)
        frame_layout.addWidget(self._template_hint)

        variables_label = QLabel("Variables (JSON)", frame)
        frame_layout.addWidget(variables_label)

        self._variables_input = QPlainTextEdit(frame)
        self._variables_input.setPlaceholderText('{"customer": "Ada", "issue": "Payment failed"}')
        self._variables_input.setFixedHeight(110)
        self._variables_input.textChanged.connect(self._update_preview)  # type: ignore[arg-type]
        frame_layout.addWidget(self._variables_input)

        schema_header = QHBoxLayout()
        schema_label = QLabel("Schema (optional)", frame)
        schema_header.addWidget(schema_label)
        self._schema_mode = QComboBox(frame)
        self._schema_mode.addItem("No validation", SchemaValidationMode.NONE.value)
        self._schema_mode.addItem("JSON Schema", SchemaValidationMode.JSON_SCHEMA.value)
        self._schema_mode.addItem("Pydantic (derived)", SchemaValidationMode.PYDANTIC.value)
        self._schema_mode.currentIndexChanged.connect(self._update_preview)  # type: ignore[arg-type]
        schema_header.addStretch(1)
        schema_header.addWidget(self._schema_mode)
        frame_layout.addLayout(schema_header)

        self._schema_input = QPlainTextEdit(frame)
        self._schema_input.setPlaceholderText(
            "{\n"
            '  "type": "object",\n'
            '  "properties": {\n'
            '    "customer": {"type": "string"}\n'
            "  },\n"
            '  "required": ["customer"]\n'
            "}"
        )
        self._schema_input.setFixedHeight(110)
        self._schema_input.textChanged.connect(self._update_preview)  # type: ignore[arg-type]
        frame_layout.addWidget(self._schema_input)

        self._variables_list = QListWidget(frame)
        self._variables_list.setObjectName("templatePreviewVariables")
        self._variables_list.setMaximumHeight(120)
        self._variables_list.setAlternatingRowColors(True)
        frame_layout.addWidget(self._variables_list)

        preview_label = QLabel("Rendered preview", frame)
        frame_layout.addWidget(preview_label)

        self._rendered_view = QPlainTextEdit(frame)
        self._rendered_view.setReadOnly(True)
        self._rendered_view.setPlaceholderText("Supply variables to render the prompt preview…")
        self._rendered_view.setMinimumHeight(140)
        frame_layout.addWidget(self._rendered_view)

        self._status_label = QLabel("", frame)
        self._status_label.setWordWrap(True)
        frame_layout.addWidget(self._status_label)

        layout.addWidget(frame)

    def _parse_variables(self) -> tuple[Dict[str, object], Optional[str]]:
        text = self._variables_input.toPlainText().strip()
        if not text:
            return {}, None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            return {}, f"Invalid variables JSON: {exc.msg}"
        if not isinstance(payload, Mapping):
            return {}, "Variables payload must be a JSON object."
        return dict(payload), None

    def _update_preview(self) -> None:
        if not self._template_text.strip():
            self._update_variable_states(set(), set(self._variable_names), set())
            self._set_status("Select a prompt to enable template previews.", is_error=False)
            return

        variables, parse_error = self._parse_variables()
        if parse_error:
            self._rendered_view.clear()
            self._update_variable_states(set(), set(self._variable_names), set())
            self._set_status(parse_error, is_error=True)
            return

        schema_mode = SchemaValidationMode.from_string(self._schema_mode.currentData())
        schema_text = self._schema_input.toPlainText()
        schema_result = self._validator.validate(variables, schema_text, mode=schema_mode)
        errors: List[str] = []
        invalid_fields: Set[str] = self._top_level_fields(schema_result.field_errors)
        if not schema_result.is_valid:
            if schema_result.errors:
                errors.extend(schema_result.errors)
            if schema_result.schema_error and not schema_result.errors:
                errors.append(schema_result.schema_error)

        render_result = self._renderer.render(self._template_text, variables)
        if render_result.errors:
            errors.extend(render_result.errors)

        missing = set(render_result.missing_variables)
        for name in self._variable_names:
            if name not in variables:
                missing.add(name)

        self._update_variable_states(set(variables.keys()), missing, invalid_fields)

        if errors:
            self._rendered_view.clear()
            self._set_status("; ".join(errors), is_error=True)
            return

        self._rendered_view.setPlainText(render_result.rendered_text)
        if missing:
            self._set_status(
                f"Missing variables: {', '.join(sorted(missing))}",
                is_error=True,
            )
            return

        if invalid_fields:
            self._set_status(
                f"Fields failing validation: {', '.join(sorted(invalid_fields))}",
                is_error=True,
            )
            return

        self._set_status("Preview ready.", is_error=False)

    def _update_variable_states(
        self,
        provided: Set[str],
        missing: Set[str],
        invalid: Set[str],
    ) -> None:
        self._variables_list.clear()
        if not self._variable_names:
            item = QListWidgetItem("No placeholders detected in the prompt body.")
            item.setForeground(QColor(self._INFO_COLOR))
            item.setFlags(Qt.ItemIsEnabled)
            self._variables_list.addItem(item)
            return

        for name in self._variable_names:
            if name in invalid:
                status = "Invalid"
                color = self._WARNING_COLOR
            elif name in missing:
                status = "Missing"
                color = self._ERROR_COLOR
            elif name in provided:
                status = "Ready"
                color = self._SUCCESS_COLOR
            else:
                status = "Unused"
                color = self._INFO_COLOR
            item = QListWidgetItem(f"{name} — {status}")
            item.setForeground(QColor(color))
            item.setFlags(Qt.ItemIsEnabled)
            self._variables_list.addItem(item)

    def _top_level_fields(self, field_paths: Sequence[str]) -> Set[str]:
        invalid: Set[str] = set()
        for path in field_paths:
            if not path:
                continue
            top_level = path.split(".", 1)[0]
            if top_level:
                invalid.add(top_level)
        return invalid

    def _set_status(self, message: str, *, is_error: bool) -> None:
        color = self._ERROR_COLOR if is_error else self._SUCCESS_COLOR
        self._status_label.setStyleSheet(f"color: {color};")
        self._status_label.setText(message)


__all__ = ["TemplatePreviewWidget"]
