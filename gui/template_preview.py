"""Workspace template preview widget with live variable validation.

Updates: v0.1.0 - 2025-11-25 - Add dynamic Jinja2 preview with custom filters and schema validation.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Set

from jinja2 import TemplateSyntaxError
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core.templating import SchemaValidationMode, SchemaValidator, TemplateRenderer


class TemplatePreviewWidget(QWidget):
    """Provide a JSON-driven variables editor with live Jinja2 previews."""

    run_requested = Signal(str, dict)

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
        self._variable_inputs: Dict[str, QLineEdit] = {}
        self._template_parse_error: Optional[str] = None
        self._schema_visible = False
        self._preview_ready = False
        self._last_rendered_text: str = ""
        self._run_enabled = False
        self._build_ui()
        self._update_preview()

    def set_template(self, template_text: str) -> None:
        """Load a new template and refresh the preview state."""

        self._template_text = template_text or ""
        self._template_parse_error = None
        if not self._template_text.strip():
            self._variable_names = []
            self._template_hint.setText("No template selected.")
            self._update_preview()
            return
        try:
            self._variable_names = self._renderer.extract_variables(self._template_text)
            status = f"Detected {len(self._variable_names)} variable(s)."
        except TemplateSyntaxError as exc:
            self._variable_names = []
            self._template_parse_error = (
                f"Template syntax error on line {exc.lineno}: {exc.message}"
            )
            status = "Template contains syntax errors."
        self._template_hint.setText(status)
        self._rebuild_variable_inputs()
        self._update_preview()

    def clear_template(self) -> None:
        """Reset the widget to an empty template state."""

        self.set_template("")

    def variables_payload(self) -> Mapping[str, object]:
        """Return the current variables dictionary assembled from form inputs."""

        return self._collect_variables()

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

        editors_layout = QHBoxLayout()
        editors_layout.setSpacing(12)

        variables_column = QVBoxLayout()
        variables_label = QLabel("Detected variables", frame)
        variables_column.addWidget(variables_label)
        self._variables_scroll = QScrollArea(frame)
        self._variables_scroll.setWidgetResizable(True)
        self._variables_widget = QWidget(self._variables_scroll)
        self._variables_layout = QVBoxLayout(self._variables_widget)
        self._variables_layout.setContentsMargins(0, 0, 0, 0)
        self._variables_layout.setSpacing(6)
        self._variables_scroll.setWidget(self._variables_widget)
        variables_column.addWidget(self._variables_scroll, 1)
        editors_layout.addLayout(variables_column, stretch=1)

        schema_column = QVBoxLayout()
        schema_header = QHBoxLayout()
        self._schema_toggle = QPushButton("Show Schema", frame)
        self._schema_toggle.setCheckable(True)
        self._schema_toggle.setChecked(False)
        self._schema_toggle.clicked.connect(self._toggle_schema_visibility)  # type: ignore[arg-type]
        schema_header.addWidget(self._schema_toggle)
        schema_header.addStretch(1)
        self._schema_mode = QComboBox(frame)
        self._schema_mode.addItem("No validation", SchemaValidationMode.NONE.value)
        self._schema_mode.addItem("JSON Schema", SchemaValidationMode.JSON_SCHEMA.value)
        self._schema_mode.addItem("Pydantic (derived)", SchemaValidationMode.PYDANTIC.value)
        self._schema_mode.currentIndexChanged.connect(self._update_preview)  # type: ignore[arg-type]
        self._schema_mode.setVisible(False)
        schema_header.addWidget(self._schema_mode)
        schema_column.addLayout(schema_header)
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
        self._schema_input.setMinimumHeight(140)
        self._schema_input.textChanged.connect(self._update_preview)  # type: ignore[arg-type]
        self._schema_input.setVisible(False)
        schema_column.addWidget(self._schema_input)
        editors_layout.addLayout(schema_column, stretch=1)

        frame_layout.addLayout(editors_layout)

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

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self._run_button = QPushButton("Run Prompt", frame)
        self._run_button.setEnabled(False)
        self._run_button.clicked.connect(self._on_run_clicked)  # type: ignore[arg-type]
        button_row.addWidget(self._run_button)
        frame_layout.addLayout(button_row)

        layout.addWidget(frame)

    def _rebuild_variable_inputs(self) -> None:
        if not hasattr(self, "_variables_layout"):
            return
        while self._variables_layout.count():
            item = self._variables_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._variable_inputs.clear()
        if not self._variable_names:
            hint = QLabel("No placeholders detected in the prompt body.", self._variables_widget)
            hint.setStyleSheet("color: #6b7280;")
            self._variables_layout.addWidget(hint)
            self._variables_layout.addStretch(1)
            return
        for name in self._variable_names:
            container = QWidget(self._variables_widget)
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(2)
            label = QLabel(name, container)
            label.setStyleSheet("font-weight: 500;")
            field = QLineEdit(container)
            field.setPlaceholderText(f"Enter value for {name}…")
            field.textChanged.connect(self._update_preview)  # type: ignore[arg-type]
            container_layout.addWidget(label)
            container_layout.addWidget(field)
            self._variables_layout.addWidget(container)
            self._variable_inputs[name] = field
        self._variables_layout.addStretch(1)

    def _collect_variables(self) -> Dict[str, str]:
        values: Dict[str, str] = {}
        for name, widget in self._variable_inputs.items():
            text = widget.text().strip()
            if text:
                values[name] = text
        return values

    def _update_preview(self) -> None:
        self._last_rendered_text = ""
        self._preview_ready = False
        if not self._template_text.strip():
            self._update_variable_states(set(), set(), set())
            self._rendered_view.clear()
            self._set_status("Select a prompt to enable template previews.", is_error=False)
            self._refresh_run_button_state()
            return

        if self._template_parse_error:
            self._rendered_view.clear()
            self._show_message_item(self._template_parse_error, self._ERROR_COLOR)
            self._set_status(self._template_parse_error, is_error=True)
            self._refresh_run_button_state()
            return

        variables = self._collect_variables()

        schema_mode = SchemaValidationMode.from_string(
            self._schema_mode.currentData()
            if self._schema_visible
            else SchemaValidationMode.NONE.value
        )
        schema_text = self._schema_input.toPlainText() if self._schema_visible else ""
        schema_result = self._validator.validate(variables, schema_text, mode=schema_mode)
        invalid_fields: Set[str] = self._top_level_fields(schema_result.field_errors)
        if not schema_result.is_valid:
            message = "; ".join(
                schema_result.errors or [schema_result.schema_error or "Schema error"]
            )
            self._rendered_view.clear()
            self._update_variable_states(set(variables.keys()), set(), invalid_fields)
            self._set_status(message, is_error=True)
            self._refresh_run_button_state()
            return

        render_result = self._renderer.render(self._template_text, variables)
        missing = set(render_result.missing_variables)
        for name in self._variable_names:
            if name not in variables:
                missing.add(name)

        self._update_variable_states(set(variables.keys()), missing, invalid_fields)

        if render_result.errors:
            if missing:
                self._rendered_view.setPlainText(self._template_text)
            else:
                self._rendered_view.clear()
            self._set_status("; ".join(render_result.errors), is_error=True)
            self._refresh_run_button_state()
            return

        self._rendered_view.setPlainText(render_result.rendered_text)
        self._last_rendered_text = render_result.rendered_text
        if missing:
            self._set_status(
                f"Missing variables: {', '.join(sorted(missing))}",
                is_error=True,
            )
            self._refresh_run_button_state()
            return

        if invalid_fields:
            self._set_status(
                f"Fields failing validation: {', '.join(sorted(invalid_fields))}",
                is_error=True,
            )
            self._refresh_run_button_state()
            return

        self._preview_ready = True
        self._set_status("Preview ready.", is_error=False)
        self._refresh_run_button_state()

    def _toggle_schema_visibility(self) -> None:
        self._schema_visible = self._schema_toggle.isChecked()
        self._schema_toggle.setText("Hide Schema" if self._schema_visible else "Show Schema")
        self._schema_mode.setVisible(self._schema_visible)
        self._schema_input.setVisible(self._schema_visible)
        self._update_preview()

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

    def _show_message_item(self, text: str, color: str) -> None:
        self._variables_list.clear()
        item = QListWidgetItem(text)
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

    def set_run_enabled(self, enabled: bool) -> None:
        """Enable or disable the Run Prompt button based on executor availability."""

        self._run_enabled = bool(enabled)
        self._refresh_run_button_state()

    def _refresh_run_button_state(self) -> None:
        if not hasattr(self, "_run_button"):
            return
        can_run = (
            self._run_enabled
            and self._preview_ready
            and bool(self._last_rendered_text.strip())
        )
        self._run_button.setEnabled(can_run)

    def _on_run_clicked(self) -> None:
        if not (self._preview_ready and self._last_rendered_text.strip()):
            self._set_status("Preview must be ready before running.", is_error=True)
            return
        variables = self._collect_variables()
        self.run_requested.emit(self._last_rendered_text, dict(variables))


__all__ = ["TemplatePreviewWidget"]
