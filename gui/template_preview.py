"""Workspace template preview widget with live variable validation.

Updates: v0.2.0 - 2025-11-27 - Persist template variables and schema settings per prompt using QSettings.
Updates: v0.1.9 - 2025-11-27 - Expose run trigger for external shortcuts and publish run state changes.
Updates: v0.1.8 - 2025-11-27 - Move schema toggle controls above editors and collapse the schema panel when hidden.
Updates: v0.1.7 - 2025-11-27 - Make the variables/schema editors and rendered preview vertically resizable.
Updates: v0.1.6 - 2025-11-27 - Consolidate template status messaging into the footer label only.
Updates: v0.1.5 - 2025-11-27 - Persist splitter sizes for the status and preview panes.
Updates: v0.1.4 - 2025-11-27 - Add contextual hints for template syntax errors.
Updates: v0.1.3 - 2025-11-27 - Keep raw templates visible while surfacing parse/render errors.
Updates: v0.1.2 - 2025-11-27 - Make rendered prompt view resizable with a splitter.
Updates: v0.1.1 - 2025-11-27 - Capture variables with multiline editors sized to four lines.
Updates: v0.1.0 - 2025-11-25 - Add dynamic Jinja2 preview with custom filters and schema validation.
"""

from __future__ import annotations

import json
from typing import Dict, List, Mapping, Optional, Sequence, Set

from jinja2 import TemplateSyntaxError
from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core.templating import (
    SchemaValidationMode,
    SchemaValidator,
    TemplateRenderer,
    format_template_syntax_error,
)


class TemplatePreviewWidget(QWidget):
    """Provide a JSON-driven variables editor with live Jinja2 previews."""

    run_requested = Signal(str, dict)
    run_state_changed = Signal(bool)

    _SUCCESS_COLOR = "#047857"
    _ERROR_COLOR = "#b91c1c"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._renderer = TemplateRenderer()
        self._validator = SchemaValidator()
        self._template_text: str = ""
        self._variable_names: List[str] = []
        self._variable_inputs: Dict[str, QPlainTextEdit] = {}
        self._template_parse_error: Optional[str] = None
        self._schema_visible = False
        self._preview_ready = False
        self._last_rendered_text: str = ""
        self._run_enabled = False
        self._last_run_ready = False
        self._current_prompt_id: Optional[str] = None
        self._state_store = QSettings("PromptManager", "TemplatePreviewState")
        self._suspend_persist = False
        self._build_ui()
        self._update_preview()

    def set_template(self, template_text: str, prompt_id: Optional[str] = None) -> None:
        """Load a new template and refresh the preview state."""

        self._template_text = template_text or ""
        self._current_prompt_id = prompt_id
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
            self._template_parse_error = format_template_syntax_error(self._template_text, exc)
            status = "Template contains syntax errors."
        self._template_hint.setText(status)
        self._rebuild_variable_inputs()
        self._restore_persisted_state()
        self._update_preview()

    def clear_template(self) -> None:
        """Reset the widget to an empty template state."""

        self.set_template("", None)

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

        schema_controls = QHBoxLayout()
        schema_controls.setContentsMargins(0, 0, 0, 0)
        schema_controls.setSpacing(6)
        self._schema_toggle = QPushButton("Show Schema", frame)
        self._schema_toggle.setCheckable(True)
        self._schema_toggle.setChecked(False)
        self._schema_toggle.clicked.connect(self._toggle_schema_visibility)  # type: ignore[arg-type]
        schema_controls.addWidget(self._schema_toggle)
        self._schema_mode = QComboBox(frame)
        self._schema_mode.addItem("No validation", SchemaValidationMode.NONE.value)
        self._schema_mode.addItem("JSON Schema", SchemaValidationMode.JSON_SCHEMA.value)
        self._schema_mode.addItem("Pydantic (derived)", SchemaValidationMode.PYDANTIC.value)
        self._schema_mode.currentIndexChanged.connect(self._update_preview)  # type: ignore[arg-type]
        self._schema_mode.setVisible(False)
        schema_controls.addWidget(self._schema_mode)
        schema_controls.addStretch(1)
        frame_layout.addLayout(schema_controls)

        editors_container = QWidget(frame)
        editors_layout = QHBoxLayout(editors_container)
        editors_layout.setContentsMargins(0, 0, 0, 0)
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

        self._schema_panel = QWidget(frame)
        schema_column = QVBoxLayout(self._schema_panel)
        schema_column.setContentsMargins(0, 0, 0, 0)
        schema_column.setSpacing(4)
        self._schema_input = QPlainTextEdit(self._schema_panel)
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
        schema_column.addWidget(self._schema_input)
        self._schema_panel.setVisible(False)
        editors_layout.addWidget(self._schema_panel, stretch=1)

        self._content_splitter = QSplitter(Qt.Vertical, frame)
        self._content_splitter.setChildrenCollapsible(False)
        self._content_splitter.addWidget(editors_container)

        preview_container = QWidget(self._content_splitter)
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(6)

        preview_label = QLabel("Rendered preview", preview_container)
        preview_layout.addWidget(preview_label)

        self._rendered_view = QPlainTextEdit(preview_container)
        self._rendered_view.setReadOnly(True)
        self._rendered_view.setPlaceholderText("Supply variables to render the prompt preview…")
        self._rendered_view.setMinimumHeight(140)
        preview_layout.addWidget(self._rendered_view, 1)

        self._status_label = QLabel("", preview_container)
        self._status_label.setWordWrap(True)
        preview_layout.addWidget(self._status_label)

        self._content_splitter.addWidget(preview_container)
        self._content_splitter.setStretchFactor(0, 1)
        self._content_splitter.setStretchFactor(1, 1)

        frame_layout.addWidget(self._content_splitter, 1)

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
            self._variables_layout.addStretch(1)
            return
        for name in self._variable_names:
            container = QWidget(self._variables_widget)
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(2)
            label = QLabel(name, container)
            label.setStyleSheet("font-weight: 500;")
            field = QPlainTextEdit(container)
            field.setPlaceholderText(f"Enter value for {name}…")
            field.setLineWrapMode(QPlainTextEdit.WidgetWidth)
            metrics = field.fontMetrics()
            default_height = (metrics.lineSpacing() * 4) + 12
            field.setFixedHeight(default_height)
            field.textChanged.connect(self._update_preview)  # type: ignore[arg-type]
            container_layout.addWidget(label)
            container_layout.addWidget(field)
            self._variables_layout.addWidget(container)
            self._variable_inputs[name] = field
        self._variables_layout.addStretch(1)

    def _collect_variables(self) -> Dict[str, str]:
        values: Dict[str, str] = {}
        for name, widget in self._variable_inputs.items():
            text = widget.toPlainText().strip()
            if text:
                values[name] = text
        return values

    def _update_preview(self) -> None:
        self._last_rendered_text = ""
        self._preview_ready = False
        try:
            if not self._template_text.strip():
                self._rendered_view.clear()
                self._set_status("Select a prompt to enable template previews.", is_error=False)
                self._refresh_run_button_state()
                return

            if self._template_parse_error:
                self._rendered_view.setPlainText(self._template_text)
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
                self._set_status(message, is_error=True)
                self._refresh_run_button_state()
                return

            render_result = self._renderer.render(self._template_text, variables)
            missing = set(render_result.missing_variables)
            for name in self._variable_names:
                if name not in variables:
                    missing.add(name)

            if render_result.errors:
                self._rendered_view.setPlainText(self._template_text)
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
        finally:
            self._persist_state()

    def _toggle_schema_visibility(self) -> None:
        self._schema_visible = self._schema_toggle.isChecked()
        self._schema_toggle.setText("Hide Schema" if self._schema_visible else "Show Schema")
        self._schema_mode.setVisible(self._schema_visible)
        if hasattr(self, "_schema_panel"):
            self._schema_panel.setVisible(self._schema_visible)
        self._update_preview()

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
        if can_run != self._last_run_ready:
            self._last_run_ready = can_run
            self.run_state_changed.emit(can_run)

    def _on_run_clicked(self) -> None:
        self.request_run()

    def request_run(self) -> bool:
        """Emit the run signal when the preview is ready; return True on success."""

        if not (self._preview_ready and self._last_rendered_text.strip()):
            self._set_status("Preview must be ready before running.", is_error=True)
            return False
        variables = self._collect_variables()
        self.run_requested.emit(self._last_rendered_text, dict(variables))
        return True

    def _restore_persisted_state(self) -> None:
        key = self._state_key()
        if key is None:
            return
        raw_value = self._state_store.value(key)
        if not isinstance(raw_value, str):
            return
        try:
            payload = json.loads(raw_value)
        except json.JSONDecodeError:
            return
        variables = payload.get("variables")
        schema_visible = payload.get("schema_visible")
        schema_text = payload.get("schema_text")
        schema_mode_value = payload.get("schema_mode")
        self._suspend_persist = True
        try:
            if isinstance(variables, dict):
                for name, value in variables.items():
                    widget = self._variable_inputs.get(name)
                    if widget is not None:
                        widget.setPlainText(str(value))
            if isinstance(schema_text, str) and self._schema_input.toPlainText() != schema_text:
                self._schema_input.setPlainText(schema_text)
            if isinstance(schema_visible, bool) and self._schema_toggle.isChecked() != schema_visible:
                self._schema_toggle.setChecked(schema_visible)
            if isinstance(schema_mode_value, str):
                index = self._schema_mode.findData(schema_mode_value)
                if index >= 0 and index != self._schema_mode.currentIndex():
                    self._schema_mode.setCurrentIndex(index)
        finally:
            self._suspend_persist = False

    def _persist_state(self) -> None:
        if self._suspend_persist:
            return
        key = self._state_key()
        if key is None:
            return
        state = {
            "variables": self._variable_text_payload(),
            "schema_visible": self._schema_toggle.isChecked(),
            "schema_text": self._schema_input.toPlainText(),
            "schema_mode": self._schema_mode.currentData(),
        }
        self._state_store.setValue(key, json.dumps(state))

    def _state_key(self) -> Optional[str]:
        if not self._current_prompt_id:
            return None
        return f"prompt/{self._current_prompt_id}"

    def _variable_text_payload(self) -> Dict[str, str]:
        return {name: widget.toPlainText() for name, widget in self._variable_inputs.items()}

    @property
    def content_splitter(self) -> QSplitter:
        """Return the splitter controlling status and preview panes."""

        return self._content_splitter


__all__ = ["TemplatePreviewWidget"]
