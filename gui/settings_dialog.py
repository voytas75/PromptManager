"""Settings dialog for configuring Prompt Manager runtime options.

Updates: v0.2.5 - 2025-11-05 - Add chat appearance controls to settings dialog.
Updates: v0.2.4 - 2025-11-05 - Add LiteLLM routing matrix for fast vs inference models.
Updates: v0.2.3 - 2025-11-05 - Introduce LiteLLM inference model field and tabbed layout.
Updates: v0.2.2 - 2025-11-26 - Add LiteLLM streaming toggle to runtime settings UI.
Updates: v0.2.1 - 2025-11-17 - Remove catalogue path configuration; imports now require explicit selection.
Updates: v0.2.0 - 2025-11-16 - Apply palette-aware border styling to match main window chrome.
Updates: v0.1.1 - 2025-11-15 - Avoid persisting LiteLLM API secrets to disk.
Updates: v0.1.0 - 2025-11-04 - Initial settings dialog implementation.
"""

from __future__ import annotations

import json
from typing import Mapping, Optional, Sequence

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QFrame,
    QMessageBox,
    QPlainTextEdit,
    QCheckBox,
    QVBoxLayout,
    QTabWidget,
    QButtonGroup,
    QRadioButton,
    QWidget,
)

from config.persistence import persist_settings_to_config
from config import DEFAULT_CHAT_USER_BUBBLE_COLOR, LITELLM_ROUTED_WORKFLOWS


class SettingsDialog(QDialog):
    """Modal dialog enabling users to configure catalogue and LiteLLM options."""

    def __init__(
        self,
        parent=None,
        *,
        litellm_model: Optional[str] = None,
        litellm_inference_model: Optional[str] = None,
        litellm_api_key: Optional[str] = None,
        litellm_api_base: Optional[str] = None,
        litellm_api_version: Optional[str] = None,
        litellm_drop_params: Optional[Sequence[str]] = None,
        litellm_reasoning_effort: Optional[str] = None,
        litellm_stream: Optional[bool] = None,
        litellm_workflow_models: Optional[Mapping[str, str]] = None,
        quick_actions: Optional[list[dict[str, object]]] = None,
        chat_user_bubble_color: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Prompt Manager Settings")
        self.setMinimumWidth(860)
        self._litellm_model = litellm_model or ""
        self._litellm_inference_model = litellm_inference_model or ""
        self._litellm_api_key = litellm_api_key or ""
        self._litellm_api_base = litellm_api_base or ""
        self._litellm_api_version = litellm_api_version or ""
        self._litellm_drop_params = ", ".join(litellm_drop_params) if litellm_drop_params else ""
        self._litellm_reasoning_effort = (litellm_reasoning_effort or "").strip()
        self._litellm_stream = bool(litellm_stream)
        original_actions = [dict(entry) for entry in (quick_actions or []) if isinstance(entry, dict)]
        self._original_quick_actions = original_actions
        self._quick_actions_value: Optional[list[dict[str, object]]] = original_actions or None
        self._workflow_models: dict[str, str] = {}
        if litellm_workflow_models:
            for key, value in litellm_workflow_models.items():
                key_str = str(key).strip()
                if key_str not in LITELLM_ROUTED_WORKFLOWS:
                    continue
                if value is None:
                    continue
                choice = str(value).strip().lower()
                if choice == "inference":
                    self._workflow_models[key_str] = "inference"
        self._workflow_groups: dict[str, QButtonGroup] = {}
        self._default_chat_color = DEFAULT_CHAT_USER_BUBBLE_COLOR
        initial_chat_color = (chat_user_bubble_color or "").strip() or self._default_chat_color
        if not QColor(initial_chat_color).isValid():
            initial_chat_color = self._default_chat_color
        self._chat_user_bubble_color = QColor(initial_chat_color).name().lower()
        self._chat_color_input: Optional[QLineEdit] = None
        self._chat_color_preview: Optional[QLabel] = None
        self._build_ui()

    def _build_ui(self) -> None:
        palette = self.palette()
        window_color = palette.color(QPalette.Window)
        border_color = QColor(
            255 - window_color.red(),
            255 - window_color.green(),
            255 - window_color.blue(),
        )
        border_color.setAlpha(255)

        outer_layout = QVBoxLayout(self)
        container = QFrame(self)
        container.setObjectName("settingsContainer")
        container.setStyleSheet(
            "#settingsContainer { "
            f"border: 1px solid {border_color.name()}; "
            "border-radius: 8px; background-color: palette(base); }"
        )
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        outer_layout.addWidget(container)

        tab_widget = QTabWidget(self)
        tab_widget.setObjectName("settingsTabs")
        layout.addWidget(tab_widget)

        litellm_tab = QWidget(self)
        litellm_form = QFormLayout(litellm_tab)
        litellm_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self._model_input = QLineEdit(self._litellm_model, litellm_tab)
        litellm_form.addRow("LiteLLM fast model", self._model_input)

        self._inference_model_input = QLineEdit(self._litellm_inference_model, litellm_tab)
        litellm_form.addRow("LiteLLM inference model", self._inference_model_input)

        self._api_key_input = QLineEdit(self._litellm_api_key, litellm_tab)
        self._api_key_input.setEchoMode(QLineEdit.Password)
        litellm_form.addRow("LiteLLM API key", self._api_key_input)

        self._api_base_input = QLineEdit(self._litellm_api_base, litellm_tab)
        litellm_form.addRow("LiteLLM API base", self._api_base_input)

        self._api_version_input = QLineEdit(self._litellm_api_version, litellm_tab)
        litellm_form.addRow("LiteLLM API version", self._api_version_input)

        self._drop_params_input = QLineEdit(self._litellm_drop_params, litellm_tab)
        self._drop_params_input.setPlaceholderText("max_tokens, temperature")
        litellm_form.addRow("LiteLLM drop params", self._drop_params_input)

        self._reasoning_effort_input = QLineEdit(self._litellm_reasoning_effort, litellm_tab)
        self._reasoning_effort_input.setPlaceholderText("minimal / medium / high")
        litellm_form.addRow("LiteLLM reasoning effort", self._reasoning_effort_input)

        self._stream_checkbox = QCheckBox("Enable streaming responses", litellm_tab)
        self._stream_checkbox.setChecked(self._litellm_stream)
        litellm_form.addRow("LiteLLM streaming", self._stream_checkbox)

        tab_widget.addTab(litellm_tab, "LiteLLM")

        routing_tab = QWidget(self)
        routing_layout = QVBoxLayout(routing_tab)
        routing_layout.setContentsMargins(0, 0, 0, 0)
        routing_hint = QLabel(
            "Assign each workflow to the fast LiteLLM model or the inference model.",
            routing_tab,
        )
        routing_hint.setWordWrap(True)
        routing_layout.addWidget(routing_hint)

        matrix_layout = QGridLayout()
        matrix_layout.setContentsMargins(0, 8, 0, 0)
        matrix_layout.setHorizontalSpacing(16)
        matrix_layout.setVerticalSpacing(6)

        header_workflow = QLabel("Workflow", routing_tab)
        header_fast = QLabel("Fast model", routing_tab)
        header_inference = QLabel("Inference model", routing_tab)
        header_workflow.setObjectName("routingHeaderWorkflow")
        header_fast.setObjectName("routingHeaderFast")
        header_inference.setObjectName("routingHeaderInference")
        matrix_layout.addWidget(header_workflow, 0, 0)
        matrix_layout.addWidget(header_fast, 0, 1)
        matrix_layout.addWidget(header_inference, 0, 2)

        for row_index, (workflow_key, workflow_label) in enumerate(
            LITELLM_ROUTED_WORKFLOWS.items(), start=1
        ):
            label = QLabel(workflow_label, routing_tab)
            matrix_layout.addWidget(label, row_index, 0)

            fast_button = QRadioButton("Fast", routing_tab)
            fast_button.setProperty("routeChoice", "fast")
            inference_button = QRadioButton("Inference", routing_tab)
            inference_button.setProperty("routeChoice", "inference")

            group = QButtonGroup(self)
            group.setExclusive(True)
            group.addButton(fast_button)
            group.addButton(inference_button)
            self._workflow_groups[workflow_key] = group

            selected_choice = self._workflow_models.get(workflow_key, "fast")
            if selected_choice == "inference":
                inference_button.setChecked(True)
            else:
                fast_button.setChecked(True)

            matrix_layout.addWidget(fast_button, row_index, 1)
            matrix_layout.addWidget(inference_button, row_index, 2)

        matrix_layout.setColumnStretch(0, 2)
        matrix_layout.setColumnStretch(1, 1)
        matrix_layout.setColumnStretch(2, 1)

        routing_layout.addLayout(matrix_layout)
        routing_layout.addStretch(1)

        tab_widget.addTab(routing_tab, "Routing")

        appearance_tab = QWidget(self)
        appearance_form = QFormLayout(appearance_tab)
        appearance_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        chat_color_input = QLineEdit(self._chat_user_bubble_color, appearance_tab)
        chat_color_input.setPlaceholderText(self._default_chat_color)
        chat_color_input.textChanged.connect(self._update_chat_color_preview)  # type: ignore[arg-type]
        self._chat_color_input = chat_color_input
        appearance_form.addRow("User chat colour", chat_color_input)

        preview_label = QLabel("You: Example message", appearance_tab)
        preview_label.setWordWrap(True)
        self._chat_color_preview = preview_label
        appearance_form.addRow("Preview", preview_label)
        self._update_chat_color_preview(self._chat_user_bubble_color)

        tab_widget.addTab(appearance_tab, "Appearance")

        quick_tab = QWidget(self)
        quick_layout = QVBoxLayout(quick_tab)
        quick_layout.setContentsMargins(0, 0, 0, 0)
        quick_actions_label = QLabel("Quick actions (JSON)", quick_tab)
        quick_layout.addWidget(quick_actions_label)

        self._quick_actions_input = QPlainTextEdit(quick_tab)
        self._quick_actions_input.setPlaceholderText(
            "Paste JSON array of quick action definitions (identifier, title, description, optional hints)."
        )
        if self._original_quick_actions:
            try:
                pretty = json.dumps(self._original_quick_actions, indent=2, ensure_ascii=False)
            except TypeError:
                pretty = ""
            self._quick_actions_input.setPlainText(pretty)
        quick_layout.addWidget(self._quick_actions_input)

        tab_widget.addTab(quick_tab, "Quick Actions")

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        button_box.accepted.connect(self.accept)  # type: ignore[arg-type]
        button_box.rejected.connect(self.reject)  # type: ignore[arg-type]
        layout.addWidget(button_box)

    def _update_chat_color_preview(self, value: str) -> None:
        """Refresh the preview bubble based on the current colour entry."""

        if self._chat_color_preview is None:
            return
        text = (value or "").strip()
        color = QColor(text) if text else QColor(self._default_chat_color)
        if not color.isValid():
            self._chat_color_preview.setStyleSheet(
                "background-color: #fff5f5; border: 1px solid #d14343; "
                "border-radius: 8px; padding: 8px; color: #d14343;"
            )
            self._chat_color_preview.setText("Invalid colour")
            return
        normalized = color.name().lower()
        border = QColor(normalized).darker(115).name()
        self._chat_color_preview.setStyleSheet(
            f"background-color: {normalized}; border: 1px solid {border}; "
            "border-radius: 8px; padding: 8px; color: #1f2933;"
        )
        self._chat_color_preview.setText("You: Example message")

    def accept(self) -> None:
        text = self._quick_actions_input.toPlainText().strip()
        if text:
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                QMessageBox.critical(self, "Invalid quick actions", f"JSON parse error: {exc}")
                return
            if not isinstance(data, list):
                QMessageBox.critical(
                    self,
                    "Invalid quick actions",
                    "Quick actions must be provided as a JSON array of objects.",
                )
                return
            for entry in data:
                if not isinstance(entry, dict):
                    QMessageBox.critical(
                        self,
                        "Invalid quick actions",
                        "Each quick action entry must be a JSON object.",
                    )
                    return
            self._quick_actions_value = data
        else:
            self._quick_actions_value = None

        color_text = self._chat_color_input.text().strip() if self._chat_color_input else ""
        if color_text:
            candidate = QColor(color_text)
            if not candidate.isValid():
                QMessageBox.critical(
                    self,
                    "Invalid chat colour",
                    "Enter a valid colour value (e.g. #e6f0ff) or leave the field blank.",
                )
                return
            self._chat_user_bubble_color = candidate.name().lower()
        else:
            self._chat_user_bubble_color = self._default_chat_color

        super().accept()

    def result_settings(self) -> dict[str, Optional[object]]:
        """Return cleaned settings data."""

        def _clean(value: str) -> Optional[str]:
            stripped = value.strip()
            return stripped or None

        drop_text = self._drop_params_input.text().strip()
        drop_params = [item.strip() for item in drop_text.split(",") if item.strip()] if drop_text else None

        workflow_models: dict[str, str] = {}
        for key, group in self._workflow_groups.items():
            button = group.checkedButton()
            if button is None:
                continue
            choice = str(button.property("routeChoice") or "").strip().lower()
            if choice == "inference":
                workflow_models[key] = "inference"

        return {
            "litellm_model": _clean(self._model_input.text()),
            "litellm_inference_model": _clean(self._inference_model_input.text()),
            "litellm_api_key": _clean(self._api_key_input.text()),
            "litellm_api_base": _clean(self._api_base_input.text()),
            "litellm_api_version": _clean(self._api_version_input.text()),
            "litellm_drop_params": drop_params,
            "litellm_reasoning_effort": _clean(self._reasoning_effort_input.text()),
            "litellm_stream": self._stream_checkbox.isChecked(),
            "litellm_workflow_models": workflow_models or None,
            "quick_actions": self._quick_actions_value,
            "chat_user_bubble_color": self._chat_user_bubble_color,
        }


__all__ = ["SettingsDialog", "persist_settings_to_config"]
