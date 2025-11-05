"""Settings dialog for configuring Prompt Manager runtime options.

Updates: v0.2.3 - 2025-11-05 - Introduce LiteLLM inference model field and tabbed layout.
Updates: v0.2.2 - 2025-11-26 - Add LiteLLM streaming toggle to runtime settings UI.
Updates: v0.2.1 - 2025-11-17 - Remove catalogue path configuration; imports now require explicit selection.
Updates: v0.2.0 - 2025-11-16 - Apply palette-aware border styling to match main window chrome.
Updates: v0.1.1 - 2025-11-15 - Avoid persisting LiteLLM API secrets to disk.
Updates: v0.1.0 - 2025-11-04 - Initial settings dialog implementation.
"""

from __future__ import annotations

import json
from typing import Optional, Sequence

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QFrame,
    QMessageBox,
    QPlainTextEdit,
    QCheckBox,
    QVBoxLayout,
    QTabWidget,
    QWidget,
)

from config.persistence import persist_settings_to_config


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
        quick_actions: Optional[list[dict[str, object]]] = None,
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
        super().accept()

    def result_settings(self) -> dict[str, Optional[object]]:
        """Return cleaned settings data."""

        def _clean(value: str) -> Optional[str]:
            stripped = value.strip()
            return stripped or None

        drop_text = self._drop_params_input.text().strip()
        drop_params = [item.strip() for item in drop_text.split(",") if item.strip()] if drop_text else None

        return {
            "litellm_model": _clean(self._model_input.text()),
            "litellm_inference_model": _clean(self._inference_model_input.text()),
            "litellm_api_key": _clean(self._api_key_input.text()),
            "litellm_api_base": _clean(self._api_base_input.text()),
            "litellm_api_version": _clean(self._api_version_input.text()),
            "litellm_drop_params": drop_params,
            "litellm_reasoning_effort": _clean(self._reasoning_effort_input.text()),
            "litellm_stream": self._stream_checkbox.isChecked(),
            "quick_actions": self._quick_actions_value,
        }


__all__ = ["SettingsDialog", "persist_settings_to_config"]
