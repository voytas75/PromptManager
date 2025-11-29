"""Settings dialog for configuring Prompt Manager runtime options.

Updates:
  v0.2.10 - 2025-11-29 - Wrap quick action/parameter parsing for Ruff line length.
  v0.2.9 - 2025-11-28 - Persist dialog geometry between sessions.
  v0.2.8 - 2025-11-23 - Surface editable LiteLLM prompt templates with reset controls.
  v0.2.7 - 2025-12-06 - Surface LiteLLM embedding model configuration.
  v0.2.6 - 2025-11-05 - Add theme mode toggle to the appearance settings.
  v0.2.5 - 2025-11-05 - Add chat appearance controls to settings dialog.
  v0.2.4 - 2025-11-05 - Add LiteLLM routing matrix for fast vs inference models.
  v0.2.3 - 2025-11-05 - Introduce LiteLLM inference model field and tabbed layout.
  v0.2.2 - 2025-11-26 - Add LiteLLM streaming toggle to runtime settings UI.
  v0.2.1 - 2025-11-17 - Remove catalogue path configuration; imports now require explicit selection.
  v0.2.0 - 2025-11-16 - Apply palette-aware border styling to match main window chrome.
  v0.1.1 - 2025-11-15 - Avoid persisting LiteLLM API secrets to disk.
  v0.1.0 - 2025-11-04 - Initial settings dialog implementation.
"""

from __future__ import annotations

import json
from functools import partial
from typing import TYPE_CHECKING

from PySide6.QtCore import QEvent, QSettings, Qt
from PySide6.QtGui import QColor, QGuiApplication, QPalette
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from config import LITELLM_ROUTED_WORKFLOWS
from config.persistence import persist_settings_to_config
from config.settings import (
    DEFAULT_CHAT_ASSISTANT_BUBBLE_COLOR,
    DEFAULT_CHAT_USER_BUBBLE_COLOR,
)
from prompt_templates import (
    DEFAULT_PROMPT_TEMPLATES,
    PROMPT_TEMPLATE_DESCRIPTIONS,
    PROMPT_TEMPLATE_KEYS,
    PROMPT_TEMPLATE_LABELS,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class SettingsDialog(QDialog):
    """Modal dialog enabling users to configure catalogue and LiteLLM options."""

    def __init__(
        self,
        parent=None,
        *,
        litellm_model: str | None = None,
        litellm_inference_model: str | None = None,
        litellm_api_key: str | None = None,
        litellm_api_base: str | None = None,
        litellm_api_version: str | None = None,
        litellm_drop_params: Sequence[str] | None = None,
        litellm_reasoning_effort: str | None = None,
        litellm_stream: bool | None = None,
        litellm_workflow_models: Mapping[str, str] | None = None,
        embedding_model: str | None = None,
        quick_actions: list[dict[str, object]] | None = None,
        chat_user_bubble_color: str | None = None,
        theme_mode: str | None = None,
        chat_colors: dict[str, str] | None = None,
        prompt_templates: dict[str, str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Prompt Manager Settings")
        self._settings = QSettings("PromptManager", "SettingsDialog")
        self.setMinimumWidth(860)
        screen = QGuiApplication.primaryScreen()
        target_height = 720
        if screen is not None:
            available_height = screen.availableGeometry().height()
            target_height = max(640, int(available_height * 0.85))
            self.setMaximumHeight(available_height)
        self.resize(self.minimumWidth(), target_height)
        self._restore_window_size(default_width=self.width(), default_height=self.height())
        self._litellm_model = litellm_model or ""
        self._litellm_inference_model = litellm_inference_model or ""
        self._litellm_api_key = litellm_api_key or ""
        self._litellm_api_base = litellm_api_base or ""
        self._litellm_api_version = litellm_api_version or ""
        self._embedding_model = embedding_model or ""
        self._litellm_drop_params = ", ".join(litellm_drop_params) if litellm_drop_params else ""
        self._litellm_reasoning_effort = (litellm_reasoning_effort or "").strip()
        self._litellm_stream = bool(litellm_stream)
        original_actions = [
            dict(entry) for entry in (quick_actions or []) if isinstance(entry, dict)
        ]
        self._original_quick_actions = original_actions
        self._quick_actions_value: list[dict[str, object]] | None = original_actions or None
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
        palette_dict = chat_colors or {}
        user_initial = palette_dict.get("user", chat_user_bubble_color) or self._default_chat_color
        if not QColor(user_initial).isValid():
            user_initial = self._default_chat_color
        self._chat_user_bubble_color = QColor(user_initial).name().lower()

        assistant_initial = palette_dict.get("assistant", DEFAULT_CHAT_ASSISTANT_BUBBLE_COLOR)
        if not QColor(assistant_initial).isValid():
            assistant_initial = DEFAULT_CHAT_ASSISTANT_BUBBLE_COLOR
        self._assistant_bubble_color = QColor(assistant_initial).name().lower()
        self._chat_color_input: QLineEdit | None = None
        self._chat_color_preview: QLabel | None = None
        self._chat_colors_value: dict[str, str] | None = None
        self._theme_mode = "dark" if (theme_mode or "").strip().lower() == "dark" else "light"
        self._theme_combo: QComboBox | None = None
        self._prompt_template_inputs: dict[str, QPlainTextEdit] = {}
        self._prompt_templates_value: dict[str, str] | None = None
        self._prompt_template_initials: dict[str, str] = {}
        provided_templates = prompt_templates or {}
        for key in PROMPT_TEMPLATE_KEYS:
            incoming = provided_templates.get(key)
            if isinstance(incoming, str) and incoming.strip():
                self._prompt_template_initials[key] = incoming.strip()
            else:
                self._prompt_template_initials[key] = DEFAULT_PROMPT_TEMPLATES.get(key, "")
        self._build_ui()

    def _restore_window_size(self, *, default_width: int, default_height: int) -> None:
        width = self._settings.value("width", default_width, type=int)
        height = self._settings.value("height", default_height, type=int)
        if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
            self.resize(width, height)

    def closeEvent(self, event: QEvent) -> None:  # type: ignore[override]
        self._settings.setValue("width", self.width())
        self._settings.setValue("height", self.height())
        super().closeEvent(event)

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

        self._embedding_model_input = QLineEdit(self._embedding_model, litellm_tab)
        self._embedding_model_input.setPlaceholderText("text-embedding-3-large")
        litellm_form.addRow("LiteLLM embedding model", self._embedding_model_input)

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

        theme_combo = QComboBox(appearance_tab)
        theme_combo.addItems(["Light", "Dark"])
        theme_combo.setCurrentIndex(1 if self._theme_mode == "dark" else 0)
        theme_combo.currentIndexChanged.connect(self._on_theme_mode_changed)  # type: ignore[arg-type]
        self._theme_combo = theme_combo
        appearance_form.addRow("Theme", theme_combo)

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

        # -------------------------------------------------------------
        # Colours tab – chat bubble customisation
        # -------------------------------------------------------------
        colors_tab = QWidget(self)
        colors_form = QFormLayout(colors_tab)
        colors_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self._color_buttons: dict[str, QPushButton] = {}

        def _make_color_row(key: str, label: str, default_hex: str) -> None:
            btn = QPushButton(colors_tab)
            btn.setText(" ")  # placeholder for minimum size
            btn.setFixedWidth(48)
            btn.setStyleSheet(f"background-color: {default_hex}; border: 1px solid #888;")

            def choose_color() -> None:  # pragma: no cover – GUI interaction
                current = QColor(btn.palette().button().color())
                chosen = QColorDialog.getColor(current, self, f"Select {label} colour")
                if not chosen.isValid():
                    return
                btn.setStyleSheet(f"background-color: {chosen.name()}; border: 1px solid #888;")
                self._color_buttons[key] = btn  # ensure stored

            btn.clicked.connect(choose_color)  # type: ignore[arg-type]
            colors_form.addRow(label, btn)
            self._color_buttons[key] = btn

        _make_color_row("user", "User bubble", self._chat_user_bubble_color)
        _make_color_row("assistant", "Assistant bubble", self._assistant_bubble_color)

        tab_widget.addTab(colors_tab, "Colors")

        quick_tab = QWidget(self)
        quick_layout = QVBoxLayout(quick_tab)
        quick_layout.setContentsMargins(0, 0, 0, 0)
        quick_actions_label = QLabel("Quick actions (JSON)", quick_tab)
        quick_layout.addWidget(quick_actions_label)

        self._quick_actions_input = QPlainTextEdit(quick_tab)
        self._quick_actions_input.setPlaceholderText(
            "Paste JSON array of quick actions (id, title, description, optional hints)."
        )
        if self._original_quick_actions:
            try:
                pretty = json.dumps(self._original_quick_actions, indent=2, ensure_ascii=False)
            except TypeError:
                pretty = ""
            self._quick_actions_input.setPlainText(pretty)
        quick_layout.addWidget(self._quick_actions_input)

        tab_widget.addTab(quick_tab, "Quick Actions")

        prompts_tab = QWidget(self)
        prompts_layout = QVBoxLayout(prompts_tab)
        prompts_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea(prompts_tab)
        scroll_area.setWidgetResizable(True)
        prompts_layout.addWidget(scroll_area)

        scroll_container = QWidget(scroll_area)
        scroll_area.setWidget(scroll_container)
        scroll_container_layout = QVBoxLayout(scroll_container)
        scroll_container_layout.setContentsMargins(0, 0, 0, 0)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        intro_label = QLabel(
            "Override the system prompts sent to LiteLLM for each supported workflow. "
            "Use descriptive, production-ready instructions; click Reset to restore defaults.",
            scroll_container,
        )
        intro_label.setWordWrap(True)
        scroll_container_layout.addWidget(intro_label)

        for key in PROMPT_TEMPLATE_KEYS:
            section = QFrame(scroll_container)
            section.setObjectName(f"promptTemplateSection_{key}")
            section.setFrameShape(QFrame.StyledPanel)
            section_layout = QVBoxLayout(section)
            section_layout.setContentsMargins(12, 8, 12, 12)

            title = QLabel(PROMPT_TEMPLATE_LABELS.get(key, key.title()), section)
            title.setObjectName(f"promptTemplateTitle_{key}")
            title.setStyleSheet("font-weight: 600;")
            section_layout.addWidget(title)

            description = QLabel(PROMPT_TEMPLATE_DESCRIPTIONS.get(key, ""), section)
            description.setWordWrap(True)
            section_layout.addWidget(description)

            editor = QPlainTextEdit(section)
            editor.setPlainText(self._prompt_template_initials.get(key, ""))
            editor.setMinimumHeight(140)
            editor.setPlaceholderText("Enter the system prompt sent to LiteLLM for this workflow.")
            section_layout.addWidget(editor)
            self._prompt_template_inputs[key] = editor

            actions_row = QHBoxLayout()
            reset_btn = QPushButton("Reset to default", section)
            reset_btn.clicked.connect(partial(self._reset_prompt_template, key))  # type: ignore[arg-type]
            actions_row.addWidget(reset_btn, alignment=Qt.AlignLeft)
            actions_row.addStretch(1)
            section_layout.addLayout(actions_row)

            scroll_container_layout.addWidget(section)

        reset_all_btn = QPushButton("Reset all to defaults", scroll_container)
        reset_all_btn.clicked.connect(self._reset_all_prompt_templates)  # type: ignore[arg-type]
        scroll_container_layout.addWidget(reset_all_btn, alignment=Qt.AlignLeft)
        scroll_container_layout.addStretch(1)

        tab_widget.addTab(prompts_tab, "Prompt Templates")

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        button_box.accepted.connect(self.accept)  # type: ignore[arg-type]
        button_box.rejected.connect(self.reject)  # type: ignore[arg-type]
        layout.addWidget(button_box)

    def _on_theme_mode_changed(self, index: int) -> None:
        """Synchronise the selected theme mode with internal state."""

        self._theme_mode = "dark" if index == 1 else "light"
        if self._chat_color_input is not None:
            self._update_chat_color_preview(self._chat_color_input.text())

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
        r, g, b, _ = color.getRgb()
        luminance = (0.299 * r) + (0.587 * g) + (0.114 * b)
        text_color = "#1f2933" if luminance >= 150 else "#f9fafc"
        self._chat_color_preview.setStyleSheet(
            f"background-color: {normalized}; border: 1px solid {border}; "
            f"border-radius: 8px; padding: 8px; color: {text_color};"
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

        if self._theme_combo is not None:
            self._theme_mode = "dark" if self._theme_combo.currentIndex() == 1 else "light"

        prompts_payload: dict[str, str] = {}
        for key, widget in self._prompt_template_inputs.items():
            text = widget.toPlainText().strip()
            if text:
                prompts_payload[key] = text
        self._prompt_templates_value = prompts_payload or None

        super().accept()

        # ------------------------------------------------------------------
        # Persist chosen colours – extract hex codes from swatch buttons
        # ------------------------------------------------------------------

        def _btn_hex(key: str, default_hex: str) -> str:
            btn = self._color_buttons.get(key)
            if btn is None:
                return default_hex
            return btn.palette().button().color().name().lower()

        self._chat_colors_value = {
            "user": _btn_hex("user", DEFAULT_CHAT_USER_BUBBLE_COLOR),
            "assistant": _btn_hex("assistant", DEFAULT_CHAT_ASSISTANT_BUBBLE_COLOR),
        }

    def result_settings(self) -> dict[str, object | None]:
        """Return cleaned settings data."""

        def _clean(value: str) -> str | None:
            stripped = value.strip()
            return stripped or None

        drop_text = self._drop_params_input.text().strip()
        drop_params = None
        if drop_text:
            drop_params = [item.strip() for item in drop_text.split(",") if item.strip()]

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
            "embedding_model": _clean(self._embedding_model_input.text()),
            "embedding_backend": "litellm",
            "litellm_api_key": _clean(self._api_key_input.text()),
            "litellm_api_base": _clean(self._api_base_input.text()),
            "litellm_api_version": _clean(self._api_version_input.text()),
            "litellm_drop_params": drop_params,
            "litellm_reasoning_effort": _clean(self._reasoning_effort_input.text()),
            "litellm_stream": self._stream_checkbox.isChecked(),
            "litellm_workflow_models": workflow_models or None,
            "quick_actions": self._quick_actions_value,
            "chat_user_bubble_color": self._chat_user_bubble_color,
            "chat_colors": self._chat_colors_value,
            "theme_mode": self._theme_mode,
            "prompt_templates": self._prompt_templates_value,
        }

    def _reset_prompt_template(self, key: str) -> None:
        """Reset a single prompt template editor to its default value."""

        editor = self._prompt_template_inputs.get(key)
        if editor is None:
            return
        editor.setPlainText(DEFAULT_PROMPT_TEMPLATES.get(key, ""))

    def _reset_all_prompt_templates(self) -> None:
        """Reset every prompt template editor to the default text."""

        for key in PROMPT_TEMPLATE_KEYS:
            self._reset_prompt_template(key)


__all__ = ["SettingsDialog", "persist_settings_to_config"]
