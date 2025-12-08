"""Dialog for editing LiteLLM prompt template overrides.

Updates:
  v0.1.1 - 2025-12-01 - Preserve trailing whitespace when saving template overrides.
  v0.1.0 - 2025-11-29 - Introduce dedicated prompt template editor dialog.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from prompt_templates import (
    DEFAULT_PROMPT_TEMPLATES,
    PROMPT_TEMPLATE_DESCRIPTIONS,
    PROMPT_TEMPLATE_KEYS,
    PROMPT_TEMPLATE_LABELS,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


class PromptTemplateEditorDialog(QDialog):
    """Modal dialog that exposes editable LiteLLM prompt templates."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        templates: Mapping[str, str] | None = None,
    ) -> None:
        """Create the editor and preload any existing template overrides."""
        super().__init__(parent)
        self.setWindowTitle("Prompt Template Overrides")
        self.setMinimumSize(720, 640)

        self._defaults: dict[str, str] = dict(DEFAULT_PROMPT_TEMPLATES)
        self._initial_overrides: dict[str, str] = {
            key: value
            for key, value in (templates or {}).items()
            if isinstance(value, str) and value.strip()
        }
        self._editors: dict[str, QPlainTextEdit] = {}
        self._status_labels: dict[str, QLabel] = {}

        layout = QVBoxLayout(self)
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

        container = QWidget(scroll_area)
        scroll_area.setWidget(container)
        container_layout = QVBoxLayout(container)
        intro = QLabel(
            "Customise the system prompts sent to LiteLLM for each workflow. "
            "Changes take effect immediately after saving.",
            container,
        )
        intro.setWordWrap(True)
        container_layout.addWidget(intro)

        for key in PROMPT_TEMPLATE_KEYS:
            section = QFrame(container)
            section.setFrameShape(QFrame.StyledPanel)
            section_layout = QVBoxLayout(section)
            section_layout.setContentsMargins(12, 10, 12, 12)

            title = QLabel(PROMPT_TEMPLATE_LABELS.get(key, key.title()), section)
            title.setStyleSheet("font-weight:600;")
            section_layout.addWidget(title)

            description = QLabel(PROMPT_TEMPLATE_DESCRIPTIONS.get(key, ""), section)
            description.setWordWrap(True)
            section_layout.addWidget(description)

            editor = QPlainTextEdit(section)
            editor.setPlaceholderText("Enter the system prompt sent to LiteLLM for this workflow.")
            editor.setMinimumHeight(150)
            editor.setPlainText(self._resolve_text(key))
            editor.textChanged.connect(partial(self._on_text_changed, key))  # type: ignore[arg-type]
            section_layout.addWidget(editor)
            self._editors[key] = editor

            status = QLabel(section)
            status.setObjectName(f"promptTemplateStatus_{key}")
            section_layout.addWidget(status)
            self._status_labels[key] = status
            self._update_status_label(key)

            actions = QHBoxLayout()
            reset_btn = QPushButton("Reset to default", section)
            reset_btn.clicked.connect(partial(self._reset_template, key))  # type: ignore[arg-type]
            actions.addWidget(reset_btn, alignment=Qt.AlignmentFlag.AlignLeft)
            actions.addStretch(1)
            section_layout.addLayout(actions)

            container_layout.addWidget(section)

        reset_all = QPushButton("Reset all to defaults", container)
        reset_all.clicked.connect(self._reset_all)  # type: ignore[arg-type]
        container_layout.addWidget(reset_all, alignment=Qt.AlignmentFlag.AlignLeft)
        container_layout.addStretch(1)

        self._buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        self._buttons.accepted.connect(self._on_accept)  # type: ignore[arg-type]
        self._buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        layout.addWidget(self._buttons)
        self._validate_inputs()

    def _resolve_text(self, key: str) -> str:
        override = self._initial_overrides.get(key)
        if override:
            return override
        return self._defaults.get(key, "")

    def _reset_template(self, key: str) -> None:
        editor = self._editors.get(key)
        if editor is None:
            return
        editor.blockSignals(True)
        editor.setPlainText(self._defaults.get(key, ""))
        editor.blockSignals(False)
        self._update_status_label(key)
        self._validate_inputs()

    def _reset_all(self) -> None:
        for key in PROMPT_TEMPLATE_KEYS:
            self._reset_template(key)

    def _on_text_changed(self, key: str) -> None:
        self._update_status_label(key)
        self._validate_inputs()

    def _update_status_label(self, key: str) -> None:
        label = self._status_labels.get(key)
        editor = self._editors.get(key)
        if not label or not editor:
            return
        raw_text = editor.toPlainText()
        text = raw_text.strip()
        default_text = self._defaults.get(key, "")
        char_count = len(raw_text)
        if not text:
            label.setStyleSheet("color:#b91c1c;")
            label.setText("Required – enter system instructions for this workflow.")
            return
        label.setStyleSheet("")
        status = "overriding default" if raw_text != default_text else "using default"
        label.setText(f"{char_count} characters • {status}")

    def _validate_inputs(self) -> None:
        valid = all(editor.toPlainText().strip() for editor in self._editors.values())
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(valid)

    def _on_accept(self) -> None:
        if not all(editor.toPlainText().strip() for editor in self._editors.values()):
            return
        self.accept()

    def result_templates(self) -> dict[str, str]:
        """Return normalised overrides excluding defaults."""
        payload: dict[str, str] = {}
        for key, editor in self._editors.items():
            text = editor.toPlainText()
            if not text.strip():
                continue
            default_text = self._defaults.get(key, "")
            if text != default_text:
                payload[key] = text
        return payload


__all__ = ["PromptTemplateEditorDialog"]
