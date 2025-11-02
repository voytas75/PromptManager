"""Settings dialog for configuring Prompt Manager runtime options.

Updates: v0.2.0 - 2025-11-16 - Apply palette-aware border styling to match main window chrome.
Updates: v0.1.1 - 2025-11-15 - Avoid persisting LiteLLM API secrets to disk.
Updates: v0.1.0 - 2025-11-04 - Initial settings dialog implementation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QFrame,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class SettingsDialog(QDialog):
    """Modal dialog enabling users to configure catalogue and LiteLLM options."""

    def __init__(
        self,
        parent=None,
        *,
        catalog_path: Optional[str] = None,
        litellm_model: Optional[str] = None,
        litellm_api_key: Optional[str] = None,
        litellm_api_base: Optional[str] = None,
        litellm_api_version: Optional[str] = None,
        quick_actions: Optional[list[dict[str, object]]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Prompt Manager Settings")
        self._catalog_path = catalog_path or ""
        self._litellm_model = litellm_model or ""
        self._litellm_api_key = litellm_api_key or ""
        self._litellm_api_base = litellm_api_base or ""
        self._litellm_api_version = litellm_api_version or ""
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

        form = QFormLayout()

        self._catalog_input = QLineEdit(self._catalog_path, self)
        browse_button = QPushButton("Browse…", self)
        browse_button.clicked.connect(self._browse_catalog)  # type: ignore[arg-type]
        catalog_row = QWidget(self)
        catalog_layout = QHBoxLayout(catalog_row)
        catalog_layout.setContentsMargins(0, 0, 0, 0)
        catalog_layout.setSpacing(6)
        catalog_layout.addWidget(self._catalog_input)
        catalog_layout.addWidget(browse_button)
        form.addRow("Catalogue path", catalog_row)

        self._model_input = QLineEdit(self._litellm_model, self)
        form.addRow("LiteLLM model", self._model_input)

        self._api_key_input = QLineEdit(self._litellm_api_key, self)
        self._api_key_input.setEchoMode(QLineEdit.Password)
        form.addRow("LiteLLM API key", self._api_key_input)

        self._api_base_input = QLineEdit(self._litellm_api_base, self)
        form.addRow("LiteLLM API base", self._api_base_input)

        self._api_version_input = QLineEdit(self._litellm_api_version, self)
        form.addRow("LiteLLM API version", self._api_version_input)

        layout.addLayout(form)

        self._quick_actions_input = QPlainTextEdit(self)
        self._quick_actions_input.setPlaceholderText(
            "Paste JSON array of quick action definitions (identifier, title, description, optional hints)."
        )
        if self._original_quick_actions:
            try:
                pretty = json.dumps(self._original_quick_actions, indent=2, ensure_ascii=False)
            except TypeError:
                pretty = ""
            self._quick_actions_input.setPlainText(pretty)
        quick_actions_label = QLabel("Quick actions (JSON)", self)
        layout.addWidget(quick_actions_label)
        layout.addWidget(self._quick_actions_input)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        button_box.accepted.connect(self.accept)  # type: ignore[arg-type]
        button_box.rejected.connect(self.reject)  # type: ignore[arg-type]
        layout.addWidget(button_box)

    def _browse_catalog(self) -> None:
        current = self._catalog_input.text().strip() or str(Path("config").resolve())
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setNameFilter("JSON files (*.json);;All files (*)")
        dialog.setDirectory(str(Path(current).parent if Path(current).exists() else Path(current)))
        if dialog.exec():
            selected = dialog.selectedFiles()
            if selected:
                self._catalog_input.setText(selected[0])

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

    def result_settings(self) -> dict[str, Optional[str | list[dict[str, object]]]]:
        """Return cleaned settings data."""

        def _clean(value: str) -> Optional[str]:
            stripped = value.strip()
            return stripped or None

        return {
            "catalog_path": _clean(self._catalog_input.text()),
            "litellm_model": _clean(self._model_input.text()),
            "litellm_api_key": _clean(self._api_key_input.text()),
            "litellm_api_base": _clean(self._api_base_input.text()),
            "litellm_api_version": _clean(self._api_version_input.text()),
            "quick_actions": self._quick_actions_value,
        }


def persist_settings_to_config(updates: dict[str, Optional[str | list[dict[str, object]]]]) -> None:
    """Merge settings into config/config.json to persist between sessions."""
    config_path = Path("config/config.json")
    config_data = {}
    if config_path.exists():
        try:
            config_data = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            config_data = {}
    secret_keys = {"litellm_api_key"}
    for key, value in updates.items():
        if key in secret_keys:
            # Always drop secrets from on-disk configuration; rely on env/secret store instead.
            config_data.pop(key, None)
            continue
        if value is not None:
            config_data[key] = value
        elif key in config_data:
            del config_data[key]
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config_data, indent=2, ensure_ascii=False), encoding="utf-8")


__all__ = ["SettingsDialog", "persist_settings_to_config"]
