"""Settings dialog for configuring Prompt Manager runtime options."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
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
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Prompt Manager Settings")
        self._catalog_path = catalog_path or ""
        self._litellm_model = litellm_model or ""
        self._litellm_api_key = litellm_api_key or ""
        self._litellm_api_base = litellm_api_base or ""
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
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

        layout.addLayout(form)

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

    def result_settings(self) -> dict[str, Optional[str]]:
        """Return cleaned settings data."""

        def _clean(value: str) -> Optional[str]:
            stripped = value.strip()
            return stripped or None

        return {
            "catalog_path": _clean(self._catalog_input.text()),
            "litellm_model": _clean(self._model_input.text()),
            "litellm_api_key": _clean(self._api_key_input.text()),
            "litellm_api_base": _clean(self._api_base_input.text()),
        }


def persist_settings_to_config(updates: dict[str, Optional[str]]) -> None:
    """Merge settings into config/config.json to persist between sessions."""
    config_path = Path("config/config.json")
    config_data = {}
    if config_path.exists():
        try:
            config_data = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            config_data = {}
    for key, value in updates.items():
        if value is not None:
            config_data[key] = value
        elif key in config_data:
            del config_data[key]
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config_data, indent=2, ensure_ascii=False), encoding="utf-8")


__all__ = ["SettingsDialog", "persist_settings_to_config"]
