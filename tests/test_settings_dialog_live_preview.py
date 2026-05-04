"""Tests for live model/routing preview in the Prompt Manager settings dialog."""

from __future__ import annotations

from typing import cast

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QLabel, QLineEdit, QRadioButton

from gui.settings_dialog import SettingsDialog


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return cast("QApplication", app)


def test_settings_dialog_updates_live_routing_preview(qt_app: QApplication) -> None:
    dialog = SettingsDialog(
        litellm_model="azure/gpt-4.1-mini",
        litellm_inference_model="azure/gpt-5.4",
        litellm_workflow_models={"scenario_generation": "inference"},
        embedding_model="azure/text-embedding-3-large",
    )

    try:
        preview = dialog.findChild(QLabel, "routingPreviewLabel")
        assert preview is not None
        assert "Fast model: azure/gpt-4.1-mini" in preview.text()
        assert "Inference model: azure/gpt-5.4" in preview.text()
        assert "Inference workflows: Scenario drafting [custom]" in preview.text()
        assert "Embeddings: litellm / azure/text-embedding-3-large [custom]" in preview.text()
        assert "🟢 Status: ready" in preview.text()
        assert "background-color: #eaf7ea" in preview.styleSheet()
        assert "color: #12351a" in preview.styleSheet()
        assert "#1f7a1f" in preview.styleSheet()

        inference_input = dialog.findChild(QLineEdit, "settingsInferenceModelInput")
        assert inference_input is not None
        inference_input.setText("")
        qt_app.processEvents()
        assert "🟡 Status: inference model missing" in preview.text()
        assert "background-color: #fff6db" in preview.styleSheet()
        assert "color: #5f3b00" in preview.styleSheet()
        assert "#a15c00" in preview.styleSheet()

        inference_input.setText("azure/gpt-5.5")
        qt_app.processEvents()
        assert "🟢 Status: ready" in preview.text()

        workflow_button = dialog.findChild(
            QRadioButton,
            "routingChoice_description_generation_inference",
        )
        assert workflow_button is not None
        workflow_button.click()
        qt_app.processEvents()

        assert "Inference workflows:" in preview.text()
        assert "Scenario drafting" in preview.text()
        assert "Prompt description synthesis" in preview.text()
        assert "[custom]" in preview.text()
        assert "🟢 Status: ready" in preview.text()

        fast_input = dialog.findChild(QLineEdit, "settingsFastModelInput")
        assert fast_input is not None
        fast_input.setText("")
        qt_app.processEvents()
        assert "🔴 Status: fast model missing" in preview.text()
        assert "background-color: #fdeaea" in preview.styleSheet()
        assert "color: #5a1616" in preview.styleSheet()
        assert "#a61b1b" in preview.styleSheet()
    finally:
        dialog.deleteLater()
