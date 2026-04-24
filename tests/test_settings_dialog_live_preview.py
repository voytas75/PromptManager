"""Tests for live model/routing preview in the Prompt Manager settings dialog."""

from __future__ import annotations

from typing import Any, cast

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QLabel

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
        litellm_workflow_models={"name_generation": "inference"},
    )

    try:
        preview = dialog.findChild(QLabel, "routingPreviewLabel")
        assert preview is not None
        assert "Fast model: azure/gpt-4.1-mini" in preview.text()
        assert "Inference model: azure/gpt-5.4" in preview.text()
        assert "Inference workflows: name_generation" in preview.text()
        assert "🟢 Status: ready" in preview.text()
        assert "background-color: #eaf7ea" in preview.styleSheet()
        assert "color: #12351a" in preview.styleSheet()
        assert "#1f7a1f" in preview.styleSheet()

        inference_input = dialog._inference_model_input
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

        workflow_groups = cast("dict[str, object]", dialog._workflow_groups)
        workflow_group = cast("Any", workflow_groups["description_generation"])
        chat_title_inference = cast("object", workflow_group.buttons()[1])
        cast("Any", chat_title_inference).click()
        qt_app.processEvents()

        assert "Inference workflows:" in preview.text()
        assert "name_generation" in preview.text()
        assert "description_generation" in preview.text()
        assert "🟢 Status: ready" in preview.text()

        fast_input = dialog._model_input
        assert fast_input is not None
        fast_input.setText("")
        qt_app.processEvents()
        assert "🔴 Status: fast model missing" in preview.text()
        assert "background-color: #fdeaea" in preview.styleSheet()
        assert "color: #5a1616" in preview.styleSheet()
        assert "#a61b1b" in preview.styleSheet()
    finally:
        dialog.deleteLater()
