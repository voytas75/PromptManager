"""Tests for the Prompt Manager settings dialog diagnostics banner."""

from __future__ import annotations

from typing import cast

import pytest

from gui.settings_dialog import SettingsDialog

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QLabel

EXPECTED_DIAGNOSTICS_STYLE_SNIPPETS = (
    "background-color: #fff6db",
    "color: #5f3b00",
    "border: 1px solid #a15c00",
)

EXPECTED_REDIS_READY_STYLE_SNIPPETS = (
    "background-color: #eaf4ff",
    "color: #12324a",
    "border: 1px solid #5b89a6",
)


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return cast("QApplication", app)


def test_settings_dialog_renders_compact_diagnostics_banner(qt_app: QApplication) -> None:
    dialog = SettingsDialog(
        config_diagnostics={
            "summary_status": "WARN",
            "items": [
                {"label": "Fast model", "status": "OK", "detail": "azure/gpt-4.1", "source": "env"},
                {
                    "label": "Inference model",
                    "status": "WARN",
                    "detail": "not configured",
                    "source": "default",
                },
                {"label": "API key", "status": "OK", "detail": "configured", "source": "env"},
                {
                    "label": "Embeddings",
                    "status": "OK",
                    "detail": "litellm / azure/UDTEMBED3L",
                    "source": "derived",
                },
                {"label": "TTS", "status": "WARN", "detail": "not configured", "source": "default"},
            ],
            "next_steps": [
                "Optional: set an inference model before routing heavier workflows to inference.",
                "Optional: configure a TTS model only if you plan to use voice output.",
            ],
            "redis_status": "Redis caching disabled (no DSN configured).",
        }
    )

    try:
        labels = [label.text() for label in dialog.findChildren(QLabel)]
        assert any("Configuration summary — WARN" in text for text in labels)
        assert any("OK | Fast model: azure/gpt-4.1 (env)" in text for text in labels)
        assert any("WARN | Inference model: not configured (default)" in text for text in labels)
        assert any("OK | Embeddings: litellm / azure/UDTEMBED3L (derived)" in text for text in labels)
        assert any("WARN | TTS: not configured (default)" in text for text in labels)
        assert any("Next steps:" in text for text in labels)
        assert any(
            "Optional: set an inference model before routing heavier workflows to inference."
            in text
            for text in labels
        )
        diagnostics_banner = dialog.findChild(QLabel, "configDiagnosticsBanner")
        assert diagnostics_banner is not None
        for snippet in EXPECTED_DIAGNOSTICS_STYLE_SNIPPETS:
            assert snippet in diagnostics_banner.styleSheet()
    finally:
        dialog.deleteLater()


def test_settings_dialog_renders_readable_redis_banner(qt_app: QApplication) -> None:
    dialog = SettingsDialog(redis_status="Redis caching enabled via redis://localhost:6379/0")

    try:
        redis_banner = dialog.findChild(QLabel, "redisStatusBanner")
        assert redis_banner is not None
        assert "Redis caching enabled via redis://localhost:6379/0" in redis_banner.text()
        for snippet in EXPECTED_REDIS_READY_STYLE_SNIPPETS:
            assert snippet in redis_banner.styleSheet()
    finally:
        dialog.deleteLater()
