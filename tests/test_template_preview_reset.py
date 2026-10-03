"""Real-Qt regressions for empty-selection template reset.

Updates:
  v0.1.0 - 2026-10-03 - Cover empty/filled reset and per-prompt settings preservation.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, QSettings
from PySide6.QtWidgets import QApplication, QPlainTextEdit, QPushButton

from gui import template_preview as preview_module
from gui.template_preview import TemplatePreviewWidget
from gui.template_preview_controller import TemplatePreviewController
from models.prompt_model import Prompt


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    """Keep one real application alive for this regression module."""
    app = QApplication.instance()
    return cast("QApplication", app) if app is not None else QApplication([])


def _settle(app: QApplication) -> None:
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


def _rendered_text(widget: TemplatePreviewWidget) -> str:
    return next(
        field.toPlainText() for field in widget.findChildren(QPlainTextEdit) if field.isReadOnly()
    )


def _run_button(widget: TemplatePreviewWidget) -> QPushButton:
    return next(
        button for button in widget.findChildren(QPushButton) if button.text() == "Run Prompt"
    )


@pytest.mark.parametrize("filled", [False, True], ids=["empty-input", "filled-input"])
def test_empty_selection_clears_live_inputs_preserving_saved_prompt_state(
    qt_app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    filled: bool,
) -> None:
    """Reset removes live fields without erasing saved values for reselect."""
    store = QSettings(str(tmp_path / "preview.ini"), QSettings.Format.IniFormat)

    def settings_factory(organization: str, application: str) -> QSettings:
        return store

    monkeypatch.setattr(preview_module, "QSettings", settings_factory)
    widget = TemplatePreviewWidget()
    shortcut = QPushButton("Run template shortcut")
    shortcut.setEnabled(False)
    controller = TemplatePreviewController(
        parent=widget,
        tab_widget=None,
        template_preview=widget,
        template_run_button=shortcut,
        execution_controller_supplier=lambda: None,
        current_prompt_supplier=lambda: None,
        error_callback=lambda title, message: None,
        status_callback=lambda text, duration: None,
    )
    widget.run_state_changed.connect(controller.handle_run_state_changed)
    run_signals: list[str] = []

    def record_run(text: str, variables: dict[str, str]) -> None:
        run_signals.append(text)

    widget.run_requested.connect(record_run)
    prompt = Prompt(
        id=uuid.UUID("00000000-0000-0000-0000-000000000431"),
        name="Reset fixture",
        description="Synthetic template reset fixture.",
        category="Probe",
        context="Recipient: {{ recipient }}",
    )
    widget.show()
    widget.set_run_enabled(True)  # Control state only: no executor is constructed.
    try:
        controller.update_preview(prompt)
        _settle(qt_app)
        field = next(
            field
            for field in widget.findChildren(QPlainTextEdit)
            if field.placeholderText() == "Enter value for recipient…"
        )
        if filled:
            field.insertPlainText("synthetic recipient")
        _settle(qt_app)
        expected = {"recipient": "synthetic recipient"} if filled else {}
        assert widget.variables_payload() == expected
        assert _run_button(widget).isEnabled() == filled
        assert shortcut.isEnabled() == filled
        saved = store.value(f"prompt/{prompt.id}")

        controller.update_preview(None)
        _settle(qt_app)
        assert widget.variables_payload() == {}
        assert not any(
            field.isVisible()
            for field in widget.findChildren(QPlainTextEdit)
            if field.placeholderText().startswith("Enter value for")
        )
        assert _rendered_text(widget) == ""
        assert not _run_button(widget).isEnabled()
        assert not shortcut.isEnabled()
        assert store.value(f"prompt/{prompt.id}") == saved
        assert run_signals == []

        controller.update_preview(prompt)
        _settle(qt_app)
        assert widget.variables_payload() == expected
        assert _run_button(widget).isEnabled() == filled
        assert shortcut.isEnabled() == filled
        if filled:
            assert _rendered_text(widget) == "Recipient: synthetic recipient"
        assert run_signals == []
    finally:
        widget.close()
        shortcut.close()
