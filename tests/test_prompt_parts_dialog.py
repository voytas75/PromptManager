"""GUI regression tests for the canonical text of a prompt part."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from core.repository import PromptRepository
from gui.dialogs.execution import ResponseStyleDialog

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    """Provide Qt without starting the event loop."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return cast("QApplication", app)


def test_dialog_preserves_snippet_separately_on_save_and_reopen(
    qt_app: QApplication, tmp_path: Path
) -> None:
    """Editing a part must not substitute its formatting field for its body."""
    repo = PromptRepository(str(tmp_path / "parts.db"))
    dialog = ResponseStyleDialog()
    dialog._name_input.setText("System policy")  # pyright: ignore[reportPrivateUsage]
    dialog._phrase_input.setPlainText("Follow these exact instructions.")  # pyright: ignore[reportPrivateUsage]
    dialog._description_input.setPlainText("When to use this policy")  # pyright: ignore[reportPrivateUsage]
    dialog._format_input.setPlainText("Use concise bullets")  # pyright: ignore[reportPrivateUsage]
    dialog._examples_input.setPlainText("Example response")  # pyright: ignore[reportPrivateUsage]
    dialog._on_accept()  # pyright: ignore[reportPrivateUsage]

    style = dialog.result_style
    assert style is not None
    assert style.snippet == "Follow these exact instructions."
    assert style.description == "When to use this policy"
    assert style.format_instructions == "Use concise bullets"
    assert style.examples == ["Example response"]
    repo.add_response_style(style)

    editor = ResponseStyleDialog(style=repo.get_response_style(style.id))
    assert editor._phrase_input.toPlainText() == "Follow these exact instructions."  # pyright: ignore[reportPrivateUsage]
    assert editor._description_input.toPlainText() == "When to use this policy"  # pyright: ignore[reportPrivateUsage]
    assert editor._format_input.toPlainText() == "Use concise bullets"  # pyright: ignore[reportPrivateUsage]
    editor._on_accept()  # pyright: ignore[reportPrivateUsage]
    edited = editor.result_style
    assert edited is not None
    repo.update_response_style(edited)
    assert repo.get_response_style(style.id).snippet == "Follow these exact instructions."


def test_new_part_does_not_duplicate_snippet_into_optional_fields(qt_app: QApplication) -> None:
    """The snippet is not silently reclassified as formatting or an example."""
    dialog = ResponseStyleDialog()
    dialog._phrase_input.setPlainText("Only the snippet text")  # pyright: ignore[reportPrivateUsage]
    dialog._on_accept()  # pyright: ignore[reportPrivateUsage]

    style = dialog.result_style
    assert style is not None
    assert style.snippet == "Only the snippet text"
    assert style.description == ""
    assert style.format_instructions is None
    assert style.examples == []
