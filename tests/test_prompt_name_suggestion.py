"""Tests for prompt name suggestion helper."""
from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path


def _install_qt_stubs() -> None:
    try:
        import PySide6  # noqa: F401
        return
    except ImportError:
        pass

    qt_widgets = types.ModuleType("PySide6.QtWidgets")

    class _Widget:
        def __init__(self, *args, **kwargs) -> None:
            return

        def resize(self, *_: object, **__: object) -> None:  # pragma: no cover - stub only
            return

    class _Dialog(_Widget):
        accepted = object()
        rejected = object()

    class _DialogButtonBox(_Widget):
        Ok = 1
        Cancel = 2

        class _Signal:
            def connect(self, *_: object, **__: object) -> None:
                return

        def __init__(self, *_: object, **__: object) -> None:
            super().__init__()
            self.accepted = self._Signal()
            self.rejected = self._Signal()

    class _LineEdit(_Widget):
        def __init__(self, *_: object, **__: object) -> None:
            self._text = ""

        def setText(self, text: str) -> None:
            self._text = text

        def text(self) -> str:
            return self._text

    class _PlainTextEdit(_Widget):
        def __init__(self, *_: object, **__: object) -> None:
            self._text = ""

        def setPlainText(self, text: str) -> None:
            self._text = text

        def toPlainText(self) -> str:
            return self._text

        def setReadOnly(self, *_: object, **__: object) -> None:  # pragma: no cover - stub only
            return

    qt_widgets.QDialog = _Dialog  # type: ignore[attr-defined]
    qt_widgets.QDialogButtonBox = _DialogButtonBox  # type: ignore[attr-defined]
    qt_widgets.QFormLayout = _Widget  # type: ignore[attr-defined]
    qt_widgets.QLineEdit = _LineEdit  # type: ignore[attr-defined]
    qt_widgets.QMessageBox = _Widget  # type: ignore[attr-defined]
    qt_widgets.QPlainTextEdit = _PlainTextEdit  # type: ignore[attr-defined]
    qt_widgets.QPushButton = _Widget  # type: ignore[attr-defined]
    qt_widgets.QVBoxLayout = _Widget  # type: ignore[attr-defined]
    qt_widgets.QHBoxLayout = _Widget  # type: ignore[attr-defined]
    qt_widgets.QWidget = _Widget  # type: ignore[attr-defined]
    qt_widgets.QApplication = _Widget  # type: ignore[attr-defined]
    qt_widgets.QLabel = _Widget  # type: ignore[attr-defined]
    class _ComboBox(_Widget):
        def __init__(self, *_: object, **__: object) -> None:
            self._items: list[tuple[str, object]] = []
            self._current_index = 0

        def addItem(self, text: str, data: object = None) -> None:
            self._items.append((text, data))
            if len(self._items) == 1:
                self._current_index = 0

        def currentData(self) -> object:
            if not self._items:
                return None
            return self._items[self._current_index][1]

        def setCurrentIndex(self, index: int) -> None:
            if 0 <= index < len(self._items):
                self._current_index = index

        def findData(self, target: object) -> int:
            for idx, (_, data) in enumerate(self._items):
                if data == target:
                    return idx
            return -1

    qt_widgets.QComboBox = _ComboBox  # type: ignore[attr-defined]

    qt_core = types.ModuleType("PySide6.QtCore")
    class _Qt:
        AA_EnableHighDpiScaling = 1
        AA_UseHighDpiPixmaps = 2
        Horizontal = 0

    qt_core.Qt = _Qt  # type: ignore[attr-defined]
    qt_core.QEvent = type("QEvent", (), {})  # type: ignore[attr-defined]
    qt_core.Signal = type("Signal", (), {"__init__": lambda self, *args, **kwargs: None})  # type: ignore[attr-defined]

    sys.modules.setdefault("PySide6", types.ModuleType("PySide6"))
    sys.modules["PySide6.QtWidgets"] = qt_widgets
    sys.modules["PySide6.QtCore"] = qt_core


_install_qt_stubs()
loader = importlib.machinery.SourceFileLoader(
    "dialog_module_for_tests", str(Path("gui/dialogs.py").resolve())
)
spec = importlib.util.spec_from_loader("dialog_module_for_tests", loader)
assert spec and spec.loader
dialogs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dialogs)  # type: ignore[attr-defined]
fallback_suggest_prompt_name = dialogs.fallback_suggest_prompt_name


def test_suggest_prompt_name_uses_first_words() -> None:
    text = "Review backend service module for race conditions and logging gaps."
    name = fallback_suggest_prompt_name(text)
    assert name.startswith("Review Backend Service Module")
    assert name.endswith("…")


def test_suggest_prompt_name_handles_empty_input() -> None:
    assert fallback_suggest_prompt_name("") == ""
    assert fallback_suggest_prompt_name("   ") == ""


def test_suggest_prompt_name_allows_short_text() -> None:
    assert fallback_suggest_prompt_name("Cleanup") == "Cleanup"
