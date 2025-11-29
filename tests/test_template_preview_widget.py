"""Tests for template preview widget error fallbacks.

Updates:
  v0.1.1 - 2025-11-29 - Extend preview helper with persistence/run state fields.
  v0.1.0 - 2025-11-27 - Ensure parse/render errors keep raw template text visible.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

from core.templating import SchemaValidator, TemplateRenderResult


def _load_template_preview_module() -> types.ModuleType:
    """Import template preview module, falling back to PySide6 stubs when absent."""

    try:
        import PySide6  # noqa: F401  # pragma: no cover - import guard only
    except ImportError:
        return _load_with_qt_stubs()
    return importlib.import_module("gui.template_preview")


def _load_with_qt_stubs() -> types.ModuleType:
    names = (
        "PySide6",
        "PySide6.QtCore",
        "PySide6.QtWidgets",
        "PySide6.QtGui",
    )
    saved: dict[str, Any] = {name: sys.modules.get(name) for name in names}

    pyside_module = types.ModuleType("PySide6")
    pyside_module.__path__ = []  # type: ignore[attr-defined]
    qt_core = types.ModuleType("PySide6.QtCore")
    qt_widgets = types.ModuleType("PySide6.QtWidgets")
    qt_gui = types.ModuleType("PySide6.QtGui")

    class _Signal:
        def __init__(self, *_: object, **__: object) -> None:
            self._slots: list[Any] = []

        def __get__(self, instance: object, owner: type | None = None) -> _Signal:
            return self

        def connect(self, slot: Any) -> None:  # pragma: no cover - stub only
            self._slots.append(slot)

        def emit(self, *args: object, **kwargs: object) -> None:  # pragma: no cover - stub only
            for slot in self._slots:
                slot(*args, **kwargs)

    class _Qt:
        Vertical = 1
        Horizontal = 2

    qt_core.Signal = _Signal  # type: ignore[attr-defined]
    qt_core.Qt = _Qt  # type: ignore[attr-defined]

    class _Widget:
        def __init__(self, *_: object, **__: object) -> None:
            return

        def deleteLater(self) -> None:  # pragma: no cover - stub only
            return

    def _make_widget(name: str) -> type:
        return type(name, (_Widget,), {})

    for widget_name in (
        "QWidget",
        "QComboBox",
        "QFrame",
        "QHBoxLayout",
        "QLabel",
        "QListWidget",
        "QListWidgetItem",
        "QPlainTextEdit",
        "QPushButton",
        "QScrollArea",
        "QSplitter",
        "QVBoxLayout",
    ):
        setattr(qt_widgets, widget_name, _make_widget(widget_name))

    class _QColor:
        def __init__(self, *_: object, **__: object) -> None:  # pragma: no cover - stub only
            return

    qt_gui.QColor = _QColor  # type: ignore[attr-defined]

    created: dict[str, types.ModuleType] = {
        "PySide6": pyside_module,
        "PySide6.QtCore": qt_core,
        "PySide6.QtWidgets": qt_widgets,
        "PySide6.QtGui": qt_gui,
    }

    for name, module in created.items():
        sys.modules[name] = module

    loader = importlib.machinery.SourceFileLoader(
        "_template_preview_for_tests",
        str(Path("gui/template_preview.py").resolve()),
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("Failed to load template preview module for tests")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)

    for name, previous in saved.items():
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous

    return module


_TEMPLATE_PREVIEW_MODULE = _load_template_preview_module()
TemplatePreviewWidget = _TEMPLATE_PREVIEW_MODULE.TemplatePreviewWidget


class _DummyTextEdit:
    def __init__(self, text: str = "") -> None:
        self._text = text
        self.blocked = False

    def setPlainText(self, text: str) -> None:
        self._text = text

    def clear(self) -> None:
        self._text = ""

    def toPlainText(self) -> str:
        return self._text

    def blockSignals(self, value: bool) -> None:  # pragma: no cover - behaviour tracked externally
        self.blocked = bool(value)


class _DummyLabel:
    def __init__(self) -> None:
        self.text_value = ""
        self.style_value = ""

    def setStyleSheet(self, value: str) -> None:
        self.style_value = value

    def setText(self, value: str) -> None:
        self.text_value = value


class _DummyButton:
    def __init__(self) -> None:
        self.enabled = False

    def setEnabled(self, value: bool) -> None:
        self.enabled = bool(value)


def _make_preview() -> TemplatePreviewWidget:
    widget = TemplatePreviewWidget.__new__(TemplatePreviewWidget)
    widget._renderer = object()
    widget._validator = SchemaValidator()
    widget._template_text = ""
    widget._template_parse_error = None
    widget._variable_names = []
    widget._variable_inputs = {}
    widget._schema_visible = False
    widget._schema_mode = types.SimpleNamespace(currentData=lambda: None)
    widget._schema_input = _DummyTextEdit()
    widget._variables_list = object()
    widget._rendered_view = _DummyTextEdit()
    widget._status_label = _DummyLabel()
    widget._template_hint = _DummyLabel()
    widget._run_button = _DummyButton()
    widget._run_enabled = False
    widget._last_run_ready = False
    widget._last_rendered_text = ""
    widget._preview_ready = False
    widget._current_prompt_id = None
    widget._suspend_persist = False
    widget._state_store = types.SimpleNamespace(value=lambda *_: None, setValue=lambda *_: None)

    class _DummyToggle:
        def __init__(self) -> None:
            self.checked = False

        def isChecked(self) -> bool:
            return self.checked

        def setChecked(self, value: bool) -> None:
            self.checked = bool(value)

        def setText(self, _: str) -> None:
            return

    widget._schema_toggle = _DummyToggle()

    def _show_message_item(self: TemplatePreviewWidget, text: str, _: str) -> None:
        self._last_message = text

    def _update_variable_states(
        self: TemplatePreviewWidget,
        provided: set[str],
        missing: set[str],
        invalid: set[str],
    ) -> None:
        self._last_variable_state = (provided, missing, invalid)

    widget._show_message_item = types.MethodType(_show_message_item, widget)
    widget._update_variable_states = types.MethodType(_update_variable_states, widget)
    return widget


def test_preview_displays_template_text_when_parse_error_occurs() -> None:
    widget = _make_preview()
    widget._template_text = "RAW TEMPLATE"
    widget._template_parse_error = "syntax issue"

    widget._update_preview()

    assert widget._rendered_view.toPlainText() == "RAW TEMPLATE"
    assert widget._status_label.text_value == "syntax issue"


def test_preview_displays_template_text_when_rendering_fails() -> None:
    widget = _make_preview()
    widget._template_text = "{{ invalid syntax }}"
    widget._template_parse_error = None

    class _StubRenderer:
        def render(self, *_: object, **__: object) -> TemplateRenderResult:
            return TemplateRenderResult(rendered_text="", errors=["boom"], missing_variables=set())

    widget._renderer = _StubRenderer()

    widget._update_preview()

    assert widget._rendered_view.toPlainText() == "{{ invalid syntax }}"
    assert widget._status_label.text_value == "boom"


def test_apply_variable_values_populates_matching_inputs() -> None:
    widget = _make_preview()
    editor = _DummyTextEdit()
    widget._variable_inputs = {"customer": editor}
    invoked: dict[str, int] = {"count": 0}

    def _stub_update(self: TemplatePreviewWidget) -> None:
        invoked["count"] += 1

    widget._update_preview = types.MethodType(_stub_update, widget)
    widget.apply_variable_values({"customer": "ACME Corp", "missing": "skip"})

    assert editor.toPlainText() == "ACME Corp"
    assert invoked["count"] == 1


def test_refresh_preview_invokes_internal_update() -> None:
    widget = _make_preview()
    invoked: dict[str, int] = {"count": 0}

    def _stub_update(self: TemplatePreviewWidget) -> None:
        invoked["count"] += 1

    widget._update_preview = types.MethodType(_stub_update, widget)
    widget.refresh_preview()

    assert invoked["count"] == 1
