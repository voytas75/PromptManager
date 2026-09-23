"""Tests for GUI application helpers.

Updates: v0.1.0 - 2025-11-05 - Ensure offscreen fallback only triggers when headless.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from typing import TYPE_CHECKING, Any, cast

import pytest

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator


@pytest.fixture()
def stub_qt(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Provide minimal PySide6 stubs so gui.application can be imported."""
    qt_core = types.ModuleType("PySide6.QtCore")
    qt_widgets = types.ModuleType("PySide6.QtWidgets")
    qt_gui = types.ModuleType("PySide6.QtGui")

    class _Qt:
        AA_EnableHighDpiScaling = object()
        AA_UseHighDpiPixmaps = object()

    qt_core.Qt = _Qt  # type: ignore[attr-defined]

    class _QApplication:
        _instance = None

        def __init__(self, _: list[str]) -> None:
            type(self)._instance = self

        @classmethod
        def instance(cls) -> _QApplication | None:
            return cls._instance

        @staticmethod
        def setAttribute(*_: object, **__: object) -> None:  # pragma: no cover - stub only
            return

        def exec(self) -> int:  # pragma: no cover - stub only
            return 0

    qt_widgets.QApplication = _QApplication  # type: ignore[attr-defined]
    qt_gui.QGuiApplication = _QApplication  # type: ignore[attr-defined]

    class _QIcon:
        def __init__(self, *_: object, **__: object) -> None:
            return

    qt_gui.QIcon = _QIcon  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "PySide6", types.ModuleType("PySide6"))
    monkeypatch.setitem(sys.modules, "PySide6.QtCore", qt_core)
    monkeypatch.setitem(sys.modules, "PySide6.QtWidgets", qt_widgets)
    monkeypatch.setitem(sys.modules, "PySide6.QtGui", qt_gui)

    stubbed_main_window = types.ModuleType("gui.main_window")

    class _MainWindow:
        def __init__(self, manager: object, settings: object | None = None) -> None:
            self.manager = manager
            self.settings = settings

    stubbed_main_window.MainWindow = _MainWindow  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gui.main_window", stubbed_main_window)

    yield

    for module_name in (
        "PySide6",
        "PySide6.QtCore",
        "PySide6.QtWidgets",
        "PySide6.QtGui",
        "gui.main_window",
    ):
        sys.modules.pop(module_name, None)


def _reload_gui_application() -> types.ModuleType:
    sys.modules.pop("gui.application", None)
    return importlib.import_module("gui.application")


def test_create_qapplication_uses_existing_instance(
    stub_qt: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _reload_gui_application()
    QApplication = module.QApplication  # type: ignore[attr-defined]
    existing = QApplication([])

    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    created = module.create_qapplication()

    assert created is existing
    assert os.environ.get("QT_QPA_PLATFORM") != "offscreen"


def test_create_qapplication_falls_back_to_offscreen_when_headless(
    stub_qt: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _reload_gui_application()

    for var in ("DISPLAY", "WAYLAND_DISPLAY", "MIR_SOCKET", "QT_QPA_PLATFORM"):
        monkeypatch.delenv(var, raising=False)

    module.create_qapplication()

    assert os.environ["QT_QPA_PLATFORM"] == "offscreen"


def test_linux_xcb_preflight_reports_missing_runtime_libraries(
    stub_qt: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing xcb runtime libraries should be named before QApplication starts."""
    module = _reload_gui_application()
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module, "_xcb_platform_plugin_path", lambda: "/tmp/libqxcb.so")

    class _CompletedProcess:
        stdout = "\n".join(
            (
                "libxcb-icccm.so.4 => not found",
                "libxcb-keysyms.so.1 => not found",
            )
        )

    def _run_ldd(*_: object, **__: object) -> Any:
        return _CompletedProcess()

    monkeypatch.setattr(module.subprocess, "run", cast("Any", _run_ldd))

    issues = module.linux_xcb_runtime_issues({"DISPLAY": ":0"})

    assert issues == ["libxcb-icccm.so.4", "libxcb-keysyms.so.1"]
    with pytest.raises(module.GuiRuntimeError, match="libxcb-icccm.so.4"):
        module.ensure_gui_runtime({"DISPLAY": ":0"})


def test_linux_xcb_preflight_skips_non_xcb_gui_modes(
    stub_qt: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Headless and explicit non-xcb modes must not be blocked by xcb checks."""
    module = _reload_gui_application()
    monkeypatch.setattr(module.sys, "platform", "linux")

    assert module.linux_xcb_runtime_issues({"QT_QPA_PLATFORM": "offscreen"}) == []
    assert module.linux_xcb_runtime_issues({"WAYLAND_DISPLAY": "wayland-0"}) == []
