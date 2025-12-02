"""Minimal PySide6.QtWidgets stubs to keep Pyright strict mode happy.

Updates:
  v0.1.1 - 2025-12-02 - Expand PySide6.QtWidgets stub classes and expose referenced attributes.
  v0.1.0 - 2025-12-02 - Initial shim exposing Qt types used by the GUI.
"""
from __future__ import annotations

from typing import Any

class _QtBase:
    """Catch-all base providing permissive attribute access on Qt objects."""

    def __init__(self, *args: object, **kwargs: object) -> None: ...

    def __call__(self, *args: object, **kwargs: object) -> Any: ...

    def __getattr__(self, name: str, /) -> Any: ...


class QAbstractItemView(_QtBase):
    NoSelection: Any
    SelectRows: Any
    SingleSelection: Any


class QAbstractSpinBox(_QtBase):
    NoButtons: Any


class QApplication(_QtBase):
    instance: Any
    restoreOverrideCursor: Any
    setAttribute: Any
    setOverrideCursor: Any


class QButtonGroup(_QtBase):
    ...


class QCheckBox(_QtBase):
    ...


class QColorDialog(_QtBase):
    getColor: Any


class QComboBox(_QtBase):
    AdjustToContents: Any
    NoInsert: Any


class QDialog(_QtBase):
    Accepted: Any


class QDialogButtonBox(_QtBase):
    AcceptRole: Any
    ActionRole: Any
    ApplyRole: Any
    Cancel: Any
    Close: Any
    DestructiveRole: Any
    Ok: Any


class QDoubleSpinBox(_QtBase):
    ...


class QFileDialog(_QtBase):
    getExistingDirectory: Any
    getOpenFileName: Any
    getSaveFileName: Any


class QFormLayout(_QtBase):
    AllNonFixedFieldsGrow: Any
    ExpandingFieldsGrow: Any


class QFrame(_QtBase):
    Plain: Any
    StyledPanel: Any


class QGridLayout(_QtBase):
    ...


class QGroupBox(_QtBase):
    ...


class QHBoxLayout(_QtBase):
    ...


class QHeaderView(_QtBase):
    ResizeToContents: Any
    Stretch: Any


class QLabel(_QtBase):
    ...


class QLayout(_QtBase):
    SetMinimumSize: Any


class QLayoutItem(_QtBase):
    ...


class QLineEdit(_QtBase):
    Password: Any


class QListView(_QtBase):
    ...


class QListWidget(_QtBase):
    ...


class QListWidgetItem(_QtBase):
    ...


class QMainWindow(_QtBase):
    ...


class QMenu(_QtBase):
    exec_: Any


class QMessageBox(_QtBase):
    No: Any
    Yes: Any
    critical: Any
    information: Any
    question: Any
    warning: Any


class QPlainTextEdit(_QtBase):
    WidgetWidth: Any


class QProgressBar(_QtBase):
    ...


class QPushButton(_QtBase):
    ...


class QRadioButton(_QtBase):
    ...


class QScrollArea(_QtBase):
    ...


class QSizePolicy(_QtBase):
    Expanding: Any
    Fixed: Any
    Minimum: Any
    Preferred: Any


class QSpinBox(_QtBase):
    ...


class QSplitter(_QtBase):
    ...


class QStackedWidget(_QtBase):
    ...


class QStatusBar(_QtBase):
    ...


class QStyle(_QtBase):
    SP_TitleBarCloseButton: Any


class QStyleFactory(_QtBase):
    create: Any


class QTabWidget(_QtBase):
    ...


class QTableWidget(_QtBase):
    NoEditTriggers: Any
    NoSelection: Any
    SelectRows: Any
    SingleSelection: Any


class QTableWidgetItem(_QtBase):
    ...


class QTextBrowser(_QtBase):
    ...


class QTextEdit(_QtBase):
    ExtraSelection: Any


class QToolBar(_QtBase):
    ...


class QToolButton(_QtBase):
    ...


class QVBoxLayout(_QtBase):
    ...


class QWidget(_QtBase):
    ...


class QWidgetItem(_QtBase):
    ...


__all__ = [
    "QAbstractItemView",
    "QAbstractSpinBox",
    "QApplication",
    "QButtonGroup",
    "QCheckBox",
    "QColorDialog",
    "QComboBox",
    "QDialog",
    "QDialogButtonBox",
    "QDoubleSpinBox",
    "QFileDialog",
    "QFormLayout",
    "QFrame",
    "QGridLayout",
    "QGroupBox",
    "QHBoxLayout",
    "QHeaderView",
    "QLabel",
    "QLayout",
    "QLayoutItem",
    "QLineEdit",
    "QListView",
    "QListWidget",
    "QListWidgetItem",
    "QMainWindow",
    "QMenu",
    "QMessageBox",
    "QPlainTextEdit",
    "QProgressBar",
    "QPushButton",
    "QRadioButton",
    "QScrollArea",
    "QSizePolicy",
    "QSpinBox",
    "QSplitter",
    "QStackedWidget",
    "QStatusBar",
    "QStyle",
    "QStyleFactory",
    "QTabWidget",
    "QTableWidget",
    "QTableWidgetItem",
    "QTextBrowser",
    "QTextEdit",
    "QToolBar",
    "QToolButton",
    "QVBoxLayout",
    "QWidget",
    "QWidgetItem",
]
