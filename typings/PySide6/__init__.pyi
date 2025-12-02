"""Namespace shim for PySide6 to back local Qt type stubs.

Updates:
  v0.1.0 - 2025-12-02 - Introduce PySide6 namespace stub pointing at bundled modules.
"""

from __future__ import annotations

from types import ModuleType

QtCore: ModuleType
QtGui: ModuleType
QtWidgets: ModuleType
QtCharts: ModuleType

__all__ = ["QtCore", "QtGui", "QtWidgets", "QtCharts"]
