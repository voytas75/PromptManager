"""Prompt Maintenance dialog wrapper.

Updates:
  v0.1.1 - 2025-12-04 - Compose the dialog from modular mixins.
  v0.1.0 - 2025-12-03 - Split maintenance dialog from the dialogs monolith.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .backend import BackendMaintenanceMixin
from .catalogue import CatalogueMaintenanceMixin
from .reset import ResetMaintenanceMixin

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Callable, Sequence

    from core.prompt_manager import PromptManager
else:  # pragma: no cover - runtime placeholders for type-only imports
    Callable = Sequence = PromptManager = object


class PromptMaintenanceDialog(
    QDialog,
    CatalogueMaintenanceMixin,
    BackendMaintenanceMixin,
    ResetMaintenanceMixin,
):
    """Expose bulk metadata and backend maintenance utilities."""

    maintenance_applied = Signal(str)

    def __init__(
        self,
        parent: QWidget,
        manager: PromptManager,
        *,
        category_generator: Callable[[str], str] | None = None,
        tags_generator: Callable[[str], Sequence[str]] | None = None,
    ) -> None:
        """Initialise the dialog and eagerly fetch current maintenance metrics."""
        super().__init__(parent)
        self._manager = manager
        self._category_generator = category_generator
        self._tags_generator = tags_generator
        self._settings = QSettings("PromptManager", "PromptMaintenanceDialog")
        self._tab_widget: QTabWidget
        self._buttons: QDialogButtonBox
        self.setWindowTitle("Prompt Maintenance")
        self.resize(780, 360)
        self._restore_window_size()
        self._build_ui()
        self._refresh_catalogue_stats()
        self._refresh_category_health()
        self._refresh_redis_info()
        self._refresh_chroma_info()
        self._refresh_storage_info()

    def _restore_window_size(self) -> None:
        width = self._settings.value("width", type=int)
        height = self._settings.value("height", type=int)
        if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
            self.resize(width, height)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Persist the current window geometry when the dialog closes."""
        self._settings.setValue("width", self.width())
        self._settings.setValue("height", self.height())
        super().closeEvent(event)

    def _build_ui(self) -> None:
        outer_layout = QVBoxLayout(self)

        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer_layout.addWidget(scroll_area, stretch=1)

        scroll_contents = QWidget(self)
        scroll_area.setWidget(scroll_contents)
        layout = QVBoxLayout(scroll_contents)

        self._tab_widget = QTabWidget(scroll_contents)
        layout.addWidget(self._tab_widget, stretch=1)

        metadata_tab = self._build_metadata_tab(self)
        self._tab_widget.addTab(metadata_tab, "Metadata")

        redis_tab = self._build_redis_tab(self)
        self._tab_widget.addTab(redis_tab, "Redis")

        chroma_tab = self._build_chroma_tab(self)
        self._tab_widget.addTab(chroma_tab, "ChromaDB")

        storage_tab = self._build_storage_tab(self)
        self._tab_widget.addTab(storage_tab, "SQLite")

        reset_tab = self._build_reset_tab(self)
        self._tab_widget.addTab(reset_tab, "Data Reset")

        self._buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        self._buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        outer_layout.addWidget(self._buttons)
