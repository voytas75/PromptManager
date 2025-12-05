"""Toolbar widget containing search and primary catalog actions.

Updates:
  v0.2.2 - 2025-12-05 - Remove Prompt Templates toolbar button.
  v0.2.1 - 2025-12-05 - Remove Prompt Chains toolbar button now that the Chain tab is embedded.
  v0.2.0 - 2025-12-04 - Add prompt chain button and signal wiring.
  v0.1.0 - 2025-11-30 - Introduce reusable prompt toolbar widget.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QStyle,
    QToolButton,
    QWidget,
)


class PromptToolbar(QWidget):
    """Expose search, CRUD, and utility actions via a compact toolbar."""

    search_requested = Signal(str)
    search_text_changed = Signal(str)
    refresh_requested = Signal()
    add_requested = Signal()
    workbench_requested = Signal()
    import_requested = Signal()
    export_requested = Signal()
    maintenance_requested = Signal()
    notifications_requested = Signal()
    info_requested = Signal()
    settings_requested = Signal()
    exit_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create toolbar controls and wire outgoing signals."""
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._search_input = QLineEdit(self)
        self._search_input.setPlaceholderText("Search prompts…")
        self._search_input.setClearButtonEnabled(True)
        self._search_input.returnPressed.connect(self._emit_search_request)  # type: ignore[arg-type]
        self._search_input.textChanged.connect(self.search_text_changed)  # type: ignore[arg-type]
        layout.addWidget(self._search_input, 2)

        self._search_button = QPushButton("Search", self)
        self._search_button.setToolTip("Search prompts with the current query")
        self._search_button.clicked.connect(self._emit_search_request)  # type: ignore[arg-type]
        layout.addWidget(self._search_button)

        self._refresh_button = QPushButton("Refresh", self)
        self._refresh_button.clicked.connect(self.refresh_requested)  # type: ignore[arg-type]
        layout.addWidget(self._refresh_button)

        self._add_button = QPushButton("Add", self)
        self._add_button.clicked.connect(self.add_requested)  # type: ignore[arg-type]
        layout.addWidget(self._add_button)

        self._workbench_button = QPushButton("🆕 New", self)
        self._workbench_button.setToolTip("Open the Enhanced Prompt Workbench.")
        self._workbench_button.clicked.connect(self.workbench_requested)  # type: ignore[arg-type]
        layout.addWidget(self._workbench_button)

        self._import_button = QPushButton("Import", self)
        self._import_button.clicked.connect(self.import_requested)  # type: ignore[arg-type]
        layout.addWidget(self._import_button)

        self._export_button = QPushButton("Export", self)
        self._export_button.clicked.connect(self.export_requested)  # type: ignore[arg-type]
        layout.addWidget(self._export_button)

        self._maintenance_button = QPushButton("Maintenance", self)
        self._maintenance_button.clicked.connect(self.maintenance_requested)  # type: ignore[arg-type]
        layout.addWidget(self._maintenance_button)

        self._notifications_button = QPushButton("Notifications", self)
        self._notifications_button.clicked.connect(self.notifications_requested)  # type: ignore[arg-type]
        layout.addWidget(self._notifications_button)

        self._info_button = QPushButton("Info", self)
        self._info_button.clicked.connect(self.info_requested)  # type: ignore[arg-type]
        layout.addWidget(self._info_button)

        self._settings_button = QPushButton("Settings", self)
        self._settings_button.clicked.connect(self.settings_requested)  # type: ignore[arg-type]
        layout.addWidget(self._settings_button)

        self._exit_button = QToolButton(self)
        self._exit_button.setIcon(self.style().standardIcon(QStyle.SP_TitleBarCloseButton))
        self._exit_button.setToolTip("Exit Prompt Manager")
        self._exit_button.setAccessibleName("Exit Prompt Manager")
        self._exit_button.setAutoRaise(True)
        self._exit_button.setCursor(Qt.PointingHandCursor)
        self._exit_button.clicked.connect(self.exit_requested)  # type: ignore[arg-type]
        layout.addWidget(self._exit_button)

    def search_text(self) -> str:
        """Return the current search field text."""
        return self._search_input.text()

    def set_search_text(self, text: str, *, block_signals: bool = False) -> None:
        """Update the search field text, optionally without emitting signals."""
        if block_signals:
            previous = self._search_input.blockSignals(True)
            try:
                self._search_input.setText(text)
            finally:
                self._search_input.blockSignals(previous)
            return
        self._search_input.setText(text)

    def set_search_placeholder(self, text: str) -> None:
        """Set the placeholder shown inside the search field."""
        self._search_input.setPlaceholderText(text)

    def set_search_enabled(self, enabled: bool) -> None:
        """Enable or disable the explicit search trigger button."""
        self._search_button.setEnabled(enabled)

    def focus_search(self) -> None:
        """Place keyboard focus in the search field."""
        self._search_input.setFocus(Qt.ShortcutFocusReason)

    def _emit_search_request(self) -> None:
        self.search_requested.emit(self.search_text())
