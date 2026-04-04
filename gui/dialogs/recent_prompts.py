"""Compact dialog for reopening recently touched prompts.

Updates:
  v0.1.0 - 2026-04-04 - Add deterministic recent prompt ordering and selection dialog.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Sequence
    from uuid import UUID

    from models.prompt_model import Prompt

_DEFAULT_RECENT_LIMIT = 10


def recent_prompts(
    prompts: Sequence[Prompt],
    *,
    limit: int = _DEFAULT_RECENT_LIMIT,
) -> list[Prompt]:
    """Return prompts ordered by most recently modified with stable tie-breakers."""
    if limit <= 0:
        return []
    return sorted(prompts, key=_recent_prompt_sort_key)[:limit]


def _recent_prompt_sort_key(prompt: Prompt) -> tuple[float, str, str]:
    """Build a deterministic key for recent prompt ordering."""
    timestamp = prompt.last_modified.timestamp() if prompt.last_modified else 0.0
    return (-timestamp, prompt.name.casefold(), str(prompt.id))


class RecentPromptsDialog(QDialog):
    """Display a compact list of recently touched prompts for quick reopening."""

    def __init__(
        self,
        prompts: Sequence[Prompt],
        parent: QWidget | None = None,
    ) -> None:
        """Initialise the dialog with the provided ordered *prompts*."""
        super().__init__(parent)
        self._prompts = list(prompts)
        self._selected_prompt_id: UUID | None = None
        self.setWindowTitle("Recent Prompts")
        self.setMinimumWidth(420)
        self.resize(480, 360)
        self._build_ui()

    @property
    def selected_prompt_id(self) -> UUID | None:
        """Return the selected prompt identifier after dialog acceptance."""
        return self._selected_prompt_id

    def _build_ui(self) -> None:
        """Create the compact recent prompt picker."""
        layout = QVBoxLayout(self)

        summary = QLabel("Reopen one of the prompts you touched most recently.", self)
        summary.setWordWrap(True)
        layout.addWidget(summary)

        self._list_widget = QListWidget(self)
        self._list_widget.itemDoubleClicked.connect(self._accept_selected_prompt)  # type: ignore[arg-type]
        layout.addWidget(self._list_widget)

        for prompt in self._prompts:
            item = QListWidgetItem(prompt.name, self._list_widget)
            item.setData(Qt.ItemDataRole.UserRole, str(prompt.id))
            item.setToolTip(self._build_tooltip(prompt))

        if self._prompts:
            self._list_widget.setCurrentRow(0)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self._open_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        if self._open_button is not None:
            self._open_button.setText("Open Prompt")
        self._sync_open_button()
        buttons.accepted.connect(self._accept_selected_prompt)  # type: ignore[arg-type]
        buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        self._list_widget.itemSelectionChanged.connect(self._sync_open_button)  # type: ignore[arg-type]
        layout.addWidget(buttons)

    def _sync_open_button(self) -> None:
        """Enable the accept action only when a prompt is selected."""
        if self._open_button is None:
            return
        self._open_button.setEnabled(self._list_widget.currentRow() >= 0)

    def _accept_selected_prompt(self) -> None:
        """Persist the current selection and close the dialog."""
        row = self._list_widget.currentRow()
        if row < 0 or row >= len(self._prompts):
            return
        self._selected_prompt_id = self._prompts[row].id
        self.accept()

    @staticmethod
    def _build_tooltip(prompt: Prompt) -> str:
        """Return a compact tooltip describing the prompt metadata."""
        modified = RecentPromptsDialog._format_timestamp(prompt.last_modified)
        category = prompt.category.strip() or "Uncategorised"
        return f"Last modified: {modified}\nCategory: {category}"

    @staticmethod
    def _format_timestamp(value: datetime) -> str:
        """Format timestamps in a compact UTC form for the picker."""
        return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")


class RecentPromptsDialogFactory:
    """Build pre-configured :class:`RecentPromptsDialog` instances."""

    def build(
        self,
        parent: QWidget,
        prompts: Sequence[Prompt],
    ) -> RecentPromptsDialog:
        """Return a recent prompt dialog bound to *parent*."""
        return RecentPromptsDialog(prompts, parent)


__all__ = ["RecentPromptsDialog", "RecentPromptsDialogFactory", "recent_prompts"]
