"""Command palette dialog and utilities for quick actions.

Updates:
  v0.1.1 - 2025-11-29 - Wrap help dialog description builder for Ruff line length.
  v0.1.0 - 2025-11-10 - Introduce keyboard-driven quick action palette.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from models.prompt_model import Prompt


@dataclass(slots=True)
class QuickAction:
    """Representation of a quick command entry."""
    identifier: str
    title: str
    description: str
    category_hint: str | None = None
    tag_hints: tuple[str, ...] = ()
    template: str | None = None
    shortcut: str | None = None
    prompt_id: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> QuickAction:
        """Create a ``QuickAction`` from a mapping, enforcing required fields."""
        identifier = str(data.get("identifier") or data.get("id") or "").strip()
        title = str(data.get("title") or "").strip()
        description = str(data.get("description") or "").strip()
        if not identifier or not title or not description:
            raise ValueError("Quick action requires identifier, title, and description")
        category_hint = data.get("category_hint") or data.get("category")
        shortcut = data.get("shortcut")
        template = data.get("template")
        prompt_id = data.get("prompt_id") or data.get("prompt")
        tag_hints_value = data.get("tag_hints") or data.get("tags") or ()
        if isinstance(tag_hints_value, str):
            tag_hints = tuple(tag.strip() for tag in tag_hints_value.split(",") if tag.strip())
        else:
            tag_hints = tuple(str(tag).strip() for tag in tag_hints_value if str(tag).strip())
        return cls(
            identifier=identifier,
            title=title,
            description=description,
            category_hint=str(category_hint).strip() if category_hint else None,
            tag_hints=tag_hints,
            template=str(template) if template is not None else None,
            shortcut=str(shortcut).strip() if shortcut else None,
            prompt_id=str(prompt_id).strip() if prompt_id else None,
        )


def rank_prompts_for_action(prompts: Iterable[Prompt], action: QuickAction) -> list[Prompt]:
    """Return prompts ordered by how well they match the quick action hints."""
    candidates: list[tuple[int, Prompt]] = []
    tags = set(tag.lower() for tag in action.tag_hints)
    category = (action.category_hint or "").lower()
    for prompt in prompts:
        score = 0
        if category and prompt.category.lower() == category:
            score += 3
        if tags:
            prompt_tags = {tag.lower() for tag in prompt.tags}
            score += len(prompt_tags & tags) * 2
        if action.identifier.lower() in prompt.name.lower():
            score += 1
        if score > 0:
            candidates.append((score, prompt))
    candidates.sort(key=lambda item: (-item[0], item[1].name.lower()))
    return [prompt for _, prompt in candidates]


class CommandPaletteDialog(QDialog):
    """Simple command palette enabling keyboard navigation across quick actions."""
    def __init__(self, actions: Sequence[QuickAction], parent=None) -> None:
        """Configure dialog widgets and preload the quick action list."""
        super().__init__(parent)
        self.setWindowTitle("Command Palette")
        self.setModal(True)
        self._actions = list(actions)
        self._filtered_actions: list[QuickAction] = list(actions)
        self._selected: QuickAction | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        search_layout = QHBoxLayout()
        search_label = QLabel("Search:", self)
        self._search_input = QLineEdit(self)
        self._search_input.setPlaceholderText("Type to filter quick actions…")
        self._search_input.setClearButtonEnabled(True)
        self._search_input.textChanged.connect(self._on_search_changed)  # type: ignore[arg-type]
        search_layout.addWidget(search_label)
        search_layout.addWidget(self._search_input, 1)
        layout.addLayout(search_layout)

        self._list_widget = QListWidget(self)
        self._list_widget.itemActivated.connect(self._on_item_activated)  # type: ignore[arg-type]
        layout.addWidget(self._list_widget, 1)

        button_box = QDialogButtonBox(QDialogButtonBox.Cancel, Qt.Horizontal, self)
        button_box.rejected.connect(self.reject)

        help_button = QPushButton("Help", self)
        help_button.clicked.connect(self._show_help)  # type: ignore[arg-type]
        button_box.addButton(help_button, QDialogButtonBox.ActionRole)
        layout.addWidget(button_box)

        self._populate_list(self._actions)
        self._search_input.setFocus(Qt.ShortcutFocusReason)

    @property
    def selected_action(self) -> QuickAction | None:
        """Return the last action accepted from the palette, if any."""
        return self._selected

    def _populate_list(self, actions: Sequence[QuickAction]) -> None:
        """Populate the list widget with the provided quick actions."""
        self._list_widget.clear()
        self._filtered_actions = list(actions)
        for action in actions:
            title = action.title
            if action.shortcut:
                title = f"{title} ({action.shortcut})"
            item = QListWidgetItem(title, self._list_widget)
            item.setToolTip(action.description)
            item.setData(Qt.UserRole, action.identifier)
        if self._list_widget.count() > 0:
            self._list_widget.setCurrentRow(0)

    def _on_search_changed(self, text: str) -> None:
        """Filter quick actions whenever the search input changes."""
        stripped = text.strip().lower()
        if not stripped:
            self._populate_list(self._actions)
            return
        matches = [
            action
            for action in self._actions
            if stripped in action.title.lower()
            or stripped in action.description.lower()
            or (action.shortcut and stripped in action.shortcut.lower())
        ]
        self._populate_list(matches)

    def _on_item_activated(self, item: QListWidgetItem) -> None:
        """Accept the dialog when an action item is activated."""
        identifier = item.data(Qt.UserRole)
        for action in self._filtered_actions:
            if action.identifier == identifier:
                self._selected = action
                break
        self.accept()

    def _show_help(self) -> None:
        """Display a modal dialog summarising all available quick actions."""
        help_lines: list[str] = []
        for action in self._actions:
            shortcut_hint = f" [{action.shortcut}]" if action.shortcut else ""
            help_lines.append(f"{action.title}: {action.description}{shortcut_hint}")
        help_text = "\n".join(help_lines)
        help_dialog = QDialog(self)
        help_dialog.setWindowTitle("Quick Actions Help")
        layout = QVBoxLayout(help_dialog)
        label = QLabel(help_text, help_dialog)
        label.setWordWrap(True)
        layout.addWidget(label)
        button_box = QDialogButtonBox(QDialogButtonBox.Close, parent=help_dialog)
        button_box.rejected.connect(help_dialog.reject)
        button_box.accepted.connect(help_dialog.accept)
        layout.addWidget(button_box)
        help_dialog.exec()


__all__ = ["QuickAction", "CommandPaletteDialog", "rank_prompts_for_action"]
