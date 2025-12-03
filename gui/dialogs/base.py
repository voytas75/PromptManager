"""Shared helpers for Prompt Manager dialog modules.

Updates:
  v0.1.0 - 2025-12-03 - Extract common utilities from gui.dialogs monolith.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import textwrap
from copy import deepcopy
from typing import TYPE_CHECKING, Any, NamedTuple

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QPlainTextEdit, QSizePolicy, QToolButton, QVBoxLayout, QWidget

if TYPE_CHECKING:
    from collections.abc import MutableMapping

logger = logging.getLogger("prompt_manager.gui.dialogs")

_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+")


class SystemInfo(NamedTuple):
    """Container describing runtime platform characteristics for display."""

    cpu: str
    architecture: str
    platform_family: str
    os_label: str


def classify_platform_family(system_name: str) -> str:
    """Return a high-level platform family string based on the system identifier."""
    name = system_name.lower()
    if name.startswith("win"):
        return "Windows"
    if name.startswith(("darwin", "mac")):
        return "macOS"
    if name.startswith(("linux", "freebsd", "openbsd", "netbsd", "aix", "hp-ux", "sunos")):
        return "Unix-like"
    if not system_name:
        return "Unknown"
    return system_name


def collect_system_info() -> SystemInfo:
    """Gather CPU and platform metadata for the info dialog."""
    uname = platform.uname()

    raw_cpu = platform.processor() or uname.processor
    if not raw_cpu:
        cpu_count = os.cpu_count()
        raw_cpu = f"{cpu_count} logical cores" if cpu_count else "Unknown"

    architecture = platform.machine() or uname.machine or "Unknown"

    system_name = platform.system() or uname.system or "Unknown"
    release = platform.release() or uname.release
    os_label = f"{system_name} {release}".strip() if release else system_name

    return SystemInfo(
        cpu=raw_cpu,
        architecture=architecture,
        platform_family=classify_platform_family(system_name),
        os_label=os_label or "Unknown",
    )


def strip_scenarios_metadata(
    metadata: MutableMapping[str, Any] | None,
) -> MutableMapping[str, Any] | None:
    """Return a deep copy of metadata without stored usage scenarios."""
    if metadata is None:
        return None
    cleaned = deepcopy(metadata)
    cleaned.pop("scenarios", None)
    return cleaned or None


class CollapsibleTextSection(QWidget):
    """Wrapper providing an expandable/collapsible plain text editor."""

    textChanged = Signal()

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        """Create a collapsed text editor labelled by *title*."""
        super().__init__(parent)
        self._title = title
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)

        self._toggle = QToolButton(self)
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(False)
        self._toggle.setArrowType(Qt.RightArrow)
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.clicked.connect(self._on_toggle_clicked)  # type: ignore[arg-type]
        self._layout.addWidget(self._toggle)
        self._apply_toggle_style()

        self._editor = QPlainTextEdit(self)
        self._editor.setVisible(False)
        self._layout.addWidget(self._editor)

        self._editor.textChanged.connect(self._on_text_changed)  # type: ignore[arg-type]
        self._editor.installEventFilter(self)
        self._collapsed_height = 0
        self._expanded_height = 100

    def event(self, event: QEvent) -> bool:  # noqa: D401 - Qt override, documentation inherited
        """Refresh toggle styling when palettes or styles change."""
        if event.type() in {QEvent.PaletteChange, QEvent.StyleChange}:
            self._apply_toggle_style()
        return super().event(event)

    def setPlaceholderText(self, text: str) -> None:
        """Set the placeholder text displayed when the field is empty."""
        self._editor.setPlaceholderText(text)

    def setPlainText(self, text: str) -> None:
        """Populate the editor and expand it when content is present."""
        self._editor.setPlainText(text)
        stripped = text.strip()
        expanded = bool(stripped)
        self._set_expanded(expanded, focus=False)

    def toPlainText(self) -> str:
        """Return the current editor contents."""
        return self._editor.toPlainText()

    def editor(self) -> QPlainTextEdit:
        """Expose the underlying ``QPlainTextEdit`` widget."""
        return self._editor

    def focusEditor(self) -> None:
        """Expand and focus the editor."""
        self._set_expanded(True, focus=True)

    def isExpanded(self) -> bool:
        """Return True when the editor is currently expanded."""
        return self._toggle.isChecked()

    def setExpanded(self, expanded: bool) -> None:
        """Force the editor into a specific expanded state."""
        self._set_expanded(expanded, focus=expanded)

    def eventFilter(self, obj, event):  # type: ignore[override]
        """Collapse the editor when focus leaves and the field is empty."""
        if obj is self._editor and event.type() == QEvent.FocusOut:
            if not self._editor.toPlainText().strip():
                self._set_expanded(False, focus=False)
        return super().eventFilter(obj, event)

    def _apply_toggle_style(self) -> None:
        """Align the toggle button background with the active theme palette."""
        palette = self._toggle.palette()
        button_color = palette.color(QPalette.Button).name()
        text_color = palette.color(QPalette.ButtonText).name()
        border_color = palette.color(QPalette.Mid).name()
        checked_color = palette.color(QPalette.Highlight).name()
        checked_text = palette.color(QPalette.HighlightedText).name()
        self._toggle.setStyleSheet(
            "QToolButton {"
            f"background-color: {button_color};"
            f"color: {text_color};"
            f"border: 1px solid {border_color};"
            "border-radius: 4px;"
            "padding: 4px 8px;"
            "text-align: left;"
            "}"
            "QToolButton:checked {"
            f"background-color: {checked_color};"
            f"color: {checked_text};"
            "}"
        )

    def _set_expanded(self, expanded: bool, *, focus: bool) -> None:
        if self._toggle.isChecked() == expanded and self._editor.isVisible() == expanded:
            return
        self._toggle.blockSignals(True)
        self._toggle.setChecked(expanded)
        self._toggle.blockSignals(False)
        self._toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._editor.setVisible(expanded)
        if expanded:
            self._editor.setFixedHeight(self._expanded_height)
        else:
            self._editor.setFixedHeight(self._collapsed_height)
        if expanded and focus:
            self._editor.setFocus(Qt.OtherFocusReason)

    def _on_toggle_clicked(self, checked: bool) -> None:
        self._set_expanded(checked, focus=checked)

    def _on_text_changed(self) -> None:
        if self._editor.toPlainText().strip():
            if not self._toggle.isChecked():
                self._set_expanded(True, focus=False)
        self.textChanged.emit()


def fallback_suggest_prompt_name(context: str, *, max_words: int = 5) -> str:
    """Generate a concise prompt name from free-form context text."""
    text = context.strip()
    if not text:
        return ""
    first_line = text.splitlines()[0]
    tokens = _WORD_PATTERN.findall(first_line)
    if not tokens:
        return ""
    words = tokens[:max_words]
    name = " ".join(word.capitalize() for word in words)
    if len(tokens) > max_words:
        name += "…"
    return name


def fallback_generate_description(context: str, *, max_length: int = 240) -> str:
    """Create a lightweight summary from the prompt body when LLMs are unavailable."""
    stripped = " ".join(context.split())
    if not stripped:
        return ""
    if len(stripped) <= max_length:
        return stripped
    trimmed = stripped[:max_length]
    last_space = trimmed.rfind(" ")
    if last_space > 0:
        trimmed = trimmed[:last_space]
    return trimmed.rstrip(".") + "…"


def fallback_generate_scenarios(context: str, *, max_items: int = 3) -> list[str]:
    """Provide heuristic usage scenarios when LLM support is unavailable."""
    cleaned = context.strip()
    if not cleaned:
        return []

    sentence_pattern = re.compile(r"(?<=[.!?])\s+|\n")
    segments = [segment.strip() for segment in sentence_pattern.split(cleaned) if segment.strip()]
    if not segments:
        segments = [cleaned]

    scenarios: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        clause = segment.rstrip(".!?")
        if not clause:
            continue
        lowered = clause[0].lower() + clause[1:] if len(clause) > 1 else clause.lower()
        scenario = f"Use when you need to {lowered}"
        if not scenario.endswith("."):
            scenario += "."
        normalised = textwrap.shorten(scenario, width=140, placeholder="…")
        key = normalised.lower()
        if key in seen:
            continue
        seen.add(key)
        scenarios.append(normalised)
        if len(scenarios) >= max_items:
            break
    return scenarios


__all__ = [
    "CollapsibleTextSection",
    "SystemInfo",
    "classify_platform_family",
    "collect_system_info",
    "fallback_generate_description",
    "fallback_generate_scenarios",
    "fallback_suggest_prompt_name",
    "logger",
    "strip_scenarios_metadata",
]
