"""Prompt detail panel shared between main and template tabs.

Updates:
  v0.1.7 - 2025-12-09 - Import QToolButton to restore GUI startup.
  v0.1.6 - 2025-12-09 - Show only the requested metadata and add close toggle.
  v0.1.5 - 2025-12-09 - Toggle metadata table off on repeat clicks and ensure full payload shows.
  v0.1.4 - 2025-12-09 - Keep metadata hidden until toggled, showing both sets together inline.
  v0.1.3 - 2025-12-09 - Display basic and full metadata together in an inline table.
  v0.1.2 - 2025-12-08 - Align palette usage with Qt ColorRole enums.
  v0.1.1 - 2025-12-08 - Reworked prompt detail layout with grouped actions.
  v0.1.0 - 2025-11-30 - Extract prompt detail widget for reuse/testing.
"""

from __future__ import annotations

import json
from html import escape
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QToolButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from models.prompt_model import Prompt


class PromptDetailWidget(QWidget):
    """Panel that summarises the currently selected prompt."""

    _CONTEXT_PREVIEW_LIMIT = 200

    delete_requested = Signal()
    edit_requested = Signal()
    fork_requested = Signal()
    version_history_requested = Signal()
    refresh_scenarios_requested = Signal()
    share_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the scrollable layout and wire up prompt action signals."""
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._scroll_area = QScrollArea(self)
        self._scroll_area.setWidgetResizable(True)
        layout.addWidget(self._scroll_area)

        content = QWidget(self._scroll_area)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(12, 12, 12, 12)

        edit_button_row = QHBoxLayout()
        edit_button_row.setContentsMargins(0, 0, 0, 0)
        edit_button_row.addStretch(1)
        self._edit_button = QPushButton("Edit Prompt", content)
        self._edit_button.setObjectName("editPromptButton")
        self._edit_button.setEnabled(False)
        self._edit_button.clicked.connect(self.edit_requested.emit)  # type: ignore[arg-type]
        edit_button_row.addWidget(self._edit_button)
        content_layout.addLayout(edit_button_row)

        self._name_label = QLabel("Select a prompt to view details", content)
        self._name_label.setObjectName("promptTitle")
        self._name_label.setWordWrap(True)
        self._name_label.setTextFormat(Qt.TextFormat.RichText)
        self._version_label = QLabel("Version unavailable", content)
        self._version_label.setObjectName("promptVersion")
        self._version_label.setWordWrap(True)
        self._version_label.setTextFormat(Qt.TextFormat.RichText)
        self._version_label.setVisible(False)
        self._rating_label = QLabel("Rating", content)
        self._rating_label.setWordWrap(True)
        self._rating_label.setTextFormat(Qt.TextFormat.RichText)
        self._rating_label.setVisible(False)
        self._meta_label = QLabel("", content)
        self._meta_label.setWordWrap(True)
        self._meta_label.setTextFormat(Qt.TextFormat.RichText)
        self._meta_label.setVisible(False)
        self._description = QLabel("", content)
        self._description.setWordWrap(True)
        self._description.setTextFormat(Qt.TextFormat.RichText)
        self._lineage_label = QLabel("", content)
        self._lineage_label.setObjectName("promptLineage")
        self._lineage_label.setWordWrap(True)
        self._lineage_label.setStyleSheet("color: #4b5563;")
        self._lineage_label.setVisible(False)
        self._lineage_label.setTextFormat(Qt.TextFormat.RichText)

        self._context = QLabel("", content)
        self._context.setWordWrap(True)
        self._context.setTextFormat(Qt.TextFormat.RichText)
        self._scenarios = QLabel("", content)
        self._scenarios.setWordWrap(True)
        self._scenarios.setTextFormat(Qt.TextFormat.RichText)
        self._examples = QLabel("", content)
        self._examples.setWordWrap(True)
        self._examples.setTextFormat(Qt.TextFormat.RichText)

        content_layout.addWidget(self._name_label)
        content_layout.addSpacing(4)
        content_layout.addWidget(self._version_label)
        content_layout.addSpacing(4)
        content_layout.addWidget(self._rating_label)
        content_layout.addSpacing(4)
        content_layout.addWidget(self._meta_label)
        content_layout.addSpacing(4)
        content_layout.addWidget(self._description)
        content_layout.addSpacing(4)
        content_layout.addWidget(self._lineage_label)
        content_layout.addSpacing(4)
        content_layout.addWidget(self._context)
        content_layout.addSpacing(4)
        content_layout.addWidget(self._scenarios)
        content_layout.addSpacing(4)
        content_layout.addWidget(self._examples)
        content_layout.addSpacing(8)

        metadata_group = QGroupBox("Metadata Views", content)
        metadata_group.setObjectName("metadataToggleGroup")
        metadata_layout = QVBoxLayout(metadata_group)
        metadata_layout.setContentsMargins(12, 8, 12, 8)
        metadata_layout.setSpacing(8)

        metadata_buttons = QHBoxLayout()
        metadata_buttons.setContentsMargins(0, 0, 0, 0)
        metadata_buttons.setSpacing(8)
        self._basic_metadata_button = QPushButton("Basic Metadata", content)
        self._basic_metadata_button.setObjectName("showBasicMetadataButton")
        self._basic_metadata_button.setEnabled(False)
        self._basic_metadata_button.clicked.connect(self._show_basic_metadata)  # type: ignore[arg-type]
        metadata_buttons.addWidget(self._basic_metadata_button)

        self._all_metadata_button = QPushButton("All Metadata", content)
        self._all_metadata_button.setObjectName("showAllMetadataButton")
        self._all_metadata_button.setEnabled(False)
        self._all_metadata_button.clicked.connect(self._show_all_metadata)  # type: ignore[arg-type]
        metadata_buttons.addWidget(self._all_metadata_button)

        metadata_buttons.addStretch(1)
        metadata_layout.addLayout(metadata_buttons)

        self._metadata_table = QTableWidget(0, 2, metadata_group)
        self._metadata_table.setObjectName("promptMetadataTable")
        self._metadata_table.setHorizontalHeaderLabels(["Type", "Metadata"])
        header = self._metadata_table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._metadata_table.verticalHeader().setVisible(False)
        self._metadata_table.setWordWrap(True)
        self._metadata_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._metadata_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._metadata_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._metadata_table.setMinimumHeight(180)
        self._metadata_table.setVisible(False)
        self._metadata_table.setEnabled(False)
        self._metadata_visible_label: str | None = None
        metadata_header = QHBoxLayout()
        metadata_header.setContentsMargins(0, 0, 0, 0)
        metadata_header.setSpacing(4)
        self._metadata_label = QLabel("", metadata_group)
        metadata_header.addWidget(self._metadata_label)
        metadata_header.addStretch(1)
        self._metadata_close_button = QToolButton(metadata_group)
        self._metadata_close_button.setText("x")
        self._metadata_close_button.setObjectName("hideMetadataButton")
        self._metadata_close_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._metadata_close_button.setAutoRaise(True)
        self._metadata_close_button.setVisible(False)
        self._metadata_close_button.setToolTip("Hide metadata")
        self._metadata_close_button.clicked.connect(self._clear_metadata_table)  # type: ignore[arg-type]
        metadata_header.addWidget(self._metadata_close_button)
        metadata_layout.addLayout(metadata_header)
        metadata_layout.addWidget(self._metadata_table)
        content_layout.addWidget(metadata_group)

        self._full_metadata_text = ""
        self._basic_metadata_text = ""
        self._current_prompt: Prompt | None = None

        fork_delete_group = QGroupBox("Fork / Delete", content)
        fork_delete_layout = QHBoxLayout(fork_delete_group)
        fork_delete_layout.setContentsMargins(12, 8, 12, 8)
        fork_delete_layout.setSpacing(8)

        self._fork_button = QPushButton("Fork Prompt", content)
        self._fork_button.setObjectName("forkPromptButton")
        self._fork_button.setEnabled(False)
        self._fork_button.clicked.connect(self.fork_requested.emit)  # type: ignore[arg-type]
        fork_delete_layout.addWidget(self._fork_button)

        self._delete_button = QPushButton("Delete Prompt", content)
        self._delete_button.setObjectName("deletePromptButton")
        self._delete_button.setEnabled(False)
        self._delete_button.setStyleSheet("color: #dc2626;")
        self._delete_button.clicked.connect(self.delete_requested.emit)  # type: ignore[arg-type]
        fork_delete_layout.addWidget(self._delete_button)
        fork_delete_layout.addStretch(1)
        content_layout.addWidget(fork_delete_group)

        history_group = QGroupBox("History & Scenarios", content)
        history_layout = QHBoxLayout(history_group)
        history_layout.setContentsMargins(12, 8, 12, 8)
        history_layout.setSpacing(8)

        self._version_history_button = QPushButton("Version History", content)
        self._version_history_button.setObjectName("versionHistoryButton")
        self._version_history_button.setEnabled(False)
        self._version_history_button.clicked.connect(  # type: ignore[arg-type]
            self.version_history_requested.emit
        )
        history_layout.addWidget(self._version_history_button)

        self._refresh_scenarios_button = QPushButton("Refresh Scenarios", content)
        self._refresh_scenarios_button.setObjectName("refreshScenariosButton")
        self._refresh_scenarios_button.setEnabled(False)
        self._refresh_scenarios_button.clicked.connect(  # type: ignore[arg-type]
            self.refresh_scenarios_requested.emit
        )
        history_layout.addWidget(self._refresh_scenarios_button)
        history_layout.addStretch(1)
        content_layout.addWidget(history_group)

        share_group = QGroupBox("Share", content)
        share_group.setObjectName("shareOptionsFrame")
        share_group.setFlat(False)
        share_layout = QVBoxLayout(share_group)
        share_layout.setContentsMargins(12, 8, 12, 8)
        share_layout.setSpacing(8)

        share_payload_row = QHBoxLayout()
        share_payload_row.setContentsMargins(0, 0, 0, 0)
        share_payload_row.setSpacing(8)

        share_payload_label = QLabel("Share data:", share_group)
        share_payload_label.setObjectName("sharePayloadLabel")
        share_payload_row.addWidget(share_payload_label)

        self._share_payload_combo = QComboBox(share_group)
        self._share_payload_combo.addItem("Prompt body only", "body")
        self._share_payload_combo.addItem("Body + description", "body_description")
        self._share_payload_combo.addItem(
            "Body + description + scenarios",
            "body_description_scenarios",
        )
        self._share_payload_combo.setEnabled(False)
        share_payload_row.addWidget(self._share_payload_combo, 1)

        self._share_metadata_checkbox = QCheckBox("Add metadata", share_group)
        self._share_metadata_checkbox.setChecked(False)
        self._share_metadata_checkbox.setEnabled(False)
        share_payload_row.addWidget(self._share_metadata_checkbox)
        share_payload_row.addStretch(1)
        share_layout.addLayout(share_payload_row)

        share_button_row = QHBoxLayout()
        share_button_row.setContentsMargins(0, 0, 0, 0)
        share_button_row.addStretch(1)
        self._share_button = QPushButton("Share Prompt", share_group)
        self._share_button.setObjectName("sharePromptButton")
        self._share_button.setEnabled(False)
        self._share_button.clicked.connect(self.share_requested.emit)  # type: ignore[arg-type]
        share_button_row.addWidget(self._share_button)
        share_layout.addLayout(share_button_row)
        content_layout.addWidget(share_group)
        content_layout.addStretch(1)

        self._scroll_area.setWidget(content)

    def display_prompt(self, prompt: Prompt) -> None:
        """Populate labels using the provided prompt."""
        header_html = self._format_prompt_header(prompt)
        self._name_label.setText(header_html)
        self._version_label.clear()
        self._version_label.setVisible(False)
        self._rating_label.clear()
        self._rating_label.setVisible(False)
        self._meta_label.clear()
        self._meta_label.setVisible(False)
        description_value = prompt.description or "No description provided."
        self._description.setText(
            self._format_label_value("Description", description_value, multiline=True)
        )
        context_text = self._format_context_preview(prompt.context)
        self._context.setText(
            self._format_label_value("Prompt Body (preview)", context_text, multiline=True)
        )
        if prompt.scenarios:
            scenario_lines = "\n".join(f"• {scenario}" for scenario in prompt.scenarios)
            scenario_text = scenario_lines
        else:
            scenario_text = "None provided."
        self._scenarios.setText(
            self._format_label_value("Scenarios", scenario_text, multiline=True)
        )
        example_lines: list[str] = []
        if prompt.example_input:
            example_lines.append(f"Example input:\n{prompt.example_input}")
        if prompt.example_output:
            example_lines.append(f"Example output:\n{prompt.example_output}")
        if example_lines:
            examples_text = "\n\n".join(example_lines)
            self._examples.setText(
                self._format_label_value("Examples", examples_text, multiline=True)
            )
        else:
            self._examples.clear()
        record_payload = prompt.to_record()
        metadata_extra = {
            key: value for key, value in prompt.to_metadata().items() if key not in record_payload
        }
        if metadata_extra:
            record_payload["metadata_extra"] = metadata_extra
        self._full_metadata_text = json.dumps(record_payload, ensure_ascii=False, indent=2)
        self._basic_metadata_text = json.dumps(
            {
                "id": str(prompt.id),
                "last_modified": prompt.last_modified.isoformat(),
                "version": prompt.version,
                "quality_score": prompt.quality_score,
                "created_at": prompt.created_at.isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        )
        self._current_prompt = prompt
        self._clear_metadata_table()
        self._basic_metadata_button.setEnabled(True)
        self._all_metadata_button.setEnabled(True)
        self._edit_button.setEnabled(True)
        self._delete_button.setEnabled(True)
        self._fork_button.setEnabled(True)
        self._version_history_button.setEnabled(True)
        self._refresh_scenarios_button.setEnabled(True)
        self._share_button.setEnabled(True)
        self._share_payload_combo.setEnabled(True)
        self._share_metadata_checkbox.setEnabled(True)

    def _format_context_preview(self, context: str | None) -> str:
        """Return a truncated, single-line context preview for the prompt summary."""
        if not context:
            return "No prompt text provided."

        limited = context[: self._CONTEXT_PREVIEW_LIMIT]
        if len(context) > self._CONTEXT_PREVIEW_LIMIT:
            limited = f"{limited.rstrip()}…"

        # Collapse newlines (and other whitespace sequences) so the preview stays on one line.
        flattened = " ".join(limited.split())
        return flattened or "No prompt text provided."

    def current_prompt(self) -> Prompt | None:
        """Return the currently displayed prompt, if any."""
        return self._current_prompt

    def _format_prompt_header(self, prompt: Prompt) -> str:
        """Return the combined prompt title/category/metadata header."""
        title = prompt.name or "Untitled prompt"
        category = prompt.category or "Uncategorised"
        language = (prompt.language or "en").strip() or "en"
        detail_parts: list[str] = [language.lower()]

        version_raw = (prompt.version or "").strip()
        if not version_raw:
            version_label = "v1"
        elif version_raw.lower().startswith("v"):
            version_label = version_raw
        else:
            version_label = f"v{version_raw}"
        detail_parts.append(version_label)

        if prompt.rating_count > 0 and prompt.quality_score is not None:
            detail_parts.append(f"rating {prompt.quality_score:.1f}/10")

        detail_text = ", ".join(detail_parts)
        detail_color = self._label_color()
        safe_title = escape(title)
        safe_category = escape(category)
        safe_details = escape(detail_text)

        tag_suffix = ""
        if prompt.tags:
            cleaned_tags = [str(tag).strip() for tag in prompt.tags if str(tag).strip()]
            if cleaned_tags:
                safe_tags = escape(", ".join(cleaned_tags))
                tag_suffix = f" [{safe_tags}]"

        return (
            f'<span style="font-weight:600;">{safe_title}</span> - '
            f"{safe_category} "
            f'<span style="color: {detail_color};">({safe_details})</span>'
            f"{tag_suffix}"
        )

    def _format_label_value(self, label: str, value: str, *, multiline: bool = False) -> str:
        """Return HTML rendering with italic label and escaped value."""
        safe_label = escape(label)
        safe_value = escape(value)
        if multiline:
            safe_value = safe_value.replace("\n", "<br/>")
        color = self._label_color()
        return (
            f'<span style="font-style: italic; color: {color};">{safe_label}:</span> {safe_value}'
        )

    def _label_color(self) -> str:
        """Return a label colour with ≥4.5 contrast for the current palette."""
        background = self.palette().color(QPalette.ColorRole.Window)
        background_lum = self._relative_luminance(
            background.redF(), background.greenF(), background.blueF()
        )
        light_candidates = ["#0b1120", "#111827"]
        dark_candidates = ["#f8fafc", "#e5e7eb"]
        preferred = light_candidates if background_lum >= 0.5 else dark_candidates
        fallback = dark_candidates if background_lum >= 0.5 else light_candidates
        for scheme in (preferred, fallback):
            for hex_code in scheme:
                candidate_lum = self._relative_luminance_hex(hex_code)
                if self._contrast_ratio(background_lum, candidate_lum) >= 4.5:
                    return hex_code
        return preferred[0]

    @staticmethod
    def _relative_luminance(red: float, green: float, blue: float) -> float:
        """Return WCAG relative luminance."""
        return (
            0.2126 * PromptDetailWidget._linearize(red)
            + 0.7152 * PromptDetailWidget._linearize(green)
            + 0.0722 * PromptDetailWidget._linearize(blue)
        )

    @staticmethod
    def _relative_luminance_hex(hex_code: str) -> float:
        color = QColor(hex_code)
        return PromptDetailWidget._relative_luminance(color.redF(), color.greenF(), color.blueF())

    @staticmethod
    def _linearize(component: float) -> float:
        if component <= 0.04045:
            return component / 12.92
        return ((component + 0.055) / 1.055) ** 2.4

    @staticmethod
    def _contrast_ratio(lum_a: float, lum_b: float) -> float:
        lighter = max(lum_a, lum_b)
        darker = min(lum_a, lum_b)
        return (lighter + 0.05) / (darker + 0.05)

    def clear(self) -> None:
        """Reset the panel to its empty state."""
        self._name_label.setText("Select a prompt to view details")
        self._version_label.clear()
        self._version_label.setVisible(False)
        self._rating_label.clear()
        self._rating_label.setVisible(False)
        self._meta_label.clear()
        self._meta_label.setVisible(False)
        self._description.clear()
        self._context.clear()
        self._scenarios.clear()
        self._examples.clear()
        self._basic_metadata_button.setEnabled(False)
        self._all_metadata_button.setEnabled(False)
        self._clear_metadata_table()
        self._full_metadata_text = ""
        self._basic_metadata_text = ""
        self._current_prompt = None
        self._edit_button.setEnabled(False)
        self._delete_button.setEnabled(False)
        self._fork_button.setEnabled(False)
        self._version_history_button.setEnabled(False)
        self._refresh_scenarios_button.setEnabled(False)
        self._share_button.setEnabled(False)
        self._share_payload_combo.setEnabled(False)
        self._share_metadata_checkbox.setChecked(False)
        self._share_metadata_checkbox.setEnabled(False)
        self._lineage_label.clear()
        self._lineage_label.setVisible(False)

    def share_button(self) -> QPushButton:
        """Return the share button so callers can anchor menus."""
        return self._share_button

    def share_payload_mode(self) -> str:
        """Return the selected share payload mode."""
        data = self._share_payload_combo.currentData()
        if data is None:
            return "body_description_scenarios"
        return str(data)

    def share_include_metadata(self) -> bool:
        """Return True when metadata should be appended to the shared payload."""
        return self._share_metadata_checkbox.isChecked()

    def _show_basic_metadata(self) -> None:
        """Show only the basic prompt metadata subset."""
        self._render_metadata("Basic", self._basic_metadata_text)

    def _show_all_metadata(self) -> None:
        """Show only the full prompt metadata payload."""
        self._render_metadata("All", self._full_metadata_text)

    def _render_metadata(self, label: str, payload: str) -> None:
        """Render a single metadata payload and toggle off when repeated."""
        if not payload:
            return
        if self._metadata_table.isVisible() and self._metadata_visible_label == label:
            self._clear_metadata_table()
            return
        self._metadata_table.clearContents()
        self._metadata_table.setRowCount(1)
        type_item = QTableWidgetItem(label)
        type_item.setFlags(type_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        payload_item = QTableWidgetItem(payload)
        payload_item.setFlags(payload_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        payload_item.setTextAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._metadata_table.setItem(0, 0, type_item)
        self._metadata_table.setItem(0, 1, payload_item)
        self._metadata_table.resizeRowsToContents()
        self._metadata_table.setVisible(True)
        self._metadata_table.setEnabled(True)
        self._metadata_table.selectRow(0)
        self._metadata_label.setText(f"Metadata ({label})")
        self._metadata_close_button.setVisible(True)
        self._metadata_visible_label = label

    def _clear_metadata_table(self) -> None:
        """Hide and clear metadata table contents."""
        self._metadata_table.clearContents()
        self._metadata_table.setRowCount(0)
        self._metadata_table.setVisible(False)
        self._metadata_table.setEnabled(False)
        self._metadata_table.clearSelection()
        self._metadata_close_button.setVisible(False)
        self._metadata_label.clear()
        self._metadata_visible_label = None

    def update_lineage_summary(self, text: str | None) -> None:
        """Display lineage/version info beneath the description."""
        if text:
            self._lineage_label.setText(self._format_label_value("Lineage", text, multiline=True))
            self._lineage_label.setVisible(True)
        else:
            self._lineage_label.clear()
            self._lineage_label.setVisible(False)


__all__ = ["PromptDetailWidget"]
