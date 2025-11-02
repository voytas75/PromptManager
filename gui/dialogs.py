"""Dialog widgets used by the Prompt Manager GUI.

Updates: v0.8.1 - 2025-11-16 - Add destructive delete control and metadata suggestion helpers to the prompt dialog.
Updates: v0.8.0 - 2025-11-16 - Add task template editor dialog.
Updates: v0.7.2 - 2025-11-02 - Collapse example sections when empty and expand on demand.
Updates: v0.7.1 - 2025-11-02 - Increase default prompt dialog size for edit and creation workflows.
Updates: v0.7.0 - 2025-11-16 - Add markdown preview dialog for rendered execution output.
Updates: v0.6.0 - 2025-11-15 - Add prompt engineering refinement button to prompt dialog.
Updates: v0.5.0 - 2025-11-09 - Capture execution ratings alongside optional notes.
Updates: v0.4.0 - 2025-11-08 - Add execution Save Result dialog with optional notes.
Updates: v0.3.0 - 2025-11-06 - Add catalogue preview dialog with diff summary output.
Updates: v0.2.0 - 2025-11-05 - Add prompt name suggestion based on context.
Updates: v0.1.0 - 2025-11-04 - Implement create/edit prompt dialog backed by Prompt dataclass.
"""

from __future__ import annotations

import logging
import re
import textwrap
import uuid
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from typing import Callable, Optional, Sequence, List

from PySide6.QtCore import Qt, QEvent, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QToolButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from core import (
    CatalogDiff,
    CatalogDiffEntry,
    NameGenerationError,
    DescriptionGenerationError,
    PromptEngineeringUnavailable,
    PromptManager,
    PromptManagerError,
    RepositoryError,
)
from core.prompt_engineering import PromptEngineeringError, PromptRefinement
from models.prompt_model import Prompt, TaskTemplate


logger = logging.getLogger("prompt_manager.gui.dialogs")

_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+")


class CollapsibleTextSection(QWidget):
    """Wrapper providing an expandable/collapsible plain text editor."""

    textChanged = Signal()

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._title = title
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

        self._editor = QPlainTextEdit(self)
        self._editor.setVisible(False)
        self._layout.addWidget(self._editor)

        self._editor.textChanged.connect(self._on_text_changed)  # type: ignore[arg-type]
        self._editor.installEventFilter(self)
        self._collapsed_height = 0
        self._expanded_height = 100

    def setPlaceholderText(self, text: str) -> None:
        self._editor.setPlaceholderText(text)

    def setPlainText(self, text: str) -> None:
        self._editor.setPlainText(text)
        stripped = text.strip()
        expanded = bool(stripped)
        self._set_expanded(expanded, focus=False)

    def toPlainText(self) -> str:
        return self._editor.toPlainText()

    def editor(self) -> QPlainTextEdit:
        return self._editor

    def focusEditor(self) -> None:
        self._set_expanded(True, focus=True)

    def isExpanded(self) -> bool:
        return self._toggle.isChecked()

    def setExpanded(self, expanded: bool) -> None:
        self._set_expanded(expanded, focus=expanded)

    def eventFilter(self, obj, event):  # type: ignore[override]
        if obj is self._editor and event.type() == QEvent.FocusOut:
            if not self._editor.toPlainText().strip():
                self._set_expanded(False, focus=False)
        return super().eventFilter(obj, event)

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


class PromptDialog(QDialog):
    """Modal dialog used for creating or editing prompt records."""

    def __init__(
        self,
        parent=None,
        prompt: Optional[Prompt] = None,
        name_generator: Optional[Callable[[str], str]] = None,
        description_generator: Optional[Callable[[str], str]] = None,
        category_generator: Optional[Callable[[str], str]] = None,
        tags_generator: Optional[Callable[[str], Sequence[str]]] = None,
        prompt_engineer: Optional[Callable[..., PromptRefinement]] = None,
    ) -> None:
        super().__init__(parent)
        self._source_prompt = prompt
        self._result_prompt: Optional[Prompt] = None
        self._name_generator = name_generator
        self._description_generator = description_generator
        self._prompt_engineer = prompt_engineer
        self._category_generator = category_generator
        self._tags_generator = tags_generator
        self._delete_requested = False
        self._delete_button: Optional[QPushButton] = None
        self._generate_category_button: Optional[QPushButton] = None
        self._generate_tags_button: Optional[QPushButton] = None
        self.setWindowTitle("Create Prompt" if prompt is None else "Edit Prompt")
        self.setMinimumWidth(760)
        self.resize(960, 700)
        self._build_ui()
        if prompt is not None:
            self._populate(prompt)

    @property
    def result_prompt(self) -> Optional[Prompt]:
        """Return the prompt produced by the dialog after acceptance."""

        return self._result_prompt

    @property
    def delete_requested(self) -> bool:
        """Return True when the user chose to delete the prompt instead of saving."""

        return self._delete_requested

    @property
    def source_prompt(self) -> Optional[Prompt]:
        """Expose the prompt supplied to the dialog for convenience."""

        return self._source_prompt

    def _build_ui(self) -> None:
        """Construct the dialog layout and wire interactions."""

        main_layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self._name_input = QLineEdit(self)
        self._generate_name_button = QPushButton("Generate", self)
        self._generate_name_button.setToolTip("Suggest a name based on the context field.")
        self._generate_name_button.clicked.connect(self._on_generate_name_clicked)  # type: ignore[arg-type]
        name_container = QWidget(self)
        name_container_layout = QHBoxLayout(name_container)
        name_container_layout.setContentsMargins(0, 0, 0, 0)
        name_container_layout.setSpacing(6)
        name_container_layout.addWidget(self._name_input)
        name_container_layout.addWidget(self._generate_name_button)
        form_layout.addRow("Name", name_container)

        self._category_input = QLineEdit(self)
        self._language_input = QLineEdit(self)
        self._tags_input = QLineEdit(self)
        self._context_input = QPlainTextEdit(self)
        self._context_input.setPlaceholderText("Paste the full prompt text here…")
        self._description_input = QPlainTextEdit(self)
        self._description_input.setFixedHeight(60)
        self._example_input = CollapsibleTextSection("Example Input", self)
        self._example_input.setPlaceholderText("Optional example input…")
        self._example_output = CollapsibleTextSection("Example Output", self)
        self._example_output.setPlaceholderText("Optional example output…")

        self._language_input.setPlaceholderText("en")
        self._tags_input.setPlaceholderText("tag-a, tag-b")

        category_container = QWidget(self)
        category_container_layout = QHBoxLayout(category_container)
        category_container_layout.setContentsMargins(0, 0, 0, 0)
        category_container_layout.setSpacing(6)
        category_container_layout.addWidget(self._category_input)
        self._generate_category_button = QPushButton("Suggest", self)
        self._generate_category_button.setToolTip("Suggest a category based on the prompt body.")
        self._generate_category_button.clicked.connect(self._on_generate_category_clicked)  # type: ignore[arg-type]
        if self._category_generator is None:
            self._generate_category_button.setEnabled(False)
            self._generate_category_button.setToolTip("Category suggestions require the main application context.")
        category_container_layout.addWidget(self._generate_category_button)
        form_layout.addRow("Category", category_container)
        form_layout.addRow("Language", self._language_input)
        tags_container = QWidget(self)
        tags_container_layout = QHBoxLayout(tags_container)
        tags_container_layout.setContentsMargins(0, 0, 0, 0)
        tags_container_layout.setSpacing(6)
        tags_container_layout.addWidget(self._tags_input)
        self._generate_tags_button = QPushButton("Suggest", self)
        self._generate_tags_button.setToolTip("Suggest tags based on the prompt body.")
        self._generate_tags_button.clicked.connect(self._on_generate_tags_clicked)  # type: ignore[arg-type]
        if self._tags_generator is None:
            self._generate_tags_button.setEnabled(False)
            self._generate_tags_button.setToolTip("Tag suggestions require the main application context.")
        tags_container_layout.addWidget(self._generate_tags_button)
        form_layout.addRow("Tags", tags_container)
        form_layout.addRow("Prompt Body", self._context_input)
        self._refine_button = QPushButton("Refine", self)
        self._refine_button.setToolTip(
            "Analyse and refine the prompt using LiteLLM when configured."
        )
        self._refine_button.clicked.connect(self._on_refine_clicked)  # type: ignore[arg-type]
        if self._prompt_engineer is None:
            try:
                self._refine_button.setEnabled(False)
            except AttributeError:  # pragma: no cover - stubbed widgets during tests
                pass
            self._refine_button.setToolTip(
                "Configure LiteLLM in Settings to enable prompt engineering."
            )
        form_layout.addRow("", self._refine_button)
        form_layout.addRow("Description", self._description_input)
        form_layout.addRow(self._example_input)
        form_layout.addRow(self._example_output)

        main_layout.addLayout(form_layout)

        self._buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        self._buttons.accepted.connect(self._on_accept)  # type: ignore[arg-type]
        self._buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        if self._source_prompt is not None:
            self._delete_button = self._buttons.addButton(
                "Delete",
                QDialogButtonBox.DestructiveRole,
            )
            self._delete_button.setToolTip("Delete this prompt from the catalogue.")
            self._delete_button.clicked.connect(self._on_delete_clicked)  # type: ignore[arg-type]
        main_layout.addWidget(self._buttons)
        self._context_input.textChanged.connect(self._on_context_changed)  # type: ignore[arg-type]

    def _populate(self, prompt: Prompt) -> None:
        """Fill inputs with the existing prompt values for editing."""

        self._name_input.setText(prompt.name)
        self._category_input.setText(prompt.category)
        self._language_input.setText(prompt.language)
        self._tags_input.setText(", ".join(prompt.tags))
        self._context_input.setPlainText(prompt.context or "")
        self._description_input.setPlainText(prompt.description)
        self._example_input.setPlainText(prompt.example_input or "")
        self._example_output.setPlainText(prompt.example_output or "")

    def _on_accept(self) -> None:
        """Validate inputs, build the prompt, and close the dialog."""

        prompt = self._build_prompt()
        if prompt is None:
            return
        self._result_prompt = prompt
        self.accept()

    def _on_delete_clicked(self) -> None:
        """Handle delete requests issued from the dialog."""

        if self._source_prompt is None:
            return
        current_name = self._name_input.text().strip()
        name = current_name or self._source_prompt.name or "this prompt"
        confirmation = QMessageBox.question(
            self,
            "Delete prompt",
            f"Are you sure you want to delete '{name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirmation != QMessageBox.Yes:
            return
        self._delete_requested = True
        self._result_prompt = None
        self.accept()

    def _on_generate_name_clicked(self) -> None:
        """Generate a prompt name from the context field."""

        try:
            suggestion = self._generate_name(self._context_input.toPlainText())
        except NameGenerationError as exc:
            QMessageBox.warning(self, "Name generation failed", str(exc))
            return
        if suggestion:
            self._name_input.setText(suggestion)

    def _on_generate_category_clicked(self) -> None:
        """Generate a category suggestion based on the prompt body."""

        if self._category_generator is None:
            return
        context = self._context_input.toPlainText()
        try:
            suggestion = (self._category_generator(context) or "").strip()
        except Exception as exc:  # noqa: BLE001 - surface generator failures to the user
            QMessageBox.warning(self, "Category suggestion failed", str(exc))
            return
        if suggestion:
            self._category_input.setText(suggestion)
        else:
            QMessageBox.information(
                self,
                "No suggestion available",
                "The assistant could not determine a suitable category.",
            )

    def _on_generate_tags_clicked(self) -> None:
        """Generate tag suggestions based on the prompt body."""

        if self._tags_generator is None:
            return
        context = self._context_input.toPlainText()
        try:
            suggestions = self._tags_generator(context) or []
        except Exception as exc:  # noqa: BLE001 - surface generator failures to the user
            QMessageBox.warning(self, "Tag suggestion failed", str(exc))
            return
        tags = [tag.strip() for tag in suggestions if str(tag).strip()]
        if tags:
            self._tags_input.setText(", ".join(tags))
        else:
            QMessageBox.information(
                self,
                "No suggestion available",
                "The assistant could not determine relevant tags.",
            )

    def _on_context_changed(self) -> None:
        """Auto-suggest a prompt name when none has been supplied."""

        if self._source_prompt is not None:
            return
        if self._name_generator is None:
            return
        current_name = self._name_input.text().strip()
        if current_name:
            return
        try:
            suggestion = self._generate_name(self._context_input.toPlainText())
        except NameGenerationError:
            return
        if suggestion:
            self._name_input.setText(suggestion)

    def _generate_name(self, context: str) -> str:
        """Generate name using LiteLLM when configured."""

        context = context.strip()
        if not context:
            return ""
        if self._name_generator is None:
            logger.info(
                "LiteLLM disabled (model not configured); using fallback name suggestion"
            )
            return fallback_suggest_prompt_name(context)
        try:
            return self._name_generator(context)
        except NameGenerationError as exc:
            message = str(exc).strip() or "unknown reason"
            if "not configured" in message.lower():
                logger.info(
                    "LiteLLM disabled (%s); using fallback name suggestion",
                    message,
                )
            else:
                logger.warning(
                    "Name generation failed (%s); using fallback suggestion",
                    message,
                    exc_info=exc,
                )
            return fallback_suggest_prompt_name(context)

    def _generate_description(self, context: str) -> str:
        """Generate description using LiteLLM when configured."""

        context = context.strip()
        if not context:
            return ""
        if self._description_generator is None:
            logger.info(
                "LiteLLM disabled (model not configured); using fallback description summary"
            )
            return fallback_generate_description(context)
        try:
            return self._description_generator(context)
        except DescriptionGenerationError as exc:
            message = str(exc).strip() or "unknown reason"
            if "not configured" in message.lower():
                logger.info(
                    "LiteLLM disabled (%s); using fallback description summary",
                    message,
                )
            else:
                logger.warning(
                    "Description generation failed (%s); using fallback summary",
                    message,
                    exc_info=exc,
                )
            return fallback_generate_description(context)

    def _on_refine_clicked(self) -> None:
        """Invoke the prompt engineer to refine the prompt body when available."""

        if self._prompt_engineer is None:
            QMessageBox.information(
                self,
                "Prompt refinement unavailable",
                "Configure LiteLLM in Settings to enable prompt engineering.",
            )
            return

        prompt_text = self._context_input.toPlainText()
        if not prompt_text.strip():
            QMessageBox.information(
                self,
                "Prompt required",
                "Enter prompt text before running refinement.",
            )
            return

        name = self._name_input.text().strip() or None
        description = self._description_input.toPlainText().strip() or None
        category = self._category_input.text().strip() or None
        tags = [tag.strip() for tag in self._tags_input.text().split(",") if tag.strip()]

        try:
            result = self._prompt_engineer(
                prompt_text,
                name=name,
                description=description,
                category=category,
                tags=tags,
            )
        except PromptEngineeringError as exc:
            QMessageBox.warning(self, "Prompt refinement failed", str(exc))
            return
        except PromptEngineeringUnavailable as exc:
            QMessageBox.warning(self, "Prompt refinement unavailable", str(exc))
            return
        except Exception as exc:  # pragma: no cover - defensive
            QMessageBox.warning(
                self,
                "Prompt refinement failed",
                f"Unexpected error: {exc}",
            )
            return

        self._context_input.setPlainText(result.improved_prompt)

        summary_parts: list[str] = []
        if result.analysis:
            summary_parts.append(result.analysis)
        if result.checklist:
            checklist = "\n".join(f"• {item}" for item in result.checklist)
            summary_parts.append(f"Checklist:\n{checklist}")
        if result.warnings:
            warnings = "\n".join(f"• {item}" for item in result.warnings)
            summary_parts.append(f"Warnings:\n{warnings}")

        message = (
            "\n\n".join(summary_parts)
            if summary_parts
            else "The prompt has been updated with the refined version."
        )
        QMessageBox.information(self, "Prompt refined", message)

    def _build_prompt(self) -> Optional[Prompt]:
        """Construct a Prompt object from the dialog inputs."""

        context_text = self._context_input.toPlainText().strip()
        name = self._name_input.text().strip()
        description = self._description_input.toPlainText().strip()

        if not name and context_text:
            generated_name = self._generate_name(context_text)
            if generated_name:
                name = generated_name
                self._name_input.setText(generated_name)

        if not description and context_text:
            generated_description = self._generate_description(context_text)
            if generated_description:
                description = generated_description
                self._description_input.setPlainText(generated_description)

        category = self._category_input.text().strip()
        if not name or not description:
            QMessageBox.warning(self, "Missing fields", "Name and description are required.")
            return None

        language = self._language_input.text().strip() or "en"
        tags = [tag.strip() for tag in self._tags_input.text().split(",") if tag.strip()]
        context = context_text or None
        example_input = self._example_input.toPlainText().strip() or None
        example_output = self._example_output.toPlainText().strip() or None

        if self._source_prompt is None:
            now = datetime.now(timezone.utc)
            return Prompt(
                id=uuid.uuid4(),
                name=name,
                description=description,
                category=category or "General",
                tags=tags,
                language=language,
                context=context,
                example_input=example_input,
                example_output=example_output,
                created_at=now,
                last_modified=now,
                version="1.0",
            )

        base = self._source_prompt
        ext2_copy = deepcopy(base.ext2) if base.ext2 is not None else None
        ext5_copy = deepcopy(base.ext5) if base.ext5 is not None else None
        return Prompt(
            id=base.id,
            name=name,
            description=description,
            category=category or base.category,
            tags=tags,
            language=language,
            context=context,
            example_input=example_input,
            example_output=example_output,
            last_modified=datetime.now(timezone.utc),
            version=base.version,
            author=base.author,
            quality_score=base.quality_score,
            usage_count=base.usage_count,
            related_prompts=base.related_prompts,
            created_at=base.created_at,
            modified_by=base.modified_by,
            is_active=base.is_active,
            source=base.source,
            checksum=base.checksum,
            ext1=base.ext1,
            ext2=ext2_copy,
            ext3=base.ext3,
            ext4=list(base.ext4) if base.ext4 is not None else None,
            ext5=ext5_copy,
        )


class PromptMaintenanceDialog(QDialog):
    """Expose bulk metadata maintenance utilities."""

    maintenance_applied = Signal(str)

    def __init__(
        self,
        parent: QWidget,
        manager: PromptManager,
        *,
        category_generator: Optional[Callable[[str], str]] = None,
        tags_generator: Optional[Callable[[str], Sequence[str]]] = None,
    ) -> None:
        super().__init__(parent)
        self._manager = manager
        self._category_generator = category_generator
        self._tags_generator = tags_generator
        self.setWindowTitle("Prompt Maintenance")
        self.resize(640, 420)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        description = QLabel(
            "Run maintenance tasks to enrich prompt metadata. Only prompts missing the "
            "selected metadata are updated.",
            self,
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        button_container = QWidget(self)
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(12)

        self._categories_button = QPushButton("Generate Missing Categories", self)
        self._categories_button.clicked.connect(self._on_generate_categories_clicked)  # type: ignore[arg-type]
        self._categories_button.setEnabled(self._category_generator is not None)
        if self._category_generator is None:
            self._categories_button.setToolTip("Category suggestions are unavailable.")
        button_layout.addWidget(self._categories_button)

        self._tags_button = QPushButton("Generate Missing Tags", self)
        self._tags_button.clicked.connect(self._on_generate_tags_clicked)  # type: ignore[arg-type]
        self._tags_button.setEnabled(self._tags_generator is not None)
        if self._tags_generator is None:
            self._tags_button.setToolTip("Tag suggestions are unavailable.")
        button_layout.addWidget(self._tags_button)

        button_layout.addStretch(1)
        layout.addWidget(button_container)

        self._log_view = QPlainTextEdit(self)
        self._log_view.setReadOnly(True)
        layout.addWidget(self._log_view, stretch=1)

        self._buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
        self._buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        layout.addWidget(self._buttons)

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self._log_view.appendPlainText(f"[{timestamp}] {message}")

    def _collect_prompts(self) -> List[Prompt]:
        try:
            return self._manager.repository.list()
        except RepositoryError as exc:
            self._append_log(f"Unable to load prompts: {exc}")
            return []

    @staticmethod
    def _prompt_context(prompt: Prompt) -> str:
        return (prompt.context or prompt.description or prompt.example_input or "").strip()

    def _on_generate_categories_clicked(self) -> None:
        if self._category_generator is None:
            self._append_log("Category generator is unavailable.")
            return
        prompts = self._collect_prompts()
        if not prompts:
            return
        updated = 0
        for prompt in prompts:
            if (prompt.category or "").strip():
                continue
            context = self._prompt_context(prompt)
            if not context:
                continue
            try:
                suggestion = (self._category_generator(context) or "").strip()
            except Exception as exc:  # noqa: BLE001
                self._append_log(f"Failed to suggest category for '{prompt.name}': {exc}")
                continue
            if not suggestion:
                continue
            updated_prompt = replace(
                prompt,
                category=suggestion,
                last_modified=datetime.now(timezone.utc),
            )
            try:
                self._manager.update_prompt(updated_prompt)
            except PromptManagerError as exc:
                self._append_log(f"Unable to update '{prompt.name}': {exc}")
                continue
            updated += 1
            self._append_log(f"Set category '{suggestion}' for '{prompt.name}'.")
        self._append_log(f"Category task completed. Updated {updated} prompt(s).")
        if updated:
            self.maintenance_applied.emit(f"Generated categories for {updated} prompt(s).")

    def _on_generate_tags_clicked(self) -> None:
        if self._tags_generator is None:
            self._append_log("Tag generator is unavailable.")
            return
        prompts = self._collect_prompts()
        if not prompts:
            return
        updated = 0
        for prompt in prompts:
            existing_tags = [tag.strip() for tag in (prompt.tags or []) if tag.strip()]
            if existing_tags:
                continue
            context = self._prompt_context(prompt)
            if not context:
                continue
            try:
                suggestions = self._tags_generator(context) or []
            except Exception as exc:  # noqa: BLE001
                self._append_log(f"Failed to suggest tags for '{prompt.name}': {exc}")
                continue
            tags = [tag.strip() for tag in suggestions if str(tag).strip()]
            if not tags:
                continue
            updated_prompt = replace(
                prompt,
                tags=tags,
                last_modified=datetime.now(timezone.utc),
            )
            try:
                self._manager.update_prompt(updated_prompt)
            except PromptManagerError as exc:
                self._append_log(f"Unable to update '{prompt.name}': {exc}")
                continue
            updated += 1
            self._append_log(f"Assigned tags {tags} to '{prompt.name}'.")
        self._append_log(f"Tag task completed. Updated {updated} prompt(s).")
        if updated:
            self.maintenance_applied.emit(f"Generated tags for {updated} prompt(s).")


class TemplateDialog(QDialog):
    """Modal dialog for creating or editing task templates."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        template: Optional[TaskTemplate] = None,
        prompts: Sequence[Prompt] = (),
    ) -> None:
        super().__init__(parent)
        self._source_template = template
        self._result_template: Optional[TaskTemplate] = None
        self._prompts = list(prompts)
        self._missing_prompt_ids: list[str] = []
        self.setWindowTitle("New Task Template" if template is None else "Edit Task Template")
        self.resize(720, 640)
        self._build_ui()
        if template is not None:
            self._populate(template)

    @property
    def result_template(self) -> Optional[TaskTemplate]:
        """Return the resulting template after dialog acceptance."""

        return self._result_template

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()

        self._name_input = QLineEdit(self)
        form.addRow("Name*", self._name_input)

        self._description_input = QPlainTextEdit(self)
        self._description_input.setPlaceholderText("Short description shown alongside the template…")
        self._description_input.setFixedHeight(80)
        form.addRow("Description*", self._description_input)

        self._category_input = QLineEdit(self)
        self._category_input.setPlaceholderText("Optional category label (e.g. Code Analysis)")
        form.addRow("Category", self._category_input)

        self._tags_input = QLineEdit(self)
        self._tags_input.setPlaceholderText("Comma-separated tags (analysis, refactor, …)")
        form.addRow("Tags", self._tags_input)

        self._version_input = QLineEdit(self)
        self._version_input.setPlaceholderText("1.0")
        form.addRow("Version", self._version_input)

        self._is_active_checkbox = QCheckBox("Template is active", self)
        self._is_active_checkbox.setChecked(True)
        form.addRow("", self._is_active_checkbox)

        layout.addLayout(form)

        layout.addWidget(QLabel("Default input", self))
        self._default_input_editor = QPlainTextEdit(self)
        self._default_input_editor.setPlaceholderText(
            "Optional starter input pasted into the workspace when the template is applied…"
        )
        self._default_input_editor.setFixedHeight(100)
        layout.addWidget(self._default_input_editor)

        layout.addWidget(QLabel("Notes", self))
        self._notes_input = QPlainTextEdit(self)
        self._notes_input.setPlaceholderText("Optional guidance or checklist shown after applying the template…")
        self._notes_input.setFixedHeight(80)
        layout.addWidget(self._notes_input)

        prompt_label = QLabel("Select prompts to include*", self)
        layout.addWidget(prompt_label)

        self._prompt_list = QListWidget(self)
        self._prompt_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self._prompt_list.setAlternatingRowColors(True)
        for prompt in self._prompts:
            label = prompt.name
            if prompt.category:
                label += f" — {prompt.category}"
            item = QListWidgetItem(label, self._prompt_list)
            item.setData(Qt.UserRole, str(prompt.id))
            item.setToolTip(prompt.description)
        layout.addWidget(self._prompt_list, stretch=1)

        self._missing_label = QLabel("", self)
        self._missing_label.setWordWrap(True)
        self._missing_label.setStyleSheet("color: #d97706;")
        self._missing_label.setVisible(False)
        layout.addWidget(self._missing_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self._on_accept)  # type: ignore[arg-type]
        buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        layout.addWidget(buttons)

    def _populate(self, template: TaskTemplate) -> None:
        self._name_input.setText(template.name)
        self._description_input.setPlainText(template.description)
        self._category_input.setText(template.category or "")
        self._tags_input.setText(", ".join(template.tags))
        self._version_input.setText(template.version)
        self._is_active_checkbox.setChecked(template.is_active)
        if template.default_input:
            self._default_input_editor.setPlainText(template.default_input)
        if template.notes:
            self._notes_input.setPlainText(template.notes)

        selected_ids = {str(pid) for pid in template.prompt_ids}
        available_ids = set()
        for index in range(self._prompt_list.count()):
            item = self._prompt_list.item(index)
            prompt_id = item.data(Qt.UserRole)
            available_ids.add(prompt_id)
            if prompt_id in selected_ids:
                item.setSelected(True)
        missing_ids = sorted(selected_ids - available_ids)
        if missing_ids:
            self._missing_prompt_ids = missing_ids
            self._missing_label.setText(
                "Some prompts referenced by this template are unavailable: "
                + ", ".join(missing_ids)
            )
            self._missing_label.setVisible(True)
        else:
            self._missing_label.setVisible(False)

    def _on_accept(self) -> None:
        name = self._name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Invalid template", "Provide a template name.")
            return
        description = self._description_input.toPlainText().strip()
        if not description:
            QMessageBox.warning(self, "Invalid template", "Provide a description for the template.")
            return

        selected_ids: list[uuid.UUID] = []
        for item in self._prompt_list.selectedItems():
            raw = item.data(Qt.UserRole)
            try:
                selected_ids.append(uuid.UUID(str(raw)))
            except (TypeError, ValueError):
                continue
        if not selected_ids and not self._missing_prompt_ids:
            QMessageBox.warning(
                self,
                "Invalid template",
                "Select at least one prompt to include in the template.",
            )
            return

        category = self._category_input.text().strip() or None
        tags = [tag.strip() for tag in self._tags_input.text().split(",") if tag.strip()]
        default_input = self._default_input_editor.toPlainText().strip() or None
        notes = self._notes_input.toPlainText().strip() or None
        version = self._version_input.text().strip() or (
            self._source_template.version if self._source_template else "1.0"
        )
        is_active = self._is_active_checkbox.isChecked()

        template_id = self._source_template.id if self._source_template else uuid.uuid4()
        created_at = self._source_template.created_at if self._source_template else datetime.now(timezone.utc)
        ext1 = self._source_template.ext1 if self._source_template else None
        ext2 = self._source_template.ext2 if self._source_template else None
        ext3 = self._source_template.ext3 if self._source_template else None

        prompt_ids = selected_ids or [uuid.UUID(value) for value in self._missing_prompt_ids]

        self._result_template = TaskTemplate(
            id=template_id,
            name=name,
            description=description,
            prompt_ids=prompt_ids,
            default_input=default_input,
            category=category,
            tags=tags,
            notes=notes,
            is_active=is_active,
            version=version,
            created_at=created_at,
            last_modified=datetime.now(timezone.utc),
            ext1=ext1,
            ext2=ext2,
            ext3=ext3,
        )
        self.accept()


class SaveResultDialog(QDialog):
    """Collect optional notes before persisting or updating a prompt execution."""

    def __init__(
        self,
        parent: Optional[QWidget],
        *,
        prompt_name: str,
        default_text: str = "",
        max_chars: int = 400,
        button_text: str = "Save",
        enable_rating: bool = True,
        initial_rating: Optional[float] = None,
    ) -> None:
        super().__init__(parent)
        self._max_chars = max_chars
        self._summary = ""
        self._rating: Optional[int] = (
            int(initial_rating) if initial_rating is not None else None
        )
        self._enable_rating = enable_rating
        self.setWindowTitle(f"{button_text} Result — {prompt_name}")
        self._build_ui(default_text, button_text)

    @property
    def note(self) -> str:
        """Return the trimmed note content."""

        return self._summary

    @property
    def rating(self) -> Optional[int]:
        """Return the selected rating, if any."""

        return self._rating

    def _build_ui(self, default_text: str, button_text: str) -> None:
        layout = QVBoxLayout(self)

        message = QLabel(
            "Add an optional summary or notes for this execution. The note will be stored with the history entry.",
            self,
        )
        message.setWordWrap(True)
        layout.addWidget(message)

        self._note_input = QPlainTextEdit(self)
        self._note_input.setPlaceholderText("Optional summary / notes…")
        if default_text:
            snippet = default_text.strip()
            if len(snippet) > self._max_chars:
                snippet = snippet[: self._max_chars - 3].rstrip() + "..."
            self._note_input.setPlainText(snippet)
        layout.addWidget(self._note_input)

        if self._enable_rating:
            rating_layout = QHBoxLayout()
            rating_label = QLabel("Rating (1-10):", self)
            self._rating_input = QComboBox(self)
            self._rating_input.addItem("No rating", None)
            for value in range(1, 11):
                self._rating_input.addItem(str(value), value)
            if self._rating is not None:
                index = self._rating_input.findData(self._rating)
                if index >= 0:
                    self._rating_input.setCurrentIndex(index)
            rating_layout.addWidget(rating_label)
            rating_layout.addWidget(self._rating_input)
            layout.addLayout(rating_layout)
        else:
            self._rating_input = None

        buttons = QDialogButtonBox(self)
        self._save_button = buttons.addButton(button_text, QDialogButtonBox.AcceptRole)
        buttons.addButton(QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)  # type: ignore[arg-type]
        buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        summary = self._note_input.toPlainText().strip()
        if summary and len(summary) > self._max_chars:
            summary = summary[: self._max_chars].rstrip()
        self._summary = summary
        if self._enable_rating and self._rating_input is not None:
            selected = self._rating_input.currentData()
            self._rating = int(selected) if selected is not None else None
        self.accept()


class MarkdownPreviewDialog(QDialog):
    """Display markdown content rendered in a read-only viewer."""

    def __init__(self, markdown_text: str, parent: Optional[QWidget], *, title: str = "Rendered Output") -> None:
        super().__init__(parent)
        self._markdown_text = markdown_text
        self.setWindowTitle(title)
        self.resize(720, 540)
        self._build_ui()

    def _build_ui(self) -> None:
        """Construct the markdown preview layout and wire controls."""

        layout = QVBoxLayout(self)
        viewer = QTextBrowser(self)
        viewer.setOpenExternalLinks(True)
        content = self._markdown_text.strip()
        if content:
            viewer.setMarkdown(content)
        else:
            viewer.setMarkdown("*No content available.*")
        layout.addWidget(viewer, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close, parent=self)
        buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        buttons.accepted.connect(self.accept)  # type: ignore[arg-type]
        layout.addWidget(buttons)


def _diff_entry_to_text(entry: CatalogDiffEntry) -> str:
    header = f"[{entry.change_type.value.upper()}] {entry.name} ({entry.prompt_id})"
    if entry.diff:
        body = textwrap.indent(entry.diff, "  ")
        return f"{header}\n{body}"
    return f"{header}\n  (no diff)"


class CatalogPreviewDialog(QDialog):
    """Show a diff preview before applying catalogue changes."""

    def __init__(self, diff: CatalogDiff, parent=None) -> None:
        super().__init__(parent)
        self._diff = diff
        self._apply = False
        self.setWindowTitle("Catalogue Preview")
        self.resize(720, 480)
        self._build_ui()

    @property
    def apply_requested(self) -> bool:
        """Return True when the user confirmed the import."""

        return self._apply

    def _build_ui(self) -> None:
        summary_text = (
            f"Added: {self._diff.added} | Updated: {self._diff.updated} | "
            f"Skipped: {self._diff.skipped} | Unchanged: {self._diff.unchanged}"
        )
        layout = QVBoxLayout(self)
        summary_label = QLabel(summary_text, self)
        summary_label.setWordWrap(True)
        layout.addWidget(summary_label)

        diff_view = QPlainTextEdit(self)
        diff_view.setReadOnly(True)
        diff_view.setPlainText(self._format_diff(self._diff))
        layout.addWidget(diff_view, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        buttons.accepted.connect(self._on_accept)  # type: ignore[arg-type]
        buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        layout.addWidget(buttons)

    @staticmethod
    def _format_diff(diff: CatalogDiff) -> str:
        if not diff.entries:
            return "No changes detected."
        entries = [
            f"Source: {diff.source or 'builtin'}",
            "",
        ]
        entries.extend(_diff_entry_to_text(entry) for entry in diff.entries)
        return "\n".join(entries)

    def _on_accept(self) -> None:
        self._apply = True
        self.accept()


__all__ = [
    "CatalogPreviewDialog",
    "MarkdownPreviewDialog",
    "PromptDialog",
    "PromptMaintenanceDialog",
    "SaveResultDialog",
    "TemplateDialog",
    "fallback_suggest_prompt_name",
    "fallback_generate_description",
]
