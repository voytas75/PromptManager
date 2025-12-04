"""Prompt editing dialog composition utilities.

Updates:
  v0.2.0 - 2025-12-04 - Modularized prompt editor using dedicated mixins and package structure.
  v0.1.0 - 2025-12-03 - Moved prompt editor dialogs out of gui.dialogs monolith.
"""

from __future__ import annotations

import uuid
from copy import deepcopy
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from models.prompt_model import Prompt

from ..base import CollapsibleTextSection, strip_scenarios_metadata
from .mixins import (
    _PromptDialogCategoryMixin,
    _PromptDialogRefinementMixin,
    _PromptDialogScenarioMixin,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from core.prompt_engineering import PromptRefinement
    from models.category_model import PromptCategory
else:  # pragma: no cover - runtime fallbacks for postponed annotations
    Callable = Sequence = PromptRefinement = PromptCategory = Any  # type: ignore[assignment]


class PromptDialog(
    _PromptDialogScenarioMixin,
    _PromptDialogCategoryMixin,
    _PromptDialogRefinementMixin,
    QDialog,
):
    """Modal dialog used for creating or editing prompt records."""

    applied = Signal(Prompt)
    execute_context_requested = Signal(Prompt, str)

    def __init__(
        self,
        parent: QWidget | None = None,
        prompt: Prompt | None = None,
        category_provider: Callable[[], Sequence[PromptCategory]] | None = None,
        name_generator: Callable[[str], str] | None = None,
        description_generator: Callable[[str], str] | None = None,
        category_generator: Callable[[str], str] | None = None,
        tags_generator: Callable[[str], Sequence[str]] | None = None,
        scenario_generator: Callable[[str], Sequence[str]] | None = None,
        prompt_engineer: Callable[..., PromptRefinement] | None = None,
        structure_refiner: Callable[..., PromptRefinement] | None = None,
        version_history_handler: Callable[[Prompt], None] | None = None,
    ) -> None:
        """Configure the prompt editor dialog, injecting optional helpers."""
        super().__init__(parent)
        self._source_prompt = prompt
        self._result_prompt: Prompt | None = None
        self._category_provider = category_provider
        self._name_generator = name_generator
        self._description_generator = description_generator
        self._prompt_engineer = prompt_engineer
        self._structure_refiner = structure_refiner
        self._category_generator = category_generator
        self._tags_generator = tags_generator
        self._scenario_generator = scenario_generator
        self._categories: list[PromptCategory] = []
        self._delete_requested = False
        self._delete_button: QPushButton | None = None
        self._apply_button: QPushButton | None = None
        self._version_label: QLabel | None = None
        self._version_history_button: QPushButton | None = None
        self._version_history_handler = version_history_handler
        self._generate_category_button: QPushButton | None = None
        self._generate_tags_button: QPushButton | None = None
        self._generate_scenarios_button: QPushButton | None = None
        self._execute_context_button: QPushButton | None = None
        self.setWindowTitle("Create Prompt" if prompt is None else "Edit Prompt")
        self.setMinimumSize(900, 780)
        self.resize(1100, 880)
        self._build_ui()
        if prompt is not None:
            self._populate(prompt)

    @property
    def result_prompt(self) -> Prompt | None:
        """Return the prompt produced by the dialog after acceptance."""
        return self._result_prompt

    @property
    def delete_requested(self) -> bool:
        """Return True when the user chose to delete the prompt instead of saving."""
        return self._delete_requested

    @property
    def source_prompt(self) -> Prompt | None:
        """Expose the prompt supplied to the dialog for convenience."""
        return self._source_prompt

    def _build_ui(self) -> None:
        """Construct the dialog layout and wire interactions."""
        main_layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignTop)
        form_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

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

        version_container = QWidget(self)
        version_container_layout = QHBoxLayout(version_container)
        version_container_layout.setContentsMargins(0, 0, 0, 0)
        version_container_layout.setSpacing(8)
        self._version_label = QLabel("Not yet saved", self)
        self._version_label.setObjectName("promptDialogVersion")
        version_container_layout.addWidget(self._version_label)
        self._version_history_button = QPushButton("Version History", self)
        self._version_history_button.setObjectName("promptDialogHistoryButton")
        self._version_history_button.setCursor(Qt.PointingHandCursor)
        self._version_history_button.clicked.connect(self._on_version_history_clicked)  # type: ignore[arg-type]
        version_container_layout.addWidget(self._version_history_button)
        version_container_layout.addStretch(1)
        form_layout.addRow("Version", version_container)

        self._category_input = QComboBox(self)
        self._category_input.setEditable(True)
        self._category_input.setInsertPolicy(QComboBox.NoInsert)
        self._category_input.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._category_input.setMinimumContentsLength(12)
        line_edit = self._category_input.lineEdit()
        if line_edit is not None:
            line_edit.setPlaceholderText("General")
        self._language_input = QLineEdit(self)
        self._author_input = QLineEdit(self)
        self._tags_input = QLineEdit(self)
        self._context_input = QPlainTextEdit(self)
        self._context_input.setPlaceholderText("Paste the full prompt text here…")
        self._context_input.setMinimumHeight(320)
        context_policy = self._context_input.sizePolicy()
        context_policy.setVerticalPolicy(QSizePolicy.Expanding)
        self._context_input.setSizePolicy(context_policy)
        self._description_input = QPlainTextEdit(self)
        self._description_input.setFixedHeight(60)
        self._description_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._example_input = CollapsibleTextSection("Example Input", self)
        self._example_input.setPlaceholderText("Optional example input…")
        self._example_output = CollapsibleTextSection("Example Output", self)
        self._example_output.setPlaceholderText("Optional example output…")

        self._language_input.setPlaceholderText("en")
        self._author_input.setPlaceholderText("Optional author name…")
        self._author_input.setMaximumWidth(260)
        self._author_input.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._tags_input.setPlaceholderText("tag-a, tag-b")

        metadata_container = QWidget(self)
        metadata_layout = QGridLayout(metadata_container)
        metadata_layout.setContentsMargins(0, 0, 0, 0)
        metadata_layout.setHorizontalSpacing(12)
        metadata_layout.setVerticalSpacing(4)

        author_label = QLabel("Author", metadata_container)
        metadata_layout.addWidget(author_label, 0, 0)
        metadata_layout.addWidget(self._author_input, 1, 0)

        category_label = QLabel("Category", metadata_container)
        metadata_layout.addWidget(category_label, 0, 1)

        category_container = QWidget(metadata_container)
        category_container_layout = QHBoxLayout(category_container)
        category_container_layout.setContentsMargins(0, 0, 0, 0)
        category_container_layout.setSpacing(6)
        category_container_layout.addWidget(self._category_input)
        self._populate_category_options()
        self._generate_category_button = QPushButton("Suggest", self)
        self._generate_category_button.setToolTip("Suggest a category based on the prompt body.")
        self._generate_category_button.clicked.connect(self._on_generate_category_clicked)  # type: ignore[arg-type]
        if self._category_generator is None:
            self._generate_category_button.setEnabled(False)
            self._generate_category_button.setToolTip(
                "Category suggestions require the main app context."
            )
        category_container_layout.addWidget(self._generate_category_button)
        metadata_layout.addWidget(category_container, 1, 1)

        tags_label = QLabel("Tags", metadata_container)
        metadata_layout.addWidget(tags_label, 0, 2)

        tags_container = QWidget(metadata_container)
        tags_container_layout = QHBoxLayout(tags_container)
        tags_container_layout.setContentsMargins(0, 0, 0, 0)
        tags_container_layout.setSpacing(6)
        tags_container_layout.addWidget(self._tags_input)
        self._generate_tags_button = QPushButton("Suggest", self)
        self._generate_tags_button.setToolTip("Suggest tags based on the prompt body.")
        self._generate_tags_button.clicked.connect(self._on_generate_tags_clicked)  # type: ignore[arg-type]
        if self._tags_generator is None:
            self._generate_tags_button.setEnabled(False)
            self._generate_tags_button.setToolTip("Tag suggestions require the main app context.")
        tags_container_layout.addWidget(self._generate_tags_button)
        metadata_layout.addWidget(tags_container, 1, 2)
        metadata_layout.setColumnStretch(0, 1)
        metadata_layout.setColumnStretch(1, 2)
        metadata_layout.setColumnStretch(2, 2)

        form_layout.addRow("Language", self._language_input)
        form_layout.addRow("", metadata_container)
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
        self._structure_refine_button = QPushButton("Refine Structure", self)
        self._structure_refine_button.setToolTip(
            "Improve formatting and section layout without altering the prompt's intent."
        )
        self._structure_refine_button.clicked.connect(self._on_refine_structure_clicked)  # type: ignore[arg-type]
        if self._structure_refiner is None:
            try:
                self._structure_refine_button.setEnabled(False)
            except AttributeError:  # pragma: no cover - stubbed widgets during tests
                pass
            self._structure_refine_button.setToolTip(
                "Configure LiteLLM in Settings to enable prompt engineering."
            )
        refine_row = QWidget(self)
        refine_row_layout = QHBoxLayout(refine_row)
        refine_row_layout.setContentsMargins(0, 0, 0, 0)
        refine_row_layout.setSpacing(8)
        refine_row_layout.addWidget(self._refine_button)
        refine_row_layout.addWidget(self._structure_refine_button)
        self._execute_context_button = QPushButton("Execute as Context", self)
        self._execute_context_button.setToolTip(
            "Run the current prompt body as context for an ad-hoc execution."
        )
        self._execute_context_button.clicked.connect(self._on_execute_context_clicked)  # type: ignore[arg-type]
        refine_row_layout.addWidget(self._execute_context_button)
        form_layout.addRow("", refine_row)
        form_layout.addRow("Description", self._description_input)
        self._scenarios_input = QPlainTextEdit(self)
        self._scenarios_input.setPlaceholderText("One scenario per line…")
        self._scenarios_input.setFixedHeight(90)
        scenarios_container = QWidget(self)
        scenarios_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        scenarios_layout = QVBoxLayout(scenarios_container)
        scenarios_layout.setContentsMargins(0, 0, 0, 0)
        scenarios_layout.setSpacing(4)
        scenarios_layout.addWidget(self._scenarios_input)
        self._generate_scenarios_button = QPushButton("Generate Scenarios", self)
        self._generate_scenarios_button.setToolTip(
            "Analyse the prompt body to suggest practical usage scenarios."
        )
        self._generate_scenarios_button.clicked.connect(self._on_generate_scenarios_clicked)  # type: ignore[arg-type]
        if self._scenario_generator is None:
            self._generate_scenarios_button.setToolTip(
                "Generate heuristic scenarios from the prompt body. Configure LiteLLM for AI help."
            )
        scenarios_layout.addWidget(self._generate_scenarios_button, alignment=Qt.AlignRight)
        form_layout.addRow("Scenarios", scenarios_container)

        examples_container = QWidget(self)
        examples_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        examples_layout = QHBoxLayout(examples_container)
        examples_layout.setContentsMargins(0, 0, 0, 0)
        examples_layout.setSpacing(12)
        examples_layout.addWidget(self._example_input, 1)
        examples_layout.addWidget(self._example_output, 1)
        form_layout.addRow("Examples", examples_container)

        main_layout.addLayout(form_layout)

        self._buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        self._buttons.accepted.connect(self._on_accept)  # type: ignore[arg-type]
        self._buttons.rejected.connect(self.reject)  # type: ignore[arg-type]
        if self._source_prompt is not None:
            apply_button = self._buttons.addButton("Apply", QDialogButtonBox.ApplyRole)
            apply_button.setToolTip("Save changes without closing the dialog.")
            apply_button.clicked.connect(self._on_apply_clicked)  # type: ignore[arg-type]
            self._apply_button = apply_button
            delete_button = self._buttons.addButton(
                "Delete",
                QDialogButtonBox.DestructiveRole,
            )
            delete_button.setToolTip("Delete this prompt from the catalogue.")
            delete_button.clicked.connect(self._on_delete_clicked)  # type: ignore[arg-type]
            self._delete_button = delete_button
        main_layout.addWidget(self._buttons)
        self._context_input.textChanged.connect(self._on_context_changed)  # type: ignore[arg-type]
        self._update_version_controls(self._source_prompt.version if self._source_prompt else None)
        self._refresh_execute_context_button()

    def prefill_from_prompt(self, prompt: Prompt) -> None:
        """Populate inputs from an existing prompt while staying in creation mode."""
        self._populate(prompt)
        self._result_prompt = None
        self._delete_requested = False

    def _populate(self, prompt: Prompt) -> None:
        """Fill inputs with the existing prompt values for editing."""
        self._name_input.setText(prompt.name)
        self._set_category_value(prompt.category)
        self._language_input.setText(prompt.language)
        self._author_input.setText(prompt.author or "")
        self._tags_input.setText(", ".join(prompt.tags))
        self._context_input.setPlainText(prompt.context or "")
        self._description_input.setPlainText(prompt.description)
        self._example_input.setPlainText(prompt.example_input or "")
        self._example_output.setPlainText(prompt.example_output or "")
        self._scenarios_input.setPlainText("\n".join(prompt.scenarios))
        self._update_version_controls(prompt.version)

    def _on_accept(self) -> None:
        """Validate inputs, build the prompt, and close the dialog."""
        prompt = self._build_prompt()
        if prompt is None:
            return
        self._result_prompt = prompt
        self.accept()

    def _on_apply_clicked(self) -> None:
        """Persist prompt edits while keeping the dialog open."""
        prompt = self._build_prompt()
        if prompt is None:
            return
        self._result_prompt = prompt
        self.applied.emit(prompt)

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

    def _on_execute_context_clicked(self) -> None:
        """Trigger the execute-as-context workflow from the dialog."""
        if self._source_prompt is None:
            QMessageBox.information(
                self,
                "Execute as context",
                "Save the prompt before executing it as context.",
            )
            return
        context_text = self._context_input.toPlainText().strip()
        if not context_text:
            QMessageBox.warning(
                self,
                "Prompt body required",
                "Enter prompt text before executing as context.",
            )
            return
        self.execute_context_requested.emit(self._source_prompt, context_text)

    def _build_prompt(self) -> Prompt | None:
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

        category = self._current_category_value()
        author = self._author_input.text().strip() or None
        if not name or not description:
            QMessageBox.warning(self, "Missing fields", "Name and description are required.")
            return None

        language = self._language_input.text().strip() or "en"
        tags = [tag.strip() for tag in self._tags_input.text().split(",") if tag.strip()]
        context = context_text or None
        example_input = self._example_input.toPlainText().strip() or None
        example_output = self._example_output.toPlainText().strip() or None
        scenarios = self._collect_scenarios()

        if self._source_prompt is None:
            now = datetime.now(UTC)
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
                scenarios=scenarios,
                created_at=now,
                last_modified=now,
                version="1",
                author=author,
            )

        base = self._source_prompt
        ext2_copy = deepcopy(base.ext2) if base.ext2 is not None else None
        ext5_copy = strip_scenarios_metadata(base.ext5)
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
            scenarios=scenarios,
            last_modified=datetime.now(UTC),
            version=base.version,
            author=author,
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

    def update_source_prompt(self, prompt: Prompt) -> None:
        """Refresh the backing prompt after an in-place update."""
        self._source_prompt = prompt
        self._populate(prompt)
        self._refresh_execute_context_button()

    def _update_version_controls(self, version: str | None) -> None:
        """Refresh the version label and history button state."""
        if self._version_label is None or self._version_history_button is None:
            return
        label_text = (version or "").strip() or "Not yet saved"
        self._version_label.setText(label_text)
        history_available = (
            self._source_prompt is not None and self._version_history_handler is not None
        )
        self._version_history_button.setVisible(history_available)
        self._version_history_button.setEnabled(history_available)

    def _refresh_execute_context_button(self) -> None:
        """Toggle execute-context control availability based on dialog state."""
        if self._execute_context_button is None:
            return
        has_source_prompt = self._source_prompt is not None
        self._execute_context_button.setEnabled(has_source_prompt)
        if has_source_prompt:
            self._execute_context_button.setToolTip(
                "Run the current prompt body as context without closing the editor."
            )
        else:
            self._execute_context_button.setToolTip(
                "Available when editing a saved prompt with a prompt body."
            )

    def _on_version_history_clicked(self) -> None:
        """Open the version history dialog via the provided handler."""
        if self._version_history_handler is None or self._source_prompt is None:
            return
        self._version_history_handler(self._source_prompt)
