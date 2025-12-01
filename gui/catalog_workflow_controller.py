"""Controllers for catalogue import/export and maintenance flows.

Updates:
  v0.1.0 - 2025-12-01 - Extracted catalogue workflows from MainWindow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable
from uuid import UUID

from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox, QWidget

from core import (
    PromptManager,
    diff_prompt_catalog,
    export_prompt_catalog,
    import_prompt_catalog,
)
from models.prompt_model import Prompt

from .dialogs import CatalogPreviewDialog, PromptMaintenanceDialog


class CatalogWorkflowController:
    """Coordinate catalogue import/export and maintenance operations."""

    def __init__(
        self,
        parent: QWidget,
        manager: PromptManager,
        *,
        load_prompts: Callable[[str], None],
        current_search_text: Callable[[], str],
        select_prompt: Callable[[UUID], None],
        current_prompt: Callable[[], Prompt | None],
        show_status: Callable[[str, int], None],
        generate_category: Callable[[str], str],
        generate_tags: Callable[[str], list[str]],
    ) -> None:
        self._parent = parent
        self._manager = manager
        self._load_prompts = load_prompts
        self._current_search_text = current_search_text
        self._select_prompt = select_prompt
        self._current_prompt = current_prompt
        self._show_status = show_status
        self._generate_category = generate_category
        self._generate_tags = generate_tags

    def open_import_dialog(self) -> None:
        """Preview catalogue diff and optionally apply updates."""

        file_path, _ = QFileDialog.getOpenFileName(
            self._parent,
            "Select catalogue file",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        catalog_path: Path | None
        if file_path:
            catalog_path = Path(file_path)
        else:
            directory = QFileDialog.getExistingDirectory(
                self._parent,
                "Select catalogue directory",
                "",
            )
            if not directory:
                return
            catalog_path = Path(directory)

        catalog_path = catalog_path.expanduser()

        try:
            preview = diff_prompt_catalog(self._manager, catalog_path)
        except Exception as exc:  # pragma: no cover - user-driven failure
            QMessageBox.warning(self._parent, "Catalogue preview failed", str(exc))
            return

        dialog = CatalogPreviewDialog(preview, self._parent)
        if dialog.exec() != QDialog.Accepted or not dialog.apply_requested:
            return

        try:
            result = import_prompt_catalog(self._manager, catalog_path)
        except Exception as exc:  # pragma: no cover - user-driven failure
            QMessageBox.critical(self._parent, "Catalogue import failed", str(exc))
            return

        message = (
            f"Catalogue applied (added {result.added}, updated {result.updated}, "
            f"skipped {result.skipped}, errors {result.errors})"
        )
        if result.errors:
            QMessageBox.warning(self._parent, "Catalogue applied with errors", message)
        else:
            self._show_status(message, 5000)
        self._load_prompts(self._current_search_text())

    def export_catalog(self) -> None:
        """Export current prompts to JSON or YAML."""

        default_path = str(Path.home() / "prompt_catalog.json")
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self._parent,
            "Export prompt catalogue",
            default_path,
            "JSON Files (*.json);;YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if not file_path:
            return
        export_path = Path(file_path)
        lower_suffix = export_path.suffix.lower()
        if selected_filter.startswith("YAML") or lower_suffix in {".yaml", ".yml"}:
            fmt = "yaml"
        else:
            fmt = "json"
        try:
            resolved = export_prompt_catalog(self._manager, export_path, fmt=fmt)
        except Exception as exc:  # pragma: no cover - user-driven failure
            QMessageBox.critical(self._parent, "Export failed", str(exc))
            return
        self._show_status(f"Catalogue exported to {resolved}", 5000)

    def open_maintenance_dialog(self) -> None:
        """Launch the maintenance dialog for batch metadata helpers."""

        dialog = PromptMaintenanceDialog(
            self._parent,
            self._manager,
            category_generator=self._generate_category,
            tags_generator=self._generate_tags,
        )
        dialog.maintenance_applied.connect(self._on_maintenance_applied)  # type: ignore[arg-type]
        dialog.exec()

    def _on_maintenance_applied(self, message: str) -> None:
        selected = self._current_prompt()
        selected_id = selected.id if selected is not None else None
        self._load_prompts(self._current_search_text())
        if isinstance(selected_id, UUID):
            self._select_prompt(selected_id)
        if message:
            self._show_status(message, 5000)


__all__ = ["CatalogWorkflowController"]
