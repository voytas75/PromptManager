"""Controllers for catalogue import/export and maintenance flows.

Updates:
  v0.1.1 - 2025-12-05 - Avoid reopening dialogs on cancel by confirming directory import.
  v0.1.0 - 2025-12-01 - Extracted catalogue workflows from MainWindow.
"""

from __future__ import annotations

from collections import abc
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox, QWidget

from core import (
    PromptManager,
    diff_prompt_catalog,
    export_prompt_catalog,
    import_prompt_catalog,
)

from .dialogs import CatalogPreviewDialog, PromptMaintenanceDialog

if TYPE_CHECKING:
    from models.prompt_model import Prompt


class CatalogWorkflowController:
    """Coordinate catalogue import/export and maintenance operations."""

    def __init__(
        self,
        parent: QWidget,
        manager: PromptManager,
        *,
        load_prompts: abc.Callable[[str], None],
        current_search_text: abc.Callable[[], str],
        select_prompt: abc.Callable[[UUID], None],
        current_prompt: abc.Callable[[], Prompt | None],
        show_status: abc.Callable[[str, int], None],
        generate_category: abc.Callable[[str], str],
        generate_tags: abc.Callable[[str], list[str]],
    ) -> None:
        """Store shared callbacks and dependencies for catalogue workflows."""
        self._parent = parent
        self._manager = manager
        self._load_prompts = self._ensure_callable(load_prompts, "load_prompts")
        self._current_search_text = self._ensure_callable(
            current_search_text,
            "current_search_text",
        )
        self._select_prompt = self._ensure_callable(select_prompt, "select_prompt")
        self._current_prompt = self._ensure_callable(current_prompt, "current_prompt")
        self._show_status = self._ensure_callable(show_status, "show_status")
        self._generate_category = self._ensure_callable(generate_category, "generate_category")
        self._generate_tags = self._ensure_callable(generate_tags, "generate_tags")

    def open_import_dialog(self) -> None:
        """Preview catalogue diff and optionally apply updates."""
        file_path, _ = QFileDialog.getOpenFileName(
            self._parent,
            "Select catalogue file",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        catalog_path: Path | None = None
        if file_path:
            catalog_path = Path(file_path)
        else:
            use_directory = QMessageBox.question(
                self._parent,
                "Import directory?",
                "No file selected. Do you want to import all JSON files from a directory instead?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if use_directory != QMessageBox.Yes:
                return
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

    @staticmethod
    def _ensure_callable(callback: Any, name: str) -> abc.Callable[..., Any]:
        """Ensure the provided callback is callable before storing it."""
        if not isinstance(callback, abc.Callable):
            raise TypeError(f"{name} callback must be callable")
        return callback


__all__ = ["CatalogWorkflowController"]
