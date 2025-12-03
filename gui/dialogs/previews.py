"""Preview dialogs like markdown and application info windows.

Updates:
  v0.1.0 - 2025-12-03 - Split preview dialogs out of gui.dialogs.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .base import collect_system_info

try:
    from ..resources import load_application_icon  # type: ignore
except ImportError:  # pragma: no cover – fallback for direct execution
    from gui.resources import load_application_icon  # type: ignore


class MarkdownPreviewDialog(QDialog):
    """Display markdown content rendered in a read-only viewer."""

    def __init__(
        self,
        markdown_text: str,
        parent: QWidget | None,
        *,
        title: str = "Rendered Output",
    ) -> None:
        """Render ``markdown_text`` inside a read-only dialog."""
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


class InfoDialog(QDialog):
    """Dialog summarising application metadata and runtime system details."""

    _TAGLINE = "Catalog, execute, and track AI prompts from a single desktop workspace."

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialise the informational dialog and load system metadata."""
        super().__init__(parent)
        self.setWindowTitle("About Prompt Manager")
        self.setModal(True)
        self.setMinimumWidth(420)
        icon = load_application_icon()
        if icon is not None:
            self.setWindowIcon(icon)

        layout = QVBoxLayout(self)

        if icon is not None:
            pixmap = icon.pixmap(96, 96)
            if not pixmap.isNull():
                icon_label = QLabel(self)
                icon_label.setAlignment(Qt.AlignHCenter)
                icon_label.setPixmap(pixmap)
                layout.addWidget(icon_label)

        title_label = QLabel("<b>Prompt Manager</b>", self)
        title_label.setTextFormat(Qt.RichText)
        layout.addWidget(title_label)

        info = collect_system_info()

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        author_label = QLabel('<a href="https://github.com/voytas75">voytas75</a>', self)
        author_label.setTextFormat(Qt.RichText)
        author_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        author_label.setOpenExternalLinks(True)
        form.addRow("Author:", author_label)

        tagline_label = QLabel(self._TAGLINE, self)
        tagline_label.setWordWrap(True)
        tagline_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("Tagline:", tagline_label)

        app_version = self._resolve_app_version()
        version_label = QLabel(app_version, self)
        version_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("Version:", version_label)

        cpu_label = QLabel(info.cpu, self)
        cpu_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("CPU:", cpu_label)

        platform_label = QLabel(f"{info.platform_family} ({info.os_label})", self)
        platform_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("Platform:", platform_label)

        architecture_label = QLabel(info.architecture, self)
        architecture_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("Architecture:", architecture_label)

        license_label = QLabel("opensource", self)
        license_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("License:", license_label)

        icon_source_label = QLabel('<a href="https://icons8.com/">Icons8</a>', self)
        icon_source_label.setTextFormat(Qt.RichText)
        icon_source_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        icon_source_label.setOpenExternalLinks(True)
        form.addRow("Icon source:", icon_source_label)

        layout.addLayout(form)

        buttons = QDialogButtonBox(parent=self)
        close_button = buttons.addButton("Close", QDialogButtonBox.AcceptRole)
        close_button.clicked.connect(self.accept)  # type: ignore[arg-type]
        layout.addWidget(buttons)

    @staticmethod
    def _resolve_app_version() -> str:
        """Return the application version preferring the local pyproject when available."""
        project_version = InfoDialog._version_from_pyproject()
        if project_version:
            return project_version

        metadata_version = InfoDialog._version_from_metadata()
        if metadata_version:
            return metadata_version

        module_version = InfoDialog._version_from_module()
        if module_version:
            return module_version

        return "dev"

    @staticmethod
    def _version_from_pyproject() -> str | None:
        try:
            import tomllib  # type: ignore[attr-defined]

            project_root = Path(__file__).resolve().parents[1]
            pyproject = project_root / "pyproject.toml"
            if not pyproject.exists():
                return None
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            project_section = data.get("project") or {}
            resolved = project_section.get("version")
            if resolved:
                return str(resolved)
        except Exception:
            return None
        return None

    @staticmethod
    def _version_from_metadata() -> str | None:
        try:
            from importlib.metadata import (  # type: ignore
                PackageNotFoundError,
                version as pkg_version,
            )

            try:
                resolved = pkg_version("prompt-manager")
                if resolved:
                    return resolved
            except PackageNotFoundError:
                return None
        except Exception:
            return None
        return None

    @staticmethod
    def _version_from_module() -> str | None:
        try:
            from importlib import import_module

            module = import_module("core")
            module_version = getattr(module, "__version__", None)
            if module_version:
                return str(module_version)
        except Exception:
            return None
        return None


__all__ = ["MarkdownPreviewDialog", "InfoDialog"]
