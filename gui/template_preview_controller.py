"""Template preview coordination helpers.

Updates:
  v0.1.0 - 2025-12-01 - Extracted preview execution and transition logic from MainWindow.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .processing_indicator import ProcessingIndicator

if TYPE_CHECKING:
    from PySide6.QtWidgets import QPushButton, QTabWidget, QWidget

    from models.prompt_model import Prompt

    from .controllers.execution_controller import ExecutionController
    from .template_preview import TemplatePreviewWidget
else:  # pragma: no cover - fallback to satisfy runtime annotations
    QWidget = QTabWidget = QPushButton = object  # type: ignore[assignment]


class TemplatePreviewController:
    """Manage preview refreshing, execution requests, and tab transitions."""
    def __init__(
        self,
        *,
        parent: QWidget,
        tab_widget: QTabWidget | None,
        template_preview: TemplatePreviewWidget | None,
        template_run_button: QPushButton | None,
        execution_controller_supplier: Callable[[], ExecutionController | None],
        current_prompt_supplier: Callable[[], Prompt | None],
        error_callback: Callable[[str, str], None],
        status_callback: Callable[[str, int], None],
    ) -> None:
        """Coordinate template-preview state with workspace and execution widgets."""
        self._parent = parent
        self._tab_widget = tab_widget
        self._template_preview = template_preview
        self._template_run_button = template_run_button
        self._execution_controller_supplier = self._ensure_callable(
            execution_controller_supplier,
            "execution_controller_supplier",
        )
        self._current_prompt_supplier = self._ensure_callable(
            current_prompt_supplier,
            "current_prompt_supplier",
        )
        self._error_callback = self._ensure_callable(error_callback, "error_callback")
        self._status_callback = self._ensure_callable(status_callback, "status_callback")
        self._transition_indicator: ProcessingIndicator | None = None

    def update_preview(self, prompt: Prompt | None) -> None:
        """Refresh the workspace template preview widget for the selected prompt."""
        if self._template_preview is None:
            return
        if prompt is None:
            self._template_preview.clear_template()
            return
        template_text = prompt.context or prompt.description or ""
        self._template_preview.set_template(template_text, str(prompt.id))

    def handle_run_requested(
        self,
        rendered_text: str,
        variables: dict[str, str],
    ) -> None:
        """Execute the selected prompt using the rendered template preview text."""
        prompt = self._current_prompt_supplier()
        if prompt is None:
            self._status_callback("Select a prompt before running from the preview.", 4000)
            return
        payload = rendered_text.strip()
        if not payload:
            self._status_callback("Render the template before running it.", 4000)
            return
        if variables:
            self._status_callback(
                f"Running template with {len(variables)} variable(s)…",
                2000,
            )
        if self._tab_widget is not None and self._tab_widget.currentIndex() != 0:
            self._show_transition_indicator()
            self._tab_widget.setCurrentIndex(0)
        controller = self._execution_controller()
        if controller is None:
            self._error_callback("Workspace unavailable", "Execution controller is not ready.")
            return
        controller.execute_prompt_with_text(
            prompt,
            payload,
            status_prefix="Executed preview",
            empty_text_message="Render the template before running it.",
            keep_text_after=False,
        )

    def handle_run_state_changed(self, can_run: bool) -> None:
        """Synchronise the Template tab shortcut button with preview availability."""
        if self._template_run_button is not None:
            self._template_run_button.setEnabled(can_run)

    def handle_template_tab_run_clicked(self) -> None:
        """Invoke the template preview execution shortcut."""
        if self._template_preview is None:
            return
        if not self._template_preview.request_run():
            self._status_callback("Render the template before running it.", 4000)

    def hide_transition_indicator(self) -> None:
        """Dismiss the template transition indicator if it is visible."""
        indicator = self._transition_indicator
        if indicator is None:
            return
        self._transition_indicator = None
        indicator.__exit__(None, None, None)

    def _show_transition_indicator(self) -> None:
        if self._transition_indicator is not None:
            return
        indicator = ProcessingIndicator(
            self._parent,
            "Opening Prompts tab…",
            title="Switching Tabs",
        )
        indicator.__enter__()
        self._transition_indicator = indicator

    def _execution_controller(self) -> ExecutionController | None:
        return self._execution_controller_supplier()

    @staticmethod
    def _ensure_callable(
        callback: Callable[..., Any],
        name: str,
    ) -> Callable[..., Any]:
        """Ensure dependency callbacks are callable."""
        if not isinstance(callback, Callable):
            raise TypeError(f"{name} must be callable")
        return callback


__all__ = ["TemplatePreviewController"]
