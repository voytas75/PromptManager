"""Share workflow helpers for Prompt Manager GUI.

Updates:
  v0.1.0 - 2025-12-01 - Extracted prompt/result share helpers from MainWindow.
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import QPushButton

from core.sharing import ShareProvider, format_prompt_for_share
from models.prompt_model import Prompt

from .controllers.execution_controller import ExecutionController
from .share_controller import ShareController
from .widgets import PromptDetailWidget


class ShareWorkflowCoordinator:
    """Handle prompt and result sharing interactions."""
    def __init__(
        self,
        share_controller: ShareController,
        *,
        detail_widget: PromptDetailWidget,
        prompt_supplier: Callable[[], Prompt | None],
        share_button_supplier: Callable[[], QPushButton | None],
        share_result_button_supplier: Callable[[], QPushButton | None],
        execution_controller_supplier: Callable[[], ExecutionController | None],
        status_callback: Callable[[str, int], None],
        error_callback: Callable[[str, str], None],
    ) -> None:
        self._share_controller = share_controller
        self._detail_widget = detail_widget
        self._prompt_supplier = prompt_supplier
        self._share_button_supplier = share_button_supplier
        self._share_result_button_supplier = share_result_button_supplier
        self._execution_controller_supplier = execution_controller_supplier
        self._show_status = status_callback
        self._show_error = error_callback

    def register_provider(self, provider: ShareProvider) -> None:
        """Store a share provider via the underlying controller."""
        self._share_controller.register_provider(provider)
        controller = self._execution_controller_supplier()
        if controller is not None:
            controller.notify_share_providers_changed()

    def share_prompt(self) -> None:
        """Display provider choices and initiate the share workflow."""
        prompt = self._detail_widget.current_prompt()
        if prompt is None:
            self._show_status("Select a prompt to share first.", 4000)
            return
        provider_button = self._share_button_supplier()
        provider_name = self._share_controller.choose_provider(provider_button)
        if not provider_name:
            return
        payload = self._build_share_payload(prompt)
        if payload is None:
            return
        self._share_controller.share_payload(
            provider_name,
            payload,
            prompt=prompt,
            prompt_name=prompt.name,
            indicator_title="Sharing Prompt",
            error_title="Unable to share prompt",
        )

    def share_result(self) -> None:
        """Display provider choices for sharing the latest output text."""
        controller = self._execution_controller_supplier()
        if controller is None:
            self._show_error("Workspace unavailable", "Execution controller is not ready.")
            return
        provider_name = self._share_controller.choose_provider(
            self._share_result_button_supplier()
        )
        if not provider_name:
            return
        controller.share_result_text(provider_name)

    def _build_share_payload(self, prompt: Prompt) -> str | None:
        mode = self._detail_widget.share_payload_mode()
        include_description = mode in {"body_description", "body_description_scenarios"}
        include_scenarios = mode == "body_description_scenarios"
        if not (prompt.context or "").strip():
            self._show_error("Unable to share prompt", "Prompt body is empty.")
            return None
        payload = format_prompt_for_share(
            prompt,
            include_description=include_description,
            include_scenarios=include_scenarios,
            include_metadata=self._detail_widget.share_include_metadata(),
        )
        if not payload.strip():
            self._show_error("Unable to share prompt", "Share payload is empty.")
            return None
        return payload


__all__ = ["ShareWorkflowCoordinator"]
