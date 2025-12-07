"""Settings workflows for the Prompt Manager main window.

Updates:
  v0.15.85 - 2025-12-07 - Accept generic run buttons to support split actions.
  v0.15.84 - 2025-12-07 - Pass SerpApi credentials through the settings workflow dialog.
  v0.15.83 - 2025-12-07 - Refresh workspace tooltip when web search settings change.
  v0.15.82 - 2025-12-01 - Extract settings + template dialogs from gui.main_window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QMessageBox

from config import DEFAULT_THEME_MODE, PromptManagerSettings
from core import NameGenerationError, PromptManager

from .prompt_templates_dialog import PromptTemplateEditorDialog
from .runtime_settings_service import RuntimeSettingsResult, RuntimeSettingsService
from .settings_dialog import SettingsDialog

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Callable

    from PySide6.QtWidgets import QAbstractButton, QWidget
else:  # pragma: no cover - runtime placeholders for type-only imports
    from typing import Any as _Any

    Callable = _Any
    QAbstractButton = _Any
    QWidget = _Any


class SettingsWorkflow:
    """Handle Settings/Template dialogs plus runtime update propagation."""

    def __init__(
        self,
        *,
        parent: QWidget,
        manager: PromptManager,
        runtime_settings_service: RuntimeSettingsService,
        runtime_settings: dict[str, object | None],
        quick_action_supplier: Callable[[], object | None],
        prompt_generation_refresher: Callable[[], None],
        appearance_controller,
        execution_controller_supplier: Callable[[], object | None],
        load_prompts: Callable[[str], None],
        current_search_text: Callable[[], str],
        run_button: QAbstractButton,
        template_preview_supplier: Callable[[], object | None],
        toast_callback: Callable[[str], None],
        web_search_tooltip_updater: Callable[[], None],
    ) -> None:
        """Persist dependencies required for settings/template workflows."""
        self._parent = parent
        self._manager = manager
        self._runtime_settings_service = runtime_settings_service
        self._runtime_settings = runtime_settings
        self._quick_action_supplier = quick_action_supplier
        self._prompt_generation_refresher = prompt_generation_refresher
        self._appearance_controller = appearance_controller
        self._execution_controller_supplier = execution_controller_supplier
        self._load_prompts = load_prompts
        self._current_search_text = current_search_text
        self._run_button = run_button
        self._template_preview_supplier = template_preview_supplier
        self._toast = toast_callback
        self._web_search_tooltip_updater = web_search_tooltip_updater

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def open_settings_dialog(self, settings: PromptManagerSettings | None = None) -> None:
        """Display the settings dialog and apply any accepted updates."""
        dialog = SettingsDialog(
            self._parent,
            litellm_model=self._runtime_settings.get("litellm_model"),
            litellm_inference_model=self._runtime_settings.get("litellm_inference_model"),
            litellm_api_key=self._runtime_settings.get("litellm_api_key"),
            litellm_api_base=self._runtime_settings.get("litellm_api_base"),
            litellm_api_version=self._runtime_settings.get("litellm_api_version"),
            litellm_drop_params=self._runtime_settings.get("litellm_drop_params"),
            litellm_reasoning_effort=self._runtime_settings.get("litellm_reasoning_effort"),
            litellm_tts_model=self._runtime_settings.get("litellm_tts_model"),
            litellm_tts_stream=self._runtime_settings.get("litellm_tts_stream"),
            litellm_stream=self._runtime_settings.get("litellm_stream"),
            litellm_workflow_models=self._runtime_settings.get("litellm_workflow_models"),
            embedding_model=self._runtime_settings.get("embedding_model"),
            quick_actions=self._runtime_settings.get("quick_actions"),
            chat_user_bubble_color=self._runtime_settings.get("chat_user_bubble_color"),
            theme_mode=self._runtime_settings.get("theme_mode"),
            chat_colors=self._runtime_settings.get("chat_colors"),
            prompt_templates=self._runtime_settings.get("prompt_templates"),
            web_search_provider=self._runtime_settings.get("web_search_provider"),
            exa_api_key=self._runtime_settings.get("exa_api_key"),
            tavily_api_key=self._runtime_settings.get("tavily_api_key"),
            serper_api_key=self._runtime_settings.get("serper_api_key"),
            serpapi_api_key=self._runtime_settings.get("serpapi_api_key"),
            auto_open_share_links=self._runtime_settings.get("auto_open_share_links"),
        )
        if dialog.exec() != SettingsDialog.Accepted:
            return
        updates = dialog.result_settings()
        self.apply_settings(updates)

    def open_prompt_templates_dialog(self) -> None:
        """Launch the prompt templates override dialog."""
        dialog = PromptTemplateEditorDialog(
            self._parent,
            templates=self._runtime_settings.get("prompt_templates"),
        )
        if dialog.exec() != PromptTemplateEditorDialog.Accepted:
            return
        overrides = dialog.result_templates()
        cleaned_overrides: dict[str, str] | None = overrides or None
        current_templates = self._runtime_settings.get("prompt_templates")
        normalised_current = current_templates or None
        if normalised_current == cleaned_overrides:
            self._toast("Prompt templates are already up to date.")
            return
        self.apply_settings({"prompt_templates": cleaned_overrides})
        self._toast("Prompt templates updated.")

    def apply_settings(self, updates: dict[str, object | None]) -> None:
        """Propagate runtime setting updates throughout the application."""
        if not updates:
            return
        try:
            result = self._runtime_settings_service.apply_updates(
                self._runtime_settings,
                updates,
            )
        except NameGenerationError as exc:
            QMessageBox.warning(self._parent, "LiteLLM configuration", str(exc))
            result = RuntimeSettingsResult(
                theme_mode=str(self._runtime_settings.get("theme_mode") or DEFAULT_THEME_MODE),
                has_executor=self._manager.executor is not None,
            )
        else:
            self._web_search_tooltip_updater()

        controller = self._quick_action_supplier()
        if controller is not None:
            controller.refresh_actions(self._runtime_settings.get("quick_actions"))

        self._prompt_generation_refresher()
        self._appearance_controller.apply_theme(result.theme_mode)

        execution_controller = self._execution_controller_supplier()
        if execution_controller is not None:
            execution_controller.refresh_chat_history_view()

        self._load_prompts(self._current_search_text())
        self._run_button.setEnabled(result.has_executor)
        template_preview = self._template_preview_supplier()
        if template_preview is not None:
            template_preview.set_run_enabled(result.has_executor)


__all__ = ["SettingsWorkflow"]
