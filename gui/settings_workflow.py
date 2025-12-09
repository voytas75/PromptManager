"""Settings workflows for the Prompt Manager main window.

Updates:
  v0.15.89 - 2025-12-09 - Keep run controls enabled but surface offline guidance.
  v0.15.88 - 2025-12-09 - Disable run controls when LLM is offline.
  v0.15.87 - 2025-12-08 - Tighten dialog typings and casts for Pyright.
  v0.15.86 - 2025-12-07 - Pass Google Programmable Search credentials through the dialog.
  v0.15.85 - 2025-12-07 - Accept generic run buttons to support split actions.
  v0.15.84 - 2025-12-07 - Pass SerpApi credentials through the settings workflow dialog.
  v0.15.83 - 2025-12-07 - Refresh workspace tooltip when web search settings change.
  v0.15.82 - 2025-12-01 - Extract settings + template dialogs from gui.main_window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from PySide6.QtWidgets import QAbstractButton, QDialog, QMessageBox, QWidget

from config import DEFAULT_THEME_MODE, PromptManagerSettings
from core import NameGenerationError, PromptManager

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from collections.abc import Callable, Mapping, Sequence

    from .appearance_controller import AppearanceController
    from .controllers.execution_controller import ExecutionController
    from .prompt_search_controller import LoadPromptsCallable
    from .prompt_templates_dialog import PromptTemplateEditorDialog
    from .quick_action_controller import QuickActionController
    from .runtime_settings_service import RuntimeSettingsResult, RuntimeSettingsService
    from .settings_dialog import SettingsDialog
    from .template_preview import TemplatePreviewWidget
else:  # pragma: no cover - runtime placeholders for type-only imports
    from typing import Any as _Any

    AppearanceController = _Any
    Callable = _Any
    ExecutionController = _Any
    LoadPromptsCallable = _Any
    Mapping = _Any
    PromptTemplateEditorDialog = _Any
    QuickActionController = _Any
    RuntimeSettingsResult = _Any
    RuntimeSettingsService = _Any
    Sequence = _Any
    SettingsDialog = _Any
    TemplatePreviewWidget = _Any


class SettingsWorkflow:
    """Handle Settings/Template dialogs plus runtime update propagation."""

    def __init__(
        self,
        *,
        parent: QWidget,
        manager: PromptManager,
        runtime_settings_service: RuntimeSettingsService,
        runtime_settings: dict[str, object | None],
        quick_action_supplier: Callable[[], QuickActionController | None],
        prompt_generation_refresher: Callable[[], None],
        appearance_controller: AppearanceController,
        execution_controller_supplier: Callable[[], ExecutionController | None],
        load_prompts: LoadPromptsCallable,
        current_search_text: Callable[[], str],
        run_button: QAbstractButton,
        template_preview_supplier: Callable[[], TemplatePreviewWidget | None],
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
        data = self._runtime_settings
        dialog = SettingsDialog(
            self._parent,
            litellm_model=cast("str | None", data.get("litellm_model")),
            litellm_inference_model=cast("str | None", data.get("litellm_inference_model")),
            litellm_api_key=cast("str | None", data.get("litellm_api_key")),
            litellm_api_base=cast("str | None", data.get("litellm_api_base")),
            litellm_api_version=cast("str | None", data.get("litellm_api_version")),
            litellm_drop_params=cast("Sequence[str] | None", data.get("litellm_drop_params")),
            litellm_reasoning_effort=cast("str | None", data.get("litellm_reasoning_effort")),
            litellm_tts_model=cast("str | None", data.get("litellm_tts_model")),
            litellm_tts_stream=cast("bool | None", data.get("litellm_tts_stream")),
            litellm_stream=cast("bool | None", data.get("litellm_stream")),
            litellm_workflow_models=cast(
                "Mapping[str, str] | None", data.get("litellm_workflow_models")
            ),
            embedding_model=cast("str | None", data.get("embedding_model")),
            quick_actions=cast("list[dict[str, object]] | None", data.get("quick_actions")),
            chat_user_bubble_color=cast("str | None", data.get("chat_user_bubble_color")),
            theme_mode=cast("str | None", data.get("theme_mode")),
            chat_colors=cast("dict[str, str] | None", data.get("chat_colors")),
            prompt_templates=cast("dict[str, str] | None", data.get("prompt_templates")),
            web_search_provider=cast("str | None", data.get("web_search_provider")),
            exa_api_key=cast("str | None", data.get("exa_api_key")),
            tavily_api_key=cast("str | None", data.get("tavily_api_key")),
            serper_api_key=cast("str | None", data.get("serper_api_key")),
            serpapi_api_key=cast("str | None", data.get("serpapi_api_key")),
            google_api_key=cast("str | None", data.get("google_api_key")),
            google_cse_id=cast("str | None", data.get("google_cse_id")),
            auto_open_share_links=cast("bool | None", data.get("auto_open_share_links")),
            redis_status=cast("str | None", data.get("redis_status")),
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        updates = dialog.result_settings()
        self.apply_settings(updates)

    def open_prompt_templates_dialog(self) -> None:
        """Launch the prompt templates override dialog."""
        dialog = PromptTemplateEditorDialog(
            self._parent,
            templates=cast(
                "Mapping[str, str] | None", self._runtime_settings.get("prompt_templates")
            ),
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
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
        llm_available = getattr(self._manager, "llm_available", False)
        run_enabled = bool(result.has_executor or not llm_available)
        if not llm_available:
            self._run_button.setToolTip(self._manager.llm_status_message("Prompt execution"))
        else:
            self._run_button.setToolTip("Run the selected prompt or the text you provide.")
        self._run_button.setEnabled(run_enabled)
        template_preview = self._template_preview_supplier()
        if template_preview is not None:
            template_preview.set_run_enabled(run_enabled)


__all__ = ["SettingsWorkflow"]
