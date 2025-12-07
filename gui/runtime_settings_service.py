"""Runtime settings helpers for Prompt Manager GUI.

Updates:
  v0.1.4 - 2025-12-07 - Track Serper web search credentials in runtime snapshots.
  v0.1.3 - 2025-12-07 - Track Tavily web search credentials in runtime snapshots.
  v0.1.2 - 2025-12-04 - Persist auto-open share preference across restarts.
  v0.1.1 - 2025-12-04 - Track web search provider/runtime secrets.
  v0.1.0 - 2025-12-01 - Extracted runtime settings snapshot + apply helpers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtGui import QColor

from config import (
    DEFAULT_CHAT_USER_BUBBLE_COLOR,
    DEFAULT_EMBEDDING_BACKEND,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_THEME_MODE,
    ChatColors,
    PromptManagerSettings,
    PromptTemplateOverrides,
)

from .appearance_controller import normalise_chat_palette, palette_differs_from_defaults
from .settings_dialog import persist_settings_to_config

if TYPE_CHECKING:
    from core import PromptManager

RuntimeSettings = dict[str, object | None]


@dataclass(slots=True)
class RuntimeSettingsResult:
    """Return data after applying runtime settings updates."""

    theme_mode: str
    has_executor: bool


class RuntimeSettingsService:
    """Coordinate runtime settings hydration and persistence."""

    def __init__(self, manager: PromptManager, settings: PromptManagerSettings | None) -> None:
        """Store references to the PromptManager and persisted settings snapshot."""
        self._manager = manager
        self._settings = settings

    def build_initial_runtime_settings(self) -> RuntimeSettings:
        """Load current settings snapshot from configuration and config files."""
        settings = self._settings
        derived_quick_actions: list[dict[str, object]] | None
        if settings and settings.quick_actions:
            derived_quick_actions = [dict(entry) for entry in settings.quick_actions]
        else:
            derived_quick_actions = None

        runtime: RuntimeSettings = {
            "litellm_model": settings.litellm_model if settings else None,
            "litellm_inference_model": settings.litellm_inference_model if settings else None,
            "litellm_api_key": settings.litellm_api_key if settings else None,
            "litellm_api_base": settings.litellm_api_base if settings else None,
            "litellm_api_version": settings.litellm_api_version if settings else None,
            "litellm_drop_params": list(settings.litellm_drop_params)
            if settings and settings.litellm_drop_params
            else None,
            "litellm_reasoning_effort": settings.litellm_reasoning_effort if settings else None,
            "litellm_tts_model": settings.litellm_tts_model if settings else None,
            "litellm_tts_stream": (settings.litellm_tts_stream if settings is not None else True),
            "litellm_stream": settings.litellm_stream if settings is not None else None,
            "litellm_workflow_models": dict(settings.litellm_workflow_models)
            if settings and settings.litellm_workflow_models
            else None,
            "embedding_backend": (
                settings.embedding_backend if settings else DEFAULT_EMBEDDING_BACKEND
            ),
            "embedding_model": (settings.embedding_model if settings else DEFAULT_EMBEDDING_MODEL),
            "quick_actions": derived_quick_actions,
            "chat_user_bubble_color": (
                settings.chat_user_bubble_color if settings else DEFAULT_CHAT_USER_BUBBLE_COLOR
            ),
            "theme_mode": settings.theme_mode if settings else DEFAULT_THEME_MODE,
            "chat_colors": (
                {
                    "user": settings.chat_colors.user,
                    "assistant": settings.chat_colors.assistant,
                }
                if settings
                else None
            ),
            "prompt_templates": (
                settings.prompt_templates.model_dump(exclude_none=True)
                if settings and settings.prompt_templates
                else None
            ),
            "web_search_provider": settings.web_search_provider if settings else None,
            "exa_api_key": settings.exa_api_key if settings else None,
            "tavily_api_key": settings.tavily_api_key if settings else None,
            "serper_api_key": settings.serper_api_key if settings else None,
            "auto_open_share_links": (
                settings.auto_open_share_links if settings is not None else True
            ),
        }

        config_path = Path("config/config.json")
        if config_path.exists():
            try:
                data = json.loads(config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
            else:
                for key in (
                    "litellm_model",
                    "litellm_inference_model",
                    "litellm_api_key",
                    "litellm_api_base",
                    "litellm_api_version",
                    "litellm_reasoning_effort",
                    "litellm_tts_model",
                    "litellm_tts_stream",
                    "embedding_backend",
                    "embedding_model",
                    "web_search_provider",
                ):
                    if runtime.get(key) is None and isinstance(data.get(key), str):
                        runtime[key] = data[key]
                if runtime.get("litellm_drop_params") is None:
                    drop_value = data.get("litellm_drop_params")
                    if isinstance(drop_value, list):
                        runtime["litellm_drop_params"] = [
                            str(item).strip() for item in drop_value if str(item).strip()
                        ]
                    elif isinstance(drop_value, str):
                        parsed = [part.strip() for part in drop_value.split(",") if part.strip()]
                        runtime["litellm_drop_params"] = parsed or None
                if runtime.get("litellm_stream") is None:
                    stream_value = data.get("litellm_stream")
                    if isinstance(stream_value, bool):
                        runtime["litellm_stream"] = stream_value
                    elif isinstance(stream_value, str):
                        lowered = stream_value.strip().lower()
                        if lowered in {"true", "1", "yes", "on"}:
                            runtime["litellm_stream"] = True
                        elif lowered in {"false", "0", "no", "off"}:
                            runtime["litellm_stream"] = False
                if runtime.get("litellm_tts_stream") is None:
                    tts_stream_value = data.get("litellm_tts_stream")
                    if isinstance(tts_stream_value, bool):
                        runtime["litellm_tts_stream"] = tts_stream_value
                    elif isinstance(tts_stream_value, str):
                        lowered = tts_stream_value.strip().lower()
                        if lowered in {"false", "0", "no", "off"}:
                            runtime["litellm_tts_stream"] = False
                        elif lowered in {"true", "1", "yes", "on"}:
                            runtime["litellm_tts_stream"] = True
                if runtime.get("litellm_workflow_models") is None:
                    routing_value = data.get("litellm_workflow_models")
                    if isinstance(routing_value, dict):
                        runtime["litellm_workflow_models"] = {
                            str(key): "inference"
                            for key, value in routing_value.items()
                            if isinstance(value, str) and value.strip().lower() == "inference"
                        }
                if runtime["quick_actions"] is None and isinstance(data.get("quick_actions"), list):
                    runtime["quick_actions"] = [
                        dict(entry) for entry in data["quick_actions"] if isinstance(entry, dict)
                    ]
                color_value = data.get("chat_user_bubble_color")
                if isinstance(color_value, str) and color_value.strip():
                    runtime["chat_user_bubble_color"] = color_value.strip()
                palette_value = data.get("chat_colors")
                palette = normalise_chat_palette(
                    palette_value if isinstance(palette_value, dict) else None
                )
                if palette:
                    runtime["chat_colors"] = palette
                theme_value = data.get("theme_mode")
                if isinstance(theme_value, str) and theme_value.strip():
                    runtime["theme_mode"] = theme_value.strip()
                share_auto_open = data.get("auto_open_share_links")
                if runtime.get("auto_open_share_links") is None and isinstance(
                    share_auto_open, bool
                ):
                    runtime["auto_open_share_links"] = share_auto_open
        raw_colour = runtime.get("chat_user_bubble_color")
        if isinstance(raw_colour, str):
            candidate_colour = QColor(raw_colour)
            runtime["chat_user_bubble_color"] = (
                candidate_colour.name().lower()
                if candidate_colour.isValid()
                else DEFAULT_CHAT_USER_BUBBLE_COLOR
            )
        else:
            runtime["chat_user_bubble_color"] = DEFAULT_CHAT_USER_BUBBLE_COLOR
        theme_value = runtime.get("theme_mode")
        if isinstance(theme_value, str):
            theme_choice = theme_value.strip().lower()
            runtime["theme_mode"] = (
                theme_choice if theme_choice in {"light", "dark"} else DEFAULT_THEME_MODE
            )
        else:
            runtime["theme_mode"] = DEFAULT_THEME_MODE
        if not isinstance(runtime.get("litellm_tts_stream"), bool):
            runtime["litellm_tts_stream"] = True
        return runtime

    def apply_updates(
        self,
        runtime: RuntimeSettings,
        updates: dict[str, object | None],
    ) -> RuntimeSettingsResult:
        """Apply updates to the runtime snapshot and persist configuration."""
        if not updates:
            return RuntimeSettingsResult(
                theme_mode=str(runtime.get("theme_mode") or DEFAULT_THEME_MODE),
                has_executor=self._manager.executor is not None,
            )

        simple_keys = (
            "litellm_model",
            "litellm_inference_model",
            "litellm_api_key",
            "litellm_api_base",
            "litellm_api_version",
            "litellm_reasoning_effort",
            "litellm_tts_model",
            "litellm_tts_stream",
            "web_search_provider",
            "exa_api_key",
            "tavily_api_key",
            "auto_open_share_links",
        )
        for key in simple_keys:
            if key in updates:
                runtime[key] = updates.get(key)

        if "embedding_backend" in updates or "embedding_model" in updates:
            backend_value = (
                updates.get("embedding_backend")
                or runtime.get("embedding_backend")
                or DEFAULT_EMBEDDING_BACKEND
            )
            model_value = (
                updates.get("embedding_model")
                or runtime.get("embedding_model")
                or DEFAULT_EMBEDDING_MODEL
            )
            runtime["embedding_backend"] = backend_value
            runtime["embedding_model"] = model_value
        embedding_backend_value = runtime.get("embedding_backend", DEFAULT_EMBEDDING_BACKEND)

        cleaned_drop_params: list[str] | None = runtime.get("litellm_drop_params")  # type: ignore[assignment]
        if "litellm_drop_params" in updates:
            drop_params_value = updates.get("litellm_drop_params")
            if isinstance(drop_params_value, list):
                cleaned_drop_params = [
                    str(item).strip() for item in drop_params_value if str(item).strip()
                ]
            elif isinstance(drop_params_value, str):
                cleaned_drop_params = [
                    part.strip() for part in drop_params_value.split(",") if part.strip()
                ]
            else:
                cleaned_drop_params = None
            runtime["litellm_drop_params"] = cleaned_drop_params

        stream_flag = bool(runtime.get("litellm_stream"))
        if "litellm_stream" in updates:
            stream_flag = bool(updates.get("litellm_stream"))
            runtime["litellm_stream"] = stream_flag

        def _normalise_workflows(value: object | None) -> dict[str, str] | None:
            if not isinstance(value, dict):
                return None
            return {
                str(key): "inference"
                for key, route in value.items()
                if isinstance(route, str) and route.strip().lower() == "inference"
            }

        cleaned_workflow_models = _normalise_workflows(runtime.get("litellm_workflow_models"))
        if "litellm_workflow_models" in updates:
            cleaned_workflow_models = _normalise_workflows(updates.get("litellm_workflow_models"))
            runtime["litellm_workflow_models"] = cleaned_workflow_models

        cleaned_quick_actions: list[dict[str, object]] | None = None
        existing_quick_actions = runtime.get("quick_actions")
        if isinstance(existing_quick_actions, list):
            cleaned_quick_actions = [
                dict(entry) for entry in existing_quick_actions if isinstance(entry, dict)
            ]
        if "quick_actions" in updates:
            quick_actions_value = updates.get("quick_actions")
            if isinstance(quick_actions_value, list):
                cleaned_quick_actions = [
                    dict(entry) for entry in quick_actions_value if isinstance(entry, dict)
                ]
            else:
                cleaned_quick_actions = None
            runtime["quick_actions"] = cleaned_quick_actions

        chat_colour = runtime.get("chat_user_bubble_color", DEFAULT_CHAT_USER_BUBBLE_COLOR)
        if "chat_user_bubble_color" in updates:
            color_value = updates.get("chat_user_bubble_color")
            if isinstance(color_value, str) and QColor(color_value).isValid():
                chat_colour = QColor(color_value).name().lower()
            else:
                chat_colour = DEFAULT_CHAT_USER_BUBBLE_COLOR
            runtime["chat_user_bubble_color"] = chat_colour

        cleaned_palette = normalise_chat_palette(
            runtime.get("chat_colors") if isinstance(runtime.get("chat_colors"), dict) else None
        )
        if "chat_colors" in updates:
            palette_input = (
                updates.get("chat_colors") if isinstance(updates.get("chat_colors"), dict) else None
            )
            cleaned_palette = normalise_chat_palette(palette_input)
            runtime["chat_colors"] = cleaned_palette or None

        theme_choice = runtime.get("theme_mode", DEFAULT_THEME_MODE)
        if "theme_mode" in updates:
            theme_value = updates.get("theme_mode")
            if isinstance(theme_value, str) and theme_value.strip().lower() in {"light", "dark"}:
                theme_choice = theme_value.strip().lower()
            else:
                theme_choice = DEFAULT_THEME_MODE
            runtime["theme_mode"] = theme_choice
        if theme_choice not in {"light", "dark"}:
            theme_choice = DEFAULT_THEME_MODE

        prompt_templates_payload: dict[str, str] | None = runtime.get("prompt_templates")  # type: ignore[assignment]
        if "prompt_templates" in updates:
            prompt_templates_value = updates.get("prompt_templates")
            if isinstance(prompt_templates_value, dict):
                cleaned_prompt_templates = {
                    str(key): value.strip()
                    for key, value in prompt_templates_value.items()
                    if isinstance(value, str) and value.strip()
                }
                prompt_templates_payload = cleaned_prompt_templates or None
            else:
                prompt_templates_payload = None
            runtime["prompt_templates"] = prompt_templates_payload

        persist_settings_to_config(
            {
                "litellm_model": runtime.get("litellm_model"),
                "litellm_inference_model": runtime.get("litellm_inference_model"),
                "litellm_api_base": runtime.get("litellm_api_base"),
                "litellm_api_version": runtime.get("litellm_api_version"),
                "litellm_reasoning_effort": runtime.get("litellm_reasoning_effort"),
                "litellm_tts_model": runtime.get("litellm_tts_model"),
                "litellm_tts_stream": runtime.get("litellm_tts_stream"),
                "litellm_workflow_models": runtime.get("litellm_workflow_models"),
                "quick_actions": runtime.get("quick_actions"),
                "litellm_drop_params": runtime.get("litellm_drop_params"),
                "litellm_stream": runtime.get("litellm_stream"),
                "litellm_api_key": runtime.get("litellm_api_key"),
                "tavily_api_key": runtime.get("tavily_api_key"),
                "embedding_backend": runtime.get("embedding_backend"),
                "embedding_model": runtime.get("embedding_model"),
                "chat_user_bubble_color": runtime.get("chat_user_bubble_color"),
                "chat_colors": (
                    runtime.get("chat_colors")
                    if palette_differs_from_defaults(runtime.get("chat_colors"))
                    else None
                ),
                "theme_mode": runtime.get("theme_mode"),
                "prompt_templates": runtime.get("prompt_templates"),
                "web_search_provider": runtime.get("web_search_provider"),
                "auto_open_share_links": runtime.get("auto_open_share_links"),
            }
        )

        settings_model = self._settings
        if settings_model is not None:
            settings_model.litellm_model = updates.get("litellm_model")
            settings_model.litellm_inference_model = updates.get("litellm_inference_model")
            settings_model.litellm_api_key = updates.get("litellm_api_key")
            settings_model.litellm_api_base = updates.get("litellm_api_base")
            settings_model.litellm_api_version = updates.get("litellm_api_version")
            settings_model.litellm_reasoning_effort = updates.get("litellm_reasoning_effort")
            settings_model.litellm_tts_model = updates.get("litellm_tts_model")
            settings_model.web_search_provider = updates.get("web_search_provider")
            settings_model.exa_api_key = updates.get("exa_api_key")
            settings_model.tavily_api_key = updates.get("tavily_api_key")
            if "auto_open_share_links" in updates:
                settings_model.auto_open_share_links = bool(updates.get("auto_open_share_links"))
            if "litellm_tts_stream" in updates:
                settings_model.litellm_tts_stream = bool(updates.get("litellm_tts_stream"))
            settings_model.litellm_workflow_models = cleaned_workflow_models
            settings_model.quick_actions = cleaned_quick_actions
            settings_model.litellm_drop_params = cleaned_drop_params
            settings_model.litellm_stream = stream_flag
            settings_model.embedding_backend = embedding_backend_value
            settings_model.embedding_model = runtime.get("embedding_model")
            settings_model.chat_user_bubble_color = chat_colour
            settings_model.theme_mode = theme_choice
            if cleaned_palette:
                palette_model = getattr(settings_model, "chat_colors", None)
                if isinstance(palette_model, ChatColors):
                    settings_model.chat_colors = palette_model.model_copy(update=cleaned_palette)
                else:
                    settings_model.chat_colors = ChatColors(**cleaned_palette)
            else:
                settings_model.chat_colors = ChatColors()
            if prompt_templates_payload:
                overrides_model = getattr(settings_model, "prompt_templates", None)
                if isinstance(overrides_model, PromptTemplateOverrides):
                    settings_model.prompt_templates = overrides_model.model_copy(
                        update=prompt_templates_payload
                    )
                else:
                    settings_model.prompt_templates = PromptTemplateOverrides(
                        **prompt_templates_payload
                    )
            else:
                settings_model.prompt_templates = PromptTemplateOverrides()

        self._manager.set_name_generator(
            runtime.get("litellm_model"),
            runtime.get("litellm_api_key"),
            runtime.get("litellm_api_base"),
            runtime.get("litellm_api_version"),
            inference_model=runtime.get("litellm_inference_model"),
            workflow_models=runtime.get("litellm_workflow_models"),
            drop_params=runtime.get("litellm_drop_params"),
            reasoning_effort=runtime.get("litellm_reasoning_effort"),
            stream=runtime.get("litellm_stream"),
            prompt_templates=runtime.get("prompt_templates"),
        )

        return RuntimeSettingsResult(
            theme_mode=theme_choice,
            has_executor=self._manager.executor is not None,
        )


__all__ = [
    "RuntimeSettingsResult",
    "RuntimeSettingsService",
]
