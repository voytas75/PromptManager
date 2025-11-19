"""Configuration helpers for Prompt Manager.

Updates: v0.2.3 - 2025-12-03 - Export assistant chat colour defaults and chat palette model.
Updates: v0.2.2 - 2025-11-05 - Expose theme defaults alongside routing constants.
Updates: v0.2.1 - 2025-11-05 - Expose LiteLLM workflow routing constants.
Updates: v0.2.0 - 2025-11-03 - Expose settings loader and configuration error types.
Updates: v0.1.0 - 2025-10-30 - Package scaffold.
"""

from .settings import (
    ChatColors,
    DEFAULT_CHAT_ASSISTANT_BUBBLE_COLOR,
    DEFAULT_CHAT_USER_BUBBLE_COLOR,
    DEFAULT_THEME_MODE,
    LITELLM_ROUTED_WORKFLOWS,
    PromptManagerSettings,
    SettingsError,
    load_settings,
)

__all__ = [
    "ChatColors",
    "DEFAULT_CHAT_ASSISTANT_BUBBLE_COLOR",
    "DEFAULT_CHAT_USER_BUBBLE_COLOR",
    "DEFAULT_THEME_MODE",
    "PromptManagerSettings",
    "SettingsError",
    "load_settings",
    "LITELLM_ROUTED_WORKFLOWS",
]
