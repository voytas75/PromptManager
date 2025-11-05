"""Configuration helpers for Prompt Manager.

Updates: v0.2.1 - 2025-11-05 - Expose LiteLLM workflow routing constants.
Updates: v0.2.0 - 2025-11-03 - Expose settings loader and configuration error types.
Updates: v0.1.0 - 2025-10-30 - Package scaffold.
"""

from .settings import (
    DEFAULT_CHAT_USER_BUBBLE_COLOR,
    LITELLM_ROUTED_WORKFLOWS,
    PromptManagerSettings,
    SettingsError,
    load_settings,
)

__all__ = [
    "DEFAULT_CHAT_USER_BUBBLE_COLOR",
    "PromptManagerSettings",
    "SettingsError",
    "load_settings",
    "LITELLM_ROUTED_WORKFLOWS",
]
