"""Configuration helpers for Prompt Manager.

Updates: v0.2.3 - 2025-12-03 - Export assistant chat colour defaults and chat palette model.
Updates: v0.2.2 - 2025-11-05 - Expose theme defaults alongside routing constants.
Updates: v0.2.1 - 2025-11-05 - Expose LiteLLM workflow routing constants.
Updates: v0.2.0 - 2025-11-03 - Expose settings loader and configuration error types.
Updates: v0.1.0 - 2025-10-30 - Package scaffold.
"""

from .settings import (
    DEFAULT_CHAT_ASSISTANT_BUBBLE_COLOR,
    DEFAULT_CHAT_FONT_COLOR,
    DEFAULT_CHAT_FONT_FAMILY,
    DEFAULT_CHAT_FONT_SIZE,
    DEFAULT_CHAT_USER_BUBBLE_COLOR,
    DEFAULT_EMBEDDING_BACKEND,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_PRIVATEBIN_COMPRESSION,
    DEFAULT_PRIVATEBIN_EXPIRATION,
    DEFAULT_PRIVATEBIN_FORMAT,
    DEFAULT_PRIVATEBIN_URL,
    DEFAULT_PROMPT_OUTPUT_FONT_COLOR,
    DEFAULT_PROMPT_OUTPUT_FONT_FAMILY,
    DEFAULT_PROMPT_OUTPUT_FONT_SIZE,
    DEFAULT_THEME_MODE,
    LITELLM_ROUTED_WORKFLOWS,
    ChatColors,
    PromptManagerSettings,
    PromptTemplateOverrides,
    SettingsError,
    load_settings,
)

__all__ = [
    "ChatColors",
    "DEFAULT_CHAT_ASSISTANT_BUBBLE_COLOR",
    "DEFAULT_CHAT_USER_BUBBLE_COLOR",
    "DEFAULT_CHAT_FONT_FAMILY",
    "DEFAULT_CHAT_FONT_SIZE",
    "DEFAULT_EMBEDDING_BACKEND",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_PRIVATEBIN_URL",
    "DEFAULT_PRIVATEBIN_EXPIRATION",
    "DEFAULT_PRIVATEBIN_FORMAT",
    "DEFAULT_PRIVATEBIN_COMPRESSION",
    "DEFAULT_PROMPT_OUTPUT_FONT_FAMILY",
    "DEFAULT_PROMPT_OUTPUT_FONT_SIZE",
    "DEFAULT_PROMPT_OUTPUT_FONT_COLOR",
    "DEFAULT_CHAT_FONT_COLOR",
    "DEFAULT_THEME_MODE",
    "PromptManagerSettings",
    "PromptTemplateOverrides",
    "SettingsError",
    "load_settings",
    "LITELLM_ROUTED_WORKFLOWS",
]
