"""Printable summaries for Prompt Manager configuration.

Updates:
  v0.1.6 - 2025-12-07 - Include PrivateBin share configuration in summaries.
  v0.1.5 - 2025-12-07 - Surface SerpApi provider credentials in CLI summaries.
  v0.1.4 - 2025-12-07 - Surface Serper provider credentials in CLI summaries.
  v0.1.3 - 2025-12-07 - Explain random web search provider behaviour in summaries.
  v0.1.2 - 2025-12-07 - Surface Tavily provider credentials in CLI summaries.
  v0.1.1 - 2025-12-04 - Surface web search provider status in CLI summary.
  v0.1.0 - 2025-12-04 - Extract CLI settings summary rendering.
"""

from __future__ import annotations

from config import (
    DEFAULT_EMBEDDING_BACKEND,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_PRIVATEBIN_COMPRESSION,
    DEFAULT_PRIVATEBIN_EXPIRATION,
    DEFAULT_PRIVATEBIN_FORMAT,
    DEFAULT_PRIVATEBIN_URL,
    LITELLM_ROUTED_WORKFLOWS,
    PromptManagerSettings,
)

from .utils import describe_path, mask_secret


def print_settings_summary(settings: PromptManagerSettings) -> None:
    """Emit a readable summary of core configuration and health checks."""
    redis_dsn = getattr(settings, "redis_dsn", None)
    litellm_model = getattr(settings, "litellm_model", None)
    litellm_inference_model = getattr(settings, "litellm_inference_model", None)
    litellm_api_key = getattr(settings, "litellm_api_key", None)
    litellm_api_base = getattr(settings, "litellm_api_base", None)
    litellm_api_version = getattr(settings, "litellm_api_version", None)
    litellm_reasoning_effort = getattr(settings, "litellm_reasoning_effort", None)
    litellm_tts_model = getattr(settings, "litellm_tts_model", None)
    litellm_tts_stream = getattr(settings, "litellm_tts_stream", True)
    litellm_stream = getattr(settings, "litellm_stream", False)
    litellm_workflow_models = getattr(settings, "litellm_workflow_models", None) or {}
    embedding_backend = getattr(settings, "embedding_backend", None)
    embedding_model = getattr(settings, "embedding_model", None)
    web_search_provider = getattr(settings, "web_search_provider", None)
    exa_api_key = getattr(settings, "exa_api_key", None)
    tavily_api_key = getattr(settings, "tavily_api_key", None)
    serper_api_key = getattr(settings, "serper_api_key", None)
    serpapi_api_key = getattr(settings, "serpapi_api_key", None)
    auto_open_share_links = getattr(settings, "auto_open_share_links", True)
    privatebin_url = getattr(settings, "privatebin_url", DEFAULT_PRIVATEBIN_URL)
    privatebin_expiration = getattr(
        settings,
        "privatebin_expiration",
        DEFAULT_PRIVATEBIN_EXPIRATION,
    )
    privatebin_format = getattr(
        settings,
        "privatebin_format",
        DEFAULT_PRIVATEBIN_FORMAT,
    )
    privatebin_compression = getattr(
        settings,
        "privatebin_compression",
        DEFAULT_PRIVATEBIN_COMPRESSION,
    )
    privatebin_burn_after = getattr(settings, "privatebin_burn_after_reading", False)
    privatebin_discussion = getattr(settings, "privatebin_open_discussion", False)

    db_path_desc = describe_path(
        settings.db_path,
        expect_directory=False,
        allow_missing_file=True,
    )
    chroma_path_desc = describe_path(settings.chroma_path, expect_directory=True)
    default_model = (
        "(auto)" if embedding_backend == "litellm" and litellm_model else DEFAULT_EMBEDDING_MODEL
    )
    resolved_model = embedding_model or default_model

    def _format_tier(value: str) -> str:
        return "Inference" if value == "inference" else "Fast"

    lines = [
        "Prompt Manager configuration summary",
        "------------------------------------",
        f"Database path: {db_path_desc}",
        f"Chroma directory: {chroma_path_desc}",
        f"Redis DSN: {redis_dsn or 'not set'}",
        f"Cache TTL (seconds): {getattr(settings, 'cache_ttl_seconds', 'n/a')}",
        "",
        "LiteLLM configuration",
        "---------------------",
        f"Fast model: {litellm_model or 'not set'}",
        f"Inference model: {litellm_inference_model or 'not set'}",
        f"TTS model: {litellm_tts_model or 'not set'}",
        f"TTS streaming: {'yes' if litellm_tts_stream else 'no'}",
        f"LiteLLM API key: {mask_secret(litellm_api_key)}",
        f"LiteLLM API base: {litellm_api_base or 'not set'}",
        f"LiteLLM API version: {litellm_api_version or 'not set'}",
        f"Reasoning effort: {litellm_reasoning_effort or 'not set'}",
        f"Streaming enabled: {'yes' if litellm_stream else 'no'}",
        "",
        "LiteLLM routing",
        "----------------",
    ]

    for workflow_key, workflow_label in LITELLM_ROUTED_WORKFLOWS.items():
        tier = litellm_workflow_models.get(workflow_key, "fast")
        lines.append(f"{workflow_label}: {_format_tier(tier)}")

    lines.extend(
        [
            "",
            "Embedding configuration",
            "-----------------------",
            f"Backend: {embedding_backend or DEFAULT_EMBEDDING_BACKEND}",
            f"Model: {resolved_model}",
        ]
    )

    provider_line = f"Provider: {web_search_provider or 'disabled'}"
    random_note = None
    if web_search_provider == "random":
        provider_line = "Provider: random (rotates between configured providers each search)"
        random_note = (
            "Random selection uses whichever providers currently have API keys configured."
        )

    lines.extend(
        [
            "",
            "Web search",
            "-----------",
            provider_line,
            f"Exa API key: {mask_secret(exa_api_key)}",
            f"Tavily API key: {mask_secret(tavily_api_key)}",
            f"Serper API key: {mask_secret(serper_api_key)}",
            f"SerpApi API key: {mask_secret(serpapi_api_key)}",
        ]
    )
    if random_note:
        lines.append(random_note)

    lines.extend(
        [
            "",
            "Sharing",
            "--------",
            f"Auto-open share links: {'yes' if auto_open_share_links else 'no'}",
            f"PrivateBin base URL: {privatebin_url}",
            f"PrivateBin expiration: {privatebin_expiration}",
            f"PrivateBin formatter: {privatebin_format}, compression: {privatebin_compression}",
            f"PrivateBin burn-after-reading: {'yes' if privatebin_burn_after else 'no'}",
            f"PrivateBin open discussion: {'yes' if privatebin_discussion else 'no'}",
        ]
    )
    print("\n".join(lines))
