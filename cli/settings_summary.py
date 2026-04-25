"""Printable summaries for Prompt Manager configuration.

Updates:
  v0.1.10 - 2026-04-25 - Add compact routing and embedding provenance summaries.
  v0.1.9 - 2026-04-12 - Surface LiteLLM drop params in CLI settings summaries.
  v0.1.8 - 2025-12-10 - Surface LiteLLM logging toggle in summaries.
  v0.1.7 - 2025-12-07 - Surface Google web search credentials in CLI summaries.
  v0.1.6 - 2025-12-07 - Include PrivateBin share configuration in summaries.
  v0.1.5 - 2025-12-07 - Surface SerpApi provider credentials in CLI summaries.
  v0.1.4 - 2025-12-07 - Surface Serper provider credentials in CLI summaries.
  v0.1.3 - 2025-12-07 - Explain random web search provider behaviour in summaries.
  v0.1.2 - 2025-12-07 - Surface Tavily provider credentials in CLI summaries.
  v0.1.1 - 2025-12-04 - Surface web search provider status in CLI summary.
  v0.1.0 - 2025-12-04 - Extract CLI settings summary rendering.
"""

from typing import Literal, cast

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
from gui.runtime_settings_service import build_config_diagnostics_items

from .utils import describe_path, mask_secret


def _stateful_value(value: str | None, *, default: str | None = None) -> str:
    """Render a scalar config value with a short effective-state label."""
    if value in (None, ""):
        return "not set"
    if default is not None and value == default:
        return f"{value} (default)"
    return f"{value} (explicit)"


def _masked_secret_state(secret: str | None) -> str:
    """Render a masked secret with set/not-set state semantics."""
    return mask_secret(secret)


def _route_state(route: str) -> str:
    """Render workflow routing tier with effective-state semantics."""
    label = "Inference" if route == "inference" else "Fast"
    state = "explicit" if route == "inference" else "default"
    return f"{label} ({state})"


def _embedding_backend_state(backend: str | None) -> str:
    """Render embedding backend with effective-state semantics."""
    resolved_backend = backend or DEFAULT_EMBEDDING_BACKEND
    if resolved_backend == DEFAULT_EMBEDDING_BACKEND:
        return f"{resolved_backend} (default)"
    return f"{resolved_backend} (explicit)"


def _embedding_model_state(
    embedding_model: str | None,
    *,
    embedding_backend: str | None,
    litellm_model: str | None,
) -> str:
    """Render embedding model with explicit/default/derived semantics."""
    backend = embedding_backend or DEFAULT_EMBEDDING_BACKEND
    if backend == "deterministic":
        return "not set"
    if embedding_model in (None, ""):
        if backend == "litellm" and litellm_model:
            return f"{litellm_model} (derived from fast model)"
        if backend == "litellm":
            return f"{DEFAULT_EMBEDDING_MODEL} (default)"
        return "not set"
    if backend == "litellm" and embedding_model == litellm_model and litellm_model:
        return f"{embedding_model} (derived from fast model)"
    if backend == DEFAULT_EMBEDDING_BACKEND and embedding_model == DEFAULT_EMBEDDING_MODEL:
        return f"{embedding_model} (default)"
    return f"{embedding_model} (explicit)"


def _redis_runtime_status(redis_dsn: str | None) -> str:
    """Render Redis availability in the same semantic shape as GUI diagnostics."""
    if isinstance(redis_dsn, str) and redis_dsn.strip():
        return f"Redis caching configured via {redis_dsn.strip()}"
    return "Redis caching disabled (no DSN configured)."


def _routing_summary_state(
    workflow_routes: dict[str, Literal["fast", "inference"]],
) -> str:
    """Render compact routing wording without overstating provenance."""
    if not workflow_routes:
        return "all workflows use the fast model [default]"

    inference_workflows = [
        LITELLM_ROUTED_WORKFLOWS.get(workflow_key, workflow_key.strip())
        for workflow_key, route in workflow_routes.items()
        if route == "inference"
    ]
    if inference_workflows:
        return "inference for: " + ", ".join(sorted(inference_workflows)) + " [custom]"
    return "all workflows use the fast model [custom]"


def _embedding_summary_state(
    embedding_model: str | None,
    *,
    embedding_backend: str | None,
    litellm_model: str | None,
) -> str:
    """Render compact embedding wording without overstating provenance."""
    backend = embedding_backend or DEFAULT_EMBEDDING_BACKEND
    model = embedding_model.strip() if isinstance(embedding_model, str) else ""
    fast_model = litellm_model.strip() if isinstance(litellm_model, str) else ""
    if model:
        if backend == "litellm" and fast_model and model == fast_model:
            return f"{backend} / {model} [derived from fast model]"
        return f"{backend} / {model} [custom]"
    if backend == "litellm" and fast_model:
        return f"{backend} / {fast_model} [derived from fast model]"
    if backend == DEFAULT_EMBEDDING_BACKEND:
        return f"{backend} / {DEFAULT_EMBEDDING_MODEL} [default]"
    return f"{backend} / not set [custom]"


def _format_cli_diagnostics_block(settings: PromptManagerSettings) -> list[str]:
    """Return CLI-ready diagnostics summary matching the GUI status model."""
    diagnostics_sources = cast(
        "dict[str, str]",
        getattr(settings, "_diagnostics_sources", None) or {},
    )
    diagnostics = build_config_diagnostics_items(
        litellm_model=getattr(settings, "litellm_model", None),
        litellm_inference_model=getattr(settings, "litellm_inference_model", None),
        litellm_api_key=getattr(settings, "litellm_api_key", None),
        embedding_backend=getattr(settings, "embedding_backend", DEFAULT_EMBEDDING_BACKEND),
        embedding_model=getattr(settings, "embedding_model", DEFAULT_EMBEDDING_MODEL),
        litellm_tts_model=getattr(settings, "litellm_tts_model", None),
        redis_status=_redis_runtime_status(getattr(settings, "redis_dsn", None)),
        sources=diagnostics_sources
        or {
            "litellm_model": "config" if getattr(settings, "litellm_model", None) else "default",
            "litellm_inference_model": (
                "config" if getattr(settings, "litellm_inference_model", None) else "default"
            ),
            "litellm_api_key": "config"
            if getattr(settings, "litellm_api_key", None)
            else "default",
            "embedding_model": "config"
            if getattr(settings, "embedding_model", None)
            else "derived",
            "litellm_tts_model": "config"
            if getattr(settings, "litellm_tts_model", None)
            else "default",
            "redis_dsn": "config" if getattr(settings, "redis_dsn", None) else "runtime",
        },
    )
    lines = [
        "Diagnostics",
        "-----------",
        f"Overall status: {diagnostics.get('summary_status', 'OK')}",
    ]
    raw_items = diagnostics.get("items")
    items = cast("list[object]", raw_items) if isinstance(raw_items, list) else []
    for raw_item in items:
        if not isinstance(raw_item, dict):
            continue
        item = cast("dict[str, object]", raw_item)
        status = str(item.get("status") or "OK").upper()
        label = str(item.get("label") or "Item")
        detail = str(item.get("detail") or "n/a")
        source = str(item.get("source") or "unknown")
        precedence = str(item.get("precedence") or "").strip()
        suffix = f"[{source}; {precedence}]" if precedence else f"[{source}]"
        lines.append(f"{status} | {label}: {detail} {suffix}")
    raw_next_steps = diagnostics.get("next_steps")
    next_steps = cast("list[object]", raw_next_steps) if isinstance(raw_next_steps, list) else []
    if next_steps:
        lines.append("Next steps:")
        for raw_step in next_steps:
            if isinstance(raw_step, str) and raw_step.strip():
                lines.append(f"- {raw_step.strip()}")
    return lines


def print_settings_summary(settings: PromptManagerSettings) -> None:
    """Emit a readable summary of core configuration and health checks."""
    redis_dsn = getattr(settings, "redis_dsn", None)
    litellm_model = getattr(settings, "litellm_model", None)
    litellm_inference_model = getattr(settings, "litellm_inference_model", None)
    litellm_api_key = getattr(settings, "litellm_api_key", None)
    litellm_api_base = getattr(settings, "litellm_api_base", None)
    litellm_api_version = getattr(settings, "litellm_api_version", None)
    litellm_reasoning_effort = getattr(settings, "litellm_reasoning_effort", None)
    litellm_drop_params = getattr(settings, "litellm_drop_params", None)
    litellm_tts_model = getattr(settings, "litellm_tts_model", None)
    litellm_tts_stream = getattr(settings, "litellm_tts_stream", True)
    litellm_stream = getattr(settings, "litellm_stream", False)
    litellm_logging_enabled = getattr(settings, "litellm_logging_enabled", False)
    litellm_workflow_models = cast(
        "dict[str, Literal['fast', 'inference']]",
        getattr(settings, "litellm_workflow_models", None) or {},
    )
    embedding_backend = getattr(settings, "embedding_backend", None)
    embedding_model = getattr(settings, "embedding_model", None)
    web_search_provider = getattr(settings, "web_search_provider", None)
    exa_api_key = getattr(settings, "exa_api_key", None)
    tavily_api_key = getattr(settings, "tavily_api_key", None)
    serper_api_key = getattr(settings, "serper_api_key", None)
    serpapi_api_key = getattr(settings, "serpapi_api_key", None)
    google_api_key = getattr(settings, "google_api_key", None)
    google_cse_id = getattr(settings, "google_cse_id", None)
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

    lines = [
        "Prompt Manager configuration summary",
        "------------------------------------",
    ]
    lines.extend(_format_cli_diagnostics_block(settings))
    lines.extend(
        [
            "",
            f"Database path: {db_path_desc}",
            f"Chroma directory: {chroma_path_desc}",
            f"Redis DSN: {_stateful_value(redis_dsn)}",
            f"Cache TTL (seconds): {getattr(settings, 'cache_ttl_seconds', 'n/a')}",
            "",
        ]
    )
    lines.extend(
        [
            "",
            "LiteLLM configuration",
            "---------------------",
            f"Fast model: {_stateful_value(litellm_model)}",
            f"Inference model: {_stateful_value(litellm_inference_model)}",
            f"TTS model: {_stateful_value(litellm_tts_model)}",
            f"TTS streaming: {'yes' if litellm_tts_stream else 'no'}",
            f"LiteLLM API key: {_masked_secret_state(litellm_api_key)}",
            f"LiteLLM API base: {_stateful_value(litellm_api_base)}",
            f"LiteLLM API version: {_stateful_value(litellm_api_version)}",
            f"Reasoning effort: {_stateful_value(litellm_reasoning_effort)}",
            f"Drop params: {', '.join(litellm_drop_params) if litellm_drop_params else 'not set'}",
            f"Streaming enabled: {'yes' if litellm_stream else 'no'}",
            f"LiteLLM logging: {'enabled' if litellm_logging_enabled else 'disabled'}",
        ]
    )

    lines.extend(
        [
            "",
            "LiteLLM routing",
            "----------------",
        ]
    )

    for workflow_key, workflow_label in LITELLM_ROUTED_WORKFLOWS.items():
        tier = litellm_workflow_models.get(workflow_key, "fast")
        lines.append(f"{workflow_label}: {_route_state(tier)}")

    lines.append(f"Routing summary: {_routing_summary_state(litellm_workflow_models)}")

    lines.extend(
        [
            "",
            "Embedding configuration",
            "-----------------------",
            f"Backend: {_embedding_backend_state(embedding_backend)}",
            f"Model: {
                _embedding_model_state(
                    embedding_model,
                    embedding_backend=embedding_backend,
                    litellm_model=litellm_model,
                )
            }",
            f"Embeddings summary: {
                _embedding_summary_state(
                    embedding_model,
                    embedding_backend=embedding_backend,
                    litellm_model=litellm_model,
                )
            }",
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
            f"Google API key: {mask_secret(google_api_key)}",
            f"Google CSE ID: {mask_secret(google_cse_id)}",
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
