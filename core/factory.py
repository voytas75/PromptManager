"""Factories for constructing PromptManager instances from validated settings.

Updates:
  v0.8.4 - 2025-12-09 - Surface offline mode when LiteLLM credentials are missing.
  v0.8.3 - 2025-12-07 - Wire Google web search provider into the factory builder.
  v0.8.2 - 2025-12-07 - Expose reusable web search service builder.
  v0.8.1 - 2025-12-07 - Wire SerpApi provider into the web search factory.
  v0.8.0 - 2025-12-07 - Wire Serper provider into the web search factory.
  v0.7.9 - 2025-12-07 - Add random provider fan-out for the web search service.
  v0.7.8 - 2025-12-07 - Add Tavily provider wiring to the web search factory.
  v0.7.7 - 2025-12-04 - Wire web search service creation into manager factory.
  v0.7.6 - 2025-11-30 - Remove docstring padding for Ruff compliance.
  v0.7.5 - 2025-11-29 - Move config imports behind type checks and wrap long strings.
  v0.7.4 - 2025-11-24 - Wire LiteLLM category suggestion helper into manager construction.
  v0.7.3-and-earlier - 2025-11-19 - Configure LiteLLM scenario generator and earlier wiring.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, cast

from .category_registry import load_category_definitions
from .embedding import EmbeddingProvider, EmbeddingSyncWorker, create_embedding_function
from .execution import CodexExecutor
from .history_tracker import HistoryTracker
from .intent_classifier import IntentClassifier
from .name_generation import (
    LiteLLMCategoryGenerator,
    LiteLLMDescriptionGenerator,
    LiteLLMNameGenerator,
)
from .notifications import NotificationCenter, notification_center as default_notification_center
from .prompt_engineering import PromptEngineer
from .prompt_manager import NameGenerationError, PromptCacheError, PromptManager
from .repository import PromptRepository
from .scenario_generation import LiteLLMScenarioGenerator
from .web_search import WebSearchService, resolve_web_search_provider

try:  # pragma: no cover - redis optional dependency
    import redis
except ImportError:  # pragma: no cover - redis optional dependency
    redis = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from chromadb.api import ClientAPI
    from redis import Redis

    from config import PromptManagerSettings
    from config.settings import PromptTemplateOverrides
else:  # pragma: no cover - typing only
    ClientAPI = Any
    PromptManagerSettings = Any
    PromptTemplateOverrides = Any
    Redis = Any

factory_logger = logging.getLogger("prompt_manager.factory")


def _resolve_redis_client(
    redis_dsn: str | None,
    redis_client: Redis | None,
) -> Redis | None:
    """Create a Redis client when a DSN is provided but no client supplied."""
    if redis_client is not None or not redis_dsn:
        return redis_client
    redis_module = cast("Any", redis)
    if redis_module is None:
        raise PromptCacheError("Redis DSN provided but redis package is not installed")
    from_url = cast("Callable[[str], Redis]", redis_module.from_url)
    return from_url(redis_dsn)


def _resolve_embedding_components(
    settings: PromptManagerSettings,
    embedding_function: Any | None,
    embedding_provider: EmbeddingProvider | None,
) -> tuple[Any | None, EmbeddingProvider]:
    """Return embedding function/provider pair respecting overrides and settings."""
    resolved_function = embedding_function
    if resolved_function is None:
        try:
            resolved_function = create_embedding_function(
                settings.embedding_backend,
                model=settings.embedding_model,
                api_key=settings.litellm_api_key,
                api_base=settings.litellm_api_base,
                device=settings.embedding_device,
            )
        except Exception as exc:
            raise RuntimeError(f"Unable to configure embedding backend: {exc}") from exc

    if embedding_provider is not None:
        return resolved_function, embedding_provider
    return resolved_function, EmbeddingProvider(resolved_function)


def _determine_llm_status(settings: PromptManagerSettings) -> tuple[bool, str | None]:
    """Return LiteLLM readiness plus a human-readable offline reason."""
    model_configured = bool(
        getattr(settings, "litellm_model", None)
        or getattr(settings, "litellm_inference_model", None)
    )
    api_key_configured = bool(getattr(settings, "litellm_api_key", None))
    if model_configured and api_key_configured:
        return True, None
    missing_parts: list[str] = []
    if not model_configured:
        missing_parts.append("model")
    if not api_key_configured:
        missing_parts.append("API key")
    missing_text = " and ".join(missing_parts) if missing_parts else "credentials"
    reason = (
        f"LiteLLM {missing_text} missing; running in offline mode. "
        "Configure PROMPT_MANAGER_LITELLM_MODEL and PROMPT_MANAGER_LITELLM_API_KEY to enable "
        "prompt generation and execution."
    )
    return False, reason


def build_web_search_service(settings: PromptManagerSettings) -> WebSearchService:
    """Return a WebSearchService configured from settings."""
    provider = resolve_web_search_provider(
        getattr(settings, "web_search_provider", None),
        exa_api_key=getattr(settings, "exa_api_key", None),
        tavily_api_key=getattr(settings, "tavily_api_key", None),
        serper_api_key=getattr(settings, "serper_api_key", None),
        serpapi_api_key=getattr(settings, "serpapi_api_key", None),
        google_api_key=getattr(settings, "google_api_key", None),
        google_cse_id=getattr(settings, "google_cse_id", None),
    )
    return WebSearchService(provider)


def build_prompt_manager(
    settings: PromptManagerSettings,
    *,
    redis_client: Redis | None = None,
    chroma_client: ClientAPI | None = None,
    embedding_function: Any | None = None,
    repository: PromptRepository | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    embedding_worker: EmbeddingSyncWorker | None = None,
    enable_background_sync: bool = True,
    notification_center: NotificationCenter | None = None,
    prompt_engineer: PromptEngineer | None = None,
) -> PromptManager:
    """Return a PromptManager configured from validated settings."""
    resolved_redis = _resolve_redis_client(settings.redis_dsn, redis_client)
    resolved_embedding_function, resolved_embedding_provider = _resolve_embedding_components(
        settings,
        embedding_function,
        embedding_provider,
    )
    name_generator = None
    description_generator = None
    scenario_generator = None
    category_generator = None
    resolved_prompt_engineer = prompt_engineer
    structure_prompt_engineer: PromptEngineer | None = None
    prompt_template_overrides: dict[str, str] = {}
    templates_source: PromptTemplateOverrides | None = getattr(
        settings,
        "prompt_templates",
        None,
    )
    if templates_source is not None:
        template_mapping: Mapping[str, object]
        try:
            template_mapping = templates_source.model_dump(exclude_none=True)  # type: ignore[attr-defined]
        except AttributeError:
            template_mapping = {}
        for key, value in template_mapping.items():
            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    prompt_template_overrides[str(key)] = stripped

    workflow_models_setting = getattr(settings, "litellm_workflow_models", None)
    workflow_routing: dict[str, str] = {}
    workflow_models_mapping: Mapping[str, object]
    if isinstance(workflow_models_setting, Mapping):
        workflow_models_mapping = cast("Mapping[str, object]", workflow_models_setting)
    else:
        workflow_models_mapping = {}
    for key, value in workflow_models_mapping.items():
        key_text = key.strip()
        value_text = str(value).strip().lower()
        if not key_text or value_text not in {"fast", "inference"}:
            continue
        workflow_routing[key_text] = value_text
    fast_model = getattr(settings, "litellm_model", None)
    inference_model = getattr(settings, "litellm_inference_model", None)

    def _select_model(workflow: str) -> str | None:
        selection = workflow_routing.get(workflow, "fast")
        if selection == "inference":
            return inference_model or fast_model
        return fast_model or inference_model

    def _construct(factory: Callable[..., Any], workflow: str, **extra: Any) -> Any | None:
        model_name = _select_model(workflow)
        if not model_name:
            return None
        try:
            return factory(
                model=model_name,
                api_key=settings.litellm_api_key,
                api_base=settings.litellm_api_base,
                api_version=settings.litellm_api_version,
                drop_params=settings.litellm_drop_params,
                **extra,
            )
        except RuntimeError as exc:
            raise NameGenerationError(
                "LiteLLM is required for configured prompt workflows. "
                "Install litellm and configure credentials."
            ) from exc

    name_generator = _construct(
        LiteLLMNameGenerator,
        "name_generation",
        system_prompt=prompt_template_overrides.get("name_generation"),
    )
    description_generator = _construct(
        LiteLLMDescriptionGenerator,
        "description_generation",
        system_prompt=prompt_template_overrides.get("description_generation"),
    )
    scenario_generator = _construct(
        LiteLLMScenarioGenerator,
        "scenario_generation",
        system_prompt=prompt_template_overrides.get("scenario_generation"),
    )
    if resolved_prompt_engineer is None:
        resolved_prompt_engineer = _construct(
            PromptEngineer,
            "prompt_engineering",
            system_prompt=prompt_template_overrides.get("prompt_engineering"),
        )
    structure_prompt_engineer = _construct(
        PromptEngineer,
        "prompt_structure_refinement",
        system_prompt=prompt_template_overrides.get("prompt_engineering"),
    )
    if structure_prompt_engineer is None:
        structure_prompt_engineer = resolved_prompt_engineer
    category_generator = _construct(
        LiteLLMCategoryGenerator,
        "category_generation",
        system_prompt=prompt_template_overrides.get("category_generation"),
    )
    intent_classifier = IntentClassifier()

    repository_instance = repository or PromptRepository(str(settings.db_path))
    history_tracker = HistoryTracker(repository_instance)
    category_definitions = load_category_definitions(
        inline_definitions=getattr(settings, "categories", None),
        path=getattr(settings, "categories_path", None),
    )
    executor = _construct(
        CodexExecutor,
        "prompt_execution",
        reasoning_effort=settings.litellm_reasoning_effort,
        stream=settings.litellm_stream,
    )
    web_search_service = build_web_search_service(settings)

    manager_kwargs: dict[str, Any] = {
        "chroma_path": str(settings.chroma_path),
        "db_path": str(settings.db_path),
        "cache_ttl_seconds": settings.cache_ttl_seconds,
        "redis_client": resolved_redis,
        "chroma_client": chroma_client,
        "embedding_function": resolved_embedding_function,
        "repository": repository_instance,
        "embedding_provider": resolved_embedding_provider,
        "embedding_worker": embedding_worker,
        "enable_background_sync": enable_background_sync,
        "name_generator": name_generator,
        "description_generator": description_generator,
        "category_definitions": category_definitions,
        "fast_model": fast_model,
        "inference_model": inference_model,
        "workflow_models": workflow_routing if workflow_routing else None,
        "intent_classifier": intent_classifier,
        "notification_center": notification_center or default_notification_center,
        "prompt_templates": prompt_template_overrides or None,
        "web_search_service": web_search_service,
    }
    if scenario_generator is not None:
        manager_kwargs["scenario_generator"] = scenario_generator
    if category_generator is not None:
        manager_kwargs["category_generator"] = category_generator
    if resolved_prompt_engineer is not None:
        manager_kwargs["prompt_engineer"] = resolved_prompt_engineer
    if structure_prompt_engineer is not None:
        manager_kwargs["structure_prompt_engineer"] = structure_prompt_engineer
    if executor is not None:
        manager_kwargs["executor"] = executor
    manager_kwargs["history_tracker"] = history_tracker

    manager = PromptManager(**manager_kwargs)
    llm_ready, llm_reason = _determine_llm_status(settings)
    if not llm_ready and llm_reason:
        factory_logger.warning(llm_reason)
    manager.set_llm_status(llm_ready, reason=llm_reason, notify=not llm_ready)
    return manager


__all__ = ["build_prompt_manager", "build_web_search_service"]
