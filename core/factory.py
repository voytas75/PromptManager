"""Factories for constructing PromptManager instances from validated settings.

Updates: v0.7.4 - 2025-11-24 - Wire LiteLLM category suggestion helper into manager construction.
Updates: v0.7.3 - 2025-11-05 - Support LiteLLM workflow routing across fast/inference models.
Updates: v0.7.2 - 2025-11-26 - Wire LiteLLM streaming flag into executor construction.
Updates: v0.7.1 - 2025-11-19 - Configure LiteLLM scenario generator for prompt metadata enrichment.
Updates: v0.7.0 - 2025-11-15 - Wire prompt engineer construction into manager factory.
Updates: v0.6.1 - 2025-11-07 - Add configurable embedding backends via settings.
Updates: v0.6.0 - 2025-11-06 - Wire intent classifier for hybrid retrieval suggestions.
Updates: v0.5.0 - 2025-11-05 - Add LiteLLM name generator wiring.
Updates: v0.1.0 - 2025-11-03 - Added build_prompt_manager helper for GUI/bootstrap reuse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, cast

from config import PromptManagerSettings
from config.settings import PromptTemplateOverrides

from .category_registry import load_category_definitions
from .embedding import EmbeddingProvider, EmbeddingSyncWorker, create_embedding_function
from .execution import CodexExecutor
from .history_tracker import HistoryTracker
from .intent_classifier import IntentClassifier
from .name_generation import (
    LiteLLMCategoryGenerator,
    LiteLLMDescriptionGenerator,
    LiteLLMNameGenerator,
    NameGenerationError,
)
from .notifications import NotificationCenter, notification_center as default_notification_center
from .prompt_engineering import PromptEngineer
from .prompt_manager import NameGenerationError, PromptCacheError, PromptManager
from .repository import PromptRepository
from .scenario_generation import LiteLLMScenarioGenerator

try:  # pragma: no cover - redis optional dependency
    import redis
except ImportError:  # pragma: no cover - redis optional dependency
    redis = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from chromadb.api import ClientAPI
    from redis import Redis
else:  # pragma: no cover - typing only
    ClientAPI = Any  # type: ignore[misc,assignment]
    Redis = Any  # type: ignore[misc,assignment]


def _resolve_redis_client(
    redis_dsn: Optional[str],
    redis_client: Optional["Redis"],
) -> Optional["Redis"]:
    """Create a Redis client when a DSN is provided but no client supplied."""
    if redis_client is not None or not redis_dsn:
        return redis_client
    if redis is None:
        raise PromptCacheError("Redis DSN provided but redis package is not installed")
    from_url = cast(Callable[[str], "Redis"], getattr(redis, "from_url"))
    return from_url(redis_dsn)


def _resolve_embedding_components(
    settings: PromptManagerSettings,
    embedding_function: Optional[Any],
    embedding_provider: Optional[EmbeddingProvider],
) -> Tuple[Optional[Any], EmbeddingProvider]:
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

def build_prompt_manager(
    settings: PromptManagerSettings,
    *,
    redis_client: Optional["Redis"] = None,
    chroma_client: Optional["ClientAPI"] = None,
    embedding_function: Optional[Any] = None,
    repository: Optional[PromptRepository] = None,
    embedding_provider: Optional[EmbeddingProvider] = None,
    embedding_worker: Optional[EmbeddingSyncWorker] = None,
    enable_background_sync: bool = True,
    notification_center: Optional[NotificationCenter] = None,
    prompt_engineer: Optional[PromptEngineer] = None,
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
    structure_prompt_engineer: Optional[PromptEngineer] = None
    prompt_template_overrides: Dict[str, str] = {}
    templates_source: Optional[PromptTemplateOverrides] = getattr(settings, "prompt_templates", None)
    if templates_source is not None:
        try:
            dumped = templates_source.model_dump(exclude_none=True)  # type: ignore[attr-defined]
        except AttributeError:
            dumped = {}
        if isinstance(dumped, dict):
            for key, value in dumped.items():
                if not isinstance(value, str):
                    continue
                stripped = value.strip()
                if stripped:
                    prompt_template_overrides[key] = stripped

    workflow_routing = getattr(settings, "litellm_workflow_models", None) or {}
    fast_model = getattr(settings, "litellm_model", None)
    inference_model = getattr(settings, "litellm_inference_model", None)

    def _select_model(workflow: str) -> Optional[str]:
        selection = workflow_routing.get(workflow, "fast")
        if selection == "inference":
            return inference_model or fast_model
        return fast_model or inference_model

    def _construct(factory: Callable[..., Any], workflow: str, **extra: Any) -> Optional[Any]:
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
                "LiteLLM is required for configured prompt workflows. Install litellm and configure credentials."
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
    history_tracker = (
        HistoryTracker(repository_instance)
        if isinstance(repository_instance, PromptRepository)
        else None
    )
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

    manager_kwargs: Dict[str, Any] = {
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
    if history_tracker is not None:
        manager_kwargs["history_tracker"] = history_tracker

    manager = PromptManager(**manager_kwargs)
    return manager


__all__ = ["build_prompt_manager"]
