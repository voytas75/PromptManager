"""Branch coverage tests for core.factory helpers.

Updates:
  v0.1.3 - 2025-12-10 - Cover LiteLLM offline guidance and embedding readiness paths.
  v0.1.2 - 2025-12-08 - Allow description generator defaults in dependency forwarding test.
  v0.1.1 - 2025-12-08 - Use real settings objects and typed casts for Pyright.
  v0.1.0 - 2025-10-30 - Add Redis client resolution unit tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from config.settings import (
    DEFAULT_EMBEDDING_BACKEND,
    DEFAULT_EMBEDDING_MODEL,
    PromptManagerSettings,
)
from core.factory import (
    PromptCacheError,
    _determine_embedding_status,
    _determine_llm_status,
    _resolve_redis_client,
    build_prompt_manager,
)

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from core.repository import PromptRepository
else:  # pragma: no cover - runtime placeholders
    PromptRepository = Any

ClientAPI = Any  # type: ignore[assignment]
Redis = Any  # type: ignore[assignment]


class _RedisStub:
    def __init__(self, label: str = "redis") -> None:
        self.label = label


def _make_settings(**overrides: object) -> PromptManagerSettings:
    defaults = {
        "chroma_path": "/tmp/chroma",
        "db_path": "/tmp/db.sqlite",
        "cache_ttl_seconds": 60,
        "redis_dsn": "redis://localhost:6379/0",
        "litellm_model": None,
        "litellm_api_key": None,
        "litellm_api_base": None,
        "litellm_drop_params": None,
        "litellm_reasoning_effort": None,
        "litellm_stream": False,
        "embedding_backend": DEFAULT_EMBEDDING_BACKEND,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "embedding_device": None,
    }
    defaults.update(overrides)
    return PromptManagerSettings(**defaults)


def _as_redis(value: object) -> Redis:
    return cast("Redis", value)


def _as_repository(value: object) -> PromptRepository:
    return cast("PromptRepository", value)


def _as_client(value: object) -> ClientAPI:
    return cast("ClientAPI", value)


def test_resolve_redis_client_returns_existing_instance() -> None:
    existing = _as_redis(_RedisStub())
    assert _resolve_redis_client("redis://localhost", existing) is existing


def test_resolve_redis_client_requires_redis_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.factory.redis", None)
    with pytest.raises(PromptCacheError):
        _resolve_redis_client("redis://localhost", None)


def test_resolve_redis_client_invokes_from_url(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class _RedisModule:
        def from_url(self, url: str) -> Redis:
            calls.append(url)
            return _as_redis(_RedisStub(url))

    monkeypatch.setattr("core.factory.redis", _RedisModule())

    result = _resolve_redis_client("redis://cache", None)
    assert isinstance(result, _RedisStub)
    assert result.label == "redis://cache"
    assert calls == ["redis://cache"]


def test_build_prompt_manager_forwards_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel_redis = _as_redis(_RedisStub("sentinel"))
    sentinel_repo = object()

    class _PromptManager:
        def __init__(
            self,
            *,
            chroma_path: str,
            db_path: str,
            cache_ttl_seconds: int,
            redis_client: object,
            chroma_client: object,
            embedding_function: object,
            repository: object,
            embedding_provider: object = None,
            embedding_worker: object = None,
            enable_background_sync: bool = True,
            name_generator: object = None,
            description_generator: object = None,
            intent_classifier: object = None,
            notification_center: object = None,
            **extra: object,
        ) -> None:
            self.kwargs = {
                "chroma_path": chroma_path,
                "db_path": db_path,
                "cache_ttl_seconds": cache_ttl_seconds,
                "redis_client": redis_client,
                "chroma_client": chroma_client,
                "embedding_function": embedding_function,
                "repository": repository,
                "embedding_provider": embedding_provider,
                "embedding_worker": embedding_worker,
                "enable_background_sync": enable_background_sync,
                "name_generator": name_generator,
                "description_generator": description_generator,
                "intent_classifier": intent_classifier,
                "notification_center": notification_center,
            }
            self.kwargs.update(extra)

    monkeypatch.setattr("core.factory.PromptManager", _PromptManager)
    monkeypatch.setattr(
        "core.factory._resolve_redis_client",
        lambda dsn, client: sentinel_redis,
    )

    settings = _make_settings()
    manager = build_prompt_manager(
        settings,
        chroma_client=_as_client("chroma"),
        embedding_function="embed",
        repository=_as_repository(sentinel_repo),
    )

    assert isinstance(manager, _PromptManager)
    assert manager.kwargs["redis_client"] is sentinel_redis
    assert manager.kwargs["repository"] is sentinel_repo
    assert manager.kwargs["chroma_client"] == "chroma"
    assert manager.kwargs["embedding_function"] == "embed"
    assert manager.kwargs["description_generator"] is not None


def test_build_prompt_manager_uses_passthrough_redis_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "core.factory._resolve_redis_client",
        lambda dsn, client: client,
    )
    monkeypatch.setattr("core.factory.PromptManager", lambda **_: "manager")

    settings = _make_settings(redis_dsn=None)
    sentinel = _as_redis(_RedisStub("passthrough"))
    result = build_prompt_manager(settings, redis_client=sentinel)
    assert result == "manager"


def test_build_prompt_manager_bubbles_embedding_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings()

    def _raise_embedding(*_: object, **__: object):
        raise ValueError("no backend")

    monkeypatch.setattr("core.factory.create_embedding_function", _raise_embedding)
    monkeypatch.setattr("core.factory.PromptManager", lambda **_: "manager")

    with pytest.raises(RuntimeError) as excinfo:
        build_prompt_manager(settings)
    assert "Unable to configure embedding backend" in str(excinfo.value)


def test_determine_llm_status_requires_models_and_keys() -> None:
    settings = _make_settings(
        litellm_model=None,
        litellm_inference_model=None,
        litellm_api_key=None,
    )

    ready, reason = _determine_llm_status(settings)

    assert ready is False
    assert reason is not None
    assert "PROMPT_MANAGER_LITELLM_MODEL" in reason
    assert "PROMPT_MANAGER_LITELLM_API_KEY" in reason


def test_determine_llm_status_requires_azure_base_and_version() -> None:
    settings = _make_settings(
        litellm_model="azure/gpt-4o-mini",
        litellm_api_key="test-key",
        litellm_api_base=None,
        litellm_api_version=None,
    )

    ready, reason = _determine_llm_status(settings)

    assert ready is False
    assert reason is not None
    assert "PROMPT_MANAGER_LITELLM_API_BASE" in reason
    assert "PROMPT_MANAGER_LITELLM_API_VERSION" in reason


def test_determine_embedding_status_marks_azure_missing_base() -> None:
    settings = _make_settings(
        embedding_backend="litellm",
        embedding_model="azure/text-embedding-3-small",
        litellm_api_key="test-key",
        litellm_api_base=None,
        litellm_api_version=None,
    )

    ready, reason = _determine_embedding_status(settings)

    assert ready is False
    assert reason is not None
    assert "PROMPT_MANAGER_LITELLM_API_BASE" in reason
    assert "PROMPT_MANAGER_LITELLM_API_VERSION" in reason


def test_determine_embedding_status_accepts_deterministic_backend() -> None:
    settings = _make_settings(
        embedding_backend="deterministic",
        embedding_model="hash",
    )

    ready, reason = _determine_embedding_status(settings)

    assert ready is True
    assert reason is None
