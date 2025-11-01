"""Branch coverage tests for core.factory helpers.

Updates: v0.1.0 - 2025-10-30 - Add Redis client resolution unit tests.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.factory import PromptCacheError, _resolve_redis_client, build_prompt_manager


def _make_settings(**overrides: object) -> SimpleNamespace:
    defaults = {
        "chroma_path": "/tmp/chroma",
        "db_path": "/tmp/db.sqlite",
        "cache_ttl_seconds": 60,
        "redis_dsn": "redis://localhost:6379/0",
        "catalog_path": None,
        "litellm_model": None,
        "litellm_api_key": None,
        "litellm_api_base": None,
        "embedding_backend": "deterministic",
        "embedding_model": None,
        "embedding_device": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_resolve_redis_client_returns_existing_instance() -> None:
    existing = object()
    assert _resolve_redis_client("redis://localhost", existing) is existing


def test_resolve_redis_client_requires_redis_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.factory.redis", None)
    with pytest.raises(PromptCacheError):
        _resolve_redis_client("redis://localhost", None)


def test_resolve_redis_client_invokes_from_url(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class _RedisModule:
        def from_url(self, url: str) -> str:
            calls.append(url)
            return f"client:{url}"

    monkeypatch.setattr("core.factory.redis", _RedisModule())

    result = _resolve_redis_client("redis://cache", None)
    assert result == "client:redis://cache"
    assert calls == ["redis://cache"]


def test_build_prompt_manager_forwards_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel_redis = object()
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

    monkeypatch.setattr("core.factory.PromptManager", _PromptManager)
    monkeypatch.setattr(
        "core.factory._resolve_redis_client",
        lambda dsn, client: sentinel_redis,
    )

    settings = _make_settings()
    manager = build_prompt_manager(
        settings,
        chroma_client="chroma",
        embedding_function="embed",
        repository=sentinel_repo,
    )

    assert isinstance(manager, _PromptManager)
    assert manager.kwargs["redis_client"] is sentinel_redis
    assert manager.kwargs["repository"] is sentinel_repo
    assert manager.kwargs["chroma_client"] == "chroma"
    assert manager.kwargs["embedding_function"] == "embed"
    assert manager.kwargs["description_generator"] is None


def test_build_prompt_manager_uses_passthrough_redis_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "core.factory._resolve_redis_client",
        lambda dsn, client: client,
    )
    monkeypatch.setattr("core.factory.PromptManager", lambda **_: "manager")

    settings = _make_settings(redis_dsn=None)
    sentinel = object()
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
