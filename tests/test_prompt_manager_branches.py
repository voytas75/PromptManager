"""Branch coverage tests for PromptManager edge cases.

Updates: v0.1.0 - 2025-10-30 - Add unit tests for error handling and caching paths.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

from core.prompt_manager import (
    PromptCacheError,
    PromptManager,
    PromptNotFoundError,
    PromptStorageError,
    RepositoryError,
    RepositoryNotFoundError,
)
from models.prompt_model import Prompt


@dataclass
class _StubCollection:
    """Chroma collection stub with injectable behaviours."""

    add_exception: Optional[BaseException] = None
    upsert_exception: Optional[BaseException] = None
    delete_exception: Optional[BaseException] = None
    query_result: Optional[Dict[str, Any]] = None
    query_exception: Optional[BaseException] = None

    def add(self, **_: Any) -> None:
        if self.add_exception:
            raise self.add_exception

    def upsert(self, **_: Any) -> None:
        if self.upsert_exception:
            raise self.upsert_exception

    def delete(self, ids: List[str]) -> None:
        if self.delete_exception:
            raise self.delete_exception
        self.deleted_ids = ids

    def query(
        self,
        *,
        query_texts: Optional[List[str]] = None,  # noqa: ARG002
        query_embeddings: Optional[List[List[float]]] = None,  # noqa: ARG002
        n_results: int,  # noqa: ARG002
        where: Optional[Dict[str, Any]] = None,  # noqa: ARG002
    ) -> Dict[str, Any]:
        if self.query_exception:
            raise self.query_exception
        if self.query_result is None:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]]}
        return self.query_result


class _StubChromaClient:
    """Return a shared collection stub for PromptManager."""

    def __init__(self, collection: _StubCollection) -> None:
        self.collection = collection
        self.get_or_create_calls = 0

    def get_or_create_collection(self, **_: Any) -> _StubCollection:
        self.get_or_create_calls += 1
        return self.collection


class _RecordingRepository:
    """Repository stand-in storing prompts in memory for tests."""

    def __init__(self) -> None:
        self.storage: Dict[uuid.UUID, Prompt] = {}
        self.deleted: List[uuid.UUID] = []

    def add(self, prompt: Prompt) -> Prompt:
        self.storage[prompt.id] = prompt
        return prompt

    def update(self, prompt: Prompt) -> Prompt:
        if prompt.id not in self.storage:
            raise RepositoryNotFoundError("missing prompt")
        self.storage[prompt.id] = prompt
        return prompt

    def delete(self, prompt_id: uuid.UUID) -> None:
        if prompt_id not in self.storage:
            raise RepositoryNotFoundError("missing delete")
        del self.storage[prompt_id]
        self.deleted.append(prompt_id)

    def get(self, prompt_id: uuid.UUID) -> Prompt:
        if prompt_id not in self.storage:
            raise RepositoryNotFoundError("missing get")
        return self.storage[prompt_id]

    def list(self, limit: Optional[int] = None) -> List[Prompt]:
        values = list(self.storage.values())
        return values[:limit] if limit is not None else values


class _RedisStub:
    """Redis client double that can be configured per test."""

    def __init__(
        self,
        *,
        payload: Optional[str] = None,
        get_exception: Optional[BaseException] = None,
        delete_exception: Optional[BaseException] = None,
        set_exception: Optional[BaseException] = None,
    ) -> None:
        self._payload = payload
        self._get_exception = get_exception
        self._delete_exception = delete_exception
        self._set_exception = set_exception
        self.setex_calls: List[Dict[str, Any]] = []
        self.deleted_keys: List[str] = []

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.setex_calls.append({"key": key, "ttl": ttl, "value": value})
        if self._set_exception:
            raise self._set_exception

    def get(self, key: str) -> Optional[bytes]:
        if self._get_exception:
            raise self._get_exception
        self.last_get = key
        if self._payload is None:
            return None
        return self._payload.encode("utf-8")

    def delete(self, key: str) -> None:
        if self._delete_exception:
            raise self._delete_exception
        self.deleted_keys.append(key)


def _sample_prompt() -> Prompt:
    return Prompt(
        id=uuid.uuid4(),
        name="Demo",
        description="Sample",
        category="test",
    )


def _build_manager(
    *,
    repository: Optional[_RecordingRepository] = None,
    collection: Optional[_StubCollection] = None,
    redis_client: Optional[_RedisStub] = None,
) -> PromptManager:
    repo = repository or _RecordingRepository()
    coll = collection or _StubCollection()
    client = _StubChromaClient(coll)
    return PromptManager(
        chroma_path="/tmp/chroma",
        db_path="/tmp/db.sqlite",
        repository=repo,
        chroma_client=client,
        redis_client=redis_client,
    )


def test_prompt_manager_requires_db_path_when_repository_missing() -> None:
    with pytest.raises(ValueError):
        PromptManager(chroma_path="/tmp/chroma")


def test_prompt_manager_wraps_repository_initialisation_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ExplodingRepo:
        def __init__(self, _: str) -> None:
            raise RepositoryError("boom")

    fake_collection = _StubCollection()
    monkeypatch.setattr("core.prompt_manager.PromptRepository", _ExplodingRepo)
    with pytest.raises(PromptStorageError):
        PromptManager(
            chroma_path="/tmp/chroma",
            db_path="/tmp/db.sqlite",
            chroma_client=_StubChromaClient(fake_collection),
        )


def test_prompt_manager_falls_back_to_persistent_client(monkeypatch: pytest.MonkeyPatch) -> None:
    collection = _StubCollection()
    repository = _RecordingRepository()

    def _failing_client(*_: Any, **__: Any) -> None:
        raise RuntimeError("client init failed")

    class _PersistentClient:
        def __init__(
            self, *, path: str | None = None
        ) -> None:  # pragma: no cover - signature parity
            self.path = path
            self.calls = 0

        def get_or_create_collection(self, **_: Any) -> _StubCollection:
            self.calls += 1
            return collection

    persistent_client = _PersistentClient(path=None)
    monkeypatch.setattr("core.prompt_manager.chromadb.Client", _failing_client)
    monkeypatch.setattr(
        "core.prompt_manager.chromadb.PersistentClient",
        lambda path: persistent_client,
    )

    manager = PromptManager(
        chroma_path="/tmp/chroma",
        db_path="/tmp/db.sqlite",
        repository=repository,
    )

    assert persistent_client.calls == 1
    assert manager.collection is collection


def test_create_prompt_rolls_back_when_chroma_add_fails() -> None:
    from core.prompt_manager import ChromaError

    class _RepoWithDeleteError(_RecordingRepository):
        def delete(self, prompt_id: uuid.UUID) -> None:  # type: ignore[override]
            super().delete(prompt_id)
            raise RepositoryError("cleanup failed")

    repository = _RepoWithDeleteError()
    collection = _StubCollection(add_exception=ChromaError("nope"))
    chroma_client = _StubChromaClient(collection)
    manager = PromptManager(
        chroma_path="/tmp/chroma",
        db_path="/tmp/db.sqlite",
        repository=repository,
        chroma_client=chroma_client,
    )

    prompt = _sample_prompt()
    with pytest.raises(PromptStorageError):
        manager.create_prompt(prompt)

    assert repository.deleted == [prompt.id]


def test_get_prompt_returns_cached_value() -> None:
    repository = _RecordingRepository()
    collection = _StubCollection()
    chroma_client = _StubChromaClient(collection)
    prompt = _sample_prompt()
    payload = prompt.to_record()
    redis_client = _RedisStub(payload=json.dumps(payload))

    manager = PromptManager(
        chroma_path="/tmp/chroma",
        db_path="/tmp/db.sqlite",
        repository=repository,
        chroma_client=chroma_client,
        redis_client=redis_client,
    )

    result = manager.get_prompt(prompt.id)
    assert result.id == prompt.id
    assert redis_client.last_get == manager._cache_key(prompt.id)


def test_get_cached_prompt_invalid_json_raises() -> None:
    repository = _RecordingRepository()
    collection = _StubCollection()
    chroma_client = _StubChromaClient(collection)
    redis_client = _RedisStub(payload="not-json")

    manager = PromptManager(
        chroma_path="/tmp/chroma",
        db_path="/tmp/db.sqlite",
        repository=repository,
        chroma_client=chroma_client,
        redis_client=redis_client,
    )

    with pytest.raises(PromptCacheError):
        manager.get_prompt(uuid.uuid4())


def test_delete_prompt_logs_when_cache_evict_fails() -> None:
    from core.prompt_manager import RedisError

    repository = _RecordingRepository()
    collection = _StubCollection()
    chroma_client = _StubChromaClient(collection)
    prompt = _sample_prompt()
    repository.add(prompt)
    failing_redis = _RedisStub(delete_exception=RedisError("boom"))

    manager = PromptManager(
        chroma_path="/tmp/chroma",
        db_path="/tmp/db.sqlite",
        repository=repository,
        chroma_client=chroma_client,
        redis_client=failing_redis,
    )

    manager.delete_prompt(prompt.id)
    assert collection.deleted_ids == [str(prompt.id)]


def test_search_prompts_handles_invalid_and_missing_entries() -> None:
    repository = _RecordingRepository()
    collection = _StubCollection()
    chroma_client = _StubChromaClient(collection)
    manager = PromptManager(
        chroma_path="/tmp/chroma",
        db_path="/tmp/db.sqlite",
        repository=repository,
        chroma_client=chroma_client,
    )

    valid_prompt = _sample_prompt()
    metadata = valid_prompt.to_metadata()
    collection.query_result = {
        "ids": [["not-a-uuid", str(valid_prompt.id)]],
        "documents": [["", valid_prompt.document]],
        "metadatas": [[{}, metadata]],
    }

    results = manager.search_prompts(query_text="hello", limit=2)
    assert len(results) == 1
    assert results[0].id == valid_prompt.id


def test_search_prompts_raises_when_repository_errors() -> None:
    class _ErroringRepo(_RecordingRepository):
        def get(self, prompt_id: uuid.UUID) -> Prompt:  # type: ignore[override]
            raise RepositoryError("db down")

    repository = _ErroringRepo()
    collection = _StubCollection(
        query_result={
            "ids": [[str(uuid.uuid4())]],
            "documents": [["doc"]],
            "metadatas": [[{"name": "n", "description": "d", "category": "c"}]],
        }
    )
    chroma_client = _StubChromaClient(collection)
    manager = PromptManager(
        chroma_path="/tmp/chroma",
        db_path="/tmp/db.sqlite",
        repository=repository,
        chroma_client=chroma_client,
    )

    with pytest.raises(PromptStorageError):
        manager.search_prompts(query_text="hi")


def test_search_prompts_requires_query_or_embedding() -> None:
    repository = _RecordingRepository()
    collection = _StubCollection()
    manager = PromptManager(
        chroma_path="/tmp/chroma",
        db_path="/tmp/db.sqlite",
        repository=repository,
        chroma_client=_StubChromaClient(collection),
    )

    with pytest.raises(ValueError):
        manager.search_prompts(query_text="", embedding=None)


def test_prompt_manager_collection_initialisation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.prompt_manager import ChromaError

    class _ExplodingClient:
        def get_or_create_collection(self, **_: Any) -> None:
            raise ChromaError("boom")

    repository = _RecordingRepository()
    with pytest.raises(PromptStorageError):
        PromptManager(
            chroma_path="/tmp/chroma",
            db_path="/tmp/db.sqlite",
            repository=repository,
            chroma_client=_ExplodingClient(),
        )


def test_create_prompt_raises_when_repository_fails() -> None:
    class _AddFailRepo(_RecordingRepository):
        def add(self, prompt: Prompt) -> Prompt:  # type: ignore[override]
            raise RepositoryError("add fail")

    repo = _AddFailRepo()
    manager = _build_manager(repository=repo)
    with pytest.raises(PromptStorageError):
        manager.create_prompt(_sample_prompt())


def test_create_prompt_logs_when_cache_write_fails() -> None:
    from core.prompt_manager import RedisError

    redis_client = _RedisStub(set_exception=RedisError("cache down"))
    repo = _RecordingRepository()
    manager = _build_manager(repository=repo, redis_client=redis_client)

    created = manager.create_prompt(_sample_prompt())
    assert created.id in repo.storage
    assert redis_client.setex_calls  # ensure attempt recorded


def test_get_prompt_raises_on_repository_error() -> None:
    class _ErrorRepo(_RecordingRepository):
        def get(self, prompt_id: uuid.UUID) -> Prompt:  # type: ignore[override]
            raise RepositoryError("db down")

    manager = _build_manager(repository=_ErrorRepo())
    with pytest.raises(PromptStorageError):
        manager.get_prompt(uuid.uuid4())


def test_get_prompt_returns_not_found() -> None:
    class _MissingRepo(_RecordingRepository):
        def get(self, prompt_id: uuid.UUID) -> Prompt:  # type: ignore[override]
            raise RepositoryNotFoundError("missing")

    manager = _build_manager(repository=_MissingRepo())
    with pytest.raises(PromptNotFoundError):
        manager.get_prompt(uuid.uuid4())


def test_get_prompt_logs_when_cache_update_fails() -> None:
    repo = _RecordingRepository()
    prompt = _sample_prompt()
    repo.add(prompt)
    manager = _build_manager(repository=repo)

    def _boom(_: Prompt) -> None:
        raise PromptCacheError("cache write failed")

    manager._cache_prompt = _boom  # type: ignore[assignment]
    result = manager.get_prompt(prompt.id)
    assert result.id == prompt.id


def test_update_prompt_handles_repository_errors() -> None:
    class _ErrorRepo(_RecordingRepository):
        def update(self, prompt: Prompt) -> Prompt:  # type: ignore[override]
            raise RepositoryError("update fail")

    repo = _ErrorRepo()
    prompt = _sample_prompt()
    repo.storage[prompt.id] = prompt
    manager = _build_manager(repository=repo)
    with pytest.raises(PromptStorageError):
        manager.update_prompt(prompt)

    class _MissingRepo(_RecordingRepository):
        def update(self, prompt: Prompt) -> Prompt:  # type: ignore[override]
            raise RepositoryNotFoundError("missing")

    repo_missing = _MissingRepo()
    repo_missing.storage[prompt.id] = prompt
    manager_missing = _build_manager(repository=repo_missing)
    with pytest.raises(PromptNotFoundError):
        manager_missing.update_prompt(prompt)


def test_update_prompt_handles_chroma_and_cache_failures() -> None:
    from core.prompt_manager import ChromaError

    prompt = _sample_prompt()
    repo = _RecordingRepository()
    repo.storage[prompt.id] = prompt
    collection = _StubCollection(upsert_exception=ChromaError("upsert"))
    manager = _build_manager(repository=repo, collection=collection)
    with pytest.raises(PromptStorageError):
        manager.update_prompt(prompt)

    collection_ok = _StubCollection()
    manager_ok = _build_manager(repository=repo, collection=collection_ok)

    def _cache_fail(_: Prompt) -> None:
        raise PromptCacheError("refresh")

    manager_ok._cache_prompt = _cache_fail  # type: ignore[assignment]
    repo.storage[prompt.id] = prompt
    updated = manager_ok.update_prompt(prompt)
    assert updated.id == prompt.id


def test_delete_prompt_handles_repository_and_chroma_errors() -> None:
    class _MissingRepo(_RecordingRepository):
        def delete(self, prompt_id: uuid.UUID) -> None:  # type: ignore[override]
            raise RepositoryNotFoundError("missing")

    manager = _build_manager(repository=_MissingRepo())
    with pytest.raises(PromptNotFoundError):
        manager.delete_prompt(uuid.uuid4())

    class _ErrorRepo(_RecordingRepository):
        def delete(self, prompt_id: uuid.UUID) -> None:  # type: ignore[override]
            raise RepositoryError("delete fail")

    manager_err = _build_manager(repository=_ErrorRepo())
    with pytest.raises(PromptStorageError):
        manager_err.delete_prompt(uuid.uuid4())

    from core.prompt_manager import ChromaError

    repo = _RecordingRepository()
    prompt = _sample_prompt()
    repo.add(prompt)
    collection = _StubCollection(delete_exception=ChromaError("delete"))
    manager_chroma = _build_manager(repository=repo, collection=collection)
    with pytest.raises(PromptStorageError):
        manager_chroma.delete_prompt(prompt.id)


def test_search_prompts_raises_when_query_fails() -> None:
    from core.prompt_manager import ChromaError

    collection = _StubCollection(query_exception=ChromaError("query"))
    manager = _build_manager(collection=collection)
    with pytest.raises(PromptStorageError):
        manager.search_prompts(query_text="hello")


def test_increment_usage_updates_repository() -> None:
    class _CountingRepo(_RecordingRepository):
        def __init__(self) -> None:
            super().__init__()
            self.update_calls = 0

        def update(self, prompt: Prompt) -> Prompt:  # type: ignore[override]
            self.update_calls += 1
            return super().update(prompt)

    repo = _CountingRepo()
    prompt = _sample_prompt()
    repo.add(prompt)
    manager = _build_manager(repository=repo)
    manager.increment_usage(prompt.id)
    assert repo.update_calls == 1
    assert repo.storage[prompt.id].usage_count == 1


def test_get_cached_prompt_returns_none_when_missing() -> None:
    redis_client = _RedisStub(payload=None)
    manager = _build_manager(redis_client=redis_client)
    assert manager._get_cached_prompt(uuid.uuid4()) is None


def test_cache_and_evict_prompt_raise_prompt_cache_error() -> None:
    from core.prompt_manager import RedisError

    redis_fail = _RedisStub(set_exception=RedisError("set"))
    manager = _build_manager(redis_client=redis_fail)
    with pytest.raises(PromptCacheError):
        manager._cache_prompt(_sample_prompt())

    redis_get_fail = _RedisStub(get_exception=RedisError("get"))
    manager_get = _build_manager(redis_client=redis_get_fail)
    with pytest.raises(PromptCacheError):
        manager_get._get_cached_prompt(uuid.uuid4())

    redis_delete_fail = _RedisStub(delete_exception=RedisError("delete"))
    manager_del = _build_manager(redis_client=redis_delete_fail)
    with pytest.raises(PromptCacheError):
        manager_del._evict_cached_prompt(uuid.uuid4())
