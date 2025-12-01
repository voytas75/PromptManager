"""Branch coverage tests for PromptManager edge cases.

Updates: v0.1.1 - 2025-11-22 - Seed version history for legacy prompts without snapshots.
Updates: v0.1.0 - 2025-10-30 - Add unit tests for error handling and caching paths.
"""
from __future__ import annotations

import json
import types
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from core.embedding import EmbeddingGenerationError
from core.intent_classifier import IntentClassifier
from core.name_generation import CategorySuggestionError, DescriptionGenerationError
from core.prompt_manager import (
    NameGenerationError,
    PromptCacheError,
    PromptManager,
    PromptManagerError,
    PromptNotFoundError,
    PromptStorageError,
    RepositoryError,
    RepositoryNotFoundError,
)
from models.category_model import PromptCategory, slugify_category
from models.prompt_model import Prompt, PromptForkLink, PromptVersion, UserProfile

if TYPE_CHECKING:
    import builtins
    from collections.abc import Sequence


def _clone_prompt(prompt: Prompt) -> Prompt:
    return Prompt.from_record(prompt.to_record())


def _clone_category(category: PromptCategory) -> PromptCategory:
    return PromptCategory.from_record(category.to_record())


@dataclass
class _StubCollection:
    """Chroma collection stub with injectable behaviours."""
    add_exception: BaseException | None = None
    upsert_exception: BaseException | None = None
    delete_exception: BaseException | None = None
    query_result: dict[str, Any] | None = None
    query_exception: BaseException | None = None

    def add(self, **_: Any) -> None:
        if self.add_exception:
            raise self.add_exception

    def upsert(self, **_: Any) -> None:
        if self.upsert_exception:
            raise self.upsert_exception

    def delete(self, ids: list[str]) -> None:
        if self.delete_exception:
            raise self.delete_exception
        self.deleted_ids = ids

    def query(
        self,
        *,
        query_texts: list[str] | None = None,  # noqa: ARG002
        query_embeddings: list[list[float]] | None = None,  # noqa: ARG002
        n_results: int,  # noqa: ARG002
        where: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> dict[str, Any]:
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
        self.storage: dict[uuid.UUID, Prompt] = {}
        self.deleted: list[uuid.UUID] = []
        self.profile = UserProfile.create_default()
        self._versions: dict[uuid.UUID, list[PromptVersion]] = {}
        self._version_index: dict[int, PromptVersion] = {}
        self._version_counter = 0
        self._fork_links: dict[int, PromptForkLink] = {}
        self._fork_counter = 0
        base_category = PromptCategory(
            slug="general",
            label="General",
            description="General prompts",
        )
        self._categories: dict[str, PromptCategory] = {base_category.slug: base_category}

    def add(self, prompt: Prompt) -> Prompt:
        self.storage[prompt.id] = _clone_prompt(prompt)
        return prompt

    def update(self, prompt: Prompt) -> Prompt:
        if prompt.id not in self.storage:
            raise RepositoryNotFoundError("missing prompt")
        self.storage[prompt.id] = _clone_prompt(prompt)
        return prompt

    def delete(self, prompt_id: uuid.UUID) -> None:
        if prompt_id not in self.storage:
            raise RepositoryNotFoundError("missing delete")
        del self.storage[prompt_id]
        self.deleted.append(prompt_id)

    def get(self, prompt_id: uuid.UUID) -> Prompt:
        if prompt_id not in self.storage:
            raise RepositoryNotFoundError("missing get")
        return _clone_prompt(self.storage[prompt_id])

    def list(self, limit: int | None = None) -> builtins.list[Prompt]:
        values = [
            _clone_prompt(prompt)
            for prompt in self.storage.values()
        ]
        return values[:limit] if limit is not None else values

    def list_categories(self, include_archived: bool = False) -> builtins.list[PromptCategory]:
        categories = [
            _clone_category(category)
            for category in self._categories.values()
            if include_archived or category.is_active
        ]
        categories.sort(key=lambda category: category.label.lower())
        return categories

    def create_category(self, category: PromptCategory) -> PromptCategory:
        slug = slugify_category(category.slug)
        if slug in self._categories:
            raise RepositoryError(f"Category {slug} already exists")
        stored = _clone_category(category)
        stored.slug = slug
        self._categories[slug] = stored
        return _clone_category(stored)

    def update_category(self, category: PromptCategory) -> PromptCategory:
        slug = slugify_category(category.slug)
        if slug not in self._categories:
            raise RepositoryNotFoundError(f"Category {slug} not found")
        stored = _clone_category(category)
        stored.slug = slug
        self._categories[slug] = stored
        self.update_prompt_category_labels(slug, stored.label)
        return _clone_category(stored)

    def set_category_active(self, slug: str, is_active: bool) -> PromptCategory:
        normalized = slugify_category(slug)
        if normalized not in self._categories:
            raise RepositoryNotFoundError(f"Category {slug} not found")
        category = _clone_category(self._categories[normalized])
        category.is_active = is_active
        category.updated_at = datetime.now(UTC)
        self._categories[normalized] = category
        return _clone_category(category)

    def sync_category_definitions(
        self,
        categories: Sequence[PromptCategory],
    ) -> builtins.list[PromptCategory]:
        created: list[PromptCategory] = []
        for category in categories:
            slug = slugify_category(category.slug)
            if slug in self._categories:
                continue
            stored = _clone_category(category)
            stored.slug = slug
            self._categories[slug] = stored
            created.append(_clone_category(stored))
        return created

    def update_prompt_category_labels(self, slug: str, label: str) -> None:
        normalized = slugify_category(slug)
        for prompt in self.storage.values():
            prompt_slug = prompt.category_slug or slugify_category(prompt.category)
            if prompt_slug != normalized:
                continue
            prompt.category = label
            prompt.category_slug = normalized

    def get_user_profile(self) -> UserProfile:
        return self.profile

    def record_user_prompt_usage(self, prompt: Prompt, *, max_recent: int = 20) -> UserProfile:  # noqa: ARG002
        self.profile.record_prompt_usage(prompt)
        return self.profile

    def record_prompt_version(
        self,
        prompt: Prompt,
        *,
        commit_message: str | None = None,
        parent_version_id: int | None = None,
    ) -> PromptVersion:
        self._version_counter += 1
        version_list = self._versions.setdefault(prompt.id, [])
        version_number = len(version_list) + 1
        version = PromptVersion(
            id=self._version_counter,
            prompt_id=prompt.id,
            version_number=version_number,
            created_at=datetime.now(UTC),
            parent_version_id=parent_version_id,
            commit_message=commit_message,
            snapshot=prompt.to_record(),
        )
        version_list.append(version)
        self._version_index[version.id] = version
        return version

    def list_prompt_versions(
        self,
        prompt_id: uuid.UUID,
        *,
        limit: int | None = None,
    ) -> builtins.list[PromptVersion]:
        versions = list(self._versions.get(prompt_id, []))
        versions.sort(key=lambda version: version.version_number, reverse=True)
        if limit is not None:
            versions = versions[:limit]
        return versions

    def get_prompt_version(self, version_id: int) -> PromptVersion:
        try:
            return self._version_index[version_id]
        except KeyError as exc:
            raise RepositoryNotFoundError("version missing") from exc

    def get_prompt_latest_version(self, prompt_id: uuid.UUID) -> PromptVersion | None:
        versions = self._versions.get(prompt_id)
        if not versions:
            return None
        return max(versions, key=lambda version: version.version_number)

    def record_prompt_fork(
        self,
        source_prompt_id: uuid.UUID,
        child_prompt_id: uuid.UUID,
    ) -> PromptForkLink:
        self._fork_counter += 1
        link = PromptForkLink(
            id=self._fork_counter,
            source_prompt_id=source_prompt_id,
            child_prompt_id=child_prompt_id,
            created_at=datetime.now(UTC),
        )
        self._fork_links[link.id] = link
        return link

    def get_prompt_parent_fork(self, prompt_id: uuid.UUID) -> PromptForkLink | None:
        for link in self._fork_links.values():
            if link.child_prompt_id == prompt_id:
                return link
        return None

    def list_prompt_children(self, prompt_id: uuid.UUID) -> builtins.list[PromptForkLink]:
        return [link for link in self._fork_links.values() if link.source_prompt_id == prompt_id]


class _RedisStub:
    """Redis client double that can be configured per test."""
    def __init__(
        self,
        *,
        payload: str | None = None,
        get_exception: BaseException | None = None,
        delete_exception: BaseException | None = None,
        set_exception: BaseException | None = None,
    ) -> None:
        self._payload = payload
        self._get_exception = get_exception
        self._delete_exception = delete_exception
        self._set_exception = set_exception
        self.setex_calls: list[dict[str, Any]] = []
        self.deleted_keys: list[str] = []

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.setex_calls.append({"key": key, "ttl": ttl, "value": value})
        if self._set_exception:
            raise self._set_exception

    def get(self, key: str) -> bytes | None:
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
        context="Base context",
    )


def _build_manager(
    *,
    repository: _RecordingRepository | None = None,
    collection: _StubCollection | None = None,
    redis_client: _RedisStub | None = None,
    name_generator: Any | None = None,
    description_generator: Any | None = None,
    category_generator: Any | None = None,
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
        intent_classifier=IntentClassifier(),
        name_generator=name_generator,
        description_generator=description_generator,
        category_generator=category_generator,
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


def test_suggest_prompts_prioritises_classifier_matches() -> None:
    repository = _RecordingRepository()
    debug_prompt = Prompt(
        id=uuid.uuid4(),
        name="Debug helper",
        description="Fix errors",
        category="Reasoning / Debugging",
        tags=["debugging"],
    )
    doc_prompt = Prompt(
        id=uuid.uuid4(),
        name="Docs",
        description="Document stuff",
        category="Documentation",
        tags=["docs"],
    )
    general_prompt = Prompt(
        id=uuid.uuid4(),
        name="General helper",
        description="General purpose",
        category="General",
        tags=["general"],
    )
    for prompt in (debug_prompt, doc_prompt, general_prompt):
        repository.add(prompt)

    collection = _StubCollection(
        query_result={
            "ids": [[str(debug_prompt.id), str(doc_prompt.id), str(general_prompt.id)]],
            "documents": [
                [debug_prompt.document, doc_prompt.document, general_prompt.document]
            ],
            "metadatas": [
                [
                    debug_prompt.to_metadata(),
                    doc_prompt.to_metadata(),
                    general_prompt.to_metadata(),
                ]
            ],
        }
    )

    manager = _build_manager(repository=repository, collection=collection)
    suggestions = manager.suggest_prompts("error during deployment", limit=2)

    assert suggestions.prediction.label.value == "debug"
    assert suggestions.prompts[0].id == debug_prompt.id
    assert not suggestions.fallback_used


def test_suggest_prompts_falls_back_to_repository_list() -> None:
    repository = _RecordingRepository()
    prompt = Prompt(
        id=uuid.uuid4(),
        name="Fallback",
        description="Baseline",
        category="General",
        tags=["misc"],
    )
    repository.add(prompt)

    empty_result = {"ids": [[]], "documents": [[]], "metadatas": [[]]}
    collection = _StubCollection(query_result=empty_result)
    manager = _build_manager(repository=repository, collection=collection)

    suggestions = manager.suggest_prompts("", limit=1)
    assert suggestions.fallback_used
    assert suggestions.prompts == repository.list(limit=1)


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
    collection = _StubCollection(upsert_exception=ChromaError("nope"))
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


def test_generate_prompt_name_requires_configured_generator() -> None:
    manager = _build_manager()
    with pytest.raises(NameGenerationError):
        manager.generate_prompt_name("Generate a catchy name from this context.")


def test_generate_prompt_description_requires_configured_generator() -> None:
    manager = _build_manager()
    with pytest.raises(DescriptionGenerationError):
        manager.generate_prompt_description(
            "Provide summary for this context.", allow_fallback=False
        )


def test_generate_prompt_description_uses_generator() -> None:
    class _DescriptionGenerator:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def generate(self, context: str) -> str:
            self.calls.append(context)
            return "Auto generated description"

    generator = _DescriptionGenerator()
    manager = _build_manager(description_generator=generator)
    summary = manager.generate_prompt_description("My prompt body")
    assert summary == "Auto generated description"
    assert generator.calls == ["My prompt body"]


def test_generate_prompt_description_falls_back_when_unconfigured() -> None:
    manager = _build_manager()
    summary = manager.generate_prompt_description("Summarise the release checklist.")
    assert "overview" in summary.lower()


def test_generate_prompt_description_falls_back_on_generator_failure() -> None:
    class _FailingGenerator:
        def generate(self, context: str) -> str:  # type: ignore[override]
            raise DescriptionGenerationError("model unavailable")

    manager = _build_manager(description_generator=_FailingGenerator())
    summary = manager.generate_prompt_description("List deployment steps and checks.")
    assert "deployment" in summary.lower()


def test_generate_prompt_category_uses_llm_when_available() -> None:
    class _CategoryGenerator:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def generate(self, context: str, *, categories: Sequence[PromptCategory]) -> str:
            self.calls.append(context)
            assert categories  # ensure catalogue passed through
            return "Documentation"

    generator = _CategoryGenerator()
    manager = _build_manager(category_generator=generator)
    suggestion = manager.generate_prompt_category("Document the deployment pipeline.")
    assert suggestion == "Documentation"
    assert generator.calls == ["Document the deployment pipeline."]


def test_generate_prompt_category_falls_back_without_llm() -> None:
    manager = _build_manager()
    suggestion = manager.generate_prompt_category("Share a weekly status summary for leadership.")
    assert suggestion == "Reporting"


def test_generate_prompt_category_falls_back_when_llm_fails() -> None:
    class _FailingGenerator:
        def generate(self, context: str, *, categories: Sequence[PromptCategory]) -> str:  # type: ignore[override]
            raise CategorySuggestionError("model unavailable")

    manager = _build_manager(category_generator=_FailingGenerator())
    suggestion = manager.generate_prompt_category("Propose enhancements to optimise performance.")
    assert suggestion == "Enhancement"


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
    repo.storage[prompt.id] = _clone_prompt(prompt)
    manager = _build_manager(repository=repo)
    with pytest.raises(PromptStorageError):
        manager.update_prompt(prompt)

    class _MissingRepo(_RecordingRepository):
        def update(self, prompt: Prompt) -> Prompt:  # type: ignore[override]
            raise RepositoryNotFoundError("missing")

    repo_missing = _MissingRepo()
    repo_missing.storage[prompt.id] = _clone_prompt(prompt)
    manager_missing = _build_manager(repository=repo_missing)
    with pytest.raises(PromptNotFoundError):
        manager_missing.update_prompt(prompt)


def test_update_prompt_handles_chroma_and_cache_failures() -> None:
    from core.prompt_manager import ChromaError

    prompt = _sample_prompt()
    repo = _RecordingRepository()
    repo.storage[prompt.id] = _clone_prompt(prompt)
    collection = _StubCollection(upsert_exception=ChromaError("upsert"))
    manager = _build_manager(repository=repo, collection=collection)
    with pytest.raises(PromptStorageError):
        manager.update_prompt(prompt)

    collection_ok = _StubCollection()
    manager_ok = _build_manager(repository=repo, collection=collection_ok)

    def _cache_fail(_: Prompt) -> None:
        raise PromptCacheError("refresh")

    manager_ok._cache_prompt = _cache_fail  # type: ignore[assignment]
    repo.storage[prompt.id] = _clone_prompt(prompt)
    updated = manager_ok.update_prompt(prompt)
    assert updated.id == prompt.id


def test_suggest_prompts_handles_repository_error_on_empty_query() -> None:
    class _Repo(_RecordingRepository):
        def list(self, limit: int | None = None) -> builtins.list[Prompt]:  # type: ignore[override]
            raise RepositoryError("list failed")

    manager = _build_manager(repository=_Repo())
    with pytest.raises(PromptStorageError):
        manager.suggest_prompts("   ", limit=1)


def test_suggest_prompts_handles_initial_search_errors() -> None:
    manager = _build_manager()

    def _fail_search(self: PromptManager, *_: Any, **__: Any) -> list[Prompt]:
        raise PromptManagerError("search failed")

    manager.search_prompts = types.MethodType(_fail_search, manager)
    result = manager.suggest_prompts("Find diagnostics")
    assert result.fallback_used is True


def test_suggest_prompts_handles_fallback_search_errors() -> None:
    manager = _build_manager()
    calls = {"count": 0}

    def _search(self: PromptManager, query: str, *, limit: int = 5) -> list[Prompt]:
        calls["count"] += 1
        if calls["count"] == 1:
            return []
        raise PromptManagerError("fallback failed")

    manager.search_prompts = types.MethodType(_search, manager)
    result = manager.suggest_prompts("Investigate ci flakes", limit=1)
    assert result.fallback_used is True


def test_suggest_prompts_raises_when_repository_fallback_fails() -> None:
    class _Repo(_RecordingRepository):
        def list(self, limit: int | None = None) -> builtins.list[Prompt]:  # type: ignore[override]
            raise RepositoryError("list fail")

    manager = _build_manager(repository=_Repo())

    def _empty_search(self: PromptManager, *_: Any, **__: Any) -> list[Prompt]:
        return []

    manager.search_prompts = types.MethodType(_empty_search, manager)
    with pytest.raises(PromptStorageError):
        manager.suggest_prompts("Observability prompts")


def test_update_prompt_requires_existing_prompt_metadata() -> None:
    prompt = _sample_prompt()

    class _MissingRepo(_RecordingRepository):
        def get(self, prompt_id: uuid.UUID) -> Prompt:  # type: ignore[override]
            raise RepositoryNotFoundError("missing get")

    manager_missing = _build_manager(repository=_MissingRepo())
    with pytest.raises(PromptNotFoundError):
        manager_missing.update_prompt(prompt)

    class _ErrorRepo(_RecordingRepository):
        def get(self, prompt_id: uuid.UUID) -> Prompt:  # type: ignore[override]
            raise RepositoryError("broken get")

    manager_error = _build_manager(repository=_ErrorRepo())
    with pytest.raises(PromptStorageError):
        manager_error.update_prompt(prompt)


def test_update_prompt_handles_version_lookup_failures() -> None:
    prompt = _sample_prompt()

    class _VersionRepo(_RecordingRepository):
        def __init__(self) -> None:
            super().__init__()
            self.storage[prompt.id] = _clone_prompt(prompt)

        def get_prompt_latest_version(self, prompt_id: uuid.UUID) -> PromptVersion | None:  # noqa: D401
            raise RepositoryError("version lookup failed")

    repo = _VersionRepo()
    manager = _build_manager(repository=repo)
    assert manager.update_prompt(prompt).id == prompt.id


def test_update_prompt_skips_version_when_body_unchanged() -> None:
    prompt = _sample_prompt()

    class _VersionRepo(_RecordingRepository):
        def __init__(self) -> None:
            super().__init__()
            self.storage[prompt.id] = _clone_prompt(prompt)

        def get_prompt_latest_version(self, prompt_id: uuid.UUID) -> PromptVersion | None:  # noqa: D401
            return PromptVersion(
                id=1,
                prompt_id=prompt_id,
                version_number=0,
                created_at=datetime.now(UTC),
                parent_version_id=None,
                commit_message=None,
                snapshot=self.storage[prompt_id].to_record(),
            )

    repo = _VersionRepo()
    manager = _build_manager(repository=repo)
    result = manager.update_prompt(prompt)
    assert result.version == prompt.version


def test_update_prompt_schedules_embedding_refresh_on_failure() -> None:
    prompt = _sample_prompt()
    repo = _RecordingRepository()
    repo.storage[prompt.id] = _clone_prompt(prompt)
    manager = _build_manager(repository=repo)

    class _FailingEmbeddingProvider:
        def embed(self, _: str) -> list[float]:
            raise EmbeddingGenerationError("fail")

    class _Worker:
        def __init__(self) -> None:
            self.scheduled: list[uuid.UUID] = []

        def schedule(self, prompt_id: uuid.UUID) -> None:
            self.scheduled.append(prompt_id)

    manager._embedding_provider = _FailingEmbeddingProvider()  # type: ignore[assignment]
    worker = _Worker()
    manager._embedding_worker = worker  # type: ignore[assignment]

    def _cache_fail(_: Prompt) -> None:
        raise PromptCacheError("cache")

    manager._cache_prompt = _cache_fail  # type: ignore[assignment]
    assert manager.update_prompt(prompt).id == prompt.id
    assert worker.scheduled == [prompt.id]


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


def test_prompt_version_commit_and_restore() -> None:
    repo = _RecordingRepository()
    manager = _build_manager(repository=repo)
    prompt = _sample_prompt()
    manager.create_prompt(prompt)

    prompt.description = "Updated"
    prompt.context = (prompt.context or "") + "\nMore detail"
    manager.update_prompt(prompt)

    versions = manager.list_prompt_versions(prompt.id)
    assert len(versions) == 2
    assert versions[0].version_number == 2

    diff = manager.diff_prompt_versions(versions[1].id, versions[0].id)
    assert "description" in diff.changed_fields
    assert diff.changed_fields["description"]["to"] == "Updated"

    restored = manager.restore_prompt_version(versions[1].id)
    assert restored.description == "Sample"


def test_metadata_only_update_does_not_create_prompt_version() -> None:
    repo = _RecordingRepository()
    manager = _build_manager(repository=repo)
    prompt = _sample_prompt()
    manager.create_prompt(prompt)

    prompt.description = "Metadata tweak"
    manager.update_prompt(prompt)

    versions = manager.list_prompt_versions(prompt.id)
    assert len(versions) == 1


def test_metadata_only_update_creates_initial_version_when_missing_history() -> None:
    repo = _RecordingRepository()
    manager = _build_manager(repository=repo)
    prompt = _sample_prompt()
    repo.add(prompt)

    prompt.description = "Legacy metadata tweak"
    manager.update_prompt(prompt)

    versions = manager.list_prompt_versions(prompt.id)
    assert len(versions) == 1
    assert versions[0].version_number == 1


def test_prompt_merge_detects_conflicts_and_can_persist() -> None:
    repo = _RecordingRepository()
    manager = _build_manager(repository=repo)
    prompt = _sample_prompt()
    manager.create_prompt(prompt)

    prompt.context = "Line 1"
    manager.update_prompt(prompt)
    base_version = manager.list_prompt_versions(prompt.id)[0]

    prompt.context = "Line 1\nLine 2"
    manager.update_prompt(prompt)
    incoming_version = manager.list_prompt_versions(prompt.id)[0]

    prompt.context = "Line 1\nLine 3"
    manager.update_prompt(prompt)

    merged, conflicts = manager.merge_prompt_versions(
        prompt.id,
        base_version_id=base_version.id,
        incoming_version_id=incoming_version.id,
        persist=True,
    )

    assert "Line 2" in (merged.context or "")
    assert "context" in conflicts


def test_prompt_fork_records_lineage() -> None:
    repo = _RecordingRepository()
    manager = _build_manager(repository=repo)
    prompt = _sample_prompt()
    manager.create_prompt(prompt)

    forked = manager.fork_prompt(prompt.id, name="Experiment")
    assert forked.name == "Experiment"
    assert forked.source == "fork"

    parent_link = manager.get_prompt_parent_fork(forked.id)
    assert parent_link is not None
    assert parent_link.source_prompt_id == prompt.id

    children = manager.list_prompt_forks(prompt.id)
    assert any(link.child_prompt_id == forked.id for link in children)
