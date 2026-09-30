"""Real SQLite guards for ordinary creation and its first version snapshot."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from typing import TYPE_CHECKING, Any, cast

import pytest

from core import PromptManager, PromptStorageError, RepositoryError, RepositoryNotFoundError
from core.embedding import EmbeddingGenerationError
from core.repository import prompts as prompt_store
from tests.test_prompt_manager_branches import (
    _TestChromaError,  # pyright: ignore[reportPrivateUsage]
)
from tests.test_prompt_manager_storage import (
    _FakeChromaClient,  # pyright: ignore[reportPrivateUsage]
    _FakeCollection,  # pyright: ignore[reportPrivateUsage]
    _FakeRedis,  # pyright: ignore[reportPrivateUsage]
    _make_prompt,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from models.prompt_model import Prompt


@pytest.fixture
def create_manager(tmp_path: Path) -> Iterator[tuple[PromptManager, _FakeCollection]]:
    collection = _FakeCollection()
    manager = PromptManager(
        chroma_path=str(tmp_path / "chroma"),
        db_path=tmp_path / "catalog.db",
        chroma_client=cast("Any", _FakeChromaClient(collection)),
        redis_client=cast("Any", _FakeRedis()),
        enable_background_sync=False,
    )
    try:
        yield manager, collection
    finally:
        manager.close()


def _reject_snapshots(tmp_path: Path) -> None:
    with closing(sqlite3.connect(tmp_path / "catalog.db")) as conn:
        conn.execute(
            """
            CREATE TRIGGER reject_initial_snapshot BEFORE INSERT ON prompt_versions
            BEGIN SELECT RAISE(ABORT, 'synthetic snapshot failure'); END;
            """
        )
        conn.commit()


@pytest.mark.parametrize("deferred", [False, True])
def test_initial_snapshot_failure_does_not_publish_creation(
    create_manager: tuple[PromptManager, _FakeCollection],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    deferred: bool,
) -> None:
    manager, collection = create_manager
    existing = manager.create_prompt(_make_prompt("Existing"), embedding=[0.1, 0.2])
    existing_row = manager.repository.get(existing.id).to_record()
    existing_versions = manager.list_prompt_versions(existing.id)
    existing_activity = manager.repository.list_prompt_activity()
    indexes = dict(cast("Any", collection)._records)
    redis = cast("Any", manager)._redis_client
    cache = dict(redis._store)
    publication: list[str] = []

    def index(*_args: object, **_kwargs: object) -> None:
        publication.append("index")

    def cached(_prompt: Prompt) -> None:
        publication.append("cache")

    def scheduled(_prompt_id: object) -> None:
        publication.append("worker")

    def embed(_text: str) -> list[float]:
        raise EmbeddingGenerationError("synthetic deferred embedding")

    monkeypatch.setattr(collection, "upsert", index)
    monkeypatch.setattr(manager, "_cache_prompt", cached)
    monkeypatch.setattr(cast("Any", manager)._embedding_worker, "schedule", scheduled)
    monkeypatch.setattr(cast("Any", manager)._embedding_provider, "embed", embed)
    _reject_snapshots(tmp_path)
    prompt = _make_prompt("Rejected creation")
    with pytest.raises(PromptStorageError) as failure:
        manager.create_prompt(prompt, embedding=None if deferred else [0.3, 0.4])
    assert isinstance(failure.value.__cause__, RepositoryError)

    with pytest.raises(RepositoryNotFoundError):
        manager.repository.get(prompt.id)
    assert manager.list_prompt_versions(prompt.id) == []
    assert publication == []
    assert manager.repository.get(existing.id).to_record() == existing_row
    assert manager.list_prompt_versions(existing.id) == existing_versions
    assert manager.repository.list_prompt_activity() == existing_activity
    assert cast("Any", collection)._records == indexes
    assert redis._store == cache


@pytest.mark.parametrize("deferred", [False, True])
def test_successful_creation_snapshot_is_readable_before_publication(
    create_manager: tuple[PromptManager, _FakeCollection],
    monkeypatch: pytest.MonkeyPatch,
    deferred: bool,
) -> None:
    manager, collection = create_manager
    prompt = _make_prompt("Committed creation")
    publication: list[str] = []
    real_upsert = collection.upsert
    real_cache = cast("Any", manager)._cache_prompt

    def committed() -> None:
        stored = manager.repository.get(prompt.id)
        versions = manager.list_prompt_versions(prompt.id)
        assert len(versions) == 1
        version = manager.repository.get_prompt_version(versions[0].id)
        assert version.version_number == 1
        assert version.parent_version_id is None
        assert version.commit_message == "initial creation"
        assert version.snapshot == stored.to_record()

    def index(*args: Any, **kwargs: Any) -> None:
        committed()
        publication.append("index")
        real_upsert(*args, **kwargs)

    def cached(current: Prompt) -> None:
        committed()
        publication.append("cache")
        real_cache(current)

    def scheduled(_prompt_id: object) -> None:
        committed()
        publication.append("worker")

    def embed(_text: str) -> list[float]:
        raise EmbeddingGenerationError("synthetic deferred embedding")

    monkeypatch.setattr(collection, "upsert", index)
    monkeypatch.setattr(manager, "_cache_prompt", cached)
    monkeypatch.setattr(cast("Any", manager)._embedding_worker, "schedule", scheduled)
    monkeypatch.setattr(cast("Any", manager)._embedding_provider, "embed", embed)
    result = manager.create_prompt(
        prompt, embedding=None if deferred else [0.3, 0.4], commit_message="initial creation"
    )
    committed()
    assert result.to_record() == manager.repository.get(prompt.id).to_record()
    assert publication == (["worker", "cache"] if deferred else ["index", "cache"])
    assert manager.get_cached_prompt(prompt.id) is not None


@pytest.mark.parametrize("mutate_first", [False, True])
def test_create_chroma_failure_compensates_catalog_and_history(
    create_manager: tuple[PromptManager, _FakeCollection],
    monkeypatch: pytest.MonkeyPatch,
    mutate_first: bool,
) -> None:
    manager, collection = create_manager
    prompt = _make_prompt("Index failure")
    real_upsert = collection.upsert

    def fail(*args: Any, **kwargs: Any) -> None:
        # The row and its snapshot must already be committed when index work starts.
        assert len(manager.list_prompt_versions(prompt.id)) == 1
        if mutate_first:
            real_upsert(*args, **kwargs)
        raise _TestChromaError("synthetic index failure")

    monkeypatch.setattr(collection, "upsert", fail)
    with pytest.raises(PromptStorageError, match="Failed to persist embedding"):
        manager.create_prompt(prompt, embedding=[0.3, 0.4])
    with pytest.raises(RepositoryNotFoundError):
        manager.repository.get(prompt.id)
    assert manager.list_prompt_versions(prompt.id) == []
    assert manager.repository.list_prompt_activity() == []
    assert manager.get_cached_prompt(prompt.id) is None
    # Existing compensation deletes SQLite only, not a mutate-then-error index record.
    assert (str(prompt.id) in cast("Any", collection)._records) is mutate_first


def test_duplicate_create_preserves_existing_row_snapshot_index_and_cache(
    create_manager: tuple[PromptManager, _FakeCollection],
) -> None:
    manager, collection = create_manager
    prompt = manager.create_prompt(_make_prompt("Original"), embedding=[0.1, 0.2])
    row = manager.repository.get(prompt.id).to_record()
    history = manager.list_prompt_versions(prompt.id)
    indexes = dict(cast("Any", collection)._records)
    cache = dict(cast("Any", manager)._redis_client._store)
    prompt.context = "Rejected duplicate body"
    with pytest.raises(PromptStorageError):
        manager.create_prompt(prompt, embedding=[0.3, 0.4])
    assert manager.repository.get(prompt.id).to_record() == row
    assert manager.list_prompt_versions(prompt.id) == history
    assert cast("Any", collection)._records == indexes
    assert cast("Any", manager)._redis_client._store == cache


def test_snapshot_serialization_failure_rolls_back_insert(
    create_manager: tuple[PromptManager, _FakeCollection], monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, collection = create_manager
    prompt = _make_prompt("Serialization failure")

    def fail(_prompt: Prompt) -> str:
        raise TypeError("synthetic snapshot serialization failure")

    monkeypatch.setattr(prompt_store, "_prompt_snapshot_json", fail)
    with pytest.raises(TypeError, match="synthetic snapshot serialization failure"):
        manager.create_prompt(prompt, embedding=[0.1, 0.2])
    with pytest.raises(RepositoryNotFoundError):
        manager.repository.get(prompt.id)
    assert manager.list_prompt_versions(prompt.id) == []
    assert cast("Any", collection)._records == {}
    assert cast("Any", manager)._redis_client._store == {}


def test_atomic_create_type_error_has_no_legacy_fallback(
    create_manager: tuple[PromptManager, _FakeCollection], monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, collection = create_manager
    calls: list[str] = []

    def fail(_prompt: Prompt, *, commit_message: str | None = None) -> None:
        calls.append("atomic")
        raise TypeError("synthetic atomic seam error")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Atomic create fell back to separate row/snapshot writes")

    monkeypatch.setattr(manager.repository, "add_with_version", fail)
    monkeypatch.setattr(manager.repository, "add", forbidden)
    monkeypatch.setattr(manager.repository, "record_prompt_version", forbidden)
    with pytest.raises(TypeError, match="synthetic atomic seam error"):
        manager.create_prompt(_make_prompt(), embedding=[0.1, 0.2])
    assert calls == ["atomic"]
    assert cast("Any", collection)._records == {}
    assert cast("Any", manager)._redis_client._store == {}


def test_real_sqlite_fork_snapshot_failure_preserves_source(
    create_manager: tuple[PromptManager, _FakeCollection], tmp_path: Path
) -> None:
    manager, collection = create_manager
    source = manager.create_prompt(_make_prompt("Fork source"), embedding=[0.1, 0.2])
    row = manager.repository.get(source.id).to_record()
    history = manager.list_prompt_versions(source.id)
    indexes = dict(cast("Any", collection)._records)
    cache = dict(cast("Any", manager)._redis_client._store)
    activity = manager.repository.list_prompt_activity()
    _reject_snapshots(tmp_path)
    with pytest.raises(PromptStorageError):
        manager.fork_prompt(source.id, name="Rejected child")
    assert [prompt.id for prompt in manager.repository.list()] == [source.id]
    assert manager.repository.get(source.id).to_record() == row
    assert manager.list_prompt_versions(source.id) == history
    assert manager.list_prompt_forks(source.id) == []
    assert manager.repository.list_prompt_activity() == activity
    assert cast("Any", collection)._records == indexes
    assert cast("Any", manager)._redis_client._store == cache


def test_real_sqlite_fork_success_has_one_initial_snapshot(
    create_manager: tuple[PromptManager, _FakeCollection],
) -> None:
    manager, _collection = create_manager
    source = manager.create_prompt(_make_prompt("Fork source"), embedding=[0.1, 0.2])
    child = manager.fork_prompt(source.id, name="Committed child")
    versions = manager.list_prompt_versions(child.id)
    assert len(versions) == 1
    assert versions[0].version_number == 1
    assert versions[0].snapshot == manager.repository.get(child.id).to_record()
    lineage = manager.get_prompt_parent_fork(child.id)
    assert lineage is not None
    assert lineage.source_prompt_id == source.id
    assert lineage.child_prompt_id == child.id
    assert len(manager.list_prompt_versions(source.id)) == 1


def test_snapshot_is_invisible_and_write_locked_before_commit(
    create_manager: tuple[PromptManager, _FakeCollection],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _collection = create_manager
    prompt = _make_prompt("Uncommitted creation")
    real_snapshot = prompt_store._prompt_snapshot_json  # pyright: ignore[reportPrivateUsage]
    observed: list[bool] = []

    def snapshot(current: Prompt) -> str:
        with pytest.raises(RepositoryNotFoundError):
            manager.repository.get(current.id)
        assert manager.list_prompt_versions(current.id) == []
        with closing(sqlite3.connect(tmp_path / "catalog.db", timeout=0)) as conn:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                conn.execute("BEGIN IMMEDIATE")
        observed.append(True)
        return real_snapshot(current)

    monkeypatch.setattr(prompt_store, "_prompt_snapshot_json", snapshot)
    manager.create_prompt(prompt, embedding=[0.1, 0.2])
    assert observed == [True]
    assert len(manager.list_prompt_versions(prompt.id)) == 1


@pytest.mark.parametrize("existing_history", [False, True])
def test_silently_ignored_snapshot_insert_rolls_back_prompt(
    create_manager: tuple[PromptManager, _FakeCollection],
    tmp_path: Path,
    existing_history: bool,
) -> None:
    manager, collection = create_manager
    if existing_history:
        source = manager.create_prompt(_make_prompt("Retained history"), embedding=[0.1, 0.2])
        source.context = "Second retained version"
        manager.update_prompt(source, embedding=[0.1, 0.2], refresh_derived_state=False)
    indexes = dict(cast("Any", collection)._records)
    cache = dict(cast("Any", manager)._redis_client._store)
    with closing(sqlite3.connect(tmp_path / "catalog.db")) as conn:
        conn.execute(
            """
            CREATE TRIGGER ignore_initial_snapshot BEFORE INSERT ON prompt_versions
            BEGIN SELECT RAISE(IGNORE); END;
            """
        )
        conn.commit()
    prompt = _make_prompt("Ignored snapshot")
    with pytest.raises(PromptStorageError) as failure:
        manager.create_prompt(prompt, embedding=[0.1, 0.2])
    assert isinstance(failure.value.__cause__, RepositoryError)
    with pytest.raises(RepositoryNotFoundError):
        manager.repository.get(prompt.id)
    assert manager.list_prompt_versions(prompt.id) == []
    assert cast("Any", collection)._records == indexes
    assert cast("Any", manager)._redis_client._store == cache
