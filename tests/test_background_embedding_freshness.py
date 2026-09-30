"""Regression guards for background-derived writes to the canonical prompt store."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import closing
from typing import TYPE_CHECKING, Any, cast

import pytest

from core import PromptManager, PromptRepository, PromptStorageError, RepositoryNotFoundError
from core.embedding import EmbeddingSyncWorker
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
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from models.prompt_model import Prompt


@pytest.fixture
def worker_manager(tmp_path: Path) -> Iterator[tuple[PromptManager, _FakeCollection]]:
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


def _persist(manager: PromptManager, prompt: Prompt, vector: Sequence[float]) -> None:
    cast("Any", manager)._persist_embedding_from_worker(prompt, vector)


def test_paused_worker_does_not_revert_saved_edit(
    worker_manager: tuple[PromptManager, _FakeCollection],
) -> None:
    manager, collection = worker_manager
    prompt = manager.create_prompt(_make_prompt(), embedding=[0.1, 0.2])
    started, release, completed = threading.Event(), threading.Event(), threading.Event()

    class Provider:
        def embed(self, _text: str) -> list[float]:
            started.set()
            assert release.wait(5), "Worker was not released"
            return [0.3, 0.4]

    def persist(source: Prompt, vector: Sequence[float]) -> None:
        try:
            _persist(manager, source, vector)
        finally:
            completed.set()

    worker = EmbeddingSyncWorker(
        cast("Any", Provider()), manager.repository.get, persist, max_attempts=1
    )
    try:
        worker.schedule(prompt.id)
        assert started.wait(5)
        edited = manager.repository.get(prompt.id)
        edited.context = "A newer saved body"
        manager.update_prompt(edited, embedding=[0.7, 0.8], refresh_derived_state=False)
        expected = manager.repository.get(prompt.id).to_record()
        release.set()
        assert completed.wait(5)
        current = manager.repository.get(prompt.id)
        assert current.to_record() == expected
        latest = manager.get_latest_prompt_version(prompt.id)
        assert latest is not None
        assert latest.snapshot["context"] == current.context
        assert cast("Any", collection)._records[str(prompt.id)]["document"] == current.document
        assert cast("Any", collection)._records[str(prompt.id)]["embedding"] == [0.7, 0.8]
    finally:
        release.set()
        worker.stop()


@pytest.mark.parametrize("change", ["context", "name", "version", "status"])
def test_stale_worker_does_not_touch_index_or_cache(
    worker_manager: tuple[PromptManager, _FakeCollection],
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    manager, collection = worker_manager
    source = manager.create_prompt(_make_prompt(), embedding=[0.1, 0.2])
    stale = manager.repository.get(source.id)
    changed = manager.repository.get(source.id)
    if change == "status":
        manager.repository.set_prompt_active(source.id, active=False, expect_active=True)
    else:
        setattr(changed, change, "2" if change == "version" else "Changed source")
        manager.repository.update(changed)
    expected = manager.repository.get(source.id).to_record()

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Stale worker reached index or cache")

    monkeypatch.setattr(collection, "upsert", forbidden)
    monkeypatch.setattr(manager, "_cache_prompt", forbidden)
    monkeypatch.setattr(manager, "_evict_cached_prompt", forbidden)
    _persist(manager, stale, [0.3, 0.4])
    assert manager.repository.get(source.id).to_record() == expected


def test_current_worker_updates_only_embedding(
    worker_manager: tuple[PromptManager, _FakeCollection],
) -> None:
    manager, collection = worker_manager
    source = manager.create_prompt(_make_prompt(), embedding=[0.1, 0.2])
    stale_counters = manager.repository.get(source.id)
    current = manager.repository.get(source.id)
    current.usage_count = 8
    current.rating_count = 2
    current.rating_sum = 15.0
    manager.repository.update(current)
    expected = manager.repository.get(source.id).to_record()
    expected["ext4"] = [0.3, 0.4]
    _persist(manager, stale_counters, [0.3, 0.4])
    assert manager.repository.get(source.id).to_record() == expected
    assert cast("Any", collection)._records[str(source.id)]["embedding"] == [0.3, 0.4]


def test_worker_index_failure_preserves_catalog(
    worker_manager: tuple[PromptManager, _FakeCollection],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, collection = worker_manager
    source = manager.create_prompt(_make_prompt(), embedding=[0.1, 0.2])
    expected = manager.repository.get(source.id).to_record()

    def fail(*_args: object, **_kwargs: object) -> None:
        raise PromptStorageError("Synthetic index failure")

    monkeypatch.setattr(collection, "upsert", fail)
    with pytest.raises(PromptStorageError):
        _persist(manager, source, [0.3, 0.4])
    assert manager.repository.get(source.id).to_record() == expected


def test_worker_handoff_holds_catalog_write_lock(
    worker_manager: tuple[PromptManager, _FakeCollection],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, collection = worker_manager
    source = manager.create_prompt(_make_prompt(), embedding=[0.1, 0.2])
    other = PromptRepository(str(tmp_path / "catalog.db"))
    competing = other.get(source.id)
    competing.context = "Competing edit"
    started, finished = threading.Event(), threading.Event()
    errors: list[Exception] = []

    def write() -> None:
        started.set()
        try:
            other.update(competing)
        except Exception as exc:
            errors.append(exc)
        finally:
            finished.set()

    writer = threading.Thread(target=write)
    original = collection.upsert

    def upsert(**kwargs: Any) -> None:
        writer.start()
        assert started.wait(2)
        assert not finished.wait(0.1), "Catalog writer crossed the index handoff"
        original(**kwargs)

    monkeypatch.setattr(collection, "upsert", upsert)
    try:
        _persist(manager, source, [0.3, 0.4])
    finally:
        if writer.ident is not None:
            writer.join(5)
    assert finished.is_set()
    assert not errors
    assert other.get(source.id).context == "Competing edit"


@pytest.mark.parametrize("versioned", [True, False])
def test_public_update_handoff_cannot_be_overwritten_by_worker(
    worker_manager: tuple[PromptManager, _FakeCollection],
    monkeypatch: pytest.MonkeyPatch,
    versioned: bool,
) -> None:
    manager, collection = worker_manager
    source = manager.create_prompt(_make_prompt(), embedding=[0.1, 0.2])
    source = manager.repository.get(source.id)
    edited = manager.repository.get(source.id)
    if versioned:
        edited.context = "New public body"
    else:
        edited.name = "New public metadata"
    edited.usage_count = 7
    ready, release = threading.Event(), threading.Event()
    errors: list[Exception] = []
    method = "update_with_version" if versioned else "update"
    original = getattr(manager.repository, method)

    def paused_update(*args: Any, **kwargs: Any) -> Any:
        ready.set()
        assert release.wait(5), "Public writer was not released"
        return original(*args, **kwargs)

    def write() -> None:
        try:
            manager.update_prompt(edited, embedding=[0.7, 0.8], refresh_derived_state=False)
        except Exception as exc:
            errors.append(exc)

    monkeypatch.setattr(manager.repository, method, paused_update)
    writer = threading.Thread(target=write)
    writer.start()
    try:
        assert ready.wait(5)
        _persist(manager, source, [0.3, 0.4])
    finally:
        release.set()
        writer.join(5)
    assert not writer.is_alive()
    assert not errors
    saved = manager.repository.get(source.id)
    indexed = cast("Any", collection)._records[str(source.id)]
    assert saved.context == edited.context
    assert saved.name == edited.name
    assert saved.usage_count == 7
    assert saved.ext4 == [0.7, 0.8]
    assert indexed["document"] == saved.document
    assert indexed["embedding"] == saved.ext4
    assert manager.get_prompt(source.id).to_record() == saved.to_record()
    versions = manager.list_prompt_versions(source.id)
    assert len(versions) == (2 if versioned else 1)
    assert saved.version == ("2" if versioned else "1")
    if versioned:
        assert versions[0].snapshot == saved.to_record()


@pytest.mark.parametrize("versioned", [True, False])
def test_public_update_index_handoff_holds_write_lock_without_cache_publication(
    worker_manager: tuple[PromptManager, _FakeCollection],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    versioned: bool,
) -> None:
    manager, collection = worker_manager
    source = manager.create_prompt(_make_prompt(), embedding=[0.1, 0.2])
    edited = manager.repository.get(source.id)
    if versioned:
        edited.context = "Versioned change"
    else:
        edited.name = "Metadata-only change"
    original = collection.upsert

    def upsert(**kwargs: Any) -> None:
        with closing(sqlite3.connect(tmp_path / "catalog.db", timeout=0)) as conn:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                conn.execute("BEGIN IMMEDIATE")
        cached = manager.get_cached_prompt(source.id)
        assert cached is not None
        assert cached.ext4 == [0.1, 0.2], "Pre-commit vector was published to cache"
        original(**kwargs)

    monkeypatch.setattr(collection, "upsert", upsert)
    manager.update_prompt(edited, embedding=[0.7, 0.8], refresh_derived_state=False)
    saved = manager.repository.get(source.id)
    assert saved.ext4 == [0.7, 0.8]
    assert manager.get_prompt(source.id).to_record() == saved.to_record()


def test_worker_postcommit_does_not_publish_over_newer_public_update(
    worker_manager: tuple[PromptManager, _FakeCollection],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, collection = worker_manager
    source = manager.create_prompt(_make_prompt(), embedding=[0.1, 0.2])
    ready, release = threading.Event(), threading.Event()
    errors: list[Exception] = []
    original = manager.repository.update_embedding_if_current

    def paused_postcommit(*args: Any, **kwargs: Any) -> Prompt | None:
        current = original(*args, **kwargs)
        ready.set()
        assert release.wait(5), "Committed worker was not released"
        return current

    def work() -> None:
        try:
            _persist(manager, source, [0.3, 0.4])
        except Exception as exc:
            errors.append(exc)

    monkeypatch.setattr(manager.repository, "update_embedding_if_current", paused_postcommit)
    worker = threading.Thread(target=work)
    worker.start()
    try:
        assert ready.wait(5)
        edited = manager.repository.get(source.id)
        edited.context = "Newer committed public body"
        manager.update_prompt(edited, embedding=[0.7, 0.8], refresh_derived_state=False)
        expected = manager.repository.get(source.id).to_record()
        assert manager.get_prompt(source.id).to_record() == expected
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert not errors
    assert manager.get_prompt(source.id).to_record() == expected
    assert manager.repository.get(source.id).to_record() == expected
    assert cast("Any", collection)._records[str(source.id)]["embedding"] == [0.7, 0.8]


def test_deleted_worker_source_is_noop(
    worker_manager: tuple[PromptManager, _FakeCollection],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, collection = worker_manager
    source = manager.create_prompt(_make_prompt(), embedding=[0.1, 0.2])
    manager.delete_prompt(source.id)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Deleted source reached index or cache")

    monkeypatch.setattr(collection, "upsert", forbidden)
    monkeypatch.setattr(manager, "_cache_prompt", forbidden)
    monkeypatch.setattr(manager, "_evict_cached_prompt", forbidden)
    _persist(manager, source, [0.3, 0.4])
    assert manager.get_cached_prompt(source.id) is None
    assert str(source.id) not in cast("Any", collection)._records
    with pytest.raises(RepositoryNotFoundError):
        manager.repository.get(source.id)


def test_worker_index_mutate_then_error_rolls_back_catalog_not_index(
    worker_manager: tuple[PromptManager, _FakeCollection],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, collection = worker_manager
    source = manager.create_prompt(_make_prompt(), embedding=[0.1, 0.2])
    expected = manager.repository.get(source.id).to_record()
    original = collection.upsert

    def mutate_then_fail(**kwargs: Any) -> None:
        original(**kwargs)
        raise PromptStorageError("Synthetic mutation-then-error")

    monkeypatch.setattr(collection, "upsert", mutate_then_fail)
    with pytest.raises(PromptStorageError, match="inspect catalog/index consistency"):
        _persist(manager, source, [0.3, 0.4])
    assert manager.repository.get(source.id).to_record() == expected
    assert manager.get_prompt(source.id).to_record() == expected
    assert cast("Any", collection)._records[str(source.id)]["embedding"] == [0.3, 0.4]


def test_public_snapshot_failure_leaves_index_cache_and_catalog_unchanged(
    worker_manager: tuple[PromptManager, _FakeCollection],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, collection = worker_manager
    source = manager.create_prompt(_make_prompt(), embedding=[0.1, 0.2])
    expected = manager.repository.get(source.id).to_record()
    edited = manager.repository.get(source.id)
    edited.context = "Must roll back with snapshot"
    with closing(sqlite3.connect(tmp_path / "catalog.db")) as conn:
        conn.execute(
            """
            CREATE TRIGGER reject_snapshot BEFORE INSERT ON prompt_versions
            BEGIN SELECT RAISE(ABORT, 'forced snapshot failure'); END;
            """
        )
        conn.commit()

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Failed snapshot reached index or cache")

    monkeypatch.setattr(collection, "upsert", forbidden)
    monkeypatch.setattr(manager, "_cache_prompt", forbidden)
    with pytest.raises(PromptStorageError, match="Failed to update prompt"):
        manager.update_prompt(edited, embedding=[0.7, 0.8], refresh_derived_state=False)
    assert manager.repository.get(source.id).to_record() == expected
    assert manager.get_prompt(source.id).to_record() == expected
    assert cast("Any", collection)._records[str(source.id)]["embedding"] == [0.1, 0.2]
    versions = manager.list_prompt_versions(source.id)
    assert len(versions) == 1
    assert versions[0].snapshot == expected


@pytest.mark.parametrize("versioned", [True, False])
@pytest.mark.parametrize("mutate_first", [True, False])
def test_public_index_error_rolls_back_catalog_without_publishing_cache(
    worker_manager: tuple[PromptManager, _FakeCollection],
    monkeypatch: pytest.MonkeyPatch,
    versioned: bool,
    mutate_first: bool,
) -> None:
    manager, collection = worker_manager
    source = manager.create_prompt(_make_prompt(), embedding=[0.1, 0.2])
    expected = manager.repository.get(source.id).to_record()
    edited = manager.repository.get(source.id)
    if versioned:
        edited.context = "Rejected body"
    else:
        edited.name = "Rejected metadata"
    original = collection.upsert

    def fail_index(**kwargs: Any) -> None:
        if mutate_first:
            original(**kwargs)
        raise _TestChromaError("Synthetic index error")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Failed index reached cache")

    monkeypatch.setattr(collection, "upsert", fail_index)
    monkeypatch.setattr(manager, "_cache_prompt", forbidden)
    with pytest.raises(PromptStorageError, match="Failed to persist embedding"):
        manager.update_prompt(edited, embedding=[0.7, 0.8], refresh_derived_state=False)
    assert manager.repository.get(source.id).to_record() == expected
    assert manager.get_prompt(source.id).to_record() == expected
    assert len(manager.list_prompt_versions(source.id)) == 1
    # Only SQLite rolls back: a Chroma mutation-then-error remains uncertain.
    indexed = cast("Any", collection)._records[str(source.id)]
    assert indexed["embedding"] == ([0.7, 0.8] if mutate_first else [0.1, 0.2])


def test_worker_evicts_cache_without_reloading_source(
    worker_manager: tuple[PromptManager, _FakeCollection],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _ = worker_manager
    source = manager.create_prompt(_make_prompt(), embedding=[0.1, 0.2])

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Worker tried to reload or publish a cache snapshot")

    monkeypatch.setattr(manager.repository, "get", forbidden)
    monkeypatch.setattr(manager, "_cache_prompt", forbidden)
    _persist(manager, source, [0.3, 0.4])
    assert manager.get_cached_prompt(source.id) is None


@pytest.mark.parametrize("versioned", [True, False])
def test_public_index_callback_typeerror_is_not_retried_unguarded(
    worker_manager: tuple[PromptManager, _FakeCollection],
    monkeypatch: pytest.MonkeyPatch,
    versioned: bool,
) -> None:
    manager, collection = worker_manager
    source = manager.create_prompt(_make_prompt(), embedding=[0.1, 0.2])
    expected = manager.repository.get(source.id).to_record()
    edited = manager.repository.get(source.id)
    if versioned:
        edited.context = "Rejected body"
    else:
        edited.name = "Rejected metadata"
    calls = 0

    def fail(**_kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        raise TypeError("Callback failure, not a protocol mismatch")

    monkeypatch.setattr(collection, "upsert", fail)
    with pytest.raises(TypeError, match="Callback failure"):
        manager.update_prompt(edited, embedding=[0.7, 0.8], refresh_derived_state=False)
    assert calls == 1
    assert manager.repository.get(source.id).to_record() == expected
    assert manager.get_prompt(source.id).to_record() == expected
    assert len(manager.list_prompt_versions(source.id)) == 1
