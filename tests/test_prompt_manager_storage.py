"""Tests covering storage coordination across SQLite, ChromaDB, and Redis facades.

Updates: v0.1.2 - 2025-12-05 - Add response style repository and manager coverage.
Updates: v0.1.1 - 2025-11-02 - Added regression tests for cache usage and search fallback paths.
Updates: v0.1.0 - 2025-10-31 - Introduce repository and manager storage integration tests.
"""

from __future__ import annotations

from datetime import datetime, timezone
import uuid
from typing import Any, Dict, List, Optional

import pytest

from core import (
    PromptManager,
    PromptRepository,
    RepositoryError,
    RepositoryNotFoundError,
)
from models.prompt_model import Prompt, PromptVersion, UserProfile
from models.response_style import ResponseStyle
from models.prompt_note import PromptNote


def _make_prompt(name: str = "Sample Prompt") -> Prompt:
    """Return a populated Prompt instance for testing."""
    return Prompt(
        id=uuid.uuid4(),
        name=name,
        description="Prompt description",
        category="general",
        tags=["test", "storage"],
        language="en",
        context="Use for validating storage layers.",
        example_input="Input example",
        example_output="Output example",
        author="tester",
        version="1",
        is_active=True,
        source="tests",
    )


def _make_response_style(name: str = "Concise Summary") -> ResponseStyle:
    """Return a populated ResponseStyle instance for manager tests."""

    now = datetime.now(timezone.utc)
    return ResponseStyle(
        id=uuid.uuid4(),
        name=name,
        description="Produce concise summaries with actionable next steps.",
        tone="concise",
        voice="mentor",
        format_instructions="Use bullet lists followed by a short paragraph.",
        guidelines="Avoid speculation, cite assumptions explicitly.",
        tags=["concise", "summary"],
        examples=["Summary:\n- Finding\n- Next step"],
        version="1.0",
        created_at=now,
        last_modified=now,
    )


def _make_prompt_note(text: str = "Investigate latency spike") -> PromptNote:
    now = datetime.now(timezone.utc)
    return PromptNote(id=uuid.uuid4(), note=text, created_at=now, last_modified=now)


class _FakeRedis:
    """Minimal Redis facade storing values in-memory for assertions."""

    def __init__(self) -> None:
        self._store: Dict[str, bytes] = {}
        self.get_calls = 0

    def setex(self, key: str, ttl: int, value: str) -> None:
        self._store[key] = value.encode("utf-8")

    def get(self, key: str) -> Optional[bytes]:
        self.get_calls += 1
        return self._store.get(key)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)


class _FakeCollection:
    """In-memory stand-in for ChromaDB collection operations."""

    def __init__(self) -> None:
        self._records: Dict[str, Dict[str, Any]] = {}

    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[Dict[str, Any]],
        embeddings: Optional[list[list[float]]] = None,
    ) -> None:
        for idx, prompt_id in enumerate(ids):
            self._records[prompt_id] = {
                "document": documents[idx],
                "metadata": metadatas[idx],
                "embedding": embeddings[idx] if embeddings else None,
            }

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[Dict[str, Any]],
        embeddings: Optional[list[list[float]]] = None,
    ) -> None:
        self.add(ids, documents, metadatas, embeddings)

    def delete(self, ids: list[str]) -> None:
        for prompt_id in ids:
            self._records.pop(prompt_id, None)

    def query(
        self,
        query_texts: Optional[list[str]],
        query_embeddings: Optional[list[list[float]]],
        n_results: int,
        where: Optional[Dict[str, Any]],
    ) -> Dict[str, list[list[Any]]]:
        prompt_ids = list(self._records.keys())[:n_results]
        documents = [self._records[prompt_id]["document"] for prompt_id in prompt_ids]
        metadatas = [self._records[prompt_id]["metadata"] for prompt_id in prompt_ids]
        return {"ids": [prompt_ids], "documents": [documents], "metadatas": [metadatas]}


class _FakeChromaClient:
    """Return a pre-seeded fake collection in place of PersistentClient."""

    def __init__(self, collection: _FakeCollection) -> None:
        self._collection = collection
        self.closed = False

    def get_or_create_collection(
        self,
        name: str,
        metadata: Dict[str, Any],
        embedding_function: Optional[Any] = None,
    ) -> _FakeCollection:
        return self._collection

    def close(self) -> None:
        self.closed = True


class _FakeRepository:
    """Minimal repository stub tracking invocation counts."""

    def __init__(self) -> None:
        self._store: Dict[uuid.UUID, Prompt] = {}
        self.get_call_count = 0
        self.fail_on_get = False
        self._profile = UserProfile.create_default()
        self._versions: Dict[uuid.UUID, List[PromptVersion]] = {}
        self._next_version_id = 1

    def add(self, prompt: Prompt) -> Prompt:
        if prompt.id in self._store:
            raise RepositoryError(f"Prompt {prompt.id} already exists")
        self._store[prompt.id] = prompt
        return prompt

    def get(self, prompt_id: uuid.UUID) -> Prompt:
        self.get_call_count += 1
        if self.fail_on_get:
            raise AssertionError("Repository.get should not be invoked when cache is primed.")
        try:
            return self._store[prompt_id]
        except KeyError as exc:
            raise RepositoryNotFoundError(f"Prompt {prompt_id} not found") from exc

    def get_user_profile(self) -> UserProfile:
        return self._profile

    def record_user_prompt_usage(self, prompt: Prompt, *, max_recent: int = 20) -> UserProfile:  # noqa: ARG002
        self._profile.record_prompt_usage(prompt)
        return self._profile

    def update(self, prompt: Prompt) -> Prompt:
        if prompt.id not in self._store:
            raise RepositoryNotFoundError(f"Prompt {prompt.id} not found")
        self._store[prompt.id] = prompt
        return prompt

    def delete(self, prompt_id: uuid.UUID) -> None:
        try:
            del self._store[prompt_id]
        except KeyError as exc:
            raise RepositoryNotFoundError(f"Prompt {prompt_id} not found") from exc

    def record_prompt_version(
        self,
        prompt: Prompt,
        *,
        commit_message: Optional[str] = None,
        parent_version_id: Optional[int] = None,
    ) -> PromptVersion:
        history = self._versions.setdefault(prompt.id, [])
        version_number = len(history) + 1
        version = PromptVersion(
            id=self._next_version_id,
            prompt_id=prompt.id,
            version_number=version_number,
            created_at=datetime.now(timezone.utc),
            parent_version_id=parent_version_id,
            commit_message=commit_message,
            snapshot=prompt.to_record(),
        )
        self._next_version_id += 1
        history.append(version)
        return version

    def list_prompt_versions(
        self,
        prompt_id: uuid.UUID,
        *,
        limit: Optional[int] = None,
    ) -> List[PromptVersion]:
        versions = self._versions.get(prompt_id, [])
        ordered = sorted(versions, key=lambda version: version.version_number, reverse=True)
        if limit is not None:
            return ordered[:limit]
        return ordered

    def get_prompt_version(self, version_id: int) -> PromptVersion:
        for history in self._versions.values():
            for version in history:
                if version.id == version_id:
                    return version
        raise RepositoryNotFoundError(f"Prompt version {version_id} not found")

    def get_prompt_latest_version(self, prompt_id: uuid.UUID) -> Optional[PromptVersion]:
        versions = self._versions.get(prompt_id)
        if not versions:
            return None
        return max(versions, key=lambda version: version.version_number)


class _StubWorker:
    def __init__(self) -> None:
        self.stopped = False

    def schedule(self, _: uuid.UUID) -> None:
        return

    def stop(self) -> None:
        self.stopped = True


class _StubRedis:
    def __init__(self) -> None:
        self.closed = False
        self.pool_closed = False

        class _Pool:
            def __init__(self, outer: "_StubRedis") -> None:
                self._outer = outer

            def disconnect(self) -> None:
                self._outer.pool_closed = True

        self.connection_pool = _Pool(self)

    def close(self) -> None:
        self.closed = True


def test_repository_roundtrip(tmp_path) -> None:
    """Ensure the repository supports CRUD operations."""
    repo_path = tmp_path / "prompt_manager.db"
    repository = PromptRepository(str(repo_path))
    prompt = _make_prompt()

    repository.add(prompt)
    loaded = repository.get(prompt.id)
    assert loaded.name == prompt.name
    assert loaded.tags == prompt.tags

    prompt.description = "Updated description"
    prompt.usage_count = 3
    repository.update(prompt)

    updated = repository.get(prompt.id)
    assert updated.description == "Updated description"
    assert updated.usage_count == 3

    repository.delete(prompt.id)
    with pytest.raises(RepositoryNotFoundError):
        repository.get(prompt.id)


def test_prompt_manager_coordinates_sqlite_and_chromadb(tmp_path) -> None:
    """Validate manager persistence across SQLite, ChromaDB, and cache facade."""
    chroma_dir = tmp_path / "chroma"
    db_path = tmp_path / "prompt_manager.db"
    manager = PromptManager(
        chroma_path=str(chroma_dir),
        db_path=str(db_path),
    )

    prompt = _make_prompt("Integration Prompt")
    embedding = [0.1, 0.2, 0.3]
    manager.create_prompt(prompt, embedding=embedding)

    stored = manager.repository.get(prompt.id)
    assert stored.id == prompt.id

    fetched = manager.get_prompt(prompt.id)
    assert fetched.name == "Integration Prompt"

    prompt.description = "Adjusted via manager"
    manager.update_prompt(prompt, embedding=embedding)
    updated = manager.repository.get(prompt.id)
    assert updated.description == "Adjusted via manager"

    results = manager.search_prompts("", embedding=embedding, limit=1)
    assert results, "Expected search results from ChromaDB"
    assert results[0].id == prompt.id
    assert results[0].description == "Adjusted via manager"

    manager.delete_prompt(prompt.id)
    with pytest.raises(RepositoryNotFoundError):
        manager.repository.get(prompt.id)
    chroma_result = manager.collection.get(ids=[str(prompt.id)])
    assert not chroma_result.get("ids"), "ChromaDB entry should be removed"
    manager.close()



def test_prompt_manager_response_style_workflow(tmp_path) -> None:
    """Ensure response style CRUD is wired through the manager."""

    chroma_dir = tmp_path / "chroma"
    db_path = tmp_path / "prompt_manager.db"
    manager = PromptManager(
        chroma_path=str(chroma_dir),
        db_path=str(db_path),
    )

    style = _make_response_style()
    manager.create_response_style(style)

    styles = manager.list_response_styles()
    assert styles and styles[0].id == style.id

    fetched = manager.get_response_style(style.id)
    assert fetched.name == style.name

    style.description = "Updated summary instructions"
    manager.update_response_style(style)
    refreshed = manager.get_response_style(style.id)
    assert refreshed.description == "Updated summary instructions"

    manager.delete_response_style(style.id)
    assert not manager.list_response_styles()

    manager.close()


def test_prompt_manager_prompt_note_workflow(tmp_path) -> None:
    """Ensure prompt note CRUD is exposed via the manager."""

    chroma_dir = tmp_path / "chroma"
    db_path = tmp_path / "prompt_manager.db"
    manager = PromptManager(
        chroma_path=str(chroma_dir),
        db_path=str(db_path),
    )

    note = _make_prompt_note()
    manager.create_prompt_note(note)

    notes = manager.list_prompt_notes()
    assert notes and notes[0].id == note.id

    fetched = manager.get_prompt_note(note.id)
    assert fetched.note == note.note

    note.note = "Update onboarding docs"
    manager.update_prompt_note(note)
    refreshed = manager.get_prompt_note(note.id)
    assert refreshed.note.startswith("Update")

    manager.delete_prompt_note(note.id)
    assert not manager.list_prompt_notes()

    manager.close()


def test_get_prompt_reads_from_cache_before_repository() -> None:
    """Ensure cached prompts short-circuit repository lookups."""
    collection = _FakeCollection()
    repo = _FakeRepository()
    redis_client = _FakeRedis()
    manager = PromptManager(
        chroma_path="unused",
        chroma_client=_FakeChromaClient(collection),
        redis_client=redis_client,
        repository=repo,
    )

    prompt = _make_prompt("Cached Prompt")
    manager.create_prompt(prompt)
    repo.fail_on_get = True

    cached = manager.get_prompt(prompt.id)
    assert cached.id == prompt.id
    assert repo.get_call_count == 0
    manager.close()


def test_search_prompts_returns_chroma_records_when_sqlite_missing() -> None:
    """Fallback to Chroma metadata when repository entry is absent."""
    collection = _FakeCollection()
    repo = _FakeRepository()
    manager = PromptManager(
        chroma_path="unused",
        chroma_client=_FakeChromaClient(collection),
        repository=repo,
    )

    prompt_id = str(uuid.uuid4())
    collection.add(
        ids=[prompt_id],
        documents=["Doc body"],
        metadatas=[
            {
                "name": "Chroma Prompt",
                "description": "Recovered from metadata",
                "category": "fallback",
                "tags": '["meta"]',
                "related_prompts": '["abc"]',
                "created_at": "2025-10-30T00:00:00+00:00",
                "last_modified": "2025-10-30T00:00:00+00:00",
                "version": "1",
                "usage_count": 0,
                "is_active": True,
            }
        ],
        embeddings=[[0.0, 0.1, 0.2]],
    )

    results = manager.search_prompts("lookup", limit=1)
    assert results
    prompt = results[0]
    assert str(prompt.id) == prompt_id
    assert prompt.description == "Recovered from metadata"
    manager.close()


def test_prompt_manager_close_shuts_down_workers_and_clients(tmp_path) -> None:
    collection = _FakeCollection()
    chroma_client = _FakeChromaClient(collection)
    repository = _FakeRepository()
    worker = _StubWorker()
    redis_client = _StubRedis()

    manager = PromptManager(
        chroma_path=str(tmp_path / "chroma"),
        db_path=None,
        chroma_client=chroma_client,
        repository=repository,  # type: ignore[arg-type]
        embedding_worker=worker,
        redis_client=redis_client,  # type: ignore[arg-type]
        enable_background_sync=False,
    )

    manager.close()
    manager.close()

    assert worker.stopped is True
    assert redis_client.closed is True
    assert redis_client.pool_closed is True
    assert chroma_client.closed is True
