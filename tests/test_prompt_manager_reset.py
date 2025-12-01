"""Tests covering PromptManager datastore reset workflows."""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from core.embedding import EmbeddingGenerationError
from core.prompt_manager import PromptManager
from core.repository import PromptRepository
from models.prompt_model import Prompt, PromptExecution, UserProfile

if TYPE_CHECKING:
    from pathlib import Path


def _make_prompt(name: str = "Diagnostics") -> Prompt:
    return Prompt(
        id=uuid.uuid4(),
        name=name,
        description="Investigate pipeline failures.",
        category="Debugging",
        tags=["ci"],
    )


def _make_execution(prompt_id: uuid.UUID) -> PromptExecution:
    return PromptExecution(
        id=uuid.uuid4(),
        prompt_id=prompt_id,
        request_text="Find failing workflow.",
        response_text="Workflow step xyz failed.",
    )


class _StubCollection:
    def __init__(self) -> None:
        self.deleted_where: dict[str, Any] | None = None
        self.upserts: list[dict[str, Any]] = []

    def delete(self, where: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self.deleted_where = where or kwargs or {}

    def upsert(self, **payload: Any) -> None:
        self.upserts.append(payload)


class _StubChromaClient:
    def __init__(self, collection: _StubCollection) -> None:
        self.collection = collection
        self.deleted = False
        self._create_calls = 0

    def get_or_create_collection(
        self,
        name: str,
        *,
        metadata: dict[str, Any] | None = None,
        embedding_function: Any | None = None,
    ) -> _StubCollection:
        self._create_calls += 1
        return self.collection

    def delete_collection(self, *, name: str) -> None:
        self.deleted = True


class _StubChromaClientNoDelete:
    def __init__(self, collection: _StubCollection) -> None:
        self.collection = collection
        self._create_calls = 0

    def get_or_create_collection(
        self,
        name: str,
        *,
        metadata: dict[str, Any] | None = None,
        embedding_function: Any | None = None,
    ) -> _StubCollection:
        self._create_calls += 1
        return self.collection


def test_repository_reset_all_data(tmp_path: Path) -> None:
    """Wipe repository data and ensure prompts plus history are removed."""
    db_path = tmp_path / "prompt_manager.db"
    repo = PromptRepository(str(db_path))

    prompt = _make_prompt()
    repo.add(prompt)

    execution = _make_execution(prompt.id)
    repo.add_execution(execution)

    profile = repo.get_user_profile()
    custom_profile = UserProfile(id=profile.id, username="custom")
    repo.save_user_profile(custom_profile)

    repo.reset_all_data()

    assert repo.list() == []
    assert repo.list_executions() == []

    refreshed_profile = repo.get_user_profile()
    assert refreshed_profile.username == "default"
    assert refreshed_profile.recent_prompts == []


def test_prompt_manager_reset_application_data(tmp_path: Path) -> None:
    """Bridge PromptManager reset helper to repository-level cleanup."""
    db_path = tmp_path / "prompt_manager.db"
    repo = PromptRepository(str(db_path))
    prompt = _make_prompt()
    repo.add(prompt)
    repo.add_execution(_make_execution(prompt.id))
    profile = repo.get_user_profile()
    repo.save_user_profile(UserProfile(id=profile.id, username="custom"))

    collection = _StubCollection()
    client = _StubChromaClient(collection)

    manager = PromptManager(
        chroma_path=str(tmp_path / "chroma"),
        db_path=str(db_path),
        chroma_client=client,
        repository=repo,
    )
    manager._logs_path = tmp_path / "logs"  # type: ignore[attr-defined]
    manager.logs_path.mkdir(parents=True, exist_ok=True)
    (manager.logs_path / "intent_usage.jsonl").write_text("{}", encoding="utf-8")

    manager.reset_application_data(clear_logs=True)

    assert repo.list() == []
    assert repo.list_executions() == []
    assert client.deleted is True
    assert client._create_calls >= 2  # initial + reset
    assert manager.user_profile is not None
    assert manager.user_profile.username == "default"
    assert not any(manager.logs_path.iterdir())


def test_reset_vector_store_without_client_delete(tmp_path: Path) -> None:
    """Avoid deleting external vector stores when no delete client is provided."""
    db_path = tmp_path / "prompt_manager.db"
    repo = PromptRepository(str(db_path))
    repo.add(_make_prompt())

    collection = _StubCollection()
    client = _StubChromaClientNoDelete(collection)

    manager = PromptManager(
        chroma_path=str(tmp_path / "chroma"),
        db_path=str(db_path),
        chroma_client=client,
        repository=repo,
    )

    manager.reset_vector_store()

    assert collection.deleted_where is not None
    assert client._create_calls >= 1


def test_clear_usage_logs_handles_missing_directory(tmp_path: Path) -> None:
    """Ignore missing usage-log directories when clearing state."""
    db_path = tmp_path / "prompt_manager.db"
    repo = PromptRepository(str(db_path))
    manager = PromptManager(
        chroma_path=str(tmp_path / "chroma"),
        db_path=str(db_path),
        chroma_client=_StubChromaClient(_StubCollection()),
        repository=repo,
    )
    custom_logs = tmp_path / "custom_logs"
    manager.clear_usage_logs(custom_logs)
    assert custom_logs.exists()
    assert list(custom_logs.iterdir()) == []


def test_rebuild_embeddings_resets_store_and_regenerates_vectors(tmp_path: Path) -> None:
    """Recreate embedding vectors after dropping the backing store."""
    db_path = tmp_path / "prompt_manager.db"
    repo = PromptRepository(str(db_path))
    prompt_one = _make_prompt("Alpha")
    prompt_two = _make_prompt("Beta")
    repo.add(prompt_one)
    repo.add(prompt_two)

    collection = _StubCollection()
    client = _StubChromaClient(collection)

    manager = PromptManager(
        chroma_path=str(tmp_path / "chroma"),
        db_path=str(db_path),
        chroma_client=client,
        repository=repo,
        enable_background_sync=False,
    )

    successes, failures = manager.rebuild_embeddings(reset_store=True)

    assert successes == 2
    assert failures == 0
    assert client.deleted is True
    assert len(collection.upserts) == 2
    stored_prompt = repo.get(prompt_one.id)
    assert stored_prompt.ext4 is not None


def test_rebuild_embeddings_counts_generation_failures(tmp_path: Path) -> None:
    """Surface failures encountered while regenerating embeddings."""
    db_path = tmp_path / "prompt_manager.db"
    repo = PromptRepository(str(db_path))
    repo.add(_make_prompt())

    class _FailingProvider:
        def embed(self, _: str) -> list[float]:
            raise EmbeddingGenerationError("boom")

    collection = _StubCollection()
    client = _StubChromaClient(collection)

    manager = PromptManager(
        chroma_path=str(tmp_path / "chroma"),
        db_path=str(db_path),
        chroma_client=client,
        repository=repo,
        embedding_provider=_FailingProvider(),  # type: ignore[arg-type]
        enable_background_sync=False,
    )

    successes, failures = manager.rebuild_embeddings()

    assert successes == 0
    assert failures == 1
    assert collection.upserts == []
