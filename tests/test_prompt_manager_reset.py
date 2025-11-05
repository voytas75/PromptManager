"""Tests covering PromptManager datastore reset workflows."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from core.prompt_manager import PromptManager
from core.repository import PromptRepository
from models.prompt_model import Prompt, PromptExecution, TaskTemplate, UserProfile


def _make_prompt(name: str = "Diagnostics") -> Prompt:
    return Prompt(
        id=uuid.uuid4(),
        name=name,
        description="Investigate pipeline failures.",
        category="Debugging",
        tags=["ci"],
    )


def _make_template(prompt_id: uuid.UUID) -> TaskTemplate:
    return TaskTemplate(
        id=uuid.uuid4(),
        name="Daily Check",
        description="Run diagnostics for daily builds.",
        prompt_ids=[prompt_id],
        default_input="Investigate latest failure logs.",
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
        self.deleted_where: Optional[Dict[str, Any]] = None

    def delete(self, where: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        self.deleted_where = where or kwargs or {}


class _StubChromaClient:
    def __init__(self, collection: _StubCollection) -> None:
        self.collection = collection
        self.deleted = False
        self._create_calls = 0

    def get_or_create_collection(
        self,
        name: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        embedding_function: Optional[Any] = None,
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
        metadata: Optional[Dict[str, Any]] = None,
        embedding_function: Optional[Any] = None,
    ) -> _StubCollection:
        self._create_calls += 1
        return self.collection


def test_repository_reset_all_data(tmp_path: Path) -> None:
    db_path = tmp_path / "prompt_manager.db"
    repo = PromptRepository(str(db_path))

    prompt = _make_prompt()
    repo.add(prompt)

    template = _make_template(prompt.id)
    repo.add_template(template)

    execution = _make_execution(prompt.id)
    repo.add_execution(execution)

    profile = repo.get_user_profile()
    custom_profile = UserProfile(id=profile.id, username="custom")
    repo.save_user_profile(custom_profile)

    repo.reset_all_data()

    assert repo.list() == []
    assert repo.list_templates() == []
    assert repo.list_executions() == []

    refreshed_profile = repo.get_user_profile()
    assert refreshed_profile.username == "default"
    assert refreshed_profile.recent_prompts == []


def test_prompt_manager_reset_application_data(tmp_path: Path) -> None:
    db_path = tmp_path / "prompt_manager.db"
    repo = PromptRepository(str(db_path))
    prompt = _make_prompt()
    repo.add(prompt)
    repo.add_template(_make_template(prompt.id))
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
    assert repo.list_templates() == []
    assert repo.list_executions() == []
    assert client.deleted is True
    assert client._create_calls >= 2  # initial + reset
    assert manager.user_profile is not None
    assert manager.user_profile.username == "default"
    assert not any(manager.logs_path.iterdir())


def test_reset_vector_store_without_client_delete(tmp_path: Path) -> None:
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
