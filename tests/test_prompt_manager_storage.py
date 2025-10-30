"""Tests covering storage coordination across SQLite, ChromaDB, and Redis facades.

Updates: v0.1.0 - 2025-10-31 - Introduce repository and manager storage integration tests.
"""

from __future__ import annotations

import uuid

import pytest

from core import (
    PromptManager,
    PromptRepository,
    RepositoryNotFoundError,
)
from models.prompt_model import Prompt


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
        version="1.0",
        is_active=True,
        source="tests",
    )


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
