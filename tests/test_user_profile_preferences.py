"""Tests covering single-user profile persistence and personalisation.

Updates: v0.1.0 - 2025-11-11 - Introduce coverage for profile tracking and suggestion biasing.
"""

from __future__ import annotations

import uuid
from typing import Any

from core.prompt_manager import PromptManager
from core.repository import PromptRepository
from models.prompt_model import Prompt, UserProfile


class _NoopCollection:
    """Minimal Chroma collection stub used for PromptManager wiring."""

    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]] | None = None,  # noqa: ARG002
    ) -> None:
        self._records = dict(zip(ids, metadatas))

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]] | None = None,  # noqa: ARG002
    ) -> None:
        self.add(ids, documents, metadatas, embeddings)

    def delete(self, ids: list[str]) -> None:  # noqa: ARG002
        return

    def query(
        self,
        query_texts: list[str] | None,  # noqa: ARG002
        query_embeddings: list[list[float]] | None,  # noqa: ARG002
        n_results: int,  # noqa: ARG002
        where: dict[str, Any] | None,  # noqa: ARG002
    ) -> dict[str, list[list[Any]]]:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]]}


class _StubChromaClient:
    def __init__(self, collection: _NoopCollection) -> None:
        self._collection = collection

    def get_or_create_collection(self, **_: Any) -> _NoopCollection:
        return self._collection


def _make_prompt(name: str, category: str, tags: list[str]) -> Prompt:
    return Prompt(
        id=uuid.uuid4(),
        name=name,
        description=f"Prompt {name}",
        category=category,
        tags=tags,
        source="tests",
    )


def test_user_profile_records_usage() -> None:
    profile = UserProfile.create_default()
    prompt = _make_prompt("Debugger", "Debug", ["debugging", "ci"])

    profile.record_prompt_usage(prompt)

    assert profile.category_weights["Debug"] == 1
    assert profile.favorite_categories() == ["Debug"]
    assert "debugging" in profile.favorite_tags()


def test_repository_persists_profile(tmp_path) -> None:
    repo = PromptRepository(str(tmp_path / "profile.db"))
    prompt = _make_prompt("Doc Helper", "Documentation", ["docs"])

    repo.record_user_prompt_usage(prompt)
    stored = repo.get_user_profile()

    assert stored.category_weights["Documentation"] == 1
    assert stored.favorite_categories() == ["Documentation"]
    assert stored.recent_prompts[0] == str(prompt.id)


def test_personalisation_biases_prompt_order(tmp_path) -> None:
    repo = PromptRepository(str(tmp_path / "prefs.db"))
    manager = PromptManager(
        chroma_path=str(tmp_path / "chroma"),
        chroma_client=_StubChromaClient(_NoopCollection()),
        repository=repo,
        enable_background_sync=False,
    )

    favourite_prompt = _make_prompt("Fixer", "Debug", ["debugging"])
    secondary_prompt = _make_prompt("Writer", "Documentation", ["docs"])
    repo.add(favourite_prompt)
    repo.add(secondary_prompt)

    manager.increment_usage(favourite_prompt.id)
    manager.increment_usage(favourite_prompt.id)

    reordered = manager._personalize_ranked_prompts([secondary_prompt, favourite_prompt])
    assert reordered[0].id == favourite_prompt.id
    assert "Debug" in manager.user_profile.favorite_categories()

    manager.close()
