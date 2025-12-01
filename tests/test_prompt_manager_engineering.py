"""Tests for PromptManager prompt refinement workflow.

Updates: v0.1.1 - 2025-11-22 - Cover structure-only refinement calls.
Updates: v0.1.0 - 2025-11-18 - Initial tests for prompt refinement workflow.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from core import PromptEngineeringUnavailable, PromptManager
from core.prompt_engineering import PromptRefinement
from core.repository import PromptRepository

if TYPE_CHECKING:
    from pathlib import Path


class _StubEmbeddingProvider:
    def embed(self, _: str) -> list[float]:
        return [0.0, 0.1, 0.2]


class _StubCollection:
    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        return None

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        return None


class _StubChromaClient:
    def __init__(self) -> None:
        self.collection = _StubCollection()

    def get_or_create_collection(
        self,
        name: str,
        metadata: dict[str, Any],
        embedding_function: Any | None = None,
    ) -> _StubCollection:
        return self.collection

    def close(self) -> None:
        return None


class _StubPromptEngineer:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.structure_calls: list[dict[str, Any]] = []

    def refine(
        self,
        prompt_text: str,
        *,
        name: str | None = None,
        description: str | None = None,
        category: str | None = None,
        tags: list[str] | None = None,
        negative_constraints: list[str] | None = None,
        structure_only: bool = False,
    ) -> PromptRefinement:
        self.calls.append(
            {
                "prompt_text": prompt_text,
                "name": name,
                "description": description,
                "category": category,
                "tags": tags,
                "negative_constraints": negative_constraints,
                "structure_only": structure_only,
            }
        )
        return PromptRefinement(
            improved_prompt=f"Improved: {prompt_text}",
            analysis="Prompt updated",
            checklist=["clarity"],
            warnings=[],
            confidence=0.9,
        )

    def refine_structure(
        self,
        prompt_text: str,
        *,
        name: str | None = None,
        description: str | None = None,
        category: str | None = None,
        tags: list[str] | None = None,
        negative_constraints: list[str] | None = None,
    ) -> PromptRefinement:
        payload = {
            "prompt_text": prompt_text,
            "name": name,
            "description": description,
            "category": category,
            "tags": tags,
            "negative_constraints": negative_constraints,
        }
        self.structure_calls.append(payload)
        return self.refine(
            prompt_text,
            name=name,
            description=description,
            category=category,
            tags=tags,
            negative_constraints=negative_constraints,
            structure_only=True,
        )


def _build_manager(
    tmp_path: Path,
    engineer: _StubPromptEngineer | None,
    *,
    structure_engineer: _StubPromptEngineer | None = None,
) -> PromptManager:
    db_path = tmp_path / "prompt_manager.db"
    chroma_path = tmp_path / "chroma"
    repository = PromptRepository(str(db_path))
    return PromptManager(
        chroma_path=str(chroma_path),
        db_path=str(db_path),
        cache_ttl_seconds=60,
        chroma_client=_StubChromaClient(),
        embedding_function=None,
        repository=repository,
        embedding_provider=_StubEmbeddingProvider(),
        enable_background_sync=False,
        prompt_engineer=engineer,
        structure_prompt_engineer=structure_engineer,
    )


def test_refine_prompt_text_returns_refinement(tmp_path: Path) -> None:
    engineer = _StubPromptEngineer()
    manager = _build_manager(tmp_path, engineer)

    result = manager.refine_prompt_text(
        "Review this code",
        name="Code Review",
        category="Analysis",
        tags=["analysis"],
    )

    assert result.improved_prompt.startswith("Improved")
    assert engineer.calls[0]["name"] == "Code Review"
    manager.close()


def test_refine_prompt_text_requires_engineer(tmp_path: Path) -> None:
    manager = _build_manager(tmp_path, engineer=None)

    with pytest.raises(PromptEngineeringUnavailable):
        manager.refine_prompt_text("Missing engineer")
    manager.close()


def test_refine_prompt_structure_uses_structure_mode(tmp_path: Path) -> None:
    engineer = _StubPromptEngineer()
    structure_engineer = _StubPromptEngineer()
    manager = _build_manager(tmp_path, engineer, structure_engineer=structure_engineer)

    result = manager.refine_prompt_structure("Streamline this prompt", tags=["structure"])

    assert result.improved_prompt.startswith("Improved")
    assert structure_engineer.structure_calls[0]["tags"] == ["structure"]
    assert structure_engineer.calls[-1]["structure_only"] is True
    assert not engineer.structure_calls
    manager.close()


def test_refine_prompt_structure_falls_back_to_general_engineer(tmp_path: Path) -> None:
    engineer = _StubPromptEngineer()
    manager = _build_manager(tmp_path, engineer)

    manager.refine_prompt_structure("Reformat only")

    assert engineer.structure_calls
    assert engineer.calls[-1]["structure_only"] is True
    manager.close()
