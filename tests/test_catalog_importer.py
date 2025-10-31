"""Tests for catalogue import and loading utilities."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Dict, List

import pytest

from core.catalog_importer import CatalogImportResult, import_prompt_catalog, load_prompt_catalog
from models.prompt_model import Prompt


class _StubRepository:
    def __init__(self) -> None:
        self._store: Dict[uuid.UUID, Prompt] = {}

    def list(self, limit: int | None = None) -> List[Prompt]:
        values = list(self._store.values())
        return values if limit is None else values[:limit]

    def add(self, prompt: Prompt) -> Prompt:
        self._store[prompt.id] = prompt
        return prompt

    def update(self, prompt: Prompt) -> Prompt:
        if prompt.id not in self._store:
            raise KeyError(prompt.id)
        self._store[prompt.id] = prompt
        return prompt


class _StubManager:
    def __init__(self) -> None:
        self.repository = _StubRepository()
        self.created: List[Prompt] = []
        self.updated: List[Prompt] = []

    def create_prompt(self, prompt: Prompt, embedding=None) -> Prompt:  # noqa: D401
        self.created.append(prompt)
        return self.repository.add(prompt)

    def update_prompt(self, prompt: Prompt, embedding=None) -> Prompt:  # noqa: D401
        self.updated.append(prompt)
        return self.repository.update(prompt)


def test_load_prompt_catalog_falls_back_to_builtin() -> None:
    prompts = load_prompt_catalog(None)
    assert prompts, "Built-in catalogue should yield prompts"
    assert all(isinstance(prompt, Prompt) for prompt in prompts)


def test_import_prompt_catalog_adds_and_updates(tmp_path: Path) -> None:
    manager = _StubManager()

    catalog_path = tmp_path / "catalog.json"
    catalog_payload = [
        {
            "name": "Diagnostics Helper",
            "description": "Assist with debugging failing CI jobs.",
            "category": "Reasoning / Debugging",
            "tags": ["ci", "debugging"],
            "language": "en",
            "quality_score": 8.1,
            "created_at": "2025-10-01T00:00:00+00:00",
            "last_modified": "2025-10-02T00:00:00+00:00",
        }
    ]
    catalog_path.write_text(json.dumps(catalog_payload), encoding="utf-8")

    result = import_prompt_catalog(manager, catalog_path)
    assert isinstance(result, CatalogImportResult)
    assert result.added == 1
    assert result.updated == 0
    assert manager.created and not manager.updated

    stored_prompt = manager.repository.list()[0]
    assert stored_prompt.name == "Diagnostics Helper"
    assert stored_prompt.category == "Reasoning / Debugging"
    assert stored_prompt.tags == ["ci", "debugging"]

    # Update entry with new quality and ensure update path is taken.
    updated_payload = dict(catalog_payload[0])
    updated_payload["quality_score"] = 9.1
    updated_payload["tags"] = ["ci", "debugging", "logs"]
    catalog_path.write_text(json.dumps([updated_payload]), encoding="utf-8")

    second_result = import_prompt_catalog(manager, catalog_path)
    assert second_result.added == 0
    assert second_result.updated == 1
    assert manager.updated
    refreshed_prompt = manager.repository.list()[0]
    assert refreshed_prompt.quality_score == pytest.approx(9.1)
    assert "logs" in refreshed_prompt.tags
