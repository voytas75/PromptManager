"""Tests for catalogue import, diff, and export utilities.

Updates: v0.2.0 - 2025-11-30 - Ensure GUI import helpers remain functional post-CLI removal.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Dict, List

import pytest

from core.catalog_importer import (
    CatalogChangeType,
    CatalogImportResult,
    diff_prompt_catalog,
    export_prompt_catalog,
    import_prompt_catalog,
    load_prompt_catalog,
)
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


def test_load_prompt_catalog_without_path_returns_empty() -> None:
    prompts = load_prompt_catalog(None)
    assert prompts == []


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
    assert result.preview is not None
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
    assert second_result.preview is not None
    assert manager.updated
    refreshed_prompt = manager.repository.list()[0]
    assert refreshed_prompt.quality_score == pytest.approx(9.1)
    assert "logs" in refreshed_prompt.tags


def test_diff_prompt_catalog_reports_expected_changes(tmp_path: Path) -> None:
    manager = _StubManager()

    catalog_path = tmp_path / "catalog.json"
    catalog_payload = [
        {
            "name": "Refactor Coach",
            "description": "Guide towards modular structure.",
            "category": "Refactoring",
            "tags": ["refactor"],
        }
    ]
    catalog_path.write_text(json.dumps(catalog_payload), encoding="utf-8")

    diff = diff_prompt_catalog(manager, catalog_path)
    assert diff.added == 1
    assert diff.updated == 0
    assert diff.entries[0].change_type is CatalogChangeType.ADD

    import_prompt_catalog(manager, catalog_path)

    updated_payload = dict(catalog_payload[0])
    updated_payload["description"] = "Emphasise separation of concerns."
    catalog_path.write_text(json.dumps([updated_payload]), encoding="utf-8")

    updated_diff = diff_prompt_catalog(manager, catalog_path)
    assert updated_diff.updated == 1
    assert updated_diff.entries[0].change_type is CatalogChangeType.UPDATE
    assert "separation of concerns" in updated_diff.entries[0].diff.lower()


def test_export_prompt_catalog_json(tmp_path: Path) -> None:
    manager = _StubManager()
    prompt = Prompt(
        id=uuid.uuid4(),
        name="Diagnostics",
        description="Investigate failures",
        category="Reasoning / Debugging",
        tags=["ci"],
    )
    manager.create_prompt(prompt)

    export_path = tmp_path / "catalog.json"
    resolved = export_prompt_catalog(manager, export_path)
    assert resolved.exists()

    data = json.loads(resolved.read_text(encoding="utf-8"))
    assert data["count"] == 1
    assert data["prompts"][0]["name"] == "Diagnostics"


def test_export_prompt_catalog_yaml(tmp_path: Path) -> None:
    yaml = pytest.importorskip("yaml")
    manager = _StubManager()
    prompt = Prompt(
        id=uuid.uuid4(),
        name="Reporter",
        description="Summarise changes",
        category="Reporting",
        tags=["summary"],
    )
    manager.create_prompt(prompt)

    export_path = tmp_path / "catalog.yaml"
    resolved = export_prompt_catalog(manager, export_path, fmt="yaml")
    assert resolved.exists()

    with resolved.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    assert payload["count"] == 1
    assert payload["prompts"][0]["category"] == "Reporting"


def test_export_prompt_catalog_respects_inactive_flag(tmp_path: Path) -> None:
    manager = _StubManager()
    active = Prompt(
        id=uuid.uuid4(),
        name="Active Prompt",
        description="Active entry",
        category="General",
        tags=[],
    )
    inactive = Prompt(
        id=uuid.uuid4(),
        name="Inactive Prompt",
        description="Inactive entry",
        category="General",
        tags=[],
        is_active=False,
    )
    manager.create_prompt(active)
    manager.create_prompt(inactive)

    default_export = export_prompt_catalog(manager, tmp_path / "default.json")
    data = json.loads(default_export.read_text(encoding="utf-8"))
    names = [entry["name"] for entry in data["prompts"]]
    assert names == ["Active Prompt"]

    full_export = export_prompt_catalog(
        manager,
        tmp_path / "full.json",
        include_inactive=True,
    )
    full_data = json.loads(full_export.read_text(encoding="utf-8"))
    full_names = sorted(entry["name"] for entry in full_data["prompts"])
    assert full_names == ["Active Prompt", "Inactive Prompt"]
