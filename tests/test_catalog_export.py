"""Tests for catalogue export utilities.

Updates: v0.1.0 - 2025-11-30 - Cover export-only workflow after removing imports.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Dict, List

import pytest

from core.catalog_importer import export_prompt_catalog
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


class _StubManager:
    def __init__(self) -> None:
        self.repository = _StubRepository()

    def create_prompt(self, prompt: Prompt, embedding=None) -> Prompt:  # noqa: D401
        return self.repository.add(prompt)


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
