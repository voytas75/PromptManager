"""Additional PromptRepository branch coverage tests.

Updates: v0.1.0 - 2025-10-30 - Add error-path coverage for repository helpers.
"""

from __future__ import annotations

import uuid
from pathlib import Path
import sqlite3

import pytest

from core.repository import (
    PromptRepository,
    RepositoryError,
    RepositoryNotFoundError,
    _json_dumps,
    _json_loads_list,
    _json_loads_optional,
)
from models.prompt_model import Prompt


def _make_prompt(name: str = "Repo Branch Test") -> Prompt:
    return Prompt(
        id=uuid.uuid4(),
        name=name,
        description="demo",
        category="tests",
    )


def test_json_helpers_cover_edge_cases() -> None:
    assert _json_dumps(None) is None
    assert _json_dumps(["a"]).startswith("[")

    assert _json_loads_list(None) == []
    assert _json_loads_list("null") == []
    assert _json_loads_list('["x",1]') == ["x", "1"]
    assert _json_loads_list("not-json") == ["not-json"]

    assert _json_loads_optional(None) is None
    assert _json_loads_optional("null") is None
    assert _json_loads_optional('{"a": 1}') == {"a": 1}
    assert _json_loads_optional("{invalid") == "{invalid"


def test_repository_add_duplicate_raises_error(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    prompt = _make_prompt()
    repo.add(prompt)
    with pytest.raises(RepositoryError):
        repo.add(prompt)


def test_repository_update_missing_prompt(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    missing = _make_prompt()
    with pytest.raises(RepositoryNotFoundError):
        repo.update(missing)


def test_repository_delete_missing_prompt(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    with pytest.raises(RepositoryNotFoundError):
        repo.delete(uuid.uuid4())


def test_repository_list_with_limit(tmp_path: Path) -> None:
    repo = PromptRepository(str(tmp_path / "repo.db"))
    prompts = [_make_prompt(f"p-{idx}") for idx in range(3)]
    for prompt in prompts:
        repo.add(prompt)

    limited = repo.list(limit=1)
    assert len(limited) == 1
    assert limited[0].id == prompts[0].id
