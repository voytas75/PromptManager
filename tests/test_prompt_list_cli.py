"""Isolated process checks for local prompt-list catalog browsing."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from core.repository import PromptRepository
from models.prompt_model import Prompt

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("installed", [False, True])
def test_prompt_list_reads_existing_catalog_without_query_or_provider(
    tmp_path: Path, installed: bool
) -> None:
    db = tmp_path / "catalog.db"
    config = tmp_path / "settings.json"
    config.write_text(
        json.dumps(
            {
                "database_path": str(db),
                "chroma_path": str(tmp_path / "chroma"),
                "embedding_backend": "deterministic",
                "redis_dsn": None,
                "litellm_model": None,
                "litellm_inference_model": None,
            }
        ),
        encoding="utf-8",
    )
    # SQLite datetime(last_modified) truncates subsecond precision in repository.list().
    now = datetime(2026, 9, 26, 12, 0, 0, 900000, tzinfo=UTC)
    repository = PromptRepository(str(db))
    older = Prompt(
        id=uuid4(),
        name="Older",
        description="old",
        category="Operations",
        tags=["triage"],
        last_modified=now - timedelta(microseconds=800000),
    )
    newer = Prompt(
        id=uuid4(),
        name="Newer",
        description="new",
        category="Operations",
        tags=["Triage"],
        last_modified=now,
    )
    repository.add(older)
    repository.add(newer)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PROMPT_MANAGER_", "AZURE_OPENAI_"))
        and key not in {"LITELLM_API_KEY", "OPENAI_API_KEY"}
    }
    env.update(
        HOME=str(tmp_path),
        PYTHONPATH=str(ROOT),
        PROMPT_MANAGER_CONFIG_JSON=str(config),
        PROMPT_MANAGER_ENV_FILE="",
        PYTHONDONTWRITEBYTECODE="1",
    )
    command = (
        [str(ROOT / ".venv/bin/prompt-manager")] if installed else [sys.executable, "-m", "main"]
    )
    result = subprocess.run(
        [*command, "prompt-list", "--tag", "triage", "--limit", "1", "--json"],
        cwd=tmp_path,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert json.loads(result.stdout) == [
        {
            "id": str(newer.id),
            "name": "Newer",
            "category": "Operations",
            "tags": ["Triage"],
            "source": "local",
            "active": True,
            "last_modified": now.isoformat(),
        },
    ]
    assert repository.get(older.id).name == "Older"
    assert repository.get(newer.id).name == "Newer"


@pytest.mark.parametrize("installed", [False, True])
def test_prompt_list_defaults_to_active_and_explicitly_shows_inactive(
    tmp_path: Path, installed: bool
) -> None:
    db = tmp_path / "catalog.db"
    config = tmp_path / "settings.json"
    config.write_text(
        json.dumps(
            {
                "database_path": str(db),
                "chroma_path": str(tmp_path / "chroma"),
                "embedding_backend": "deterministic",
                "redis_dsn": None,
                "litellm_model": None,
                "litellm_inference_model": None,
            }
        ),
        encoding="utf-8",
    )
    repository = PromptRepository(str(db))
    active = Prompt(id=uuid4(), name="Visible", description="test", category="Operations")
    inactive = Prompt(id=uuid4(), name="Hidden", description="test", category="Operations")
    repository.add(active)
    repository.add(inactive)
    repository.set_prompt_active(inactive.id, active=False, expect_active=True)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PROMPT_MANAGER_", "AZURE_OPENAI_"))
        and key not in {"LITELLM_API_KEY", "OPENAI_API_KEY"}
    }
    env.update(
        HOME=str(tmp_path),
        PYTHONPATH=str(ROOT),
        PROMPT_MANAGER_CONFIG_JSON=str(config),
        PROMPT_MANAGER_ENV_FILE="",
        PYTHONDONTWRITEBYTECODE="1",
    )
    command = (
        [str(ROOT / ".venv/bin/prompt-manager")] if installed else [sys.executable, "-m", "main"]
    )
    for filters, expected in [
        ([], active.id),
        (["--active", "true"], active.id),
        (["--active", "false"], inactive.id),
    ]:
        result = subprocess.run(
            [*command, "prompt-list", "--limit", "1", "--json", *filters],
            cwd=tmp_path,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stderr == ""
        assert [row["id"] for row in json.loads(result.stdout)] == [str(expected)]
