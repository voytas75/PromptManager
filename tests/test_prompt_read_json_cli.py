"""Real-process JSON outcome checks for selected asset-read commands."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from core.repository import PromptRepository
from models.prompt_model import Prompt

ROOT = Path(__file__).resolve().parents[1]


def _run(
    tmp_path: Path, config: Path, installed: bool, *args: str
) -> subprocess.CompletedProcess[str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PROMPT_MANAGER_", "AZURE_", "OPENAI_", "LITELLM_"))
        and key not in {"OPENAI_API_KEY", "LITELLM_API_KEY"}
    }
    env.update(
        HOME=str(tmp_path),
        PYTHONPATH=str(ROOT),
        PROMPT_MANAGER_CONFIG_JSON=str(config),
        PROMPT_MANAGER_ENV_FILE="",
        PYTHONDONTWRITEBYTECODE="1",
        CHROMA_ANONYMIZED_TELEMETRY="0",
    )
    front = (
        [str(ROOT / ".venv/bin/prompt-manager")] if installed else [sys.executable, "-m", "main"]
    )
    return subprocess.run(
        [*front, *args],
        cwd=tmp_path,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=35,
        check=False,
    )


@pytest.mark.parametrize("installed", [False, True])
def test_prompt_read_json_process_outcomes(tmp_path: Path, installed: bool) -> None:
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
    missing = str(uuid.uuid4())
    show_missing = _run(tmp_path, config, installed, "prompt-show", missing, "--json")
    assert show_missing.returncode == 4 and show_missing.stdout == "", show_missing
    assert json.loads(show_missing.stderr)["error"]["code"] == "PROMPT_NOT_FOUND"

    empty = _run(tmp_path, config, installed, "prompt-find", "unmatched", "--json")
    assert empty.returncode == 0 and json.loads(empty.stdout) == [], empty
    assert empty.stderr == ""

    invalid = _run(
        tmp_path, config, installed, "prompt-find", "query", "--active", "invalid", "--json"
    )
    assert invalid.returncode == 5 and invalid.stdout == "", invalid
    assert json.loads(invalid.stderr)["error"]["code"] == "INVALID_ACTIVE"

    text = _run(tmp_path, config, installed, "prompt-show", missing)
    assert text.returncode == 4 and "Prompt not found" in text.stdout

    prompt = Prompt(
        id=uuid.uuid4(),
        name="Readback Prompt",
        description="Local readback.",
        category="Testing",
        context="Prompt body",
        source="local",
    )
    repository.add(prompt)
    shown = _run(tmp_path, config, installed, "prompt-show", str(prompt.id), "--json")
    assert shown.returncode == 0 and json.loads(shown.stdout)["id"] == str(prompt.id), shown
    assert shown.stderr == ""
    assert repository.get(prompt.id).context == "Prompt body"
