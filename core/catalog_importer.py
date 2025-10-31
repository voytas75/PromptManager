"""Utilities for importing prompt catalogues into Prompt Manager.

Updates: v0.5.0 - 2025-11-05 - Seed SQLite/Chroma from packaged or user-provided catalogues.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from importlib.resources import as_file
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from catalog import builtin_catalog_resource
from models.prompt_model import Prompt

from .prompt_manager import PromptManager, PromptStorageError

logger = logging.getLogger("prompt_manager.catalog")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return [str(value)]


def _read_json(path: Path) -> List[Dict[str, Any]]:
    try:
        contents = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FileNotFoundError(f"Cannot read prompt catalogue: {path}") from exc
    try:
        payload = json.loads(contents)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}") from exc
    if isinstance(payload, dict):
        if "prompts" in payload and isinstance(payload["prompts"], list):
            return payload["prompts"]
        return [payload]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Prompt catalogue {path} must contain a JSON object or list")


def _load_entries_from_path(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.is_dir():
        entries: List[Dict[str, Any]] = []
        for candidate in sorted(path.glob("*.json")):
            entries.extend(_read_json(candidate))
        return entries
    return _read_json(path)


def _load_builtin_entries() -> List[Dict[str, Any]]:
    resource = builtin_catalog_resource()
    with as_file(resource) as resolved_path:
        return _load_entries_from_path(Path(resolved_path))


def _entry_to_prompt(entry: Dict[str, Any]) -> Prompt:
    if "name" not in entry or "description" not in entry:
        raise ValueError("Prompt catalogue entries must include 'name' and 'description'")

    record: Dict[str, Any] = {
        "id": entry.get("id")
        or str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"prompt-manager:{entry['name'].strip().lower()}",
            )
        ),
        "name": entry["name"],
        "description": entry["description"],
        "category": entry.get("category") or "General",
        "tags": _ensure_list(entry.get("tags")),
        "language": entry.get("language") or "en",
        "context": entry.get("context"),
        "example_input": entry.get("example_input"),
        "example_output": entry.get("example_output"),
        "version": entry.get("version") or "1.0",
        "author": entry.get("author"),
        "quality_score": entry.get("quality_score"),
        "usage_count": int(entry.get("usage_count") or 0),
        "related_prompts": _ensure_list(entry.get("related_prompts")),
        "created_at": entry.get("created_at") or _now_iso(),
        "last_modified": entry.get("last_modified") or _now_iso(),
        "modified_by": entry.get("modified_by"),
        "is_active": entry.get("is_active", True),
        "source": entry.get("source") or "catalog",
        "checksum": entry.get("checksum"),
        "ext1": entry.get("ext1"),
        "ext2": entry.get("ext2"),
        "ext3": entry.get("ext3"),
        "ext4": entry.get("ext4"),
        "ext5": entry.get("ext5"),
    }
    return Prompt.from_record(record)


def load_prompt_catalog(catalog_path: Optional[Path]) -> List[Prompt]:
    """Load prompts from a user-provided path or packaged defaults."""
    entries: List[Dict[str, Any]] = []
    if catalog_path is not None:
        try:
            entries = _load_entries_from_path(catalog_path)
        except FileNotFoundError:
            logger.warning(
                "Prompt catalogue not found at %s; falling back to built-in entries.",
                catalog_path,
            )
        except ValueError as exc:
            logger.error("Skipping catalogue %s: %s", catalog_path, exc)

    if not entries:
        try:
            entries = _load_builtin_entries()
        except Exception as exc:  # pragma: no cover - IO failures
            logger.error("Unable to load built-in prompt catalogue: %s", exc)
            return []

    prompts: List[Prompt] = []
    for entry in entries:
        try:
            prompt = _entry_to_prompt(entry)
        except Exception as exc:
            logger.error("Skipping invalid prompt entry: %s", exc)
            continue
        prompts.append(prompt)
    return prompts


def _merge_prompt(existing: Prompt, incoming: Prompt) -> Prompt:
    return replace(
        incoming,
        id=existing.id,
        usage_count=existing.usage_count,
        created_at=existing.created_at,
        last_modified=datetime.now(timezone.utc),
    )


class CatalogImportResult:
    """Aggregate statistics from a catalogue import operation."""

    def __init__(self) -> None:
        self.added = 0
        self.updated = 0
        self.skipped = 0
        self.errors = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "added": self.added,
            "updated": self.updated,
            "skipped": self.skipped,
            "errors": self.errors,
        }


def import_prompt_catalog(
    manager: PromptManager,
    catalog_path: Optional[Path],
    *,
    overwrite: bool = True,
) -> CatalogImportResult:
    """Import prompts into the manager, updating existing entries when appropriate."""
    prompts = load_prompt_catalog(catalog_path)
    result = CatalogImportResult()
    if not prompts:
        return result

    try:
        existing_prompts = manager.repository.list()
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Unable to inspect existing prompts: %s", exc)
        result.errors = len(prompts)
        return result

    existing_by_name = {prompt.name.strip().lower(): prompt for prompt in existing_prompts}

    for prompt in prompts:
        key = prompt.name.strip().lower()
        if key in existing_by_name:
            if not overwrite:
                result.skipped += 1
                continue
            merged = _merge_prompt(existing_by_name[key], prompt)
            try:
                manager.update_prompt(merged)
            except PromptStorageError as exc:
                logger.error("Failed to update prompt '%s': %s", prompt.name, exc)
                result.errors += 1
                continue
            existing_by_name[key] = merged
            result.updated += 1
            continue

        try:
            manager.create_prompt(prompt)
        except PromptStorageError as exc:
            logger.error("Failed to import prompt '%s': %s", prompt.name, exc)
            result.errors += 1
            continue
        existing_by_name[key] = prompt
        result.added += 1

    return result


__all__ = ["load_prompt_catalog", "import_prompt_catalog", "CatalogImportResult"]
