"""Catalogue export utilities for Prompt Manager.

Updates: v0.7.0 - 2025-11-30 - Remove catalogue import helpers; retain export support only.
Updates: v0.6.1 - 2025-11-17 - Require explicit catalogue paths; remove built-in fallback prompts.
Updates: v0.6.0 - 2025-11-06 - Add diff previews, export helpers, and bulk directory support.
Updates: v0.5.0 - 2025-11-05 - Seed SQLite/Chroma from packaged or user-provided catalogues.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from models.prompt_model import Prompt

from .prompt_manager import PromptManager, PromptStorageError

try:  # pragma: no cover - optional dependency for YAML export
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime when exporting YAML
    yaml = None  # type: ignore


def _now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat()


def _prompt_to_catalog_dict(prompt: Prompt) -> Dict[str, Any]:
    """Convert a prompt record into a serialisable dictionary."""

    record = {
        "id": str(prompt.id),
        "name": prompt.name,
        "description": prompt.description,
        "category": prompt.category,
        "tags": list(prompt.tags),
        "language": prompt.language,
        "context": prompt.context,
        "example_input": prompt.example_input,
        "example_output": prompt.example_output,
        "version": prompt.version,
        "author": prompt.author,
        "quality_score": prompt.quality_score,
        "usage_count": prompt.usage_count,
        "related_prompts": list(prompt.related_prompts),
        "created_at": prompt.created_at.isoformat(),
        "last_modified": prompt.last_modified.isoformat(),
        "modified_by": prompt.modified_by,
        "is_active": prompt.is_active,
        "source": prompt.source,
        "checksum": prompt.checksum,
        "ext1": prompt.ext1,
        "ext2": prompt.ext2,
        "ext3": prompt.ext3,
        "ext4": list(prompt.ext4) if prompt.ext4 is not None else None,
        "ext5": prompt.ext5,
    }
    return record


def export_prompt_catalog(
    manager: PromptManager,
    output_path: Path,
    *,
    fmt: str = "json",
    include_inactive: bool = False,
) -> Path:
    """Export the current prompt repository to JSON or YAML."""

    fmt_lower = fmt.lower()
    if fmt_lower not in {"json", "yaml"}:
        raise ValueError("fmt must be 'json' or 'yaml'")

    try:
        prompts = manager.repository.list()
    except Exception as exc:  # pragma: no cover - defensive
        raise PromptStorageError("Unable to load prompts for export") from exc

    exportable = [
        _prompt_to_catalog_dict(prompt)
        for prompt in prompts
        if include_inactive or prompt.is_active
    ]

    payload = {
        "generated_at": _now_iso(),
        "count": len(exportable),
        "prompts": exportable,
    }

    resolved_path = output_path.expanduser()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt_lower == "json":
        resolved_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    else:
        if yaml is None:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "PyYAML is required for YAML export. Install it with `pip install pyyaml`."
            )
        with resolved_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)

    return resolved_path


__all__ = ["export_prompt_catalog"]
