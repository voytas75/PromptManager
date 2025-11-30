"""Import, preview, and export prompt catalogues for Prompt Manager.

Updates:
  v0.7.2 - 2025-11-30 - Document diff helpers and fix docstring spacing for lint compliance.
  v0.7.1 - 2025-11-30 - Retained GUI helpers after removing the CLI command.
  v0.6.1 - 2025-11-17 - Required explicit catalogue paths; removed fallback prompts.
  v0.6.0 - 2025-11-06 - Added diff previews, export helpers, and bulk directory support.
  v0.5.0 - 2025-11-05 - Seeded SQLite/Chroma from packaged or user catalogues.
"""

from __future__ import annotations

import difflib
import json
import logging
import textwrap
import uuid
from collections.abc import Iterable as IterableABC, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path

from models.prompt_model import Prompt

from .prompt_manager import PromptManager, PromptStorageError

CatalogEntry = dict[str, Any]


def _entry_list_factory() -> list[CatalogDiffEntry]:
    return []


def _prompt_list_factory() -> list[Prompt]:
    return []


def _prompt_pair_list_factory() -> list[tuple[Prompt, Prompt]]:
    return []


logger = logging.getLogger("prompt_manager.catalog")

try:  # pragma: no cover - optional dependency for YAML export
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime when exporting YAML
    yaml = None  # type: ignore


def _now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).isoformat()


def _ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, IterableABC):
        iterable = cast("IterableABC[Any]", value)
        return [str(item) for item in iterable]
    return [str(value)]


def _read_json(path: Path) -> list[CatalogEntry]:
    try:
        contents = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FileNotFoundError(f"Cannot read prompt catalogue: {path}") from exc
    try:
        payload: object = json.loads(contents)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}") from exc
    if isinstance(payload, dict):
        if "prompts" in payload and isinstance(payload["prompts"], list):
            entries: list[CatalogEntry] = []
            raw_prompts = cast("list[object]", payload["prompts"])
            for raw_entry in raw_prompts:
                if not isinstance(raw_entry, dict):
                    raise ValueError("Catalogue prompts must be JSON objects")
                entries.append(cast("CatalogEntry", raw_entry))
            return entries
        return [cast("CatalogEntry", payload)]
    if isinstance(payload, list):
        entries = []
        raw_entries = cast("list[object]", payload)
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict):
                raise ValueError("Prompt entries must be JSON objects")
            entries.append(cast("CatalogEntry", raw_entry))
        return entries
    raise ValueError(f"Prompt catalogue {path} must contain a JSON object or list")


def _load_entries_from_path(path: Path) -> list[CatalogEntry]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.is_dir():
        entries: list[CatalogEntry] = []
        for candidate in sorted(path.glob("*.json")):
            entries.extend(_read_json(candidate))
        return entries
    return _read_json(path)


def _entry_to_prompt(entry: CatalogEntry) -> Prompt:
    if "name" not in entry or "description" not in entry:
        raise ValueError("Prompt catalogue entries must include 'name' and 'description'")

    record: dict[str, Any] = {
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


def load_prompt_catalog(catalog_path: Path | None) -> list[Prompt]:
    """Load prompts from a user-provided path."""
    if catalog_path is None:
        logger.debug("No prompt catalogue path provided; returning empty list.")
        return []

    try:
        entries = _load_entries_from_path(catalog_path)
    except FileNotFoundError:
        logger.error("Prompt catalogue not found at %s", catalog_path)
        return []
    except ValueError as exc:
        logger.error("Skipping catalogue %s: %s", catalog_path, exc)
        return []

    prompts: list[Prompt] = []
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
        last_modified=datetime.now(UTC),
    )


class CatalogChangeType(str, Enum):
    """Categorise catalogue changes detected during preview."""

    ADD = "add"
    UPDATE = "update"
    SKIP = "skip"
    UNCHANGED = "unchanged"


@dataclass(slots=True)
class CatalogDiffEntry:
    """Single change entry rendered in a preview or diff dialog."""

    prompt_id: uuid.UUID
    name: str
    change_type: CatalogChangeType
    diff: str


@dataclass(slots=True)
class CatalogDiff:
    """Aggregate diff describing catalogue changes."""

    entries: list[CatalogDiffEntry] = field(default_factory=_entry_list_factory)
    added: int = 0
    updated: int = 0
    skipped: int = 0
    unchanged: int = 0
    source: str | None = None

    def has_changes(self) -> bool:
        """Return True when at least one prompt will be created or updated."""
        return self.added > 0 or self.updated > 0

    def summary(self) -> dict[str, int]:
        """Return a dictionary of diff counters for downstream reporting."""
        return {
            "added": self.added,
            "updated": self.updated,
            "skipped": self.skipped,
            "unchanged": self.unchanged,
        }


@dataclass(slots=True)
class CatalogChangePlan:
    """Plan describing how a catalogue import should be applied."""

    create: list[Prompt] = field(default_factory=_prompt_list_factory)
    update: list[tuple[Prompt, Prompt]] = field(default_factory=_prompt_pair_list_factory)
    skip: list[Prompt] = field(default_factory=_prompt_list_factory)
    diff: CatalogDiff = field(default_factory=CatalogDiff)


_DIFF_FIELDS: Sequence[str] = (
    "name",
    "description",
    "category",
    "tags",
    "language",
    "context",
    "example_input",
    "example_output",
    "version",
    "author",
    "quality_score",
    "related_prompts",
    "is_active",
    "source",
    "ext1",
    "ext2",
    "ext3",
    "ext4",
    "ext5",
)


def _prompt_to_catalog_dict(prompt: Prompt) -> dict[str, Any]:
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


def _format_prompt_json(prompt: Prompt) -> str:
    return json.dumps(
        _prompt_to_catalog_dict(prompt),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def _diff_prompts(existing: Prompt, incoming: Prompt) -> str:
    before = _format_prompt_json(existing).splitlines()
    after = _format_prompt_json(_merge_prompt(existing, incoming)).splitlines()
    diff_lines = difflib.unified_diff(
        before,
        after,
        fromfile=f"{existing.name} (existing)",
        tofile=f"{incoming.name} (incoming)",
        lineterm="",
    )
    return "\n".join(diff_lines)


def _build_change_plan(
    manager: PromptManager,
    prompts: Sequence[Prompt],
    *,
    overwrite: bool,
) -> CatalogChangePlan:
    try:
        existing_prompts = manager.repository.list()
    except Exception as exc:  # pragma: no cover - defensive
        raise PromptStorageError("Unable to inspect existing prompts") from exc

    plan = CatalogChangePlan()
    existing_by_name: dict[str, Prompt] = {
        prompt.name.strip().lower(): prompt for prompt in existing_prompts
    }

    for prompt in prompts:
        key = prompt.name.strip().lower()
        existing = existing_by_name.get(key)
        if existing is None:
            plan.create.append(prompt)
            plan.diff.entries.append(
                CatalogDiffEntry(
                    prompt_id=prompt.id,
                    name=prompt.name,
                    change_type=CatalogChangeType.ADD,
                    diff=textwrap.dedent(_format_prompt_json(prompt)),
                )
            )
            plan.diff.added += 1
            existing_by_name[key] = prompt
            continue

        if not overwrite:
            plan.diff.entries.append(
                CatalogDiffEntry(
                    prompt_id=existing.id,
                    name=existing.name,
                    change_type=CatalogChangeType.SKIP,
                    diff="Overwrite disabled; existing prompt retained.",
                )
            )
            plan.diff.skipped += 1
            plan.skip.append(existing)
            continue

        diff_text = _diff_prompts(existing, prompt)
        if diff_text.strip():
            plan.update.append((existing, prompt))
            plan.diff.entries.append(
                CatalogDiffEntry(
                    prompt_id=existing.id,
                    name=existing.name,
                    change_type=CatalogChangeType.UPDATE,
                    diff=diff_text,
                )
            )
            plan.diff.updated += 1
            existing_by_name[key] = _merge_prompt(existing, prompt)
        else:
            plan.diff.unchanged += 1

    return plan


@dataclass(slots=True)
class CatalogImportResult:
    """Aggregate statistics from a catalogue import operation."""

    added: int = 0
    updated: int = 0
    skipped: int = 0
    errors: int = 0
    preview: CatalogDiff | None = None

    def summary(self) -> dict[str, int]:
        """Return aggregate counts from the previous import run."""
        return {
            "added": self.added,
            "updated": self.updated,
            "skipped": self.skipped,
            "errors": self.errors,
        }


def diff_prompt_catalog(
    manager: PromptManager,
    catalog_path: Path | None,
    *,
    overwrite: bool = True,
) -> CatalogDiff:
    """Return a diff preview describing how a catalogue import would change prompts."""
    prompts = load_prompt_catalog(catalog_path)
    plan = _build_change_plan(manager, prompts, overwrite=overwrite)
    plan.diff.source = str(catalog_path) if catalog_path else None
    return plan.diff


def import_prompt_catalog(
    manager: PromptManager,
    catalog_path: Path | None,
    *,
    overwrite: bool = True,
) -> CatalogImportResult:
    """Apply catalogue changes to the repository and return a summary result."""
    prompts = load_prompt_catalog(catalog_path)
    plan = _build_change_plan(manager, prompts, overwrite=overwrite)

    result = CatalogImportResult(preview=plan.diff, skipped=len(plan.skip))

    for prompt in plan.create:
        try:
            manager.create_prompt(prompt)
            result.added += 1
        except Exception as exc:
            logger.error("Unable to create prompt %s: %s", prompt.name, exc)
            result.errors += 1

    for existing, incoming in plan.update:
        try:
            merged = _merge_prompt(existing, incoming)
            manager.update_prompt(merged)
            result.updated += 1
        except Exception as exc:
            logger.error("Unable to update prompt %s: %s", existing.name, exc)
            result.errors += 1

    return result


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


__all__ = [
    "CatalogChangePlan",
    "CatalogChangeType",
    "CatalogDiff",
    "CatalogDiffEntry",
    "CatalogImportResult",
    "diff_prompt_catalog",
    "export_prompt_catalog",
    "import_prompt_catalog",
    "load_prompt_catalog",
]
