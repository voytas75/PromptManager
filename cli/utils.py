"""Shared CLI utility functions for Prompt Manager commands.

Updates:
  v0.1.0 - 2025-12-04 - Extract stdout logging, masking, path helpers, and exporters.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from collections.abc import Mapping, Sequence
    from logging import Logger
else:  # pragma: no cover - runtime placeholders for type-only imports
    Mapping = Sequence = Logger = Any


def print_and_log(logger: Logger, level: int, message: str) -> None:
    """Log *message* at *level* and mirror it to stdout."""
    logger.log(level, message)
    print(message)


def mask_secret(value: str | None) -> str:
    """Return an obfuscated representation of secret configuration values."""
    if not value:
        return "not set"
    secret = value.strip()
    if len(secret) <= 6:
        return "set (****)"
    prefix = secret[:4]
    suffix = secret[-4:]
    return f"set ({prefix}...{suffix})"


def describe_path(
    path_value: object,
    *,
    expect_directory: bool,
    allow_missing_file: bool = False,
) -> str:
    """Return a human-friendly description of *path_value* suitability."""
    try:
        path = Path(path_value) if path_value is not None else None
    except TypeError:
        path = None
    if path is None:
        return "not set"

    resolved = path.expanduser()
    if resolved.exists():
        if expect_directory and not resolved.is_dir():
            return f"{resolved} (exists but is not a directory)"
        if not expect_directory and resolved.is_dir():
            return f"{resolved} (exists but is a directory)"
        return f"{resolved} (exists)"

    message = f"{resolved} (missing)"
    if not expect_directory and allow_missing_file:
        message = f"{resolved} (missing - created on demand)"
    parent = resolved.parent
    if not parent.exists():
        message += f", parent missing: {parent}"
    return message


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    """Persist *rows* to CSV at *path* and return the resolved destination."""
    if not rows:
        raise ValueError("No rows available for export")
    headers: list[str] = []
    seen_keys: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen_keys:
                continue
            seen_keys.add(key)
            headers.append(str(key))

    resolved = path.expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})
    return resolved


def resolve_export_format(path: Path, explicit_format: str | None) -> str:
    """Return an export format slug based on *path* or *explicit_format*."""
    if explicit_format:
        return explicit_format.lower()
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return "yaml"
    return "json"


def format_metric(value: float | None, *, suffix: str = "") -> str:
    """Return display-friendly metric text with optional *suffix*."""
    if value is None:
        return "n/a"
    formatted = f"{value:.2f}" if abs(value) < 1000 else f"{value:.0f}"
    return f"{formatted}{suffix}" if suffix else formatted
