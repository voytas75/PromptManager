"""Inspect recent prompt executions and token usage.

Updates:
  v0.1.0 - 2025-12-10 - Add helper to dump stored and estimated token usage from the DB.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping, Sequence


def _estimate_usage(model: str, messages: Sequence[Mapping[str, str]], text: str) -> dict[str, int]:
    """Estimate token usage with litellm.token_counter when usage is missing."""
    try:
        import litellm  # type: ignore
    except Exception:
        return {}
    try:
        usage = litellm.token_counter(model=model, messages=list(messages), text=text)  # type: ignore[attr-defined]
    except Exception:
        return {}
    if not isinstance(usage, Mapping):
        return {}
    prompt_value = _coerce_int(usage.get("prompt_tokens"))
    completion_value = _coerce_int(usage.get("completion_tokens"))
    total_value = usage.get("total_tokens")
    total_tokens = _coerce_int(total_value) if total_value is not None else prompt_value + completion_value
    return {
        "prompt_tokens": prompt_value,
        "completion_tokens": completion_value,
        "total_tokens": total_tokens,
    }


def _coerce_int(value: object | None) -> int:
    """Safely coerce numbers that may arrive as strings or None."""
    if value in (None, ""):
        return 0
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        try:
            return int(float(value))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0


def _load_rows(db_path: Path, limit: int) -> list[sqlite3.Row]:
    """Return recent execution rows ordered by executed_at descending."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    query = (
        "SELECT id, prompt_id, executed_at, request_text, response_text, metadata "
        "FROM prompt_executions "
        "ORDER BY datetime(executed_at) DESC LIMIT ?;"
    )
    return list(conn.execute(query, (limit,)))


def main() -> None:
    """Entry point for the token usage inspection helper."""
    parser = argparse.ArgumentParser(
        description="Inspect stored token usage and estimate when usage is missing."
    )
    parser.add_argument(
        "--db",
        default="data/prompt_manager.db",
        help="Path to prompt_manager SQLite database (default: data/prompt_manager.db).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of recent executions to inspect (default: 5).",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    rows = _load_rows(db_path, max(1, args.limit))
    if not rows:
        print("No executions found.")
        return

    for idx, row in enumerate(rows, start=1):
        metadata_text = row["metadata"] or "{}"
        try:
            metadata = json.loads(metadata_text)
        except json.JSONDecodeError:
            metadata = {}
        usage = metadata.get("usage") or {}
        model = (
            metadata.get("context", {})
            .get("execution", {})
            .get("model", "unknown-model")
        )
        conversation = metadata.get("conversation") or []
        if not conversation:
            conversation = [{"role": "user", "content": row["request_text"] or ""}]
        response_text = row["response_text"] or ""

        estimated = {}
        if not usage:
            estimated = _estimate_usage(model, conversation, response_text)

        print(f"[{idx}] executed_at={row['executed_at']}")
        print(f"     prompt_id={row['prompt_id']}")
        print(f"     model={model}")
        print(f"     stored usage: {usage if usage else 'EMPTY'}")
        if estimated:
            print(f"     estimated usage: {estimated}")
        print()


if __name__ == "__main__":
    main()
