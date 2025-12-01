"""Usage logging utilities for intent workspace interactions.

Updates:
  v0.3.2 - 2025-11-29 - Wrap logging helpers for Ruff line-length compliance.
  v0.3.1 - 2025-12-08 - Remove task template logging after feature retirement.
  v0.2.0 - 2025-11-09 - Record manual save ratings alongside notes.
  v0.1.0 - 2025-11-07 - Introduce JSONL logger for detect/suggest/copy events.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from core.intent_classifier import IntentPrediction

logger = logging.getLogger("prompt_manager.gui.usage")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class IntentUsageLogger:
    """Persist anonymised analytics for intent workspace interactions."""
    def __init__(self, path: Path | str | None = None, *, enabled: bool = True) -> None:
        """Optionally disable logging or override the JSONL output path."""
        self._enabled = enabled
        default_path = Path("data") / "logs" / "intent_usage.jsonl"
        self._path = Path(path) if path is not None else default_path

    @property
    def log_path(self) -> Path:
        """Return the resolved path for the usage log."""
        return self._path

    def log_detect(self, *, prediction: IntentPrediction, query_text: str) -> None:
        """Log details for an intent detection event."""
        record = self._base_record("detect", query_text)
        record.update(
            {
                "label": prediction.label.value,
                "confidence": prediction.confidence,
                "category_hints": prediction.category_hints,
                "tag_hints": prediction.tag_hints,
                "language_hints": prediction.language_hints,
            }
        )
        self._append(record)

    def log_suggest(
        self,
        *,
        prediction: IntentPrediction,
        query_text: str,
        prompts: Sequence[object],
        fallback_used: bool,
    ) -> None:
        """Log details for a suggestion event."""
        record = self._base_record("suggest", query_text)
        record.update(
            {
                "label": prediction.label.value,
                "confidence": prediction.confidence,
                "results_count": len(prompts),
                "fallback_used": fallback_used,
                "top_prompts": [getattr(prompt, "name", "") for prompt in prompts[:5]],
            }
        )
        self._append(record)

    def log_copy(self, *, prompt_name: str, prompt_has_body: bool) -> None:
        """Log that a prompt has been copied to the clipboard."""
        record = {
            "timestamp": _now_iso(),
            "event": "copy",
            "prompt_name": prompt_name,
            "has_body": prompt_has_body,
        }
        self._append(record)

    def log_share(self, *, provider: str, prompt_name: str, payload_chars: int) -> None:
        """Log that a prompt was shared via an external provider."""
        record = {
            "timestamp": _now_iso(),
            "event": "share",
            "provider": provider,
            "prompt_name": prompt_name,
            "chars": payload_chars,
        }
        self._append(record)

    def log_execute(
        self,
        *,
        prompt_name: str,
        success: bool,
        duration_ms: int | None,
        error: str | None = None,
    ) -> None:
        """Log prompt execution outcomes."""
        record = {
            "timestamp": _now_iso(),
            "event": "execute",
            "prompt_name": prompt_name,
            "success": success,
            "duration_ms": duration_ms,
        }
        if error:
            record["error"] = error[:200]
        self._append(record)

    def log_history_view(self, *, total: int) -> None:
        """Log when the execution history dialog is opened."""
        record = {
            "timestamp": _now_iso(),
            "event": "history",
            "entries": total,
        }
        self._append(record)

    def log_save(self, *, prompt_name: str, note_length: int, rating: float | None = None) -> None:
        """Log that a prompt result was manually saved."""
        record = {
            "timestamp": _now_iso(),
            "event": "save",
            "prompt_name": prompt_name,
            "note_length": note_length,
        }
        if rating is not None:
            record["rating"] = rating
        self._append(record)

    def log_note_edit(self, *, note_length: int) -> None:
        """Log note edits performed in the history dialog."""
        record = {
            "timestamp": _now_iso(),
            "event": "edit_note",
            "note_length": note_length,
        }
        self._append(record)

    def log_history_export(self, *, entries: int, path: str) -> None:
        """Log that history was exported."""
        record = {
            "timestamp": _now_iso(),
            "event": "history_export",
            "entries": entries,
            "path": path,
        }
        self._append(record)

    def _base_record(self, event: str, query_text: str) -> dict[str, object]:
        digest = hashlib.blake2s(query_text.encode("utf-8"), digest_size=8).hexdigest()
        preview = " ".join(query_text.split())[:120]
        return {
            "timestamp": _now_iso(),
            "event": event,
            "query_chars": len(query_text),
            "query_hash": digest,
            "query_preview": preview,
        }

    def _append(self, record: dict[str, object]) -> None:
        if not self._enabled:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")
        except OSError:
            logger.debug(
                "Unable to write usage analytics",
                extra={"event": record.get("event")},
                exc_info=True,
            )


__all__ = ["IntentUsageLogger"]
