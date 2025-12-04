"""Prompt chain data models.

Updates:
  v0.1.0 - 2025-12-04 - Introduce prompt chain and step dataclasses.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Mapping, MutableMapping, Sequence


def _utc_now() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(UTC)


def _ensure_uuid(value: uuid.UUID | str | None) -> uuid.UUID:
    """Return a uuid.UUID for ``value`` defaulting to a new identifier."""
    if value is None:
        return uuid.uuid4()
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


def _ensure_datetime(value: datetime | str | None) -> datetime:
    """Return an aware datetime parsed from ``value`` or now when missing."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if value is None:
        return _utc_now()
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _coerce_metadata(value: Any | None) -> MutableMapping[str, Any] | None:
    """Return metadata as a mutable mapping when possible."""
    if value is None:
        return None
    if isinstance(value, MutableMapping):
        return value
    if isinstance(value, Mapping):
        return dict(value)
    return {"value": value}


@dataclass(slots=True)
class PromptChainStep:
    """Single prompt invocation that belongs to a prompt chain."""

    id: uuid.UUID
    chain_id: uuid.UUID
    prompt_id: uuid.UUID
    order_index: int
    input_template: str
    output_variable: str
    condition: str | None = None
    stop_on_failure: bool = True
    metadata: MutableMapping[str, Any] | None = None

    def to_record(self) -> dict[str, Any]:
        """Return a dictionary suitable for SQLite persistence."""
        return {
            "id": str(self.id),
            "chain_id": str(self.chain_id),
            "prompt_id": str(self.prompt_id),
            "order_index": int(self.order_index),
            "input_template": self.input_template,
            "output_variable": self.output_variable,
            "condition": self.condition,
            "stop_on_failure": 1 if self.stop_on_failure else 0,
            "metadata": self.metadata,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> PromptChainStep:
        """Hydrate a step from a SQLite row payload."""
        return cls(
            id=_ensure_uuid(row.get("id")),
            chain_id=_ensure_uuid(row.get("chain_id")),
            prompt_id=_ensure_uuid(row.get("prompt_id")),
            order_index=int(row.get("order_index", 0)),
            input_template=str(row.get("input_template") or ""),
            output_variable=str(row.get("output_variable") or ""),
            condition=row.get("condition"),
            stop_on_failure=bool(int(row.get("stop_on_failure", 1))),
            metadata=_coerce_metadata(row.get("metadata")),
        )


@dataclass(slots=True)
class PromptChain:
    """Composable workflow that executes multiple prompts in sequence."""

    id: uuid.UUID
    name: str
    description: str
    is_active: bool = True
    variables_schema: MutableMapping[str, Any] | None = None
    metadata: MutableMapping[str, Any] | None = None
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)
    steps: list[PromptChainStep] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        """Return a dictionary suitable for SQLite persistence."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "is_active": 1 if self.is_active else 0,
            "variables_schema": self.variables_schema,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def with_steps(self, steps: Sequence[PromptChainStep]) -> PromptChain:
        """Return a copy of the chain with ``steps`` assigned."""
        return replace(self, steps=list(sorted(steps, key=lambda step: step.order_index)))

    @classmethod
    def from_row(
        cls,
        row: Mapping[str, Any],
        *,
        steps: Sequence[PromptChainStep] | None = None,
    ) -> PromptChain:
        """Hydrate a chain from a SQLite row payload."""
        chain = cls(
            id=_ensure_uuid(row.get("id")),
            name=str(row.get("name") or ""),
            description=str(row.get("description") or ""),
            is_active=bool(int(row.get("is_active", 1))),
            variables_schema=_coerce_metadata(row.get("variables_schema")),
            metadata=_coerce_metadata(row.get("metadata")),
            created_at=_ensure_datetime(row.get("created_at")),
            updated_at=_ensure_datetime(row.get("updated_at")),
            steps=list(sorted(steps or [], key=lambda step: step.order_index)),
        )
        return chain

