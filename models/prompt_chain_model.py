"""Prompt chain data models.

Updates:
  v0.2.0 - 2025-12-04 - Add chain_from_payload helper for GUI and CLI imports.
  v0.1.0 - 2025-12-04 - Introduce prompt chain and step dataclasses.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any


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


__all__ = [
    "PromptChain",
    "PromptChainStep",
    "chain_from_payload",
]


def chain_from_payload(payload: Mapping[str, Any]) -> PromptChain:
    """Return a :class:`PromptChain` built from a JSON-like mapping.

    Args:
        payload: Mapping containing ``name``, ``description``, and ``steps`` fields.

    Returns:
        PromptChain: The hydrated chain with ordered steps attached.

    Raises:
        ValueError: If required keys are missing or malformed.
    """
    name = str(payload.get("name") or "").strip()
    if not name:
        raise ValueError("Prompt chain requires a non-empty 'name' field.")
    chain_id_text = payload.get("id")
    chain_id = uuid.uuid4() if not chain_id_text else uuid.UUID(str(chain_id_text))
    description = str(payload.get("description") or "")
    steps_payload = payload.get("steps") or []
    steps: list[PromptChainStep] = []
    if not isinstance(steps_payload, list):
        raise ValueError("'steps' must be an array.")
    for index, step_payload in enumerate(steps_payload, start=1):
        if not isinstance(step_payload, Mapping):
            raise ValueError(f"Step {index} must be an object.")
        prompt_id_value = step_payload.get("prompt_id")
        if not prompt_id_value:
            raise ValueError(f"Step {index} is missing 'prompt_id'.")
        step_id_value = step_payload.get("id")
        order_index = int(step_payload.get("order_index") or index)
        input_template = str(step_payload.get("input_template") or "")
        output_variable = str(step_payload.get("output_variable") or f"step_{order_index}")
        condition_text = str(step_payload.get("condition") or "").strip()
        steps.append(
            PromptChainStep(
                id=uuid.uuid4() if not step_id_value else uuid.UUID(str(step_id_value)),
                chain_id=chain_id,
                prompt_id=uuid.UUID(str(prompt_id_value)),
                order_index=order_index,
                input_template=input_template,
                output_variable=output_variable,
                condition=condition_text or None,
                stop_on_failure=bool(step_payload.get("stop_on_failure", True)),
                metadata=_coerce_metadata(step_payload.get("metadata")),
            )
        )
    chain = PromptChain(
        id=chain_id,
        name=name,
        description=description,
        is_active=bool(payload.get("is_active", True)),
        variables_schema=_coerce_metadata(payload.get("variables_schema")),
        metadata=_coerce_metadata(payload.get("metadata")),
    )
    return chain.with_steps(steps)
