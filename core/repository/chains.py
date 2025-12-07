"""Prompt chain persistence helpers.

Updates:
  v0.1.1 - 2025-12-07 - Restrict Path import to type-checking contexts.
  v0.1.0 - 2025-12-04 - Introduce prompt chain and step CRUD mixin.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any

from models.prompt_chain_model import PromptChain, PromptChainStep

from .base import (
    RepositoryError,
    RepositoryNotFoundError,
    connect as _connect,
    json_dumps as _json_dumps,
    json_loads_optional as _json_loads_optional,
    stringify_uuid as _stringify_uuid,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import uuid
    from collections.abc import Mapping, Sequence
    from pathlib import Path


class ChainStoreMixin:
    """Mixin exposing CRUD helpers for prompt chain definitions."""

    _db_path: Path

    _CHAIN_COLUMNS = (
        "id",
        "name",
        "description",
        "is_active",
        "variables_schema",
        "metadata",
        "created_at",
        "updated_at",
    )
    _STEP_COLUMNS = (
        "id",
        "chain_id",
        "prompt_id",
        "order_index",
        "input_template",
        "output_variable",
        "condition",
        "stop_on_failure",
        "metadata",
    )

    def list_chains(
        self,
        *,
        include_inactive: bool = False,
        include_steps: bool = True,
    ) -> list[PromptChain]:
        """Return all prompt chains optionally including inactive entries."""
        where_clause = "" if include_inactive else "WHERE is_active = 1"
        query = f"SELECT * FROM prompt_chains {where_clause} ORDER BY datetime(created_at) DESC;"
        try:
            with _connect(self._db_path) as conn:
                rows = conn.execute(query).fetchall()
                steps_map = (
                    self._load_steps_for_chains(conn, [row["id"] for row in rows])
                    if include_steps and rows
                    else {}
                )
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to list prompt chains") from exc
        return [self._row_to_chain(row, steps=steps_map.get(row["id"], [])) for row in rows]

    def get_chain(self, chain_id: uuid.UUID, *, include_steps: bool = True) -> PromptChain:
        """Return a single prompt chain by identifier."""
        try:
            with _connect(self._db_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM prompt_chains WHERE id = ?;",
                    (_stringify_uuid(chain_id),),
                )
                row = cursor.fetchone()
                if row is None:
                    raise RepositoryNotFoundError(f"Prompt chain {chain_id} not found")
                steps = (
                    self._load_steps_for_chains(conn, [row["id"]]).get(row["id"], [])
                    if include_steps
                    else []
                )
        except RepositoryNotFoundError:
            raise
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to load prompt chain {chain_id}") from exc
        return self._row_to_chain(row, steps=steps)

    def add_chain(self, chain: PromptChain) -> PromptChain:
        """Persist a new prompt chain and its steps."""
        payload = self._chain_to_row(chain)
        steps = chain.steps or []
        placeholders = ", ".join(f":{column}" for column in self._CHAIN_COLUMNS)
        columns = ", ".join(self._CHAIN_COLUMNS)
        query = f"INSERT INTO prompt_chains ({columns}) VALUES ({placeholders});"
        try:
            with _connect(self._db_path) as conn:
                conn.execute(query, payload)
                if steps:
                    self._insert_steps(conn, steps)
        except sqlite3.IntegrityError as exc:
            raise RepositoryError(f"Prompt chain {chain.id} already exists") from exc
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to create prompt chain") from exc
        return chain

    def update_chain(self, chain: PromptChain) -> PromptChain:
        """Replace an existing prompt chain definition."""
        payload = self._chain_to_row(chain)
        assignments = ", ".join(
            f"{column} = :{column}" for column in self._CHAIN_COLUMNS if column != "id"
        )
        query = f"UPDATE prompt_chains SET {assignments} WHERE id = :id;"
        try:
            with _connect(self._db_path) as conn:
                cursor = conn.execute(query, payload)
                if cursor.rowcount == 0:
                    raise RepositoryNotFoundError(f"Prompt chain {chain.id} not found")
                conn.execute(
                    "DELETE FROM prompt_chain_steps WHERE chain_id = ?;",
                    (payload["id"],),
                )
                if chain.steps:
                    self._insert_steps(conn, chain.steps)
        except RepositoryNotFoundError:
            raise
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to update prompt chain {chain.id}") from exc
        return chain

    def delete_chain(self, chain_id: uuid.UUID) -> None:
        """Delete a prompt chain definition."""
        try:
            with _connect(self._db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM prompt_chains WHERE id = ?;",
                    (_stringify_uuid(chain_id),),
                )
                if cursor.rowcount == 0:
                    raise RepositoryNotFoundError(f"Prompt chain {chain_id} not found")
        except RepositoryNotFoundError:
            raise
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to delete prompt chain {chain_id}") from exc

    def chain_exists(self, chain_id: uuid.UUID) -> bool:
        """Return True when the specified chain is stored."""
        try:
            with _connect(self._db_path) as conn:
                cursor = conn.execute(
                    "SELECT 1 FROM prompt_chains WHERE id = ? LIMIT 1;",
                    (_stringify_uuid(chain_id),),
                )
                row = cursor.fetchone()
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to check prompt chain existence") from exc
        return row is not None

    # Internal helpers ------------------------------------------------- #

    def _chain_to_row(self, chain: PromptChain) -> dict[str, Any]:
        payload = chain.to_record()
        payload["is_active"] = 1 if chain.is_active else 0
        payload["variables_schema"] = _json_dumps(payload["variables_schema"])
        payload["metadata"] = _json_dumps(payload["metadata"])
        return payload

    def _insert_steps(self, conn: sqlite3.Connection, steps: Sequence[PromptChainStep]) -> None:
        payloads = [self._step_to_row(step) for step in steps]
        placeholders = ", ".join(f":{column}" for column in self._STEP_COLUMNS)
        query = (
            f"INSERT INTO prompt_chain_steps ({', '.join(self._STEP_COLUMNS)}) "
            f"VALUES ({placeholders});"
        )
        conn.executemany(query, payloads)

    def _step_to_row(self, step: PromptChainStep) -> dict[str, Any]:
        payload = step.to_record()
        payload["stop_on_failure"] = 1 if step.stop_on_failure else 0
        payload["metadata"] = _json_dumps(payload["metadata"])
        return payload

    def _row_to_chain(
        self,
        row: Mapping[str, Any],
        *,
        steps: Sequence[PromptChainStep],
    ) -> PromptChain:
        chain_row = dict(row)
        chain_row["variables_schema"] = _json_loads_optional(chain_row.get("variables_schema"))
        chain_row["metadata"] = _json_loads_optional(chain_row.get("metadata"))
        return PromptChain.from_row(chain_row, steps=steps)

    def _load_steps_for_chains(
        self,
        conn: sqlite3.Connection,
        chain_ids: Sequence[str],
    ) -> dict[str, list[PromptChainStep]]:
        if not chain_ids:
            return {}
        placeholders = ",".join("?" for _ in chain_ids)
        query = (
            "SELECT * FROM prompt_chain_steps "
            f"WHERE chain_id IN ({placeholders}) "
            "ORDER BY chain_id, order_index;"
        )
        rows = conn.execute(query, list(chain_ids)).fetchall()
        result: dict[str, list[PromptChainStep]] = {}
        for row in rows:
            chain_id = row["chain_id"]
            if chain_id not in result:
                result[chain_id] = []
            row_map = dict(row)
            row_map["metadata"] = _json_loads_optional(row_map.get("metadata"))
            result[chain_id].append(PromptChainStep.from_row(row_map))
        return result
