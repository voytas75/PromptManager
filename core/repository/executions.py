"""Prompt execution persistence, filtering, and analytics helpers.

Updates:
  v0.11.0 - 2025-12-04 - Extract execution CRUD and analytics into mixin.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from models.prompt_model import PromptExecution

from .base import (
    RepositoryError,
    RepositoryNotFoundError,
    connect as _connect,
    json_dumps as _json_dumps,
    stringify_uuid as _stringify_uuid,
)

if TYPE_CHECKING:
    import uuid
    from datetime import datetime


class ExecutionStoreMixin:
    """Execution persistence helpers shared across repository implementations."""

    _db_path: Path

    _EXECUTION_COLUMNS: ClassVar[tuple[str, ...]] = (
        "id",
        "prompt_id",
        "request_text",
        "response_text",
        "status",
        "error_message",
        "duration_ms",
        "executed_at",
        "input_hash",
        "rating",
        "metadata",
    )

    def add_execution(self, execution: PromptExecution) -> PromptExecution:
        """Persist a prompt execution history entry."""
        payload = self._execution_to_row(execution)
        placeholders = ", ".join(f":{column}" for column in self._EXECUTION_COLUMNS)
        query = (
            f"INSERT INTO prompt_executions ({', '.join(self._EXECUTION_COLUMNS)}) "
            f"VALUES ({placeholders});"
        )
        try:
            with _connect(self._db_path) as conn:
                conn.execute(query, payload)
        except sqlite3.IntegrityError as exc:
            raise RepositoryError(f"Execution {execution.id} already exists") from exc
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to insert execution {execution.id}") from exc
        return execution

    def get_execution(self, execution_id: uuid.UUID) -> PromptExecution:
        """Fetch a single execution by identifier."""
        try:
            with _connect(self._db_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM prompt_executions WHERE id = ?;",
                    (_stringify_uuid(execution_id),),
                )
                row = cursor.fetchone()
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to load execution {execution_id}") from exc
        if row is None:
            raise RepositoryNotFoundError(f"Execution {execution_id} not found")
        return self._row_to_execution(row)

    def update_execution(self, execution: PromptExecution) -> PromptExecution:
        """Persist changes to an existing execution entry."""
        payload = self._execution_to_row(execution)
        assignments = ", ".join(
            f"{column} = :{column}" for column in self._EXECUTION_COLUMNS if column != "id"
        )
        query = f"UPDATE prompt_executions SET {assignments} WHERE id = :id;"
        try:
            with _connect(self._db_path) as conn:
                cursor = conn.execute(query, payload)
                if cursor.rowcount == 0:
                    raise RepositoryNotFoundError(f"Execution {execution.id} not found")
        except RepositoryNotFoundError:
            raise
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to update execution {execution.id}") from exc
        return execution

    def list_executions(
        self,
        *,
        limit: int | None = None,
    ) -> list[PromptExecution]:
        """Return executions ordered by most recent first."""
        clause = "ORDER BY datetime(executed_at) DESC"
        if limit is not None:
            clause += f" LIMIT {int(limit)}"
        query = f"SELECT * FROM prompt_executions {clause};"
        try:
            with _connect(self._db_path) as conn:
                rows = conn.execute(query).fetchall()
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to fetch executions") from exc
        return [self._row_to_execution(row) for row in rows]

    def list_executions_for_prompt(
        self,
        prompt_id: uuid.UUID,
        *,
        limit: int | None = None,
    ) -> list[PromptExecution]:
        """Return execution history for a given prompt."""
        clause = "WHERE prompt_id = ? ORDER BY datetime(executed_at) DESC"
        params: list[Any] = [_stringify_uuid(prompt_id)]
        if limit is not None:
            clause += f" LIMIT {int(limit)}"
        query = f"SELECT * FROM prompt_executions {clause};"
        try:
            with _connect(self._db_path) as conn:
                rows = conn.execute(query, params).fetchall()
        except sqlite3.Error as exc:
            raise RepositoryError(f"Failed to fetch execution history for {prompt_id}") from exc
        return [self._row_to_execution(row) for row in rows]

    def list_executions_filtered(
        self,
        *,
        status: str | None = None,
        prompt_id: uuid.UUID | None = None,
        search: str | None = None,
        limit: int | None = None,
        order_desc: bool = True,
    ) -> list[PromptExecution]:
        """Return executions filtered by status, prompt, and search term."""
        query = "SELECT * FROM prompt_executions"
        conditions: list[str] = []
        params: list[Any] = []

        if status:
            conditions.append("status = ?")
            params.append(status)
        if prompt_id:
            conditions.append("prompt_id = ?")
            params.append(_stringify_uuid(prompt_id))
        if search:
            like = f"%{search.lower()}%"
            conditions.append(
                "("
                "LOWER(request_text) LIKE ? OR "
                "LOWER(response_text) LIKE ? OR "
                "LOWER(COALESCE(metadata, '')) LIKE ?"
                ")"
            )
            params.extend([like, like, like])

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        order = "DESC" if order_desc else "ASC"
        query += f" ORDER BY datetime(executed_at) {order}"
        if limit is not None:
            query += f" LIMIT {int(limit)}"

        try:
            with _connect(self._db_path) as conn:
                rows = conn.execute(query, params).fetchall()
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to fetch filtered executions") from exc
        return [self._row_to_execution(row) for row in rows]

    def get_execution_analytics(
        self,
        *,
        since: datetime | None = None,
        limit: int = 5,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Return overall and per-prompt execution aggregates."""
        limit_value = max(1, int(limit)) if limit else 5
        filters: list[str] = []
        params: list[Any] = []
        if since is not None:
            filters.append("datetime(executed_at) >= ?")
            params.append(since.isoformat())
        where_clause = f" WHERE {' AND '.join(filters)}" if filters else ""

        summary_query = (
            "SELECT "
            "COUNT(*) AS total_runs, "
            "SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS success_runs, "
            "AVG(duration_ms) AS avg_duration_ms, "
            "AVG(rating) AS avg_rating "
            "FROM prompt_executions"
            f"{where_clause};"
        )

        prompt_query = (
            "SELECT "
            "prompt_id, "
            "MAX(p.name) AS prompt_name, "
            "COUNT(*) AS total_runs, "
            "SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS success_runs, "
            "AVG(duration_ms) AS avg_duration_ms, "
            "AVG(rating) AS avg_rating, "
            "MAX(executed_at) AS last_executed_at "
            "FROM prompt_executions "
            "LEFT JOIN prompts p ON p.id = prompt_executions.prompt_id "
            f"{where_clause} "
            "GROUP BY prompt_id "
            "ORDER BY success_runs DESC, avg_rating DESC, total_runs DESC "
            "LIMIT ?;"
        )

        def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any]:
            if row is None:
                return {}
            return {key: row[key] for key in row.keys()}

        try:
            with _connect(self._db_path) as conn:
                summary_row = conn.execute(summary_query, params).fetchone()
                prompt_params = list(params)
                prompt_params.append(limit_value)
                prompt_rows = conn.execute(prompt_query, prompt_params).fetchall()
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to compute execution analytics") from exc

        summary = _row_to_dict(summary_row)
        aggregates = [_row_to_dict(row) for row in prompt_rows]
        return summary, aggregates

    def get_model_usage_breakdown(
        self,
        *,
        since: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Return aggregated token usage grouped by model identifier."""
        filters = ["metadata IS NOT NULL", "json_valid(metadata)"]
        params: list[Any] = []
        if since is not None:
            filters.append("datetime(executed_at) >= ?")
            params.append(since.isoformat())
        where_clause = f" WHERE {' AND '.join(filters)}" if filters else ""

        query = (
            "SELECT "
            "COALESCE("
            "json_extract(metadata, '$.context.execution.model'), "
            "json_extract(metadata, '$.model'), "
            "'unknown'"
            ") AS model, "
            "COUNT(*) AS run_count, "
            "SUM(COALESCE(json_extract(metadata, '$.usage.prompt_tokens'), 0)) AS prompt_tokens, "
            "SUM(COALESCE(json_extract(metadata, '$.usage.completion_tokens'), 0)) "
            "AS completion_tokens, "
            "SUM(COALESCE(json_extract(metadata, '$.usage.total_tokens'), 0)) AS total_tokens "
            "FROM prompt_executions "
            f"{where_clause} "
            "GROUP BY model "
            "ORDER BY run_count DESC;"
        )

        try:
            with _connect(self._db_path) as conn:
                rows = conn.execute(query, params).fetchall()
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to compute model usage breakdown") from exc
        results: list[dict[str, Any]] = []
        for row in rows:
            results.append({key: row[key] for key in row.keys()})
        return results

    def get_benchmark_execution_stats(
        self,
        *,
        since: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Return aggregated metrics for persisted benchmark executions."""
        filters = [
            "metadata IS NOT NULL",
            "json_valid(metadata)",
            "COALESCE(json_extract(metadata, '$.benchmark'), 0) = 1",
        ]
        params: list[Any] = []
        if since is not None:
            filters.append("datetime(executed_at) >= ?")
            params.append(since.isoformat())
        where_clause = f" WHERE {' AND '.join(filters)}" if filters else ""

        query = (
            "SELECT "
            "COALESCE("
            "json_extract(metadata, '$.model'), "
            "json_extract(metadata, '$.context.execution.model'), "
            "'unknown'"
            ") AS model, "
            "COUNT(*) AS total_runs, "
            "SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS success_runs, "
            "AVG(duration_ms) AS avg_duration_ms, "
            "SUM(COALESCE(json_extract(metadata, '$.usage.total_tokens'), 0)) AS total_tokens "
            "FROM prompt_executions "
            f"{where_clause} "
            "GROUP BY model "
            "ORDER BY total_runs DESC;"
        )

        try:
            with _connect(self._db_path) as conn:
                rows = conn.execute(query, params).fetchall()
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to compute benchmark execution stats") from exc
        results: list[dict[str, Any]] = []
        for row in rows:
            results.append({key: row[key] for key in row.keys()})
        return results

    def get_prompt_execution_statistics(
        self,
        prompt_id: uuid.UUID,
        *,
        since: datetime | None = None,
    ) -> dict[str, Any]:
        """Return aggregate execution metrics for a single prompt."""
        filters = ["prompt_id = ?"]
        params: list[Any] = [_stringify_uuid(prompt_id)]
        if since is not None:
            filters.append("datetime(executed_at) >= ?")
            params.append(since.isoformat())
        where_clause = f" WHERE {' AND '.join(filters)}"

        query = (
            "SELECT "
            "COUNT(*) AS total_runs, "
            "SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS success_runs, "
            "AVG(duration_ms) AS avg_duration_ms, "
            "AVG(rating) AS avg_rating, "
            "MAX(executed_at) AS last_executed_at, "
            "MAX(p.name) AS prompt_name "
            "FROM prompt_executions "
            "LEFT JOIN prompts p ON p.id = prompt_executions.prompt_id"
            f"{where_clause};"
        )

        def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any]:
            if row is None:
                return {}
            return {key: row[key] for key in row.keys()}

        try:
            with _connect(self._db_path) as conn:
                row = conn.execute(query, params).fetchone()
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to compute prompt execution statistics") from exc
        return _row_to_dict(row)

    def get_category_execution_statistics(self) -> dict[str, dict[str, Any]]:
        """Return execution aggregates grouped by prompt category."""
        query = (
            "SELECT "
            "COALESCE(p.category_slug, '') AS slug, "
            "COUNT(*) AS total_runs, "
            "SUM(CASE WHEN pe.status = 'success' THEN 1 ELSE 0 END) AS success_runs, "
            "MAX(pe.executed_at) AS last_executed_at "
            "FROM prompt_executions pe "
            "LEFT JOIN prompts p ON p.id = pe.prompt_id "
            "GROUP BY slug;"
        )
        try:
            with _connect(self._db_path) as conn:
                rows = conn.execute(query).fetchall()
        except sqlite3.Error as exc:
            raise RepositoryError("Failed to compute category execution statistics") from exc
        stats: dict[str, dict[str, Any]] = {}
        for row in rows:
            slug = row["slug"] if row["slug"] is not None else ""
            stats[str(slug)] = {
                "total_runs": int(row["total_runs"] or 0),
                "success_runs": int(row["success_runs"] or 0),
                "last_executed_at": row["last_executed_at"],
            }
        return stats

    def _execution_to_row(self, execution: PromptExecution) -> dict[str, Any]:
        """Serialise PromptExecution into SQLite mapping."""
        record = execution.to_record()
        record["metadata"] = _json_dumps(record["metadata"])
        return record

    def _row_to_execution(self, row: sqlite3.Row) -> PromptExecution:
        """Hydrate PromptExecution from a SQLite row."""
        payload: dict[str, Any] = {
            "id": row["id"],
            "prompt_id": row["prompt_id"],
            "request_text": row["request_text"],
            "response_text": row["response_text"],
            "status": row["status"],
            "error_message": row["error_message"],
            "duration_ms": row["duration_ms"],
            "executed_at": row["executed_at"],
            "input_hash": row["input_hash"],
            "rating": row["rating"],
            "metadata": row["metadata"],
        }
        return PromptExecution.from_record(payload)


__all__ = ["ExecutionStoreMixin"]
