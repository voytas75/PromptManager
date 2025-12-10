# Token Usage Aggregation Plan

Last updated: 2025-12-10

## Goals
- Surface token usage at three scopes: per-query (single execution), per-session (workspace/CLI run grouping), and global (all history).
- Keep aggregation cheap by reusing existing `prompt_executions.metadata.usage` payloads and SQLite rollups.
- Expose identical summaries in GUI and CLI with busy indicators for long queries and toast notifications for quick exports.

## Data Model
- **Execution-level (existing)**: `prompt_executions.metadata.usage` holds `prompt_tokens`, `completion_tokens`, `total_tokens`.
- **Session-level (new)**: introduce `execution_sessions` table with `id (uuid)`, `started_at`, `ended_at`, `run_count`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `last_execution_id`. Stamp `session_id` on each `PromptExecution` (column + metadata) so HistoryTracker can roll up by session.
- **Global rollups (new helper)**: add `token_usage_rollups` helper view or cached table keyed by `window_start`, `window_end`, `scope ("all" | "session" | "prompt")`, storing aggregates for dashboard/CLI without full scans.

## Persistence & Computation
- HistoryTracker to accept optional `session_id` and write it into `PromptExecution` plus `execution_sessions`.
- Extend repository analytics to:
  - fetch per-session totals (`get_token_usage_by_session(session_id)`),
  - fetch per-prompt totals for a window (`get_token_usage_by_prompt(since=None)`),
  - reuse `get_token_usage_totals(since=None)` for global.
- Background rollup hook: when HistoryTracker flushes an execution, update or insert the corresponding session row; nightly job (or on-demand command) can refresh global rollups for dashboards.

## UX Surfaces
- **Workspace header**: show live per-query usage, running session totals, and all-time totals; tooltip explains session scoping and offline mode behaviour.
- **History tab**: add session filter + per-session summary row; CSV export includes `session_id`.
- **Analytics dashboard**: new token usage chart grouped by scope (query/session/global) with window selector; reuse busy indicator already used for other analytics fetches.
- **Prompt detail**: show per-prompt totals (current window + all-time) with link to session breakdown.
- **Chain runner**: display accumulated tokens per chain run and expose session id in logs.

## CLI & Automation
- Extend `python -m main history-analytics` with `--session-id` and `--token-scope {query,session,global}` to mirror GUI views.
- Add `python -m main usage-report --session-id <uuid> [--since DAYS]` to export CSV/JSON rollups for automation.
- Keep outputs deterministic for tests; mock external services (`respx`/`vcrpy`) and use fixed timestamps in coverage suites.

## Testing & Quality Gates
- Unit-test rollup helpers (session/global) with synthetic executions and fixed timestamps.
- Property-based tests for malformed usage metadata (missing numbers, strings) to ensure coercion and zero-fill behaviours.
- End-to-end GUI/CLI smoke to verify busy indicators/toasts trigger for long queries and that session filters behave as documented.

