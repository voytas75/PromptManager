# PromptManager — Prompt activity events v1

**Status:** in progress
**Owner:** PromptManager Team

## Goal

Add one durable, local activity ledger for successful **prompt mutations** so the same semantic operation is recorded whether it comes from the GUI or CLI.

## Confirmed baseline

- Prompt versions record snapshots but do not record a general operator action or interface origin.
- Execution history is separate from prompt-asset mutation history.
- SQLite has no prompt activity-event table.
- GUI and CLI both converge on `PromptManager` lifecycle/version methods for prompt mutation.

## Scope

- Persist `created`, `updated`, `forked`, `restored`, and `deleted` events.
- Store event timestamp, prompt ID, operation, and `origin` (`gui` or `cli`).
- Emit once after the authoritative mutation succeeds.
- Keep fork/restore semantic events singular rather than duplicating their delegated create/update event.
- Thread CLI origin explicitly through catalog import, fork, restore, and scenario refresh.
- Preserve GUI compatibility through the central default origin; no new GUI panel or command is included.

## Out of scope

- Search/query history.
- Logging keystrokes, opened views, filters, or reads.
- New GUI history browser or CLI activity-read command.
- User/account identity, telemetry, remote transmission, content snapshots, or prompt-body storage in events.
- Changes to execution history or version-history semantics.

## Execution order

1. Add focused provider-free RED tests for repository persistence and one semantic event per central mutation.
2. Add the SQLite table, typed activity record, repository read/write helpers, and central event emission.
3. Thread explicit CLI origin through the existing mutating paths; GUI remains covered by the central default.
4. Run focused tests, Ruff, strict Pyright on changed Python files, and SQLite smoke verification.
5. Update this ledger and `docs/CHANGELOG.md` with confirmed delivery evidence.

## Done criteria

- Fresh and existing SQLite databases expose the activity table safely.
- Successful create/update/fork/restore/delete operations produce exactly one event with the correct prompt ID, operation, and origin.
- Failed mutations and fork rollback do not produce a success event.
- CLI mutations record `origin="cli"`; existing GUI-domain calls record `origin="gui"` by default.

## Completion update

**Status: completed — 2026-09-21**

Delivered:

- Added the SQLite-backed `prompt_activity_events` ledger with `prompt_id`, semantic operation, origin, and UTC timestamp; it deliberately has no prompt-body payload and no foreign key, so a deletion event remains readable after the asset is removed.
- Added repository `record_prompt_activity(...)` and newest-first `list_prompt_activity(...)` methods; no GUI panel or CLI reader was added in this slice.
- Centralized one event per successful domain mutation: `created`, `updated`, `forked`, `restored`, and `deleted`.
- Prevented duplicate generic events for fork/restore and prevented rollback cleanup from emitting a false user-visible deletion event.
- Threaded `origin="cli"` through catalog import / `prompt-add`, scenario refresh, prompt fork, and version restore. Existing GUI paths retain the explicit central default `origin="gui"` without UI rewrites.
- Kept search, filters, view opens, and keystrokes out of the ledger.

Verified:

```bash
.venv/bin/pytest tests/test_repository_branches.py tests/test_prompt_manager_branches.py tests/test_catalog_importer.py tests/test_main_entry.py -q
# 182 passed

.venv/bin/ruff check <12 changed Python paths>
.venv/bin/ruff format --check <12 changed Python paths>
# passed

.venv/bin/pyright core/repository/activity.py core/repository/__init__.py \
  core/repository/maintenance.py core/catalog_importer.py \
  core/prompt_manager/generation.py core/prompt_manager/lifecycle.py \
  core/prompt_manager/versioning.py tests/test_repository_branches.py \
  tests/test_prompt_manager_branches.py tests/test_catalog_importer.py \
  tests/test_main_entry.py
# 0 errors

.venv/bin/python <SQLite schema smoke>
# prompt_activity_events columns: id, prompt_id, operation, origin, occurred_at
```

Deferred unchanged: any activity browser, activity-read CLI command, search history, user identity, telemetry, remote transmission, and event payloads containing prompt content.
