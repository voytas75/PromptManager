# PromptManager — CLI tag operations v1

**Status:** completed — 2026-09-21
**Owner:** PromptManager Team

## Product decision

Implement the three tag convenience commands retained in the CLI backlog:

```bash
prompt-manager tag-list [--json]
prompt-manager tag-show <tag> [--json]
prompt-manager prompt-tag <uuid-or-exact-name> add|remove <tag> [--dry-run] [--json]
```

## Confirmed seams

- `Prompt.tags` is persisted prompt metadata.
- Existing `prompt-find --tag` is a retrieval filter, not a tag-catalogue or mutation interface.
- Existing `PromptManager.update_prompt(..., origin="cli")` owns persistence, cache/embedding refresh, version behavior, and activity ledger emission for successful changes.
- `_resolve_prompt_reference()` already gives UUID-first and exact-name fallback resolution with existing ambiguity/not-found exit semantics.

## Contract

### Read commands

- `tag-list` aggregates each logical tag at most once per prompt, matching case-insensitively.
- It reports `tag`, `prompt_count`, and `active_prompt_count`; records sort by descending prompt count, then case-insensitive display value.
- `tag-show` matches one exact logical tag case-insensitively and returns compact prompt records sorted by name.
- Empty catalogues and missing tags are successful empty reads, not errors.

### Write command

- `prompt-tag` accepts a non-blank tag and `add` or `remove`.
- Membership is case-insensitive; original existing spelling is retained.
- The operation is idempotent: an already-present add or absent remove produces `changed: false` and does not write.
- `--dry-run` uses a detached prompt copy and does not mutate persistence or the loaded in-memory prompt.
- A changed non-dry-run delegates to `update_prompt` with `commit_message="Tag <action>: <tag>"` and `origin="cli"`.

## Boundaries

- No mass rename/delete/merge, tag taxonomy persistence, provider-generated tags, semantic inference, or GUI change.
- No new storage schema: tags remain prompt metadata.
- No automatic normalization of pre-existing stored tag spellings beyond the logical case-insensitive view.

## Execution order

1. Add a provider-free tag catalogue/mutation helper and unit tests.
2. Add parser, handlers, dispatch, and entrypoint tests for read, dry-run, write, and idempotence.
3. Update CLI docs, changelog, and backlog status.
4. Verify focused tests, full suite, static checks, help, live temporary-database proof, and diff hygiene.

## Done criteria

- All three commands are visible in root help.
- `tag-list` and `tag-show` are deterministic, read-only, and JSON-capable.
- `prompt-tag --dry-run` has no write effect; actual changed operation persists via the lifecycle seam exactly once.
- No provider/model call is introduced.

## Completion update

**Status: completed — 2026-09-21**

Delivered:

- Added `tag-list [--json]`, `tag-show <tag> [--json]`, and `prompt-tag <uuid-or-exact-name> add|remove <tag> [--dry-run] [--json]`.
- Added `core/prompt_tagging.py` for one canonical case-insensitive tag aggregation/membership contract.
- Preserved the existing prompt lifecycle for changed writes, activity ledger, cache, and version behavior.
- Added the narrow `refresh_derived_state=False` lifecycle option used solely by `prompt-tag`, preventing category-generator and embedding-provider work for metadata-only tag edits.
- Added a detached-copy preview, so `--dry-run` does not mutate the loaded object or persistence.

Verified:

```bash
.venv/bin/pytest -q
# 873 passed, 1 skipped
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pyright main.py config models
# all passed
```

Provider-free live proof with isolated SQLite confirmed `prompt-tag ... --dry-run` does not write; the subsequent write persisted `['Ops', 'review']` and emitted a final `prompt_activity_events` row `('updated', 'cli')`, without provider-derived refresh output.

Deferred: mass tag rename/delete/merge, taxonomy persistence, tag generation/inference, GUI changes, and semantic tag search.
