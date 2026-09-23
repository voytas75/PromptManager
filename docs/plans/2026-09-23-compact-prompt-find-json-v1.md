# PromptManager — Compact `prompt-find --json` v1

Status: completed
Owner: PromptManager Team

## Goal

Align structured semantic-search results with the established bounded prompt-detail JSON contract:

```bash
python -m main --no-gui prompt-find "research prompt" --json
python -m main --no-gui prompt-find "research prompt" --json --full
```

Default JSON must remain compact and omit raw embedding vectors. The existing `--full` option is the sole explicit complete-record opt-in.

## Confirmed baseline

- `prompt-show --json` already omits `ext4`, reports bounded embedding metadata, and makes `--json --full` the complete-record path.
- Before this slice, `prompt-find --json` serialized `Prompt.to_record()` directly, exposing `ext4`; a real Windows-host CLI probe returned a 3,072-value vector for one result.
- `Prompt.to_record()` is a persisted-record contract used beyond CLI reads. It must not change.

## Decision

Add `--full` to `prompt-find`, valid only with `--json`. Reuse one CLI-local JSON payload renderer for `prompt-show` and `prompt-find` so their compact/full embedding behavior cannot drift.

## Scope

- `cli/parser.py`: add `prompt-find --full` and require `--json`.
- `cli/commands.py`: emit compact prompt records by default in `prompt-find --json`; retain full records only with `--json --full`.
- Tests, CLI developer index, changelog, status, package/lock version metadata, and this ledger.

## Out of scope

- Semantic ranking, filtering, embedding generation/storage, persistence schema, provider calls, GUI behavior, and changes to `Prompt.to_record()`.
- Commit, tag, push, or publish.

## Acceptance criteria

1. `prompt-find --json` excludes `ext4` and reports `embedding.present` and `embedding.dimensions`.
2. `prompt-find --json --full` includes the exact vector.
3. `prompt-find --full` fails in parser validation because `--json` is absent.
4. `prompt-show` preserves its existing compact/full behavior.
5. Targeted tests, Ruff, strict Pyright on changed Python files, relevant CLI smoke, `uv lock --check`, and `git diff --check` pass.

## Completion update

Completed 2026-09-23:

- Added `prompt-find --full`, valid only with `--json`.
- Reused one CLI-local prompt JSON renderer so `prompt-show` and `prompt-find` now share the compact/full embedding contract.
- Default `prompt-find --json` omits `ext4` and emits `embedding.present` plus `embedding.dimensions`; `prompt-find --json --full` retains the persisted `ext4` vector.
- Advanced package and lock metadata from `0.23.1` to `0.23.2`.
- Updated the developer CLI index, changelog, and compact status ledger.

Verification completed:

- focused contract regressions: 3 passed;
- complete `tests/test_main_entry.py`: 109 passed;
- `ruff check .`, `ruff format --check .`, `uv lock --check`, and `git diff --check`: passed;
- full configured strict Pyright: 0 errors;
- real repository CLI smoke with a stored 3,072-dimensional embedding: compact JSON omitted `ext4` and reported 3,072 dimensions, full JSON returned the 3,072-element vector, and invalid `--full` exited 2 with the expected message;
- `uv sync --locked --all-extras` rebuilt the editable package, and `importlib.metadata.version("prompt-manager")` reported `0.23.2`.

Unchanged: semantic ranking, filters, persistence, provider calls, embedding generation/storage, GUI behavior, `Prompt.to_record()`, tagging, and publishing.