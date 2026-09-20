# PromptManager — Bounded `prompt-show --json` v1

Status: completed
Owner: PromptManager Team

## Goal

Make the normal JSON read model compact and safe for agent/operator use while retaining an explicit full-record escape hatch:

```bash
python -m main --no-gui prompt-show <uuid-or-name> --json
python -m main --no-gui prompt-show <uuid-or-name> --json --full
```

## Confirmed baseline

- `prompt-show --json` emits `Prompt.to_record()`, including `ext4` as the entire embedding vector.
- An isolated runtime probe confirmed `ext4` is a JSON list (32 deterministic dimensions in the probe); production dimensions can be much larger.
- `Prompt.to_record()` is also used by persistence, versioning, cache, GUI, and export paths; changing it would broaden the contract unnecessarily.

## Decision

Create a `prompt-show`-owned compact JSON renderer. Default `--json` omits `ext4` and exposes only embedding presence/dimension metadata. `--json --full` retains the current complete `Prompt.to_record()` output including `ext4`. `--full` without `--json` is rejected at parser validation.

## Execution order

1. Add RED regressions for compact JSON, full JSON, and invalid `--full` without `--json`.
2. Add the parser option and the narrow renderer/validation in the CLI layer.
3. Update help and developer CLI index.
4. Run focused tests, real isolated CLI compact/full smoke, static gates, and diff checks.

## Out of scope

- Changing `Prompt.to_record()`.
- Changing `prompt-find --json` or `prompt-history --json` in this slice.
- Embedding generation, storage, export semantics, provider calls, commit, or push.

## Completion update

Completed 2026-09-20:

- Added a `prompt-show`-owned compact JSON read model that omits `ext4` and reports `embedding.present` plus `embedding.dimensions`.
- Added `prompt-show --json --full` to explicitly emit the complete `Prompt.to_record()` payload, including the full embedding vector.
- Made `prompt-show --full` without `--json` fail at parser validation with exit 2 and a clear requirement message.
- Preserved `Prompt.to_record()` and all storage, versioning, GUI, cache, and export contracts unchanged.
- Updated runtime help and the developer CLI index.

Verification completed:

- focused compact/full/parser regressions: 3 passed;
- Ruff lint/format: passed;
- real isolated CLI smoke after deterministic `reembed`: compact JSON omitted `ext4` and reported a present 32-dimensional embedding; full JSON emitted `ext4` with the same 32 dimensions;
- `git diff --check`: passed.

Deferred unchanged: bounded JSON views for `prompt-find` and `prompt-history`, embedding generation/storage, commit, and push.
