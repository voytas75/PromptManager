# Pyright Strict Expansion Roadmap

## Current enforced CI scope
- `pyright main.py config`

## Verified baseline
- local verification on 2026-04-23: `pyright main.py config` → `0 errors, 0 warnings, 0 informations`
- GitHub Actions verification on 2026-04-23: run `24840076043` (`Quality Gates`) succeeded for commit `503e546`
- current gate is intentionally narrow and should stay stable while typed coverage expands incrementally

## Next candidate scope
- `pyright main.py config models`

## Phase backlog
1. `models`
2. `core` (small typed-safe modules first)
3. `core` (remaining modules)
4. `gui`
5. `tests`

## Rules
- never widen CI scope before a local green run for the exact next scope
- widen CI one area at a time
- no blanket suppressions
- no whole-repo strict claims until CI really enforces them
- update `docs/README-DEV.md` whenever enforced scope changes
- after each scope expansion, confirm the corresponding GitHub Actions run is green

## Per-phase checklist
1. choose the smallest next scope worth enforcing
2. run local Pyright for that exact scope
3. fix real type issues without broad `type: ignore` escapes
4. re-run local Pyright until green
5. update CI to the new scope
6. update developer docs to match CI reality
7. push and confirm green GitHub Actions

## Exit criteria for each phase
- local Pyright for the next scope is green
- CI updated to match the verified local scope
- GitHub Actions green after the change
- docs aligned with the enforced scope

## Guardrails
- do not revert to `include: ["."]`
- do not expand straight from `main.py + config` to full repo strict mode
- do not leave documentation ahead of actual CI enforcement
- do not treat test-only local success as sufficient without a matching green GH run
