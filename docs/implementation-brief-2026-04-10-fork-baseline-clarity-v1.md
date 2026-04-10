# PromptManager — Implementation Brief

Date: 2026-04-10
Status: ready-for-delegation
Feature: Fork Baseline Clarity v1
Primary sources:
- `docs/product-boundary-ssot.md`
- `docs/product-backlog-ssot.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`

## Goal

Implement one bounded **Fork Baseline Clarity** improvement so a newly created fork clearly starts as its own lineage baseline and the existing lineage summary uses human-readable prompt names instead of raw UUIDs where possible.

## Product intent

This slice strengthens the core loop at:
- inspect,
- refine,
- reuse.

It should reduce friction between:
- "I forked this prompt or I am inspecting a forked prompt"
- and
- "I can understand whether this is its own branch and where it came from without decoding low-signal lineage metadata."

## Scope

### In scope
- reset a newly created fork to a fresh visible version baseline
- improve the existing lineage summary so parent lineage uses prompt names instead of raw IDs when the related prompt can be resolved
- keep child-fork summary compact and bounded
- add focused deterministic regression coverage for:
  - fork baseline version behavior
  - human-readable parent lineage summary

### Out of scope
- cross-prompt diff UI
- multi-hop ancestry graphs
- rationale capture for why a fork exists
- merge/review workflows
- schema changes
- broad redesign of detail or history surfaces

## Recommended UX posture

Prefer one quiet clarity pass on existing fork/version signals.

Suggested v1:
- a new fork starts visibly as `v1`
- lineage summary says `Forked from <prompt name>` when resolvable
- if a name cannot be resolved safely, use the current safe fallback rather than widening the flow
- keep child summary compact, for example count-based rather than full graph rendering

Default recommendation:
- **make the current lineage readable**, not richer

Reason:
- small scope
- directly addresses current confusion in fork/version surfaces
- creates a cleaner base before any future compare UX

## Data source

Use only existing prompt/version/fork records.

Relevant seams:
- prompt visible version field
- recorded parent/child fork links
- existing prompt retrieval by id

Do not add new persistence in this slice.

## Likely implementation seam

### Fork creation
- `core/prompt_manager/versioning.py`
  - `fork_prompt(...)`
  - ensure the forked prompt starts as a fresh visible baseline rather than inheriting the parent version label

### Lineage summary
- `gui/workspace_history_controller.py`
  - `_update_prompt_lineage_summary(...)`
  - resolve parent names from existing ids and render a compact human-readable summary

### Tests
- `tests/test_prompt_manager_branches.py`
- add one focused controller test if needed for lineage-summary formatting

## Happy-path scenario

1. User forks a prompt.
2. New prompt appears as its own branch with visible baseline `v1`.
3. User opens the fork in detail view.
4. Lineage summary says `Forked from <source prompt name>` instead of showing a raw UUID.
5. User can distinguish this fork from the parent faster.

That is enough for v1.

## Acceptance checks

1. A new fork starts with a fresh visible version baseline.
2. Existing fork lineage summary uses human-readable parent prompt names when available.
3. Child-fork summary remains compact.
4. No new schema or new lineage UI surface is introduced.
5. Focused regression coverage protects the bounded behavior.

## Suggested test

One or two focused tests are enough.

Recommended shape:
- fork a prompt with an inherited version and verify the new fork starts at `v1`
- drive lineage summary for a forked prompt and verify the rendered text uses the parent prompt name rather than raw UUID text

## Rollback

Rollback should be one isolated patch:
- revert the fork baseline reset
- revert the lineage summary formatting improvement
- remove the focused regression tests
- leave the rest of versioning/fork flow untouched

## Anti-goals

- do not add diffing
- do not add ancestry graphs
- do not add fork rationale metadata
- do not widen into merge/review workflow
- do not redesign the detail widget

## Notes for implementation

- Keep the slice boring.
- Human-readable lineage beats more lineage.
- Fresh fork baseline should reflect branch clarity, not history erasure.
