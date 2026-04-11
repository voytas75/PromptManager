# PromptManager — Implementation Brief

Date: 2026-04-11
Status: delivered and spot-verified
Feature: Similarity Strength Cue v1
Primary sources:
- `docs/product-boundary-ssot.md`
- `docs/product-backlog-ssot.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`
- `docs/implementation-brief-2026-04-10-similar-match-preview-v1.md`

Validation:
- `pytest -q tests/test_draft_promote_dialog.py`
- result: `8 passed`

## Goal

Implement one bounded **Similarity Strength Cue** improvement inside the existing **Draft Promote** advisory list so operators can notice very close existing prompt matches faster without exposing raw similarity scores in the visible row or widening into a duplicate-management flow.

## Product intent

This slice strengthens the core loop at:
- normalize,
- inspect,
- refine.

It should reduce friction between:
- "I see similar prompts exist"
- and
- "I can tell quickly when one of them is probably very close to what I am about to promote"

without turning Draft Promote into a scoring-heavy review screen.

## Scope

### In scope
- derive one quiet qualitative cue from the existing `similarity` value already attached to similar prompt matches
- render the cue only in the existing Draft Promote visible advisory row
- show the cue only for very close matches
- keep preview text, tooltip metadata, and promote flow otherwise unchanged except for one stronger open-existing button label when the selected match is very close
- add focused regression coverage for visible and absent states

### Out of scope
- visible raw numeric similarity scores in the row
- new thresholds UI or settings
- ranking changes
- second review screen
- duplicate blocking, merge, or replace-existing flows
- schema changes

## Recommended UX posture

Prefer one small confidence cue over richer scoring.

Suggested v1:
- show `Very close match` only when the existing similarity signal is strong enough to matter
- nudge the operator further by renaming the open-existing action to `Open Very Close Match` when that state is selected
- keep weaker matches on the current clean label path with the calmer `Open Existing Match` wording
- let tooltip metadata continue carrying the raw score for deeper inspection

Default recommendation:
- **make very close matches easier to notice, not more analytical**

## Likely implementation seam

### Advisory label rendering
- `gui/dialogs/draft_promote.py`
  - extend the current visible row label builder with one bounded qualitative cue helper
  - keep the helper local unless another surface truly needs it

### Tests
- `tests/test_draft_promote_dialog.py`
- add focused coverage for:
  - visible `Very close match` cue for strong similarity
  - no visible cue for weaker matches
  - threshold-bound visible state and stronger summary copy when a very close match exists

## Happy-path scenario

1. User opens Promote Draft.
2. Similar prompts are listed as today.
3. A very close existing prompt shows one visible cue such as `Very close match`.
4. The summary copy also nudges the operator more clearly when a very close existing prompt is present.
5. The operator notices the likely duplicate faster before deciding whether to open the existing prompt or continue promoting as new.

That is enough for v1.

## Acceptance checks

1. A similar prompt with strong similarity shows one visible `Very close match` cue in the advisory row.
2. Weaker similar prompts keep the current clean row without the cue.
3. Existing preview text still works as before.
4. Existing actions remain behaviorally unchanged, with only bounded button-copy strengthening for very close selected matches.
5. Tooltip metadata remains unchanged.
5. No new dialog, schema, ranking change, or duplicate-management flow is introduced.
6. Focused regression coverage protects the bounded behavior.

## Rollback

Rollback should be one isolated patch:
- remove the qualitative similarity-strength helper
- remove the visible cue from advisory rows
- remove the focused regression tests
- leave Similar Match Preview v1 and the current Draft Promote advisory flow intact

## Anti-goals

- do not show raw numeric scores in the row
- do not redesign the Draft Promote dialog
- do not add duplicate blocking or auto-merge behavior
- do not widen into a separate compare or dedupe workflow
- do not touch retrieval ranking or semantic search logic

## Notes for implementation

- Keep the slice boring.
- One visible confidence cue is enough.
- Stronger button copy is acceptable when it helps the operator make the safer ingest choice faster.
- Quiet wording cleanup is worth doing when it removes awkward phrasing from the decision moment.
- Tooltip can stay more detailed than the row.
