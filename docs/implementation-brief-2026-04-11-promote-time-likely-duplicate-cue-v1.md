# PromptManager — Implementation Brief

Date: 2026-04-11
Status: delivered and spot-verified
Feature: Promote-time Likely Duplicate Cue v1
Primary sources:
- `docs/product-boundary-ssot.md`
- `docs/product-backlog-ssot.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`

Validation:
- `pytest -q tests/test_draft_promote_dialog.py`
- result: `9 passed`

## Goal

Implement one bounded **promote-time likely-duplicate cue** inside the existing `Promote Draft` advisory flow so operators get a stronger warning when the captured draft appears to duplicate an existing prompt asset.

## Product intent

This slice strengthens the core loop at:
- capture,
- normalize,
- retrieve.

It should improve operator judgment at the exact ingest moment where prompt libraries quietly decay: promoting a draft that is not merely similar, but likely already exists.

The slice must stay advisory.
It must not turn PromptManager into a duplicate-management subsystem.

## Scope

### In scope
- add one bounded deterministic duplicate heuristic for the existing draft-promote advisory list
- surface one stronger visible cue when an existing prompt is likely a duplicate of the current draft
- strengthen summary copy and selected-button wording for the likely-duplicate case
- keep `Promote as New` available
- add focused regression coverage for the bounded behavior

### Out of scope
- auto-merge
- blocking promotion
- new compare screen
- duplicate review dashboard
- duplicate taxonomy/registry
- broad similarity heuristic redesign
- import pipeline redesign

## Recommended UX posture

Prefer one calm but unmistakable warning over score-heavy UI.

Suggested v1:
- if a similar prompt is not only very close, but likely the same prompt after bounded normalization, show:
  - row cue: **`Likely duplicate`**
  - summary copy stronger than the current very-close wording
  - selected action copy: **`Open Likely Duplicate`**
- keep the rest of the advisory list intact
- keep the operator in control

Default recommendation:
- **be stronger in wording, not broader in workflow**

## Likely implementation seam

### Advisory dialog
- `gui/dialogs/draft_promote.py`
  - add one helper that decides whether a candidate is a likely duplicate
  - reuse current advisory rendering path
  - keep existing `Very close match` logic for non-duplicate high-similarity cases

### Normalization posture
Use a small deterministic comparison only.
Recommended first cut:
- normalize draft body and candidate body by:
  - trimming outer whitespace
  - collapsing internal whitespace
- treat bodies as likely duplicate only when the normalized bodies are equal and non-empty

This is intentionally boring.
Do not widen into semantic or fuzzy duplicate detection for v1.

### Tests
- `tests/test_draft_promote_dialog.py`
  - cover visible likely-duplicate row cue
  - cover stronger summary copy
  - cover stronger open-existing button wording
  - cover non-duplicate path staying on current wording
  - cover continue-as-new still working

## Happy-path scenario

1. User quick-captures a prompt.
2. User opens `Promote Draft`.
3. Advisory list finds an existing prompt whose normalized body matches the draft body.
4. Dialog shows a stronger `Likely duplicate` cue.
5. Operator can open the existing prompt or still continue with `Promote as New`.

That is enough for v1.

## Acceptance checks

1. A candidate with the same normalized body as the draft shows a visible `Likely duplicate` cue.
2. The advisory summary copy is stronger for likely-duplicate cases than for ordinary similar-match cases.
3. The open-existing button changes to `Open Likely Duplicate` only for the selected likely-duplicate case.
4. Non-duplicate very-close matches still use the current `Very close match` path.
5. `Promote as New` remains available and behaves as before.
6. Focused regression tests pass.

## Rollback

Rollback should be one isolated patch:
- remove the likely-duplicate helper
- revert the advisory cue and wording changes
- remove the focused regression tests
- leave the rest of the similarity advisory flow untouched

## Anti-goals

- do not auto-merge anything
- do not block promotion
- do not introduce a compare workflow
- do not redesign similarity scoring
- do not widen into a general dedupe system
- do not use this slice to refactor adjacent capture/promote code

## Notes for implementation

- Keep the slice boring.
- Deterministic body equality is enough for v1.
- Stronger operator judgment matters more than heuristic cleverness.
- If equality cannot be established confidently, fall back to the current similarity path.