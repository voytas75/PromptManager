# PromptManager — Implementation Brief

Date: 2026-04-11
Status: delivered and spot-verified
Feature: Duplicate Reason Cue v1
Primary sources:
- `docs/product-boundary-ssot.md`
- `docs/product-backlog-ssot.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`
- `docs/implementation-brief-2026-04-11-promote-time-likely-duplicate-cue-v1.md`
- `docs/implementation-brief-2026-04-11-similarity-strength-cue-v1.md`

Validation:
- `pytest -q tests/test_draft_promote_dialog.py`
- result: `10 passed`

## Goal

Implement one bounded **Duplicate Reason Cue** improvement inside the existing `Promote Draft` advisory flow so the operator can understand *why* the currently selected existing prompt is being flagged as `Likely duplicate` or `Very close match` without opening a second screen or reading raw similarity details.

## Product intent

This slice strengthens the core loop at:
- capture,
- normalize,
- inspect.

It should improve operator judgment at the exact ingest moment where PromptManager already warns about near-duplicates, by making the warning slightly more explainable without widening the workflow.

The slice must stay quiet and local.
It must not turn Draft Promote into a compare tool, scoring UI, or duplicate-management subsystem.

## Scope

### In scope
- add one compact visible reason cue for the currently selected advisory match only
- support the two current bounded advisory states:
  - `Likely duplicate`
  - `Very close match`
- derive the reason only from signals already used in the current dialog
- keep the current advisory list, actions, and dialog structure intact
- add focused regression coverage for visible and absent reason states

### Out of scope
- raw numeric score display in the visible row
- adding reason text to every row in the advisory list
- compare or diff screen
- additional dialog or expandable detail panel
- duplicate blocking, merge, or replace-existing flows
- new heuristic families beyond current bounded signals
- schema or storage changes
- retrieval ranking changes

## Recommended UX posture

Prefer one short human-readable reason for the selected match over richer analytics.

Suggested v1:
- if the selected match is `Likely duplicate`, show one quiet reason such as:
  - `Same normalized body`
- if the selected match is `Very close match`, show one quiet reason such as:
  - `Very similar prompt body`
- render the reason in existing selected-state wording or summary text, not as a new always-visible row segment for every item
- keep the reason short, deterministic, and subordinate to the main state cue
- do not expose raw thresholds or scoring math in the visible row

Default recommendation:
- **make the selected warning more legible, not the whole list more analytical**

## Likely implementation seam

### Advisory dialog
- `gui/dialogs/draft_promote.py`
  - extend the existing selected-state summary with one bounded reason helper
  - reuse current likely-duplicate and very-close-match state detection
  - keep reason generation local unless another surface truly needs it

### Reason posture
Use only current bounded signals.

Recommended first cut:
- for `Likely duplicate`:
  - show `Same normalized body`
- for `Very close match`:
  - show `Very similar prompt body`
- for weaker or ordinary matches:
  - show no extra reason cue
- only show the reason for the currently selected match

This is intentionally boring.
Do not widen into field-by-field compare explanations or fuzzy natural-language rationale.

### Tests
- `tests/test_draft_promote_dialog.py`
  - cover visible reason cue for selected likely duplicates
  - cover visible reason cue for selected very close matches
  - cover absence of reason cue for weaker matches
  - cover non-selected rows staying compact and deterministic

## Happy-path scenario

1. User opens `Promote Draft` for a captured prompt.
2. Advisory list finds one existing prompt that is either a likely duplicate or a very close match.
3. The list stays compact and uses the current bounded row cues.
4. When the operator selects a strong match, the dialog shows one short reason that explains the warning in plain language.
5. Operator can judge the warning faster without opening the tooltip or existing prompt first.

That is enough for v1.

## Acceptance checks

1. A selected likely duplicate can show one compact reason cue derived from current duplicate logic.
2. A selected very close match can show one compact reason cue derived from current similarity-strength logic.
3. Ordinary or weaker matches keep the current clean row without extra filler.
4. Non-selected rows stay compact and do not gain extra reason segments.
5. Existing dialog actions and summary flow remain behaviorally unchanged except for the bounded selected-state explanation.
6. No new dialog, panel, schema, or compare workflow is introduced.
7. Focused regression tests pass.

## Rollback

Rollback should be one isolated patch:
- remove the reason helper
- remove the bounded selected-state reason cue
- remove the focused regression tests
- leave the existing likely-duplicate and very-close-match behavior untouched

## Anti-goals

- do not add a compare screen
- do not add explainability paragraphs or expanded metadata panels
- do not show raw similarity thresholds in the row
- do not add reason text to every advisory row
- do not redesign the Draft Promote dialog
- do not widen into a general dedupe workflow
- do not use this slice to refactor adjacent capture/promote logic

## Notes for implementation

- Keep the slice boring.
- One short reason is enough.
- Explanation should reduce hesitation, not add analysis overhead.
- Prefer selected-state explanation over row-level verbosity.
- If no strong current-state reason exists, keep the list clean.
