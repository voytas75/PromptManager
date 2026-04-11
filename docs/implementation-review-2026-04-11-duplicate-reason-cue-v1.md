# PromptManager — Implementation Review

Date: 2026-04-11
Target: `Duplicate Reason Cue v1`
Expected source: `docs/implementation-brief-2026-04-11-duplicate-reason-cue-v1.md`
Reviewer: main

## Verdict

**Aligned.**

The delivered change matches the bounded brief closely. It adds one selected-state explanation inside the existing `Promote Draft` advisory flow, keeps the visible list compact, and avoids widening into compare, scoring, or duplicate-management behavior.

## What matches

### 1. The change stays in the intended seam
The implementation remains local to:
- `gui/dialogs/draft_promote.py`
- `tests/test_draft_promote_dialog.py`

That matches the brief's bounded advisory-dialog posture.

### 2. The reason cue is selected-state only
The dialog now appends one short reason to the advisory summary for the currently selected strong match:
- `Reason: Same normalized body.` for likely duplicates
- `Reason: Very similar prompt body.` for very close matches

This matches the safer brief revision and avoids row-level verbosity across the whole list.

### 3. Existing row labels remain compact
The advisory rows still show only the existing bounded cues (`Likely duplicate`, `Very close match`) plus the existing preview signal where applicable.
No extra reason fragments were added to every row.

### 4. The current promote/open behavior remains intact
The existing button-label logic for:
- `Open Existing Match`
- `Open Very Close Match`
- `Open Likely Duplicate`

remains in place, and the new change does not alter the underlying actions.

### 5. Focused regression coverage exists
Focused tests cover:
- likely-duplicate selected reason
- very-close selected reason
- weak-match absence of reason
- non-selected rows staying clean

Validation passed:
- `pytest -q tests/test_draft_promote_dialog.py`
- result: `10 passed`

## What is missing

Nothing material relative to the brief.

A richer explanation such as field-by-field reasoning was not added, which is correct for this slice and should stay out of scope.

## What drifted / widened

No meaningful scope drift is visible.

One small behavioral nuance is worth noting: the base advisory summary still reflects whether any strong match exists in the current list, even when the currently selected row is weaker. The newly added reason cue, however, remains correctly limited to the selected strong match only. This is still within the bounded posture and does not widen the workflow.

## What is unverified

### 1. Live readability at longer copy lengths
This review did not perform a visual GUI pass to judge whether the extra summary line wraps cleanly in all window sizes or font settings.

### 2. Multi-match real-world judgment quality
The tests confirm bounded behavior, but this review did not assess whether operators actually make faster promote decisions with mixed strong/weak advisory lists.

## Recommended next action

Treat `Duplicate Reason Cue v1` as delivered.

Do not expand it into richer explainability.
If a follow-up is ever needed, it should be a separate tiny UX slice only if real operator friction remains visible in live use.

## Sources reviewed

- `docs/implementation-brief-2026-04-11-duplicate-reason-cue-v1.md`
- `gui/dialogs/draft_promote.py`
- `tests/test_draft_promote_dialog.py`
- focused test result: `pytest -q tests/test_draft_promote_dialog.py` → `10 passed`
