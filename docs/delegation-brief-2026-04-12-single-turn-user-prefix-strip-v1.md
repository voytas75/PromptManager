# PromptManager — Delegation Brief

Date: 2026-04-12
Status: ready-for-delegation
Target: dev / Codex
Feature: Single-Turn User Prefix Strip v1
Primary brief:
- `docs/implementation-brief-2026-04-12-single-turn-user-prefix-strip-v1.md`

## Mission

Implement one **small Single-Turn User Prefix Strip v1** slice.

Goal:
- let Quick Capture remove one obvious outer `User:` wrapper when the captured input is clearly just one wrapped user turn
- keep the slice inside the existing Quick Capture cleanup seam
- avoid widening into transcript parsing

This is **not** a generic role-strip feature.
It is one bounded cleanup rule for one narrow input shape.

## Required posture

Keep this slice:
- small,
- boring,
- local in effect,
- deterministic,
- free of adjacent cleanup.

If the work starts to look like transcript parsing, import normalization, or broader role handling, stop and simplify.

## Source anchors

Read first:
- `docs/implementation-brief-2026-04-12-single-turn-user-prefix-strip-v1.md`
- `docs/product-ssot.md`
- `docs/analysis-brief-2026-04-11-quick-capture-real-input-review-v1.md`
- `docs/analysis-review-2026-04-11-quick-capture-real-input-review-v1.md`

Likely implementation seam:
- `gui/dialogs/quick_capture.py`

Likely tests to extend:
- `tests/test_quick_capture_dialog.py`

## Deliverable

Ship exactly one bounded patch that does all of the following:
1. adds one `User:`-only cleanup rule in Quick Capture draft conversion
2. strips only one outer prefix from clearly single-turn input
3. leaves transcript-like, ambiguous, and other role-prefixed input unchanged
4. adds focused deterministic regression coverage

## Do now

### 1. Add one narrow cleanup rule
Implement the smallest possible helper for:
- `User: prompt text...`
- `User:\nprompt text...`

Only strip when real non-empty body clearly follows.

### 2. Keep all other role handling out
Do not strip:
- `Assistant:`
- `System:`
- multi-turn role text
- stacked role prefixes

### 3. Add focused tests
Cover at least:
- stripped inline `User: ...`
- stripped multiline `User:\n...`
- unchanged bare `User:`
- unchanged transcript-like `User:\nAssistant: ...`
- unchanged `System: ...`
- unchanged `Assistant: ...`

## Acceptance checks

1. Input `User: Summarize the deployment risks for this release.` stores as `Summarize the deployment risks for this release.`
2. Input `User:\nSummarize the deployment risks for this release.` stores as `Summarize the deployment risks for this release.`
3. Input `User:` with no real body remains unchanged.
4. Input `User:\nAssistant: ok, here is the summary` remains unchanged.
5. Input `User: first line\nAssistant: second line` remains unchanged.
6. Input `System: Keep the answer terse.` remains unchanged.
7. Input `Assistant: Rewrite this prompt for executives.` remains unchanged.
8. No new UI, settings, schema, or non-Quick-Capture behavior is introduced.
9. Focused regression coverage protects stripped and unchanged paths.

## Validation

Run focused validation only.
Prefer the narrowest reasonable test set proving the slice.

## Required final report

Return:
1. what changed
2. exact files changed
3. validation run and results
4. whether the slice stayed bounded
5. whether there was any temptation toward broader transcript parsing and how it was avoided

## Anti-goals

- do not parse full transcripts
- do not strip `Assistant:` or `System:` in v1
- do not strip multiple stacked role labels
- do not infer conversation structure
- do not widen into import cleanup, markdown cleanup, or generic transcript normalization
- do not touch unrelated Quick Capture or prompt-storage seams

## Rollback

Rollback should be one isolated revert of:
- the `User:` stripping helper
- the focused regression tests
