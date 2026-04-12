# PromptManager — Implementation Brief

Date: 2026-04-12
Status: ready
Feature: Single-Turn User Prefix Strip v1
Primary sources:
- `docs/product-boundary-ssot.md`
- `docs/product-backlog-ssot.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`
- `docs/analysis-brief-2026-04-11-quick-capture-real-input-review-v1.md`
- `docs/analysis-review-2026-04-11-quick-capture-real-input-review-v1.md`
- `docs/implementation-brief-2026-04-11-prompt-label-strip-v1.md`

## Goal

Implement one bounded **Single-Turn User Prefix Strip** improvement inside the existing Quick Capture flow so a pasted prompt that is clearly wrapped as a single `User:` turn can be stored without that one outer wrapper.

The intended v1 behavior is narrow:

> `User: <real prompt body>` -> store `<real prompt body>`

This is **not** a generic transcript parser.
It is one small cleanup rule for one plausible wrapper-noise case.

## Product intent

This slice strengthens the core loop at:
- capture,
- normalize,
- reuse.

It should reduce friction between:
- "I pasted one prompt copied from a chat turn"
- and
- "the saved draft body is already usable without manually deleting `User:` first."

## Why this is narrower than a transcript-cleanup slice

Recent Quick Capture analysis explicitly argued against reopening broad cleanup without strong evidence.
That means this slice must stay **smaller** than a generic role-strip or transcript-normalization pass.

So the safe posture is:
- support only `User:`
- support only one outer prefix
- support only clearly single-turn shapes
- leave `Assistant:` and `System:` untouched
- leave transcript-like input untouched

Default recommendation:
- **prefer false negatives over destructive cleanup**

## Scope

### In scope
- one deterministic Quick Capture cleanup step
- strip exactly one outer `User:` prefix when it appears at the very start of the captured body
- support only these two bounded shapes:
  - `User: prompt text...`
  - `User:\nprompt text...`
- strip only when real non-empty prompt text clearly follows
- preserve the remaining prompt body unchanged apart from removing the outer prefix or prefix line
- keep the existing draft model, fields, and dialog flow unchanged
- add focused deterministic regression coverage for stripped and non-stripped cases

### Out of scope
- `Assistant:` prefix stripping
- `System:` prefix stripping
- full chat transcript parsing
- multi-turn conversation cleanup
- stacked role-label stripping
- markdown cleanup beyond existing bounded seams
- import pipeline expansion
- schema changes
- changes outside the Quick Capture draft conversion seam

## Recommended UX posture

Prefer one quiet cleanup step at save time.

Suggested v1:
- if the captured body starts with one obvious `User:` prefix and real prompt text follows, strip it before storing
- if the prefix is incomplete, transcript-like, ambiguous, or likely part of real content, keep it unchanged
- do not infer conversation structure
- do not try to be smart about multiple role turns

Reason:
- still close to real capture friction
- much lower risk than a broader role-strip rule
- stays consistent with the product's current anti-drift posture

## Data source

Use only the pasted Quick Capture body.

Do not add new persistence in this slice.

## Likely implementation seam

### Draft conversion
- `gui/dialogs/quick_capture.py`
  - add one small helper that strips one outer `User:` prefix from the body before prompt creation
  - call it from `QuickCaptureDraft.to_prompt()`
  - keep it local unless another existing cleanup helper already fits naturally

### Tests
- `tests/test_quick_capture_dialog.py`
  - add focused stripped and non-stripped cases for `User:` only

## Happy-path scenario

1. User opens Quick Capture.
2. User pastes:
   `User: Summarize the deployment risks for this release.`
3. User saves draft.
4. Stored prompt body is:
   `Summarize the deployment risks for this release.`

That is enough for v1.

## Acceptance checks

1. Quick Capture strips `User: Summarize the deployment risks for this release.` to `Summarize the deployment risks for this release.`
2. Quick Capture strips `User:\nSummarize the deployment risks for this release.` to `Summarize the deployment risks for this release.`
3. Input `User:` with no real body remains unchanged.
4. Input `User:\nAssistant: ok, here is the summary` remains unchanged.
5. Input `User: first line\nAssistant: second line` remains unchanged.
6. Input `System: Keep the answer terse.` remains unchanged.
7. Input `Assistant: Rewrite this prompt for executives.` remains unchanged.
8. No new UI, settings, schema, or non-Quick-Capture behavior is introduced.
9. Focused regression coverage protects stripped and unchanged paths.

## Suggested test set

A small deterministic set is enough:
- stripped inline `User: ...`
- stripped multiline `User:\n...`
- unchanged bare `User:` with no real body
- unchanged transcript-like `User:\nAssistant: ...`
- unchanged mixed multi-turn `User: first line\nAssistant: second line`
- unchanged `System: ...`
- unchanged `Assistant: ...`

## Rollback

Rollback should be one isolated patch:
- remove the `User:` stripping helper
- remove the focused regression tests
- leave the rest of Quick Capture untouched

## Anti-goals

- do not parse whole chat transcripts
- do not strip `Assistant:` or `System:` in v1
- do not strip multiple stacked role labels
- do not infer speaker intent or conversation structure
- do not rewrite prompt content beyond removing one outer `User:` prefix
- do not widen into import cleanup, markdown cleanup, or generic transcript normalization
- do not use this slice to refactor unrelated Quick Capture logic

## Notes for implementation

- Keep the slice boring.
- Prefer false negatives over destructive cleanup.
- Strip at most one outer prefix.
- If the case is not obviously one wrapped user turn, keep it as-is.
- If implementation pressure starts adding broader role handling, the slice is drifting.
