# PromptManager — Implementation Brief

Date: 2026-04-11
Status: delivered and spot-verified
Feature: Blockquote Unwrap v1
Primary sources:
- `docs/product-boundary-ssot.md`
- `docs/product-backlog-ssot.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`
- `docs/implementation-brief-2026-04-10-fence-unwrap-v1.md`
- `docs/implementation-brief-2026-04-11-prompt-label-strip-v1.md`

## Goal

Implement one bounded **Blockquote Unwrap** improvement inside the existing Quick Capture flow so raw prompt text pasted from chat, notes, or markdown does not get stored with one obvious outer markdown blockquote wrapper when the captured body is clearly just quoted prompt content.

## Product intent

This slice strengthens the core loop at:
- capture,
- normalize,
- reuse.

It should reduce friction between:
- "I pasted a quoted prompt from somewhere messy"
- and
- "the saved draft body is already usable without manually deleting one outer `>` wrapper first."

## Scope

### In scope
- one deterministic cleanup step for quick-capture body text
- unwrap exactly one obvious outer markdown blockquote wrapper when the whole captured body is clearly quoted content
- support only the bounded v1 shape where every non-empty line starts with `>`
- preserve the inner prompt text unchanged apart from removing the outer quote marker and one following optional space per line
- keep the existing draft model, fields, and dialog flow unchanged
- add focused deterministic regression coverage for unwrapped and non-unwrapped cases

### Out of scope
- generic markdown normalization beyond one obvious outer blockquote wrapper
- nested or repeated unwrap passes
- transcript parsing
- AI-generated cleanup or rewriting
- import pipeline expansion
- schema changes
- changes outside the Quick Capture draft conversion seam

## Recommended UX posture

Prefer one quiet cleanup step at save time.

Suggested v1:
- if the captured body is entirely wrapped as one obvious outer blockquote, unwrap it before storing
- if the body contains mixed quoted and non-quoted lines, keep it unchanged
- if unwrapping would leave no real prompt text, keep it unchanged
- do not try to interpret partial quoting, nested quoting, or conversation structure

Default recommendation:
- **remove only one obvious outer quote wrapper**, nothing more

Reason:
- very small scope
- common copy/paste noise from chats and notes
- low risk when bounded to all-lines-quoted input only

## Data source

Use only the pasted quick-capture body.

Do not add new persistence in this slice.

## Likely implementation seam

### Draft conversion
- `gui/dialogs/quick_capture.py`
  - add one small helper that unwraps one obvious outer blockquote wrapper before prompt creation
  - call it from `QuickCaptureDraft.to_prompt()` alongside the existing bounded cleanup steps
  - keep it local unless another cleanup seam truly needs the same helper

### Tests
- `tests/test_quick_capture_dialog.py`
  - add one happy-path quoted-body test
  - add one multiline quoted-body test
  - add one mixed or ambiguous case that must remain unchanged

## Happy-path scenario

1. User opens Quick Capture.
2. User pastes text such as:
   `> Summarize deployment risks for this release.`
3. User saves draft.
4. Stored prompt body is:
   `Summarize deployment risks for this release.`

That is enough for v1.

## Acceptance checks

1. Quick Capture unwraps one obvious outer blockquote wrapper before storing the prompt body.
2. The remaining prompt body is preserved apart from removing the wrapper markers.
3. Bodies with mixed quoted and non-quoted content remain unchanged.
4. Wrapper-only or effectively empty results remain unchanged.
5. No new schema or UI surface is introduced.
6. Focused regression coverage protects unwrapped and non-unwrapped paths.

## Suggested test

One or two focused tests are enough.

Recommended shape:
- wrapped case: `> ...` becomes stored body without the quote marker
- wrapped multiline case: each quoted line is unwrapped once
- non-wrapped case: mixed prose plus quoted lines remains unchanged
- non-wrapped case: wrapper-only or empty-after-unwrap input remains unchanged

## Rollback

Rollback should be one isolated patch:
- remove the blockquote-unwrapping helper
- remove its call from `QuickCaptureDraft.to_prompt()`
- remove the focused regression tests
- leave the rest of Quick Capture untouched

## Anti-goals

- do not build a generic markdown cleanup framework
- do not unwrap nested blockquotes repeatedly
- do not parse conversations or transcript roles
- do not rewrite prompt content
- do not add settings or heuristics UI
- do not use this slice to refactor unrelated Quick Capture logic

## Notes for implementation

- Keep the slice boring.
- Prefer false negatives over destructive cleanup.
- Only unwrap when every non-empty line is clearly quoted.
- Remove one outer wrapper layer only.
- If a case is not obviously wrapper noise, keep it as-is.

## Delivery note

Delivered in:
- `gui/dialogs/quick_capture.py`
- `tests/test_quick_capture_dialog.py`

Focused validation:
- `QT_QPA_PLATFORM=offscreen .venv/bin/pytest -q tests/test_quick_capture_dialog.py`
- result: `14 passed`
