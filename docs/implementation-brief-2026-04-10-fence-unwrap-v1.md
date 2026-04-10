# PromptManager — Implementation Brief

Date: 2026-04-10
Status: ready-for-delegation
Feature: Fence Unwrap v1
Primary sources:
- `docs/product-boundary-ssot.md`
- `docs/product-backlog-ssot.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`

## Goal

Implement one bounded **Fence Unwrap** improvement inside the existing Quick Capture flow so raw prompt text pasted from chat, markdown, or notes does not get stored with an unnecessary outer markdown code fence when the entire captured body is clearly wrapped inside one fenced block.

## Product intent

This slice strengthens the core loop at:
- capture,
- normalize,
- reuse.

It should reduce friction between:
- "I pasted a raw prompt from somewhere messy"
- and
- "the saved draft body is already usable without manual cleanup of outer wrapper noise."

## Scope

### In scope
- one deterministic normalization step for quick-capture body text
- unwrap exactly one outer fenced markdown block when the whole pasted body is enclosed by it
- preserve the inner prompt text unchanged apart from removing the outer wrapper
- support common fence forms such as:
  - ```
  - ```text
  - ```markdown
- keep the existing draft model, fields, and dialog flow unchanged
- focused deterministic regression coverage for wrapped and non-wrapped cases

### Out of scope
- AI-generated cleanup
- multi-block parsing or broad markdown normalization
- semantic prompt cleanup or rewriting
- import pipeline expansion
- schema changes
- changes outside the Quick Capture draft conversion seam

## Recommended UX posture

Prefer one quiet normalization step at save time.

Suggested v1:
- if the entire body is wrapped in one obvious outer code fence, unwrap it before storing
- if the body is not clearly one outer fenced block, keep it unchanged
- do not try to be clever about partial fences, multiple fenced sections, or mixed prose + blocks

Default recommendation:
- **remove only obvious outer wrapper noise**, nothing more

Reason:
- very small scope
- high practical value for copied prompts from chat or docs
- low risk of changing legitimate prompt content when bounded properly

## Data source

Use only the pasted quick-capture body.

Do not add new persistence in this slice.

## Likely implementation seam

### Draft conversion
- `gui/dialogs/quick_capture.py`
  - add one small helper that unwraps a single outer fence before prompt creation
  - call it from `QuickCaptureDraft.to_prompt()`

### Tests
- `tests/test_quick_capture_dialog.py`
  - add one wrapped-body happy-path test
  - add one non-wrapped or ambiguous-body test

## Happy-path scenario

1. User opens Quick Capture.
2. User pastes a prompt copied from chat wrapped in:
   ```text
   ...prompt...
   ```
3. User saves draft.
4. Stored prompt body contains the inner prompt text, not the outer markdown fence.

That is enough for v1.

## Acceptance checks

1. Quick Capture unwraps one obvious outer fenced block before storing the prompt body.
2. Inner prompt text is preserved.
3. Bodies that are not clearly one outer fenced block remain unchanged.
4. No new schema or UI surface is introduced.
5. Focused regression coverage protects wrapped and non-wrapped cases.

## Suggested test

One or two focused tests are enough.

Recommended shape:
- wrapped case: ```text ... ``` becomes stored inner body only
- non-wrapped case: mixed prose plus fenced text remains unchanged

## Rollback

Rollback should be one isolated patch:
- remove the fence-unwrapping helper
- remove the focused regression tests
- leave the rest of Quick Capture untouched

## Anti-goals

- do not normalize the entire markdown universe
- do not rewrite prompt content
- do not parse multiple blocks
- do not add import-mode settings
- do not widen this into a cleanup framework

## Notes for implementation

- Keep the slice boring.
- Prefer false negatives over destructive cleanup.
- Only unwrap when the outer fence is obvious and complete.
