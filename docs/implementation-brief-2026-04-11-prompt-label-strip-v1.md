# PromptManager — Implementation Brief

Date: 2026-04-11
Status: implemented
Feature: Prompt Label Strip v1
Primary sources:
- `docs/product-boundary-ssot.md`
- `docs/product-backlog-ssot.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`
- `docs/implementation-brief-2026-04-10-fence-unwrap-v1.md`

## Goal

Implement one bounded **Prompt Label Strip** improvement inside the existing Quick Capture flow so raw prompt text pasted from chat, notes, or docs does not get stored with one obvious outer wrapper label such as `Prompt:` or `System prompt:` when that label is clearly just noise before the real prompt body.

## Product intent

This slice strengthens the core loop at:
- capture,
- normalize,
- reuse.

It should reduce friction between:
- "I pasted a prompt from somewhere messy"
- and
- "the saved draft body is already usable without manually deleting one obvious top label first."

## Scope

### In scope
- one deterministic cleanup step for quick-capture body text
- strip exactly one obvious outer top label when it appears at the start of the captured body
- support only a tiny bounded first allowlist:
  - `Prompt:`
  - `User prompt:`
  - `System prompt:`
- strip only when the label appears as a top-of-body prefix and real non-empty prompt text clearly follows
- support only these two bounded shapes:
  - `Label: prompt text...`
  - `Label:\nprompt text...`
- preserve the remaining prompt body unchanged apart from removing the outer label line or prefix
- keep the existing draft model, fields, and dialog flow unchanged
- add focused deterministic regression coverage for stripped and non-stripped cases

### Out of scope
- general chat transcript parsing
- multi-line role parsing such as full `User:` / `Assistant:` conversations
- markdown cleanup beyond the already-bounded fence unwrap seam
- AI-generated cleanup or rewriting
- import pipeline expansion
- schema changes
- changes outside the Quick Capture draft conversion seam

## Recommended UX posture

Prefer one quiet cleanup step at save time.

Suggested v1:
- if the captured body starts with one obvious allowed wrapper label and real prompt text follows, strip that label before storing
- if the prefix is ambiguous, incomplete, transcript-like, or looks like real prompt content, keep it unchanged
- do not try to parse broader conversation structure or stacked labels

Default recommendation:
- **remove only one obvious top-level prompt label**, nothing more

Reason:
- very small scope
- common real-world copy/paste noise
- low risk when kept to a tiny allowlist and first-line/top-prefix posture

## Data source

Use only the pasted quick-capture body.

Do not add new persistence in this slice.

## Likely implementation seam

### Draft conversion
- `gui/dialogs/quick_capture.py`
  - add one small helper that strips one obvious top label from the body before prompt creation
  - call it from `QuickCaptureDraft.to_prompt()`
  - keep it local unless another cleanup seam truly needs the same helper

### Tests
- `tests/test_quick_capture_dialog.py`
  - add one happy-path stripped-label test
  - add one ambiguous or non-label case that must remain unchanged
  - optionally add one allowed-variant case such as `System prompt:`

## Happy-path scenario

1. User opens Quick Capture.
2. User pastes text such as:
   `Prompt: Summarize the deployment risks for this release.`
3. User saves draft.
4. Stored prompt body is:
   `Summarize the deployment risks for this release.`

That is enough for v1.

## Acceptance checks

1. Quick Capture can strip one obvious allowed top label before storing the prompt body.
2. The remaining prompt body is preserved.
3. Bodies without an allowed wrapper label remain unchanged.
4. Ambiguous or broader transcript-like input remains unchanged.
5. No new schema or UI surface is introduced.
6. Focused regression coverage protects stripped and non-stripped paths.

## Suggested test

One or two focused tests are enough.

Recommended shape:
- stripped case: `Prompt: ...` becomes stored body without the label
- stripped multiline case: `System prompt:\n...` becomes stored body without the label line
- non-stripped case: text that merely contains `prompt:` later in the body remains unchanged
- non-stripped case: bare `Prompt:` without following body remains unchanged
- non-stripped case: transcript-like `User:` / `Assistant:` style content remains unchanged

## Rollback

Rollback should be one isolated patch:
- remove the label-stripping helper
- remove the focused regression tests
- leave the rest of Quick Capture untouched

## Anti-goals

- do not parse whole chat transcripts
- do not strip multiple stacked labels
- do not support generic `User:` / `Assistant:` / `System:` role parsing in v1
- do not rewrite prompt content
- do not widen into markdown or import cleanup broadly
- do not add settings or heuristics UI
- do not use this slice to refactor unrelated Quick Capture logic

## Notes for implementation

- Keep the slice boring.
- Prefer false negatives over destructive cleanup.
- Restrict v1 to one tiny allowlist and one top-of-body seam.
- Strip at most one outer label.
- If a case is not obviously wrapper noise, keep it as-is.
