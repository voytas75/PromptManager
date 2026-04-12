# PromptManager — Implementation Brief

Date: 2026-04-12
Status: delivered and spot-verified
Feature: Edit Dialog Promote Draft Shortcut v1
Primary sources:
- `docs/product-boundary-ssot.md`
- `docs/product-backlog-ssot.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`
- `docs/next-slice-brief-2026-04-04-draft-promote-normalize.md`
- GitHub issue: `#8` `Edit prompt which is as draft`

Validation:
- `pytest -q tests/test_prompt_editor_flow.py tests/test_prompt_dialog_refinement.py`
- result: `11 passed`

## Goal

Implement one bounded shortcut from **`Edit Prompt`** into the existing **`Promote Draft`** workflow so an operator editing a draft does not need to back out to the detail view just to switch into promotion.

The desired experience is simple:

> open draft in editor → realize this should be promoted → use one visible shortcut → continue in the existing promote flow

## Product intent

This slice strengthens the same core loop that already exists:
- capture,
- normalize,
- inspect,
- reuse.

`Promote Draft` already exists and belongs in the core usage path.
The gap is only that the edit dialog does not offer a short handoff into that flow when the current prompt is still a draft.

This should stay a **routing improvement**, not a lifecycle redesign.

## Scope

### In scope
- show one visible **`Promote Draft…`** action inside `Edit Prompt` only when the edited prompt is still a draft
- route that action into the existing promote flow
- keep the existing promote dialog, duplicate cues, and save/update path as they are
- define one boring rule for unsaved changes before handoff
- add focused regression coverage for the visible shortcut and handoff behavior

### Out of scope
- redesigning the editor dialog
- merging edit and promote into one giant form
- introducing a new prompt lifecycle system
- adding new draft states or schema fields
- duplicate-management changes
- compare view changes
- new AI suggestions during handoff
- broad refactor of editor flow plumbing unless strictly required for the shortcut

## Recommended UX posture

Prefer one visible escape hatch over a richer editor redesign.

Suggested v1:
- if the prompt is a draft, show **`Promote Draft…`** alongside existing edit actions
- if there are no unsaved changes, open the existing promote flow immediately
- if there are unsaved changes, ask one short question:
  - **`Apply changes and continue to Promote Draft?`**
- on confirm:
  - apply the current editor changes through the existing update path
  - then open the existing promote dialog on the refreshed prompt
- on cancel:
  - stay in the editor with no hidden state changes

Default recommendation:
- **apply first, then promote**

This is safer and more boring than trying to transfer unsaved form state directly into the promote dialog.

## Likely implementation seam

### Edit dialog
- `gui/dialogs/prompt_editor/dialog.py`
  - expose one draft-only `Promote Draft…` action near existing edit controls
  - keep it hidden for non-draft prompts
  - if needed, emit one bounded signal or result flag for promote handoff

### Editor flow
- `gui/prompt_editor_flow.py`
  - coordinate the handoff from edit dialog to existing `promote_draft_prompt(...)`
  - reuse current save/apply behavior rather than inventing a second temporary-state path
  - after applying unsaved changes, reopen or continue with the stored prompt version

### Draft detection
- reuse the existing draft check already used by the detail view / draft promotion flow
- do not create a second definition of draft state

### Tests
- likely seam:
  - `tests/test_prompt_editor_dialog.py`
  - and/or `tests/test_prompt_editor_flow.py`
- cover at least:
  - shortcut visible for draft prompts
  - shortcut hidden for non-draft prompts
  - unsaved-change confirm path applies and opens promote
  - cancel path leaves editor state untouched

## Happy-path scenarios

### Scenario A: clean editor state
1. User opens `Edit Prompt` for a draft prompt.
2. User decides this should be promoted, not just generically edited.
3. User clicks `Promote Draft…`.
4. Existing promote dialog opens immediately.
5. User completes promotion through the current bounded promote flow.

### Scenario B: unsaved edits exist
1. User opens `Edit Prompt` for a draft prompt.
2. User changes title/tags/description.
3. User clicks `Promote Draft…`.
4. The editor asks: `Apply changes and continue to Promote Draft?`
5. On confirm, the current changes are saved through the existing path.
6. The existing promote dialog opens using the refreshed prompt.

That is enough for v1.

## Acceptance checks

1. Draft prompts opened in `Edit Prompt` expose one visible **`Promote Draft…`** action.
2. Non-draft prompts do not expose that action.
3. When no unsaved changes exist, the shortcut opens the existing promote flow directly.
4. When unsaved changes exist, the user gets one short confirm step before handoff.
5. Confirming the handoff applies current changes through the existing save/update path before promotion.
6. Cancelling the handoff leaves the editor open and does not silently change stored data.
7. Existing promote behavior remains unchanged after handoff.
8. Focused regression tests pass.

## Rollback

Rollback should be one isolated patch:
- remove the `Promote Draft…` shortcut from the editor dialog
- remove the bounded handoff logic from the editor flow
- remove focused regression tests
- leave existing detail-view promotion intact

## Anti-goals

- do not turn the editor into a draft dashboard
- do not duplicate the full promote form inside the editor
- do not widen into draft queues, bulk triage, or lifecycle management
- do not add schema churn for transient handoff state
- do not use this slice to redesign generic edit/save/apply behavior
- do not change duplicate cues or selection logic in `Promote Draft`

## Notes for implementation

- Keep the slice boring.
- This is a shortcut, not a new workflow.
- Reuse the existing promote seam as-is.
- Prefer one confirm step over clever unsaved-state plumbing.
- If the handoff logic starts demanding broad dialog refactors, the slice is drifting.
