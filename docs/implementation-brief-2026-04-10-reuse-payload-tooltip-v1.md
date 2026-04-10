# PromptManager — Implementation Brief

Date: 2026-04-10
Status: ready-for-delegation
Feature: Reuse Payload Tooltip v1
Primary sources:
- `docs/product-boundary-ssot.md`
- `docs/product-backlog-ssot.md`
- `docs/implementation-review-2026-04-06-reuse-polish-v1.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`

## Goal

Implement one bounded **Reuse Payload Tooltip** improvement inside the existing prompt detail view so the operator can immediately understand what `Copy Prompt` and `Open in Workspace` will use, especially when workspace handoff falls back to description because no stored prompt body exists.

## Product intent

This slice strengthens the core loop at:
- inspect,
- reuse.

It should reduce friction between:
- "I found the prompt"
- and
- "I know exactly what the reuse action will do before I click it."

## Scope

### In scope
- one bounded tooltip pass for the existing quick reuse buttons in the shared detail widget
- explain payload semantics using only existing prompt data and existing action behavior
- cover the important states:
  - prompt body available
  - description-only fallback for workspace handoff
  - no reusable payload available
- keep button labels, actions, and layout unchanged
- add focused deterministic regression coverage for the tooltip states

### Out of scope
- new buttons, panels, badges, or visible helper rows
- auto-run behavior or execution changes
- changes to copy/open payload semantics
- changes to prompt actions controller behavior
- analytics, sharing, chains, voice, or retrieval work

## Recommended UX posture

Prefer one quiet explanatory layer on the current buttons.

Suggested v1:
- `Copy Prompt` tooltip explains that it copies the stored prompt body when one exists
- when no prompt body exists, the disabled tooltip explains why copy is unavailable
- `Open in Workspace` tooltip explains whether it will use:
  - the prompt body, or
  - description fallback, or
  - nothing because no reusable payload exists
- keep the existing labels stable

Default recommendation:
- **tooltips only**, not new visible chrome

Reason:
- very small scope
- directly addresses reuse confidence
- aligns with the prior implementation review note about disabled-state clarity

## Data source

Use existing prompt data only.

Relevant source fields:
- `prompt.context`
- `prompt.description`

Do not add new persistence in this slice.

## Likely implementation seam

### UI / behavior
- `gui/widgets/prompt_detail_widget.py`
  - set dynamic tooltips for quick reuse buttons during `display_prompt()`
  - reset tooltips in `clear()`
  - keep enable/disable behavior unchanged

### Tests
- `tests/test_prompt_detail_widget.py`
  - add one focused tooltip-state test for body-backed reuse
  - add one focused tooltip-state test for description-only fallback / copy disabled case

## Happy-path scenario

1. User opens a prompt in detail view.
2. Quick Reuse buttons remain the same.
3. Hovering a button explains what payload it will use.
4. If the prompt has no body, `Copy Prompt` stays disabled and the tooltip explains why.
5. If the prompt has only description, `Open in Workspace` tooltip explains the fallback clearly.

That is enough for v1.

## Acceptance checks

1. `Copy Prompt` tooltip explains body-only copy behavior when a prompt body exists.
2. Disabled `Copy Prompt` explains why copy is unavailable when no prompt body exists.
3. `Open in Workspace` tooltip explains whether it will use prompt body or description fallback.
4. Existing labels, layout, and button behavior remain unchanged.
5. Focused regression coverage protects the tooltip states.

## Suggested test

One or two focused widget tests are enough.

Recommended shape:
- one prompt with `context` and verify both tooltips describe prompt-body reuse
- one description-only prompt and verify:
  - `Copy Prompt` is disabled with explanatory tooltip
  - `Open in Workspace` tooltip explains description fallback

## Rollback

Rollback should be one isolated patch:
- remove the tooltip-setting helper logic from the detail widget
- remove the focused tooltip tests
- leave the current quick reuse actions and payload semantics untouched

## Anti-goals

- do not add a visible reuse helper panel
- do not change copy/open behavior
- do not widen this into controller refactors
- do not add telemetry or analytics
- do not redesign the detail view

## Notes for implementation

- Keep the slice boring.
- Use plain direct wording.
- Explain current behavior, do not invent smarter behavior.
