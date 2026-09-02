# PromptManager — Implementation Brief

Date: 2026-04-12
Status: delivered and spot-verified
Feature: Template Missing Variables Cue v1
Primary sources:
- `docs/product-ssot.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`
- `gui/template_preview.py`

Validation:
- `pytest -q tests/test_template_preview_widget.py`
- result: `7 passed`

## Goal

Implement one bounded **missing variables cue** inside `Template Preview` so an operator can immediately see which template variables are still required before a valid render/run is possible.

The desired experience is simple:

> open template preview → fill some variables → immediately see what is still missing

## Product intent

This slice strengthens the core loop at:
- inspect,
- fill,
- render,
- run.

The product already exposes template-variable awareness in the shared detail flow.
This slice should move the next practical answer into the actual preview/render seam:

> what exactly is still missing right now?

This should stay a **bounded validation cue**, not a wizard or autofill workflow.

## Scope

### In scope
- show one compact missing-variables cue in `Template Preview` only when render input is incomplete
- derive the cue from the existing template render/validation path
- keep the cue short, quiet, and subordinate to the current preview/status messaging
- add focused regression coverage for visible and hidden states

### Out of scope
- autofill helpers
- sample JSON generation
- copy-missing-keys actions
- template onboarding or wizard redesign
- new panels, dialogs, or tabs
- execution flow changes
- persistence/schema changes
- detail-view changes

## Recommended UX posture

Prefer one short, actionable cue over richer validation UI.

Suggested v1 wording:
- `Missing variables: customer_name`
- `Missing variables: customer_name, region`
- `Missing variables: customer_name, region +2`

Recommended rules:
- show the cue only when template syntax is valid and one or more required variables are missing
- cap visible names at 2, then append `+N`
- hide the cue when render input is complete
- hide the cue when a parse/syntax error is already the primary issue
- keep the cue near the existing preview/status surface instead of adding a separate validation panel

Default recommendation:
- **show what blocks render, without turning preview into a form builder**

## Likely implementation seam

### Preview widget
- `gui/template_preview.py`
  - extend the current validation/render state update path
  - reuse the existing renderer result or missing-variable signal
  - keep layout changes minimal and local

### Tests
- likely seam:
  - `tests/test_template_preview_widget.py`
- cover at least:
  - cue visible when variables are missing
  - bounded summary for 3+ missing names
  - cue hidden when render succeeds
  - cue hidden when parse error is active

## Happy-path scenarios

### Scenario A: partially filled template
1. User opens a valid template with required variables.
2. User fills some but not all variables.
3. Preview shows one compact cue such as `Missing variables: customer_name, region`.
4. User immediately knows what still blocks a clean render.

### Scenario B: fully renderable template
1. User provides all required variables.
2. Preview renders normally.
3. No missing-variable cue is shown.

### Scenario C: syntax issue instead of missing inputs
1. Template has parse/syntax errors.
2. Existing syntax error remains the primary signal.
3. Missing-variable cue does not add noise on top.

That is enough for v1.

## Acceptance checks

1. A valid template with incomplete variables shows one compact `Missing variables: ...` cue.
2. The cue uses a bounded summary of variable names.
3. A fully renderable template does not show the cue.
4. Parse/syntax errors remain primary and do not also show a missing-variable cue.
5. No new panel, wizard, or execution behavior is introduced.
6. Focused regression tests protect the bounded behavior.

## Rollback

Rollback should be one isolated patch:
- remove the missing-variables cue from `Template Preview`
- remove the focused regression tests
- leave existing preview rendering, syntax handling, and run gating untouched

## Anti-goals

- do not add autofill actions
- do not add a copy-missing-keys workflow
- do not add sample payload generation
- do not redesign the preview surface into a validation dashboard
- do not change template execution semantics

## Notes for implementation

- Keep the slice boring.
- The operator needs one clear blocker summary, not a helper system.
- If the UI starts asking for buttons, payload generation, or form scaffolding, the slice is drifting.
