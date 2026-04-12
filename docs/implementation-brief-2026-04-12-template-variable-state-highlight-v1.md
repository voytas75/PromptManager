# PromptManager — Implementation Brief

Date: 2026-04-12
Status: delivered and spot-verified
Feature: Template Variable State Highlight v1
Primary sources:
- `docs/product-boundary-ssot.md`
- `docs/product-backlog-ssot.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`
- `gui/template_preview.py`

Validation:
- `QT_QPA_PLATFORM=offscreen .venv/bin/pytest -q tests/test_template_preview_widget.py`
- result: `10 passed`

## Goal

Implement one bounded **variable-state highlight** improvement inside `Template Preview` so an operator can immediately see which specific variable inputs are currently:
- missing,
- invalid,
- or ready.

The desired experience is simple:

> see preview status → immediately know which exact field to fix

## Product intent

This slice strengthens the core loop at:
- fill,
- validate,
- render,
- run.

The product already shows:
- detected template variables,
- bounded missing-variable summaries,
- preview-ready vs blocked status.

This slice should add the next practical answer directly at the field seam:

> which input needs attention right now?

This should stay a **bounded field-state cue**, not a form builder or validation dashboard.

## Scope

### In scope
- show one quiet per-field state cue in `Template Preview`
- distinguish at least:
  - missing fields,
  - invalid fields,
  - neutral/ready fields
- derive the state from the existing preview + schema validation path
- keep the visuals subtle and local to the existing variable inputs
- add focused regression coverage for state transitions

### Out of scope
- autofill helpers
- field auto-focus or jump-to-error behavior
- new validation panels or summaries
- sample payload generation
- copy-missing-keys actions
- schema redesign
- execution flow changes
- persistence/schema changes

## Recommended UX posture

Prefer subtle field-level feedback over louder workflow mechanics.

Suggested v1 behavior:
- missing variable label/input gets one quiet attention cue
- schema-invalid field gets one distinct invalid cue
- valid/provided fields stay neutral or lightly ready
- when state changes, the field styling updates without adding popups or extra rows

Recommended rules:
- do not add buttons or helper actions
- do not add text-heavy explanations per field
- do not duplicate the global status summary at every field
- use just enough contrast to guide correction without making the preview look like a form wizard

Default recommendation:
- **show where attention is needed, without turning editing into a workflow**

## Likely implementation seam

### Preview widget
- `gui/template_preview.py`
  - extend the current variable-input creation path to retain label/input references per variable
  - extend `_update_preview()` or an adjacent helper to apply state styling based on:
    - missing variables,
    - schema-invalid top-level fields,
    - otherwise neutral/ready state
  - keep layout and behavior unchanged

### Tests
- likely seam:
  - `tests/test_template_preview_widget.py`
- cover at least:
  - missing field gets attention state
  - invalid field gets invalid state
  - state resets when input becomes valid

## Happy-path scenarios

### Scenario A: missing input
1. User opens a valid template preview.
2. One required field is empty.
3. That specific field shows a quiet missing-state cue.
4. User knows exactly where to type next.

### Scenario B: invalid provided input
1. User provides a value that fails schema validation.
2. That specific field shows an invalid-state cue.
3. Other unrelated fields stay quiet.

### Scenario C: corrected input
1. User fixes the missing or invalid field.
2. The field returns to neutral/ready styling.
3. No extra workflow is introduced.

That is enough for v1.

## Acceptance checks

1. Missing variables get a visible but quiet field-level cue.
2. Schema-invalid fields get a distinct field-level cue.
3. Valid fields do not keep stale warning/error styling.
4. No new panel, wizard, or execution behavior is introduced.
5. Focused regression tests protect the bounded behavior.

## Rollback

Rollback should be one isolated patch:
- remove variable field-state styling
- remove focused regression tests
- leave existing preview rendering, status summaries, and run gating untouched

## Anti-goals

- do not add autofill or correction suggestions
- do not add click-to-fix or jump navigation
- do not add per-field help text blocks
- do not redesign the preview into a validation dashboard
- do not change template execution semantics

## Notes for implementation

- Keep the slice boring.
- The status line answers *what* is wrong; this slice should answer *where*.
- If implementation starts adding new controls, panels, or explanation text, the slice is drifting.
- Implemented in the existing `gui/template_preview.py` variable-input seam with subtle label/editor styling for missing and schema-invalid fields plus neutral reset behavior for corrected inputs.
