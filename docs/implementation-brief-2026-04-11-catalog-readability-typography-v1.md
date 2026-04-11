# PromptManager — Implementation Brief

Date: 2026-04-11
Status: delivered and spot-verified
Feature: Catalog Readability Typography v1
Primary sources:
- `docs/product-boundary-ssot.md`
- `docs/product-backlog-ssot.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`

Validation:
- `pytest -q tests/test_prompt_list_model.py tests/test_prompt_detail_widget.py`
- result: `15 passed`

## Goal

Implement one bounded **Catalog Readability Typography** improvement so the main prompt catalog surfaces are easier to read by default without introducing a global zoom system, a full typography settings matrix, or broad theming work.

## Product intent

This slice strengthens the core loop at:
- retrieve,
- inspect.

It should reduce friction between:
- "I found a prompt"
- and
- "I can comfortably read the key distinguishing and inspection text without squinting"

while staying inside the existing prompt list/detail flow.

## Scope

### In scope
- improve default readability in the main prompt list preview row
- improve default readability in the shared prompt detail surface
- keep the current layout, actions, and information structure intact
- add focused regression coverage for the bounded typography behavior

### Out of scope
- global UI zoom
- whole-app font settings or app-wide typography redesign
- broad stylesheet cleanup
- prompt chains, analytics, maintenance dialogs, or workspace output/chat typography
- new panels or layout redesign

## Recommended UX posture

Prefer one quiet default-readability pass over a settings-heavy solution.

Suggested v1:
- stop shrinking the second-line prompt-list preview text below the row's base font size
- slightly enlarge the key text labels in the shared prompt detail widget
- preserve hierarchy through weight, color, and spacing rather than tiny type

Default recommendation:
- **make the catalog calmer and more readable, not more configurable**

## Likely implementation seam

### Main prompt list
- `gui/prompt_list_delegate.py`
  - remove the forced one-point preview shrink
  - keep the existing two-line structure and muted preview color

### Shared detail surface
- `gui/widgets/prompt_detail_widget.py`
  - apply one bounded readability font pass to the title and detail text labels
  - keep all existing cues, labels, and actions intact

### Tests
- `tests/test_prompt_list_model.py`
- `tests/test_prompt_detail_widget.py`
- add focused coverage for:
  - preview font no longer shrinking below the row base font
  - detail widget title/body text becoming more readable by default

## Happy-path scenario

1. User opens the main catalog.
2. Prompt-list preview lines remain readable instead of looking visibly undersized.
3. User opens a prompt in the detail pane.
4. The title and key inspection text are easier to read without changing workflow or layout.

That is enough for v1.

## Acceptance checks

1. Prompt-list preview text no longer uses a smaller point size than the row base font.
2. Shared prompt-detail title text is larger than the widget base font.
3. Shared prompt-detail body/inspection text is at least slightly larger than the widget base font.
4. No new settings surface, zoom control, or layout redesign is introduced.
5. Focused regression coverage protects the bounded behavior.

## Rollback

Rollback should be one isolated patch:
- revert the prompt-list preview font sizing change
- revert the prompt-detail readability font pass
- remove the focused regression tests
- leave retrieval preview, inspection cues, and layout behavior untouched

## Anti-goals

- do not introduce app-wide zoom
- do not widen into global theme architecture
- do not redesign the detail widget structure
- do not touch workspace output/chat typography in this slice
- do not sweep every dialog in the app just because fonts exist there too

## Notes for implementation

- Keep the slice boring.
- Better defaults beat more knobs for v1.
- Readability hierarchy should come from structure and emphasis, not tiny preview text.
