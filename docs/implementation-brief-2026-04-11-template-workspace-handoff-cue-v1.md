# PromptManager — Implementation Brief

Date: 2026-04-11
Status: delivered and spot-verified
Feature: Template Workspace Handoff Cue v1
Primary sources:
- `docs/product-boundary-ssot.md`
- `docs/product-backlog-ssot.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`
- `docs/implementation-brief-2026-04-11-template-variable-cue-v1.md`
- `docs/next-slice-brief-2026-04-04-quick-reuse-handoff-v1.md`

## Goal

Implement one bounded **Template Workspace Handoff Cue** improvement inside the existing prompt detail view so the operator can immediately understand that `Open in Workspace` is the correct next reuse path when a prompt still requires template variables.

## Product intent

This slice strengthens the core loop at:
- inspect,
- reuse.

It should reduce friction between:
- "I can see this prompt requires variables"
- and
- "I know the next safe reuse action without guessing what `Open in Workspace` will do."

The slice must stay quiet and local.
It must not turn the detail view into a template editor, preview surface, or second workflow.

## Scope

### In scope
- make the existing `Open in Workspace` tooltip template-aware when the current prompt body contains detected template variables
- reuse the same bounded variable-detection seam already used by `Template Variable Cue v1`
- keep the current button labels unchanged:
  - `Copy Prompt`
  - `Open in Workspace`
- keep current action behavior unchanged
- show a short bounded variable summary in the tooltip with a hard v1 limit of at most two explicit variable names, then a count suffix when needed, such as:
  - `Open the prompt in Workspace to fill variables: customer_name, region.`
  - `Open the prompt in Workspace to fill variables: customer_name, region +2.`
- preserve the current non-template tooltip behavior for plain prompts
- add focused regression coverage for template-aware and non-template tooltip states

### Out of scope
- changing `Open in Workspace` action semantics
- changing `Copy Prompt` semantics
- adding variable editing in detail view
- adding inline preview or validation in detail view
- auto-opening template preview modes
- new labels, new buttons, or new panels
- schema or persistence changes
- changes outside the shared detail-view reuse seam

## Recommended UX posture

Prefer one quiet handoff cue over richer template workflow guidance.

Suggested v1:
- if the prompt body contains detected template variables, keep the existing visible cue and also make the `Open in Workspace` tooltip explain that workspace is the handoff path for filling variables
- if the prompt body is plain text, keep the existing tooltip behavior unchanged
- keep the tooltip short, operator-facing, and subordinate to the current reuse controls
- do not expose a full variable list when many variables exist

Default recommendation:
- **clarify the next action**, not the whole template workflow

## Data source

Use only the current prompt body (`Prompt.context`) and the existing template-variable extraction seam.

Prefer reusing:
- `core.templating.TemplateRenderer.extract_variables(...)`
- the existing bounded variable-summary posture from `Template Variable Cue v1`

Do not add new persistence in this slice.

## Likely implementation seam

### Detail view reuse tooltips
- `gui/widgets/prompt_detail_widget.py`
  - extend the existing reuse-tooltip helper
  - when prompt body contains template variables, make `Open in Workspace` tooltip explain that variables can be filled there
  - keep `Copy Prompt` tooltip behavior unchanged unless a tiny wording alignment is strictly required for consistency
  - prefer sharing a local bounded variable-summary helper instead of duplicating slightly different summary rules

### Tests
- `tests/test_prompt_detail_widget.py`
  - cover template-aware `Open in Workspace` tooltip text
  - cover bounded summary when many variables exist
  - cover ordinary plain-prompt tooltip behavior staying unchanged

## Happy-path scenario

1. User opens a prompt in the detail view.
2. The prompt body contains Jinja-style variables such as `{{ customer_name }}` and `{{ region }}`.
3. The detail view already shows the existing template-variable cue.
4. The operator hovers `Open in Workspace`.
5. Tooltip explains that workspace is the place to fill the required variables.
6. The operator understands the next reuse step without trial and error.

That is enough for v1.

## Acceptance checks

1. A prompt with detected template variables shows a template-aware `Open in Workspace` tooltip.
2. The template-aware tooltip stays bounded, with at most two explicit variable names before `+N`.
3. Plain prompts keep the current `Open in Workspace` tooltip behavior.
4. Button labels remain unchanged.
5. `Copy Prompt` action behavior remains unchanged.
6. `Open in Workspace` remains a non-executing handoff.
7. No new panel, editor, validation surface, or workspace redesign is introduced.
8. Focused regression coverage protects template-aware and ordinary tooltip behavior.

## Suggested test

Two or three focused tests are enough.

Recommended shape:
- prompt with `{{ customer_name }}` and `{{ region }}` gives `Open in Workspace` a template-aware tooltip
- prompt with many variables gives a bounded tooltip summary plus `+N`
- plain prompt keeps the existing ordinary tooltip text

## Rollback

Rollback should be one isolated patch:
- remove the template-aware tooltip branch from the detail widget
- remove any small helper added only for the bounded variable-summary reuse
- remove the focused regression tests
- leave the existing template-variable cue and reuse actions untouched

## Anti-goals

- do not add variable editing to the detail view
- do not change button labels
- do not auto-open or auto-run anything
- do not redesign workspace handoff behavior
- do not widen into template-preview or validation UX
- do not touch Quick Capture, retrieval list, or fork flows in this slice

## Notes for implementation

- Keep the slice boring.
- Reuse the existing template-awareness seam instead of inventing a second detection path.
- Tooltip clarity is enough; do not add new visible chrome.
- If the variable list is long, prefer bounded summary over completeness.

## Delivery note

Delivered in:
- `gui/widgets/prompt_detail_widget.py`
- `tests/test_prompt_detail_widget.py`

Focused validation:
- `QT_QPA_PLATFORM=offscreen .venv/bin/pytest -q tests/test_prompt_detail_widget.py`
