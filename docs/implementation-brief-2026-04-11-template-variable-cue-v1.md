# PromptManager — Implementation Brief

Date: 2026-04-11
Status: delivered and spot-verified
Feature: Template Variable Cue v1
Primary sources:
- `docs/product-boundary-ssot.md`
- `docs/product-backlog-ssot.md`
- `docs/session-restart-brief-2026-04-06-slice-guidelines.md`
- `docs/implementation-brief-2026-04-10-context-lead-usage-cue-v1.md`

## Goal

Implement one bounded **Template Variable Cue** improvement inside the existing prompt detail view so the operator can immediately see when a prompt body requires template variables before it is directly reusable.

## Product intent

This slice strengthens the core loop at:
- inspect,
- reuse.

It should reduce friction between:
- "I opened this prompt"
- and
- "I know right away whether I can copy/use it directly or whether it still expects variables."

The slice must stay quiet and local.
It must not turn the detail view into a template dashboard.

## Scope

### In scope
- add one compact variable-requirement cue to the existing shared detail view
- derive the cue only from the current prompt body using the existing templating seam
- show the cue only when real template variables are detected
- display a short bounded summary with a hard v1 limit of at most two explicit variable names, then a count suffix when needed, such as:
  - `Requires variables: customer_name, region`
  - `Requires variables: customer_name, region +2`
- keep the current detail layout, actions, and metadata flow unchanged
- add focused regression coverage for visible, absent, and bounded-summary cases

### Out of scope
- variable editors in detail view
- schema or validation UI in detail view
- template execution or preview changes
- support for every possible templating dialect
- new persistence fields or schema changes
- workspace or workbench redesign
- changes outside the shared detail-view seam

## Recommended UX posture

Prefer one quiet inspect cue over richer template tooling.

Suggested v1:
- if the prompt body contains detected template variables, show one compact cue in the existing detail information flow
- render it as a bounded inspect cue, not inside `When to use`, not inside metadata dump, and not as a new panel
- if the prompt body is plain text, show nothing
- keep the summary short and operator-facing rather than rendering a full variable panel
- in v1, show at most two explicit variable names, then a `+N` suffix for the rest

Default recommendation:
- **show just enough to warn about required variables, not enough to create a second workflow**

## Data source

Use only the current prompt body (`Prompt.context`) and the existing template-variable extraction seam.

Prefer reusing:
- `core.templating.TemplateRenderer.extract_variables(...)`

Do not add new persistence in this slice.

## Likely implementation seam

### Detail view
- `gui/widgets/prompt_detail_widget.py`
  - add one local helper that resolves a bounded variable-requirement cue from `Prompt.context`
  - render the cue inside the existing detail information flow without adding a new panel
  - keep it separate from the existing `When to use` cue and separate from metadata rendering

### Tests
- `tests/test_prompt_detail_widget.py`
  - cover visible cue when template variables exist
  - cover no cue for plain prompts
  - cover bounded summary when multiple variables exist

## Happy-path scenario

1. User opens a prompt in the detail view.
2. The prompt body contains Jinja-style variables such as `{{ customer_name }}` and `{{ region }}`.
3. The detail view shows one compact cue such as:
   `Requires variables: customer_name, region`
4. The operator immediately understands that direct reuse needs variable substitution.

That is enough for v1.

## Acceptance checks

1. The detail view can show one compact template-variable cue when variables are detected from the prompt body.
2. Plain prompts without detected variables show no extra cue.
3. The cue stays short and bounded when many variables exist, with a hard v1 limit of at most two explicit variable names before `+N`.
4. The cue is rendered in the existing detail information flow, not in `When to use`, metadata dump, or a new panel.
5. No new panel, editor, schema surface, or execution flow is introduced.
6. Focused regression coverage protects visible, absent, and bounded-summary behavior.

## Suggested test

One or two focused tests are enough.

Recommended shape:
- prompt with `{{ customer_name }}` and `{{ region }}` shows a visible cue
- plain prompt shows no cue
- prompt with many variables shows a truncated summary plus count suffix

## Rollback

Rollback should be one isolated patch:
- remove the variable-cue helper from the detail widget
- remove the cue rendering
- remove the focused regression tests
- leave the rest of detail view and templating behavior untouched

## Anti-goals

- do not add variable editing to the detail view
- do not add validation status or schema rendering here
- do not redesign the detail widget structure
- do not place this cue inside `When to use` or metadata dump
- do not widen into workbench/template-preview workflow
- do not add support for unrelated templating syntaxes in this slice

## Notes for implementation

- Keep the slice boring.
- Prefer the existing Jinja-focused extraction seam rather than inventing another parser.
- Show no cue at all when no variable requirement exists.
- Favor one compact warning over completeness.

## Delivery note

Delivered in:
- `gui/widgets/prompt_detail_widget.py`
- `tests/test_prompt_detail_widget.py`

Focused validation:
- `QT_QPA_PLATFORM=offscreen .venv/bin/pytest -q tests/test_prompt_detail_widget.py`
- result: `13 passed`
